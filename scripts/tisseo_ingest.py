"""
tisseo_ingest.py
================
Ingest a Tisséo GTFS feed into optimised Parquet / GeoParquet files.

Usage:
    python scripts/tisseo_ingest.py
    python scripts/tisseo_ingest.py --gtfs /path/to/gtfs.zip --out /path/to/out/

Output (written to --out, default: data/parquet/):
    stops.geoparquet       stop locations as WGS-84 Point geometries
    routes.parquet         transit lines (bus, metro, tram …)
    trips.parquet          trips enriched with calendar data
    stop_times.parquet     timetable (largest file)
    shapes.geoparquet      route shapes as LineString geometries
    calendar_dates.parquet service exceptions (public holidays …)
    _meta.json             ingestion summary (row counts, duration)
"""

import argparse
import json
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GTFS_ZIP, PARQUET

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import LineString, Point


# ── GTFS column schemas (subset of useful columns) ────────────────────────────

GTFS_DTYPES = {
    "stops": {
        "stop_id":            str,
        "stop_name":          str,
        "stop_lat":           float,
        "stop_lon":           float,
        "location_type":      "Int8",
        "parent_station":     str,
        "wheelchair_boarding": "Int8",
    },
    "routes": {
        "route_id":         str,
        "agency_id":        str,
        "route_short_name": str,
        "route_long_name":  str,
        "route_type":       "Int16",
        "route_color":      str,
        "route_text_color": str,
    },
    "trips": {
        "route_id":      str,
        "service_id":    str,
        "trip_id":       str,
        "trip_headsign": str,
        "direction_id":  "Int8",
        "shape_id":      str,
    },
    "stop_times": {
        "trip_id":        str,
        "arrival_time":   str,   # "HH:MM:SS" — may exceed 24h (GTFS spec)
        "departure_time": str,
        "stop_id":        str,
        "stop_sequence":  "Int32",
        "pickup_type":    "Int8",
        "drop_off_type":  "Int8",
    },
    "calendar": {
        "service_id": str,
        "monday":     "Int8",
        "tuesday":    "Int8",
        "wednesday":  "Int8",
        "thursday":   "Int8",
        "friday":     "Int8",
        "saturday":   "Int8",
        "sunday":     "Int8",
        "start_date": str,
        "end_date":   str,
    },
    "calendar_dates": {
        "service_id":     str,
        "date":           str,
        "exception_type": "Int8",
    },
    "shapes": {
        "shape_id":           str,
        "shape_pt_lat":       float,
        "shape_pt_lon":       float,
        "shape_pt_sequence":  "Int32",
        "shape_dist_traveled": float,
    },
}

ROUTE_TYPE_LABELS = {
    0: "Tram",
    1: "Metro",
    2: "Train",
    3: "Bus",
    4: "Ferry",
    5: "Cable car",
    6: "Aerial lift",
    7: "Funicular",
    11: "Trolleybus",
    12: "Monorail",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def read_gtfs_file(zf: zipfile.ZipFile, filename: str, dtypes: dict) -> pd.DataFrame:
    """Read a CSV file from a GTFS ZIP with explicit column types."""
    # Some Tisséo exports place files inside a sub-folder
    candidates = [n for n in zf.namelist() if n.endswith(filename)]
    if not candidates:
        log(f"  ⚠  {filename} not found in ZIP — skipped")
        return pd.DataFrame()
    name = candidates[0]
    log(f"  → reading {name}")
    with zf.open(name) as f:
        df = pd.read_csv(
            f,
            dtype={k: v for k, v in dtypes.items() if v not in ("Int8", "Int16", "Int32")},
            low_memory=False,
            encoding="utf-8-sig",   # handle optional BOM
        )
    # Apply nullable integer types separately
    for col, t in dtypes.items():
        if col in df.columns and t in ("Int8", "Int16", "Int32"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(t)
    df = df.dropna(axis=1, how="all")
    log(f"     {len(df):,} rows, {len(df.columns)} columns")
    return df


def gtfs_time_to_seconds(series: pd.Series) -> pd.Series:
    """
    Convert "HH:MM:SS" strings to seconds since midnight (Int32).
    GTFS allows hours > 24 for overnight trips (e.g. 25:30:00).
    """
    parts = series.str.split(":", expand=True)
    if parts.shape[1] < 3:
        return pd.Series(pd.NA, index=series.index, dtype="Int32")
    h = pd.to_numeric(parts[0], errors="coerce")
    m = pd.to_numeric(parts[1], errors="coerce")
    s = pd.to_numeric(parts[2], errors="coerce")
    return (h * 3600 + m * 60 + s).astype("Int32")


def route_type_label(series: pd.Series) -> pd.Series:
    return series.map(ROUTE_TYPE_LABELS).fillna("Unknown")


# ── Data builders ─────────────────────────────────────────────────────────────

def build_stops(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert stops DataFrame to GeoDataFrame with Point geometry."""
    df = df.copy()
    df["geometry"] = df.apply(
        lambda r: Point(r["stop_lon"], r["stop_lat"])
        if pd.notna(r.get("stop_lon")) and pd.notna(r.get("stop_lat"))
        else None,
        axis=1,
    )
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def build_shapes(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert shapes DataFrame to GeoDataFrame with LineString geometry per shape_id.
    The shapes.txt file lists ordered points; we group them into lines.
    """
    if df.empty:
        return gpd.GeoDataFrame()

    df = df.sort_values(["shape_id", "shape_pt_sequence"])

    def points_to_line(grp):
        coords = list(zip(grp["shape_pt_lon"], grp["shape_pt_lat"]))
        return LineString(coords) if len(coords) >= 2 else None

    geom = df.groupby("shape_id").apply(points_to_line)
    return gpd.GeoDataFrame(
        geom.reset_index().rename(columns={0: "geometry"}),
        geometry="geometry",
        crs="EPSG:4326",
    )


def build_stop_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich stop_times with:
      arrival_sec / departure_sec  — seconds since midnight (Int32)
      hour_of_day                  — departure hour 0–29 (Int8)
      is_overnight                 — 1 if hour > 24
    """
    df = df.copy()
    df["arrival_sec"]   = gtfs_time_to_seconds(df["arrival_time"])
    df["departure_sec"] = gtfs_time_to_seconds(df["departure_time"])
    df["hour_of_day"]   = (df["departure_sec"] // 3600).astype("Int8")
    df["is_overnight"]  = (df["hour_of_day"] >= 24).astype("Int8")
    return df


def build_trips_with_calendar(trips: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Merge trips with calendar to add operating-days information per trip."""
    if calendar.empty:
        return trips
    merged = trips.merge(calendar, on="service_id", how="left")

    day_cols = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    day_abbr = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def days_label(row):
        active = [abbr for col, abbr in zip(day_cols, day_abbr) if row.get(col, 0) == 1]
        return "-".join(active) if active else "none"

    if all(c in merged.columns for c in day_cols):
        merged["days_label"] = merged.apply(days_label, axis=1)

    return merged


# ── Parquet export ────────────────────────────────────────────────────────────

def save_parquet(df: pd.DataFrame, path: Path, row_group_size: int = 100_000):
    """Write a DataFrame to Parquet with Zstd compression."""
    if df.empty:
        log(f"  ⚠  Empty DataFrame — {path.name} not written")
        return
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table, path,
        compression="zstd",
        row_group_size=row_group_size,
        write_statistics=True,
    )
    size_mb = path.stat().st_size / 1_048_576
    log(f"  ✓ {path.name} ({len(df):,} rows, {size_mb:.1f} MB)")


def save_geoparquet(gdf: gpd.GeoDataFrame, path: Path):
    """Write a GeoDataFrame to GeoParquet with Zstd compression."""
    if gdf.empty:
        log(f"  ⚠  Empty GeoDataFrame — {path.name} not written")
        return
    gdf.to_parquet(path, compression="zstd", index=False)
    size_mb = path.stat().st_size / 1_048_576
    log(f"  ✓ {path.name} ({len(gdf):,} rows, {size_mb:.1f} MB)")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def ingest(gtfs_path: Path, out_dir: Path):
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": str(gtfs_path),
        "generated_at": datetime.now().isoformat(),
        "files": {},
    }

    log(f"Opening ZIP: {gtfs_path}")
    with zipfile.ZipFile(gtfs_path, "r") as zf:
        log(f"  Files in archive: {', '.join(zf.namelist())}")

        # ── 1. Stops ──────────────────────────────────────────────────────────
        log("\n[1/6] Stops")
        stops_gdf = build_stops(read_gtfs_file(zf, "stops.txt", GTFS_DTYPES["stops"]))
        save_geoparquet(stops_gdf, out_dir / "stops.geoparquet")
        meta["files"]["stops"] = len(stops_gdf)

        # ── 2. Routes ─────────────────────────────────────────────────────────
        log("\n[2/6] Routes")
        routes_df = read_gtfs_file(zf, "routes.txt", GTFS_DTYPES["routes"])
        if not routes_df.empty:
            routes_df["mode"] = route_type_label(routes_df["route_type"])
        save_parquet(routes_df, out_dir / "routes.parquet")
        meta["files"]["routes"] = len(routes_df)

        # ── 3. Calendar + Trips ───────────────────────────────────────────────
        log("\n[3/6] Calendar + Trips")
        calendar_df       = read_gtfs_file(zf, "calendar.txt",       GTFS_DTYPES["calendar"])
        calendar_dates_df = read_gtfs_file(zf, "calendar_dates.txt", GTFS_DTYPES["calendar_dates"])
        trips_df          = read_gtfs_file(zf, "trips.txt",          GTFS_DTYPES["trips"])
        trips_enriched    = build_trips_with_calendar(trips_df, calendar_df)
        save_parquet(trips_enriched,    out_dir / "trips.parquet")
        save_parquet(calendar_dates_df, out_dir / "calendar_dates.parquet")
        meta["files"]["trips"]          = len(trips_enriched)
        meta["files"]["calendar_dates"] = len(calendar_dates_df)

        # ── 4. Stop times ─────────────────────────────────────────────────────
        log("\n[4/6] Stop times (large file — please wait …)")
        stop_times_df = build_stop_times(
            read_gtfs_file(zf, "stop_times.txt", GTFS_DTYPES["stop_times"])
        )
        save_parquet(stop_times_df, out_dir / "stop_times.parquet", row_group_size=250_000)
        meta["files"]["stop_times"] = len(stop_times_df)

        # ── 5. Shapes ─────────────────────────────────────────────────────────
        log("\n[5/6] Shapes")
        shapes_gdf = build_shapes(read_gtfs_file(zf, "shapes.txt", GTFS_DTYPES["shapes"]))
        save_geoparquet(shapes_gdf, out_dir / "shapes.geoparquet")
        meta["files"]["shapes"] = len(shapes_gdf)

    # ── 6. Metadata ───────────────────────────────────────────────────────────
    log("\n[6/6] Summary")
    meta["duration_sec"] = round(time.time() - t0, 1)
    (out_dir / "_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )
    log("  ✓ _meta.json")

    print("\n" + "═" * 55)
    print(f"  Ingestion complete in {meta['duration_sec']} s")
    print(f"  Output directory: {out_dir.resolve()}")
    print("  Files written:")
    for fname, count in meta["files"].items():
        ext = "geoparquet" if fname in ("stops", "shapes") else "parquet"
        print(f"    {fname}.{ext:12s}  {count:>10,} rows")
    print("═" * 55)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest a Tisséo GTFS feed into Parquet / GeoParquet"
    )
    parser.add_argument(
        "--gtfs", type=Path, default=GTFS_ZIP,
        help="Path to the GTFS .zip file (default: config.GTFS_ZIP)",
    )
    parser.add_argument(
        "--out", type=Path, default=PARQUET,
        help="Output directory for Parquet files (default: config.PARQUET)",
    )
    args = parser.parse_args()

    if not args.gtfs.exists():
        print(f"ERROR: GTFS file not found → {args.gtfs.resolve()}")
        print("Download from: https://data.toulouse-metropole.fr/explore/dataset/tisseo-gtfs/")
        print(f"Then place it at: {GTFS_ZIP.parent.resolve()}")
        raise SystemExit(1)

    ingest(args.gtfs, args.out)
