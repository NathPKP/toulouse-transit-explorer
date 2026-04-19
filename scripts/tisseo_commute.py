"""
tisseo_commute.py  (v3 — 2 transfers + inter-stop walking)
===========================================================
Estimate transit travel time from every IRIS district to a given destination.

Algorithm (3-round RAPTOR-style):
    Round 0 — Direct: board at origin stop → arrive at destination
    Round 1 — 1 transfer: leg1 + [walk ≤ WALK_TRANSFER_MAX m] + leg2
    Round 2 — 2 transfers: leg1 + [walk] + leg2 + [walk] + leg3
    At each round the set of "reachable" stops is expanded to walking neighbours.
    The best time across all rounds is kept per origin stop.

Usage:
    python scripts/tisseo_commute.py --dest "Place du Capitole, Toulouse"
    python scripts/tisseo_commute.py --dest 43.6047,1.4442
    python scripts/tisseo_commute.py --dest "Airbus ZAC" --heure 9 --marche 1000
"""

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GEO, PARQUET, ROOT

import duckdb
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from branca.colormap import LinearColormap

# ── Constants ─────────────────────────────────────────────────────────────────

INDICATORS_FILE      = GEO    / "quartiers_indicators.parquet"
STOPS_FILE           = PARQUET / "stops.geoparquet"
STOP_TIMES_FILE      = PARQUET / "stop_times.parquet"
TRIPS_FILE           = PARQUET / "trips.parquet"
CALENDAR_DATES_FILE  = PARQUET / "calendar_dates.parquet"
OUTPUT_DIR           = ROOT   / "outputs"

WALK_SPEED_MPM    = 80     # metres per minute (~4.8 km/h)
WALK_MAX_DEST     = 500    # metres — search radius around destination
WALK_TRANSFER_MAX = 400    # metres — max walk between stops for a transfer
TRANSFER_PENALTY  = 8      # fixed minutes per transfer (boarding wait, etc.)
DEFAULT_WAIT      = 8.0    # fallback wait time when frequency is unknown (min)
COMMUTE_CAP       = 75     # minutes — colormap ceiling


# ── Geocoding ─────────────────────────────────────────────────────────────────

_TOULOUSE_BOX = "1.05,43.85,1.75,43.30"
_HEADERS      = {"User-Agent": "toulouse-transit-explorer/1.0"}


def _nominatim(q: str, **params) -> list:
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": q, "format": "json", "limit": 1, **params},
        headers=_HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def parse_dest(dest_str: str) -> tuple[float, float, str]:
    """
    Parse a destination string into (lat, lon, label).
    Accepts: "lat,lon" coordinates, embedded coordinates, or a place name
    that is geocoded via OpenStreetMap Nominatim.
    """
    s = dest_str.strip()

    # Bare "lat,lon"
    m = re.match(r"^(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)$", s)
    if m:
        return float(m.group(1)), float(m.group(2)), s

    # Embedded coordinates in a longer string
    m = re.search(r"\(?\s*(\d{2,3}\.\d+)\s*,\s*(\d{1,3}\.\d+)\s*\)?", s)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if 42.0 <= lat <= 45.0 and 0.0 <= lon <= 4.0:
            label = re.sub(r"\(?\s*\d+\.\d+\s*,\s*\d+\.\d+\s*\)?", "", s).strip(" ,-()")
            return lat, lon, label or s

    # Nominatim geocoding
    time.sleep(1)
    results = (
        _nominatim(s, viewbox=_TOULOUSE_BOX, bounded=1, countrycodes="fr")
        or _nominatim(s, viewbox=_TOULOUSE_BOX, countrycodes="fr")
        or _nominatim(s, countrycodes="fr")
        or _nominatim(s)
    )
    if not results:
        raise ValueError(
            f"Address not found: {dest_str!r}\n"
            "Try using coordinates instead: 43.604,1.444"
        )
    lat   = float(results[0]["lat"])
    lon   = float(results[0]["lon"])
    label = results[0].get("display_name", dest_str).split(",")[0]
    return lat, lon, label


# ── Geometry utilities ────────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float,
                lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorised haversine distance in metres."""
    R = 6_371_000
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a  = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def sql_in(ids: list[str]) -> str:
    return ", ".join(f"'{i}'" for i in ids)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    indicators  = gpd.read_parquet(INDICATORS_FILE)
    stops_all   = gpd.read_parquet(STOPS_FILE)
    stop_points = stops_all[stops_all["location_type"] == 0].copy()
    stop_points["lat"] = stop_points.geometry.y
    stop_points["lon"] = stop_points.geometry.x
    return indicators, stop_points


def find_dest_stops(sp: gpd.GeoDataFrame,
                    dest_lat: float, dest_lon: float) -> list[str]:
    """Return stop_ids within WALK_MAX_DEST metres of the destination."""
    dists  = haversine_m(dest_lat, dest_lon, sp["lat"].values, sp["lon"].values)
    nearby = sp[dists <= WALK_MAX_DEST]
    if nearby.empty:
        raise ValueError(
            f"No Tisséo stop within {WALK_MAX_DEST} m of the destination.\n"
            "Try more precise coordinates."
        )
    return nearby["stop_id"].tolist()


# ── DuckDB connection ─────────────────────────────────────────────────────────

def _open_con(day_type: str = "weekday") -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB connection with a `st` view filtered by day type.
    day_type: "weekday" | "saturday" | "sunday"
    Uses pandas for date parsing to avoid DuckDB version edge-cases.
    """
    con = duckdb.connect()

    if CALENDAR_DATES_FILE.exists() and TRIPS_FILE.exists():
        # pandas dayofweek: 0=Mon … 4=Fri, 5=Sat, 6=Sun
        _dow_map = {"weekday": [0, 1, 2, 3, 4], "saturday": [5], "sunday": [6]}
        valid_dow = _dow_map.get(day_type, [0, 1, 2, 3, 4])

        cal = pd.read_parquet(CALENDAR_DATES_FILE)
        cal["_dt"] = pd.to_datetime(cal["date"].astype(str), format="%Y%m%d")
        valid_svc = set(
            cal[(cal["exception_type"] == 1) & cal["_dt"].dt.dayofweek.isin(valid_dow)]
            ["service_id"]
        )

        trips_df = pd.read_parquet(TRIPS_FILE, columns=["trip_id", "service_id"])
        valid_trips = trips_df[trips_df["service_id"].isin(valid_svc)][["trip_id"]]

        con.register("_valid_trips", valid_trips)
        con.execute(f"""
            CREATE VIEW st AS
            SELECT s.*
            FROM read_parquet('{STOP_TIMES_FILE}') s
            JOIN _valid_trips vt ON s.trip_id = vt.trip_id
        """)
    else:
        con.execute(f"CREATE VIEW st AS SELECT * FROM read_parquet('{STOP_TIMES_FILE}')")

    return con


# ── DuckDB queries ────────────────────────────────────────────────────────────

def compute_direct(con: duckdb.DuckDBPyConnection,
                   dest_ids: list[str], heure: int) -> pd.DataFrame:
    """Best direct (single-vehicle) transit time from each stop to destination."""
    return con.execute(f"""
        SELECT
            o.stop_id                                      AS stop_id,
            MIN((d.arrival_sec - o.departure_sec) / 60.0) AS vehicle_min_direct,
            COUNT(DISTINCT o.trip_id)                      AS nb_trips_directs
        FROM st o
        JOIN st d
            ON  o.trip_id       = d.trip_id
            AND d.stop_sequence > o.stop_sequence
            AND d.arrival_sec   > o.departure_sec
        WHERE d.stop_id     IN ({sql_in(dest_ids)})
          AND o.stop_id NOT IN ({sql_in(dest_ids)})
          AND o.hour_of_day  = {heure}
        GROUP BY o.stop_id
        HAVING vehicle_min_direct > 0
    """).df()


def query_arrivals(con: duckdb.DuckDBPyConnection,
                   exclude_ids: list[str],
                   target_ids: list[str],
                   heure: int) -> pd.DataFrame:
    """
    Best single-vehicle transit time from any origin stop to any stop in target_ids.
    Returns (origin_id, arrive_id, ride_min).
    """
    if not target_ids:
        return pd.DataFrame(columns=["origin_id", "arrive_id", "ride_min"])
    return con.execute(f"""
        SELECT
            o.stop_id                                      AS origin_id,
            d.stop_id                                      AS arrive_id,
            MIN((d.arrival_sec - o.departure_sec) / 60.0) AS ride_min
        FROM st o
        JOIN st d
            ON  o.trip_id       = d.trip_id
            AND d.stop_sequence > o.stop_sequence
            AND d.arrival_sec   > o.departure_sec
        WHERE o.hour_of_day   = {heure}
          AND o.stop_id NOT IN ({sql_in(exclude_ids)})
          AND d.stop_id       IN ({sql_in(target_ids)})
        GROUP BY o.stop_id, d.stop_id
        HAVING ride_min > 0
    """).df()


def compute_wait(con: duckdb.DuckDBPyConnection, heure: int) -> pd.DataFrame:
    """Average wait time (min) per stop: 60 / trips_per_hour / 2."""
    return con.execute(f"""
        SELECT stop_id,
               LEAST(ROUND(60.0 / NULLIF(COUNT(*), 0) / 2, 1), 20) AS wait_min
        FROM st
        WHERE hour_of_day = {heure}
        GROUP BY stop_id
    """).df()


# ── Inter-stop walking ────────────────────────────────────────────────────────

def compute_walk_pairs(stop_points: gpd.GeoDataFrame,
                       max_m: int = WALK_TRANSFER_MAX) -> pd.DataFrame:
    """
    Return DataFrame(from_id, to_id, walk_min) for all stop pairs
    within max_m metres. Computed in chunks to limit memory usage.
    """
    sp   = stop_points.drop_duplicates(subset=["stop_id"]).reset_index(drop=True)
    lats = sp["lat"].values
    lons = sp["lon"].values
    ids  = sp["stop_id"].values
    n    = len(sp)
    R    = 6_371_000
    CHUNK = 500

    from_ids, to_ids, walk_mins = [], [], []
    lat2_r = np.radians(lats)
    lon2_r = np.radians(lons)

    for start in range(0, n, CHUNK):
        end  = min(start + CHUNK, n)
        lat1 = np.radians(lats[start:end])[:, None]
        lon1 = np.radians(lons[start:end])[:, None]
        dp   = lat2_r[None, :] - lat1
        dl   = lon2_r[None, :] - lon1
        a    = (np.sin(dp / 2) ** 2
                + np.cos(lat1) * np.cos(lat2_r[None, :]) * np.sin(dl / 2) ** 2)
        dists = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        mask  = (dists > 0) & (dists <= max_m)
        ii, jj = np.where(mask)
        from_ids.extend(ids[start + ii])
        to_ids.extend(ids[jj])
        walk_mins.extend(dists[ii, jj] / WALK_SPEED_MPM)

    return pd.DataFrame({"from_id": from_ids, "to_id": to_ids, "walk_min": walk_mins})


def expand_connections(times_dict: dict, walk_pairs: pd.DataFrame) -> dict:
    """
    Extend {stop_id: transit_time_to_dest} to include walking neighbours.
    A passenger arrives at stop A (from_id) and can walk to stop B (to_id)
    which is in times_dict; extra cost = walk_min.
    Returns an updated dict with the best time per stop.
    """
    expanded = dict(times_dict)
    if walk_pairs.empty or not times_dict:
        return expanded
    rel = walk_pairs[walk_pairs["to_id"].isin(times_dict)].copy()
    if rel.empty:
        return expanded
    rel["conn_time"] = rel["to_id"].map(times_dict) + rel["walk_min"]
    best = rel.groupby("from_id")["conn_time"].min()
    for fid, ct in best.items():
        if fid not in expanded or expanded[fid] > ct:
            expanded[fid] = ct
    return expanded


# ── Main routing: 3 rounds (direct + 1 transfer + 2 transfers) ───────────────

def build_stop_times(con: duckdb.DuckDBPyConnection,
                     dest_ids: list[str],
                     stop_points: gpd.GeoDataFrame,
                     heure: int,
                     walk_transfer_max: int = WALK_TRANSFER_MAX,
                     walk_pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Compute the best transit time from each origin stop to the destination,
    covering direct trips, 1 transfer, and 2 transfers with optional inter-stop
    walking at each transfer point.

    Returns DataFrame(stop_id, vehicle_min, is_transfer, n_transfers,
                      inter_stop_name, nb_trips_directs).
    """
    if walk_pairs is None:
        walk_pairs = compute_walk_pairs(stop_points, walk_transfer_max)
    sp_names = stop_points.set_index("stop_id")["stop_name"].to_dict()

    # ── Round 0: direct trips ─────────────────────────────────────────────────
    direct_df   = compute_direct(con, dest_ids, heure)
    direct_cnt  = dict(zip(direct_df["stop_id"], direct_df["nb_trips_directs"]))
    direct_dict = dict(zip(direct_df["stop_id"], direct_df["vehicle_min_direct"]))

    conn0 = expand_connections(direct_dict, walk_pairs)

    # ── Round 1: 1 transfer ───────────────────────────────────────────────────
    leg1_df  = query_arrivals(con, dest_ids, list(conn0.keys()), heure)
    r1_dict  = {}
    r1_inter = {}

    if not leg1_df.empty:
        leg1_df["conn_time"] = leg1_df["arrive_id"].map(conn0)
        leg1_df["total"]     = leg1_df["ride_min"] + TRANSFER_PENALTY + leg1_df["conn_time"]
        leg1_df              = leg1_df.dropna(subset=["total"])
        if not leg1_df.empty:
            best_idx = leg1_df.groupby("origin_id")["total"].idxmin()
            best1    = leg1_df.loc[best_idx]
            r1_dict  = dict(zip(best1["origin_id"], best1["total"]))
            r1_inter = dict(zip(best1["origin_id"], best1["arrive_id"]))

    conn1 = expand_connections(r1_dict, walk_pairs) if r1_dict else {}

    # ── Round 2: 2 transfers ──────────────────────────────────────────────────
    r2_dict  = {}
    r2_inter = {}

    if conn1:
        leg2_df = query_arrivals(con, dest_ids, list(conn1.keys()), heure)
        if not leg2_df.empty:
            leg2_df["conn_time"] = leg2_df["arrive_id"].map(conn1)
            leg2_df["total"]     = leg2_df["ride_min"] + TRANSFER_PENALTY + leg2_df["conn_time"]
            leg2_df              = leg2_df.dropna(subset=["total"])
            if not leg2_df.empty:
                best_idx2 = leg2_df.groupby("origin_id")["total"].idxmin()
                best2     = leg2_df.loc[best_idx2]
                r2_dict   = dict(zip(best2["origin_id"], best2["total"]))
                r2_inter  = dict(zip(best2["origin_id"], best2["arrive_id"]))

    # ── Combine: best time per stop ───────────────────────────────────────────
    all_ids = set(direct_dict) | set(r1_dict) | set(r2_dict)
    rows = []
    for sid in all_ids:
        options = []
        if sid in direct_dict:
            options.append((direct_dict[sid], False, 0, ""))
        if sid in r1_dict:
            options.append((r1_dict[sid], True, 1,
                            sp_names.get(r1_inter.get(sid, ""), "")))
        if sid in r2_dict:
            options.append((r2_dict[sid], True, 2,
                            sp_names.get(r2_inter.get(sid, ""), "")))
        vehicle_min, is_transfer, n_transfers, inter_stop = min(options, key=lambda x: x[0])
        rows.append({
            "stop_id"         : sid,
            "vehicle_min"     : vehicle_min,
            "is_transfer"     : is_transfer,
            "n_transfers"     : n_transfers,
            "inter_stop_name" : inter_stop,
            "nb_trips_directs": direct_cnt.get(sid, 0),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "stop_id", "vehicle_min", "is_transfer", "n_transfers",
            "inter_stop_name", "nb_trips_directs",
        ])
    return pd.DataFrame(rows)


# ── IRIS aggregation ──────────────────────────────────────────────────────────

def compute_iris_commute(indicators: gpd.GeoDataFrame,
                         stop_points: gpd.GeoDataFrame,
                         stop_times_df: pd.DataFrame,
                         wait: pd.DataFrame,
                         walk_max_m: int) -> gpd.GeoDataFrame:

    sp = (stop_points[["stop_id", "stop_name", "lat", "lon"]]
          .merge(stop_times_df, on="stop_id", how="inner")
          .merge(wait, on="stop_id", how="left"))
    sp["wait_min"] = sp["wait_min"].fillna(DEFAULT_WAIT)

    lat_arr   = sp["lat"].values
    lon_arr   = sp["lon"].values
    veh_arr   = sp["vehicle_min"].values
    wait_arr  = sp["wait_min"].values
    name_arr  = sp["stop_name"].values
    trips_arr = sp["nb_trips_directs"].values
    tr_arr    = sp["is_transfer"].values
    ntr_arr   = (sp["n_transfers"].values
                 if "n_transfers" in sp.columns else np.zeros(len(sp), int))
    inter_arr = (sp["inter_stop_name"].values
                 if "inter_stop_name" in sp.columns
                 else np.full(len(sp), ""))

    records = []
    for _, row in indicators.iterrows():
        c    = row.geometry.centroid
        dist = haversine_m(c.y, c.x, lat_arr, lon_arr)
        mask = dist <= walk_max_m

        if not mask.any():
            records.append({
                "iris_code"           : row["iris_code"],
                "commute_total_min"   : np.nan,
                "commute_walk_min"    : np.nan,
                "commute_wait_min"    : np.nan,
                "commute_vehicle_min" : np.nan,
                "commute_stop_name"   : None,
                "commute_nb_trips"    : 0,
                "commute_is_transfer" : False,
                "commute_n_transfers" : 0,
                "commute_inter_stop"  : None,
            })
            continue

        walk_min = dist[mask] / WALK_SPEED_MPM
        total    = walk_min + wait_arr[mask] + veh_arr[mask]
        i        = int(np.argmin(total))

        records.append({
            "iris_code"           : row["iris_code"],
            "commute_total_min"   : round(float(total[i]),          1),
            "commute_walk_min"    : round(float(walk_min[i]),        1),
            "commute_wait_min"    : round(float(wait_arr[mask][i]), 1),
            "commute_vehicle_min" : round(float(veh_arr[mask][i]),  1),
            "commute_stop_name"   : str(name_arr[mask][i]),
            "commute_nb_trips"    : int(trips_arr[mask][i]),
            "commute_is_transfer" : bool(tr_arr[mask][i]),
            "commute_n_transfers" : int(ntr_arr[mask][i]),
            "commute_inter_stop"  : str(inter_arr[mask][i]) or None,
        })

    return gpd.GeoDataFrame(
        indicators.merge(pd.DataFrame(records), on="iris_code", how="left"),
        geometry="geometry", crs="EPSG:4326",
    )


# ── Map ───────────────────────────────────────────────────────────────────────

PALETTE    = ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"]
CAT_COLORS = {
    "bien desservi": "#1a9850", "moyen": "#fc8d59",
    "sous-desservi": "#d73027", "non desservi": "#aaaaaa",
}

_BADGE = {
    0: ("<span style='background:#10b981;color:#fff;font-size:10px;"
        "padding:1px 6px;border-radius:8px;margin-left:6px'>Direct</span>"),
    1: ("<span style='background:#f59e0b;color:#fff;font-size:10px;"
        "padding:1px 6px;border-radius:8px;margin-left:6px'>1 transfer</span>"),
    2: ("<span style='background:#ef6820;color:#fff;font-size:10px;"
        "padding:1px 6px;border-radius:8px;margin-left:6px'>2 transfers</span>"),
}


def _fmt(v, unit="", dec=0):
    return f"{v:.{dec}f} {unit}".strip() if not pd.isna(v) else "n/a"


def popup_html(row: pd.Series, dest_label: str) -> str:
    total   = row.get("commute_total_min")
    walk    = row.get("commute_walk_min")
    wait    = row.get("commute_wait_min")
    vehicle = row.get("commute_vehicle_min")
    stop    = row.get("commute_stop_name") or "—"
    trips   = int(row.get("commute_nb_trips", 0))
    n_tr    = int(row.get("commute_n_transfers", 0))
    inter   = row.get("commute_inter_stop")
    score   = row.get("score_desserte")
    cat     = str(row.get("categorie") or "non desservi")
    nom     = row.get("iris_nom", row.get("iris_code", "—"))
    code    = row.get("iris_code", "—")

    cat_color  = CAT_COLORS.get(cat, "#aaa")
    score_w    = int(score) if not pd.isna(score) else 0
    mode_badge = _BADGE.get(n_tr, _BADGE[2])

    if pd.isna(total):
        trajet_block = """
        <div style="background:#f5f5f5;border-radius:6px;padding:10px;
             text-align:center;color:#999;font-size:12px">
          No transit link identified<br>from this district
        </div>"""
    else:
        cap     = min(total, COMMUTE_CAP)
        bar_w   = int(cap / COMMUTE_CAP * 100)
        bar_col = ("#1a9850" if total < 20 else
                   "#fee08b" if total < 35 else
                   "#fc8d59" if total < 50 else "#d73027")
        inter_row = (
            f"<tr><td style='color:#666;padding:2px 0'>Transfer at</td>"
            f"<td style='text-align:right;font-weight:bold'>{inter}</td></tr>"
            if n_tr > 0 and inter else ""
        )
        trajet_block = f"""
        <div style="margin:8px 0">
          <div style="font-size:22px;font-weight:bold;text-align:center;color:{bar_col}">
            {_fmt(total, "min")} {mode_badge}
          </div>
          <div style="font-size:11px;color:#888;text-align:center;margin-bottom:5px">
            to {dest_label}
          </div>
          <div style="background:#eee;border-radius:4px;height:10px">
            <div style="background:{bar_col};height:10px;border-radius:4px;
                 width:{bar_w}%"></div>
          </div>
        </div>
        <table style="width:100%;font-size:11px;color:#555;border-collapse:collapse">
          <tr><td style="padding:3px 0">Walk to stop</td>
              <td style="text-align:right">{_fmt(walk,"min",1)}</td></tr>
          <tr><td>Wait</td>
              <td style="text-align:right">{_fmt(wait,"min",1)}</td></tr>
          <tr><td>Transit ride</td>
              <td style="text-align:right">{_fmt(vehicle,"min",1)}</td></tr>
          {inter_row}
          <tr style="border-top:1px solid #eee">
            <td style="padding-top:4px">Departure stop</td>
            <td style="text-align:right;font-weight:bold">{stop}</td></tr>
          <tr><td>Trips / h</td>
              <td style="text-align:right">{trips}</td></tr>
        </table>"""

    return f"""
    <div style="font-family:Arial,sans-serif;font-size:13px;min-width:255px">
      <b style="font-size:15px">{nom}</b>
      <span style="color:#aaa;font-size:11px;margin-left:4px">{code}</span>
      <hr style="margin:6px 0;border-color:#eee">
      {trajet_block}
      <hr style="margin:8px 0;border-color:#eee">
      <div style="font-size:11px;color:#555;margin-bottom:3px">Network coverage</div>
      <div style="margin-bottom:4px">
        <span style="background:{cat_color};color:#fff;padding:1px 7px;
              border-radius:8px;font-size:11px">{cat}</span>
        <span style="float:right;font-size:11px;color:#666">
          {_fmt(score,"/100",1)}</span>
      </div>
      <div style="background:#eee;border-radius:3px;height:7px">
        <div style="background:{cat_color};height:7px;border-radius:3px;
             width:{score_w}%"></div>
      </div>
    </div>"""


def build_map(gdf: gpd.GeoDataFrame,
              dest_lat: float, dest_lon: float,
              dest_label: str, heure: int) -> folium.Map:

    valid    = gdf["commute_total_min"].dropna()
    vmin     = max(0.0, float(valid.min()) - 2) if not valid.empty else 0
    vmax     = float(min(COMMUTE_CAP, valid.quantile(0.95))) if not valid.empty else COMMUTE_CAP
    colormap = LinearColormap(
        colors=PALETTE, vmin=vmin, vmax=vmax,
        caption=f"Travel time to {dest_label} (min) — {heure}h",
    )

    c = gdf.geometry.union_all().centroid
    m = folium.Map(location=[c.y, c.x], zoom_start=12,
                   tiles="CartoDB positron",
                   attr="© OpenStreetMap contributors © CARTO",
                   scrollWheelZoom=False)

    fg = folium.FeatureGroup(name="Travel time", show=True)
    for _, row in gdf.iterrows():
        t_val = row.get("commute_total_min")
        n_tr  = int(row.get("commute_n_transfers", 0))
        color = colormap(min(t_val, COMMUTE_CAP)) if not pd.isna(t_val) else "#cccccc"
        nom   = row.get("iris_nom", row.get("iris_code", ""))
        suffix = ("" if n_tr == 0 else
                  " (1 transfer)" if n_tr == 1 else " (2 transfers)")
        tip   = (f"{nom} — {t_val:.0f} min{suffix}"
                 if not pd.isna(t_val) else f"{nom} — no link")

        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, c=color: {
                "fillColor": c, "color": "#555",
                "weight": 0.6, "fillOpacity": 0.75,
            },
            tooltip=folium.Tooltip(tip, sticky=True),
            popup=folium.Popup(popup_html(row, dest_label), max_width=295),
        ).add_to(fg)
    fg.add_to(m)
    colormap.add_to(m)

    folium.Marker(
        location=[dest_lat, dest_lon],
        tooltip=dest_label,
        popup=folium.Popup(
            f"<b>{dest_label}</b><br>"
            f"<span style='font-size:11px;color:#888'>Destination — {heure}h</span>",
            max_width=200,
        ),
        icon=folium.Icon(color="red", icon="briefcase", prefix="fa"),
    ).add_to(m)

    t_col = gdf["commute_total_min"]
    ntr   = gdf["commute_n_transfers"].fillna(0)
    stats = {
        "lt20": int((t_col < 20).sum()),
        "lt35": int(((t_col >= 20) & (t_col < 35)).sum()),
        "lt50": int(((t_col >= 35) & (t_col < 50)).sum()),
        "gt50": int((t_col >= 50).sum()),
        "none": int(t_col.isna().sum()),
    }
    n_dir = int(((ntr == 0) & t_col.notna()).sum())
    n_r1  = int((ntr == 1).sum())
    n_r2  = int((ntr == 2).sum())

    legend = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
         background:white;padding:13px 17px;border-radius:9px;
         box-shadow:0 2px 8px rgba(0,0,0,.22);font-family:Arial;font-size:12px">
      <b style="font-size:13px">Commute to {dest_label}</b>
      <div style="font-size:11px;color:#888;margin-bottom:8px">{heure}h &bull;
        {n_dir} direct &bull; {n_r1} x1 transfer &bull; {n_r2} x2 transfers</div>
      <table style="border-collapse:collapse">
        <tr><td style="padding:2px 8px 2px 0;font-size:16px;color:#1a9850">&#9632;</td>
            <td>under 20 min ({stats['lt20']} districts)</td></tr>
        <tr><td style="font-size:16px;color:#fee08b">&#9632;</td>
            <td>20–35 min ({stats['lt35']} districts)</td></tr>
        <tr><td style="font-size:16px;color:#fc8d59">&#9632;</td>
            <td>35–50 min ({stats['lt50']} districts)</td></tr>
        <tr><td style="font-size:16px;color:#d73027">&#9632;</td>
            <td>over 50 min ({stats['gt50']} districts)</td></tr>
        <tr><td style="font-size:16px;color:#cccccc">&#9632;</td>
            <td>no link ({stats['none']} districts)</td></tr>
      </table>
    </div>"""

    title = f"""
    <div style="position:fixed;top:14px;left:50%;transform:translateX(-50%);
         z-index:1000;background:white;padding:8px 20px;border-radius:8px;
         box-shadow:0 2px 8px rgba(0,0,0,.22);font-family:Arial;
         font-size:15px;font-weight:bold;white-space:nowrap">
      Tisséo commute to: {dest_label}
    </div>"""

    m.get_root().html.add_child(folium.Element(legend))
    m.get_root().html.add_child(folium.Element(title))

    b = gdf.total_bounds
    m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ── Utilities ─────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s_-]+", "_", text).strip("_")[:40]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate Tisséo transit commute time to a destination"
    )
    p.add_argument("--dest",   required=True, help="Destination address or lat,lon")
    p.add_argument("--heure",  type=int, default=8,   help="Departure hour (default: 8)")
    p.add_argument("--marche", type=int, default=800, help="Max walk to stop in metres (default: 800)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not INDICATORS_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {INDICATORS_FILE}\n"
            "Run first: python scripts/tisseo_indicators.py"
        )

    print("1/5  Parsing destination …")
    dest_lat, dest_lon, dest_label = parse_dest(args.dest)

    print("2/5  Loading data …")
    indicators, stop_points = load_data()
    print(f"     {len(indicators)} districts | {len(stop_points)} stops")

    print(f"3/5  Finding destination stops (radius {WALK_MAX_DEST} m) …")
    dest_stops = find_dest_stops(stop_points, dest_lat, dest_lon)
    print(f"     {len(dest_stops)} stops found")

    print(f"4/5  Computing travel times at {args.heure}h (direct + 1 + 2 transfers) …")
    con        = _open_con()
    wait       = compute_wait(con, args.heure)
    stop_times = build_stop_times(con, dest_stops, stop_points, args.heure)
    con.close()

    n_dir = int((stop_times["n_transfers"] == 0).sum())
    n_r1  = int((stop_times["n_transfers"] == 1).sum())
    n_r2  = int((stop_times["n_transfers"] == 2).sum())
    print(f"     {len(stop_times)} stops: {n_dir} direct, {n_r1} x1 transfer, {n_r2} x2 transfers")

    print("5/5  Aggregating by district + building map …")
    result  = compute_iris_commute(indicators, stop_points, stop_times, wait, args.marche)
    t_col   = result["commute_total_min"]
    n_conn  = int(t_col.notna().sum())
    print(f"     {n_conn}/{len(result)} districts connected"
          f" | median {t_col.median():.0f} min | min {t_col.min():.0f} min")

    m = build_map(result, dest_lat, dest_lon, dest_label, args.heure)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"carte_commute_{slugify(dest_label)}.html"
    m.save(str(out))

    from _map_index import generate_index
    generate_index(OUTPUT_DIR)

    print(f"\nMap   → {out}")
    print(f"Index → {OUTPUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
