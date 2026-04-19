"""
tisseo_check.py
===============
Post-ingestion data quality check for the Tisséo GTFS Parquet files.
Run after tisseo_ingest.py to validate data and get a network overview.

Usage:
    python scripts/tisseo_check.py
    python scripts/tisseo_check.py --data /path/to/parquet/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PARQUET

import duckdb
import geopandas as gpd
import pandas as pd

SEP = "─" * 60


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def check(data_dir: Path):
    con = duckdb.connect()

    parquet_files = {
        "stops":          data_dir / "stops.geoparquet",
        "routes":         data_dir / "routes.parquet",
        "trips":          data_dir / "trips.parquet",
        "stop_times":     data_dir / "stop_times.parquet",
        "calendar_dates": data_dir / "calendar_dates.parquet",
    }

    for name, path in parquet_files.items():
        if path.exists():
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
        else:
            print(f"⚠  {path.name} missing")

    # ── 1. Network overview ────────────────────────────────────────────────────
    section("Network overview")
    overview = con.execute("""
        SELECT
            (SELECT COUNT(*) FROM stops)      AS stops,
            (SELECT COUNT(*) FROM routes)     AS routes,
            (SELECT COUNT(*) FROM trips)      AS trips,
            (SELECT COUNT(*) FROM stop_times) AS stop_times
    """).df()
    for col in overview.columns:
        print(f"  {col:<20} {overview[col].iloc[0]:>12,}")

    # ── 2. Mode breakdown ──────────────────────────────────────────────────────
    section("Breakdown by transport mode")
    modes = con.execute("""
        SELECT mode, COUNT(*) AS routes
        FROM routes
        GROUP BY mode
        ORDER BY routes DESC
    """).df()
    print(modes.to_string(index=False))

    # ── 3. Hourly service coverage ─────────────────────────────────────────────
    section("Hourly coverage — trips per hour across full network (daytime only)")
    hourly = con.execute("""
        SELECT
            hour_of_day                   AS hour,
            COUNT(*)                      AS passages,
            COUNT(DISTINCT stop_id)       AS active_stops,
            COUNT(DISTINCT trip_id)       AS trips
        FROM stop_times
        WHERE hour_of_day BETWEEN 5 AND 24
        GROUP BY hour_of_day
        ORDER BY hour
    """).df()
    print(hourly.to_string(index=False))

    # ── 4. Top 10 busiest stops ────────────────────────────────────────────────
    section("Top 10 stops — theoretical daily passages")
    top_stops = con.execute("""
        SELECT
            st.stop_id,
            s.stop_name,
            COUNT(*)                AS total_passages,
            COUNT(DISTINCT trip_id) AS trips,
            MIN(hour_of_day)        AS first_hour,
            MAX(hour_of_day)        AS last_hour
        FROM stop_times st
        JOIN stops s USING (stop_id)
        WHERE hour_of_day BETWEEN 5 AND 25
        GROUP BY st.stop_id, s.stop_name
        ORDER BY total_passages DESC
        LIMIT 10
    """).df()
    print(top_stops.to_string(index=False))

    # ── 5. Peak-hour frequency by line ────────────────────────────────────────
    section("Peak-hour frequency (7–9 am) by line — top 15")
    freq_peak = con.execute("""
        WITH passages AS (
            SELECT
                t.route_id,
                st.stop_id,
                COUNT(*) AS passages_2h
            FROM stop_times st
            JOIN trips t USING (trip_id)
            WHERE st.hour_of_day IN (7, 8)
            GROUP BY t.route_id, st.stop_id
        ),
        freq_per_line AS (
            SELECT
                route_id,
                AVG(passages_2h)              AS avg_passages_2h,
                120.0 / NULLIF(AVG(passages_2h), 0) AS interval_min
            FROM passages
            GROUP BY route_id
        )
        SELECT
            r.route_short_name            AS line,
            r.mode,
            ROUND(f.avg_passages_2h, 1)   AS passages_7h_9h,
            ROUND(f.interval_min, 0)      AS avg_interval_min
        FROM freq_per_line f
        JOIN routes r USING (route_id)
        ORDER BY f.avg_passages_2h DESC
        LIMIT 15
    """).df()
    print(freq_peak.to_string(index=False))

    # ── 6. Service span by line ────────────────────────────────────────────────
    section("Service span by line (first / last trip)")
    span = con.execute("""
        SELECT
            r.route_short_name AS line,
            r.mode,
            MIN(st.departure_sec) / 3600  AS start_h,
            MAX(st.departure_sec) / 3600  AS end_h,
            (MAX(st.departure_sec) - MIN(st.departure_sec)) / 3600.0 AS span_h
        FROM stop_times st
        JOIN trips t USING (trip_id)
        JOIN routes r USING (route_id)
        GROUP BY r.route_short_name, r.mode
        ORDER BY span_h DESC
        LIMIT 20
    """).df()
    print(span.to_string(index=False))

    # ── 7. Data quality ────────────────────────────────────────────────────────
    section("Data quality checks")
    nulls = con.execute("""
        SELECT
            COUNT(*) FILTER (WHERE stop_lat IS NULL OR stop_lon IS NULL) AS stops_missing_coords,
            COUNT(*) FILTER (WHERE stop_name IS NULL)                     AS stops_missing_name
        FROM stops
    """).df()
    print("  Stops:")
    for col in nulls.columns:
        print(f"    {col:<35} {nulls[col].iloc[0]:>6}")

    st_nulls = con.execute("""
        SELECT
            COUNT(*) FILTER (WHERE arrival_sec IS NULL)   AS missing_arrival,
            COUNT(*) FILTER (WHERE departure_sec IS NULL) AS missing_departure,
            COUNT(*) FILTER (WHERE stop_id IS NULL)       AS missing_stop_id
        FROM stop_times
    """).df()
    print("  Stop times:")
    for col in st_nulls.columns:
        print(f"    {col:<35} {st_nulls[col].iloc[0]:>6}")

    # ── 8. Geographic extent ───────────────────────────────────────────────────
    section("Network geographic extent")
    bbox = con.execute("""
        SELECT
            ROUND(MIN(stop_lat), 4) AS lat_min,
            ROUND(MAX(stop_lat), 4) AS lat_max,
            ROUND(MIN(stop_lon), 4) AS lon_min,
            ROUND(MAX(stop_lon), 4) AS lon_max
        FROM stops
        WHERE stop_lat IS NOT NULL
    """).df()
    print(bbox.to_string(index=False))

    con.close()
    print(f"\n{SEP}")
    print("  Check complete.")
    print(SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-ingestion data quality check")
    parser.add_argument(
        "--data", type=Path, default=PARQUET,
        help="Parquet directory (default: config.PARQUET)",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"ERROR: directory not found → {args.data.resolve()}")
        print("Run first: python scripts/tisseo_ingest.py")
        raise SystemExit(1)

    check(args.data)
