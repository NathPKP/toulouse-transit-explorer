"""
test_pipeline.py
================
Sanity checks for the processed Parquet / GeoParquet data files.
Run with: pytest tests/
"""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

ROOT    = Path(__file__).parent.parent
PARQUET = ROOT / "data" / "parquet"
GEO     = ROOT / "data" / "geo"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def stops():
    return gpd.read_parquet(PARQUET / "stops.geoparquet")

@pytest.fixture(scope="session")
def stop_times():
    return pd.read_parquet(PARQUET / "stop_times.parquet")

@pytest.fixture(scope="session")
def trips():
    return pd.read_parquet(PARQUET / "trips.parquet")

@pytest.fixture(scope="session")
def routes():
    return pd.read_parquet(PARQUET / "routes.parquet")

@pytest.fixture(scope="session")
def indicators():
    return gpd.read_parquet(GEO / "quartiers_indicators.parquet")

@pytest.fixture(scope="session")
def hourly():
    return pd.read_parquet(GEO / "hourly_indicators.parquet")


# ── File existence ─────────────────────────────────────────────────────────────

def test_parquet_files_exist():
    for name in ["stops.geoparquet", "stop_times.parquet", "trips.parquet",
                 "routes.parquet", "calendar_dates.parquet"]:
        assert (PARQUET / name).exists(), f"Missing: {name}"

def test_geo_files_exist():
    assert (GEO / "iris_toulouse.geojson").exists()
    assert (GEO / "quartiers_indicators.parquet").exists()
    assert (GEO / "hourly_indicators.parquet").exists()

def test_meta_json_exists():
    meta = PARQUET / "_meta.json"
    assert meta.exists()
    data = json.loads(meta.read_text())
    assert "generated_at" in data


# ── Stops ─────────────────────────────────────────────────────────────────────

def test_stops_columns(stops):
    for col in ("stop_id", "stop_name", "geometry"):
        assert col in stops.columns

def test_stops_no_null_ids(stops):
    assert stops["stop_id"].notna().all(), "Some stop_id values are null"

def test_stops_crs(stops):
    assert stops.crs is not None
    assert stops.crs.to_epsg() == 4326

def test_stops_geometry_valid(stops):
    pts = stops[stops["location_type"] == 0]
    assert pts.geometry.notna().all()
    assert (pts.geometry.geom_type == "Point").all()

def test_stops_toulouse_bbox(stops):
    pts = stops[stops["location_type"] == 0]
    assert pts.geometry.x.between(1.1, 1.8).all(), "Stops outside Toulouse longitude range"
    assert pts.geometry.y.between(43.3, 43.9).all(), "Stops outside Toulouse latitude range"


# ── Stop times ────────────────────────────────────────────────────────────────

def test_stop_times_columns(stop_times):
    for col in ("trip_id", "stop_id", "hour_of_day", "departure_sec", "is_overnight"):
        assert col in stop_times.columns

def test_stop_times_hour_range(stop_times):
    day = stop_times[stop_times["is_overnight"] == 0]
    assert day["hour_of_day"].between(0, 23).all()

def test_stop_times_no_null_trip(stop_times):
    assert stop_times["trip_id"].notna().all()

def test_stop_times_volume(stop_times):
    assert len(stop_times) > 10_000, "Suspiciously few stop_times rows"


# ── Trips ─────────────────────────────────────────────────────────────────────

def test_trips_columns(trips):
    for col in ("trip_id", "route_id", "service_id"):
        assert col in trips.columns

def test_trips_no_null_ids(trips):
    assert trips["trip_id"].notna().all()
    assert trips["route_id"].notna().all()

def test_trips_join_stop_times(trips, stop_times):
    trip_ids_st = set(stop_times["trip_id"].dropna())
    trip_ids_tr = set(trips["trip_id"].dropna())
    orphans = trip_ids_st - trip_ids_tr
    assert len(orphans) == 0, f"{len(orphans)} trip_ids in stop_times not found in trips"


# ── Routes ────────────────────────────────────────────────────────────────────

def test_routes_columns(routes):
    for col in ("route_id", "route_short_name"):
        assert col in routes.columns

def test_routes_no_null_ids(routes):
    assert routes["route_id"].notna().all()


# ── Coverage indicators ───────────────────────────────────────────────────────

def test_indicators_columns(indicators):
    for col in ("iris_code", "iris_nom", "score_desserte", "categorie",
                "amplitude_h", "coverage_pct", "area_km2", "geometry"):
        assert col in indicators.columns

def test_indicators_score_range(indicators):
    scored = indicators["score_desserte"].dropna()
    assert (scored >= 0).all() and (scored <= 100).all()

def test_indicators_categories(indicators):
    valid = {"bien desservi", "moyen", "sous-desservi"}
    cats  = set(indicators["categorie"].dropna().astype(str).unique())
    assert cats.issubset(valid), f"Unexpected categories: {cats - valid}"

def test_indicators_coverage_range(indicators):
    cov = indicators["coverage_pct"].dropna()
    assert (cov >= 0).all() and (cov <= 100).all()

def test_indicators_crs(indicators):
    assert indicators.crs is not None
    assert indicators.crs.to_epsg() == 4326


# ── Hourly indicators ─────────────────────────────────────────────────────────

def test_hourly_columns(hourly):
    for col in ("iris_code", "iris_nom", "heure", "nb_passages",
                "nb_lignes", "nb_arrets", "frequence_moy_min"):
        assert col in hourly.columns

def test_hourly_hour_range(hourly):
    assert hourly["heure"].between(5, 23).all()

def test_hourly_no_negative_passages(hourly):
    assert (hourly["nb_passages"] >= 0).all()

def test_hourly_iris_codes_consistent(indicators, hourly):
    ind_codes    = set(indicators["iris_code"].dropna())
    hourly_codes = set(hourly["iris_code"].dropna())
    unknown = hourly_codes - ind_codes
    assert len(unknown) == 0, f"{len(unknown)} iris_codes in hourly not in indicators"
