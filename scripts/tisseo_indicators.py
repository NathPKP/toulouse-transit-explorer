"""
tisseo_indicators.py
====================
Compute transit coverage indicators per IRIS district and per hour
for the Tisséo network in Toulouse.

Usage:
    python scripts/tisseo_indicators.py

Prerequisites:
    data/geo/iris_toulouse.geojson  — IRIS district boundaries (IGN/INSEE)
    Download: https://geoservices.ign.fr/contoursiris
    Or run:   python scripts/tisseo_prepare_iris.py

Output:
    data/geo/quartiers_indicators.parquet   — synthetic per-district scores
    data/geo/hourly_indicators.parquet      — raw hourly counts per district
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PARQUET, GEO

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

HOUR_MIN       = 5
HOUR_MAX       = 24
AMPLITUDE_MAX  = HOUR_MAX - HOUR_MIN   # 19 h
FREQ_IDEAL_MIN = 5                     # "excellent" frequency threshold (min)
FREQ_MAX_MIN   = 60                    # "poor" frequency threshold (min)

STOPS_FILE      = PARQUET / "stops.geoparquet"
STOP_TIMES_FILE = PARQUET / "stop_times.parquet"
TRIPS_FILE      = PARQUET / "trips.parquet"
IRIS_FILE       = GEO / "iris_toulouse.geojson"
OUTPUT_FILE     = GEO / "quartiers_indicators.parquet"
HOURLY_FILE     = GEO / "hourly_indicators.parquet"

# Candidate column names in IRIS GeoJSON files (IGN BDTOPO / ContourIRIS)
_IRIS_ID_CANDIDATES   = ["CODE_IRIS", "code_iris", "DCOMIRIS", "GRD_QUART", "iris_code"]
_IRIS_NAME_CANDIDATES = ["NOM_IRIS",  "nom_iris",  "LIBIRIS",  "iris_nom",  "NOM_COM"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_col(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Column '{label}' not found in the IRIS GeoJSON.\n"
        f"Available columns : {df.columns.tolist()}\n"
        f"Expected one of   : {candidates}"
    )


def load_iris() -> gpd.GeoDataFrame:
    if not IRIS_FILE.exists():
        raise FileNotFoundError(
            f"\nIRIS file not found: {IRIS_FILE}\n\n"
            "Download IRIS boundaries from:\n"
            "  https://geoservices.ign.fr/contoursiris\n"
            "Or run: python scripts/tisseo_prepare_iris.py\n"
            "Filter to Toulouse (INSEE 31555) and save as data/geo/iris_toulouse.geojson"
        )
    iris = gpd.read_file(IRIS_FILE)
    if iris.crs is None:
        iris = iris.set_crs("EPSG:4326")
    elif iris.crs.to_epsg() != 4326:
        iris = iris.to_crs("EPSG:4326")
    return iris


# ── Step 1: spatial join stops x IRIS ─────────────────────────────────────────

def join_stops_iris(iris: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return a DataFrame with (stop_id, iris_code, iris_nom) for each stop."""
    iris_id_col   = _detect_col(iris, _IRIS_ID_CANDIDATES,   "IRIS identifier")
    iris_name_col = _detect_col(iris, _IRIS_NAME_CANDIDATES, "IRIS name")

    stops = gpd.read_parquet(STOPS_FILE)
    if stops.crs is None:
        stops = stops.set_crs("EPSG:4326")
    elif stops.crs.to_epsg() != 4326:
        stops = stops.to_crs("EPSG:4326")

    joined = gpd.sjoin(
        stops[["stop_id", "geometry"]],
        iris[[iris_id_col, iris_name_col, "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined.rename(columns={iris_id_col: "iris_code", iris_name_col: "iris_nom"})
    return joined[["stop_id", "iris_code", "iris_nom"]].reset_index(drop=True)


# ── Step 2: hourly indicators via DuckDB ──────────────────────────────────────

def compute_hourly(stops_iris: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per (iris_code, iris_nom, hour):
        nb_passages       — total transit passages
        nb_lignes         — distinct routes serving the district
        nb_arrets         — distinct active stops
        frequence_moy_min — average headway in minutes: 60 * stops / passages
    """
    con = duckdb.connect()
    con.execute(f"CREATE VIEW stop_times AS SELECT * FROM read_parquet('{STOP_TIMES_FILE}')")
    con.execute(f"CREATE VIEW trips      AS SELECT * FROM read_parquet('{TRIPS_FILE}')")
    con.register("stops_iris", stops_iris)

    hourly = con.execute("""
        SELECT
            si.iris_code,
            si.iris_nom,
            st.hour_of_day                                    AS heure,
            COUNT(*)                                          AS nb_passages,
            COUNT(DISTINCT t.route_id)                        AS nb_lignes,
            COUNT(DISTINCT st.stop_id)                        AS nb_arrets,
            ROUND(
                60.0 * COUNT(DISTINCT st.stop_id)
                / NULLIF(COUNT(*), 0),
                1
            )                                                 AS frequence_moy_min
        FROM stop_times st
        JOIN trips      t  USING (trip_id)
        JOIN stops_iris si ON si.stop_id = st.stop_id
        WHERE st.hour_of_day BETWEEN ? AND ?
          AND si.iris_code IS NOT NULL
          AND st.is_overnight = 0
        GROUP BY si.iris_code, si.iris_nom, st.hour_of_day
        ORDER BY si.iris_code, st.hour_of_day
    """, [HOUR_MIN, HOUR_MAX]).df()

    con.close()
    return hourly


# ── Step 3: spatial coverage metric ───────────────────────────────────────────

def compute_coverage(iris: gpd.GeoDataFrame,
                     stops: gpd.GeoDataFrame,
                     buffer_m: int = 500) -> pd.DataFrame:
    """
    For each IRIS district compute:
        coverage_pct — % of district area within buffer_m metres of an active stop.
                       Large districts with sparse stop coverage are penalised.
        area_km2     — district area in km².

    Uses Lambert-93 (EPSG:2154) for accurate metric distances.
    """
    iris_id_col = _detect_col(iris, _IRIS_ID_CANDIDATES, "IRIS identifier")

    iris_proj  = iris[[iris_id_col, "geometry"]].to_crs("EPSG:2154").copy()
    stops_proj = stops[["geometry"]].to_crs("EPSG:2154").copy()

    iris_proj["area_m2"] = iris_proj.geometry.area

    buffers = stops_proj.geometry.buffer(buffer_m)
    try:
        stop_union = buffers.union_all()
    except AttributeError:                  # geopandas < 0.14
        stop_union = buffers.unary_union

    iris_proj["covered_m2"]   = iris_proj.geometry.intersection(stop_union).area
    iris_proj["coverage_pct"] = (
        iris_proj["covered_m2"] / iris_proj["area_m2"] * 100
    ).clip(0, 100).round(1)
    iris_proj["area_km2"] = (iris_proj["area_m2"] / 1e6).round(4)

    return iris_proj.rename(columns={iris_id_col: "iris_code"})[
        ["iris_code", "coverage_pct", "area_km2"]
    ]


# ── Step 4: synthetic composite score ─────────────────────────────────────────

def compute_synthetic(hourly: pd.DataFrame,
                      iris: gpd.GeoDataFrame,
                      stops: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute per iris_code:
        amplitude_h    — hours between first and last active hour
        score_desserte — composite 0-100 score
        categorie      — 'bien desservi' / 'moyen' / 'sous-desservi'

    Score weights:
        25% amplitude  +  35% peak frequency  +  20% line diversity  +  20% spatial coverage

    Spatial coverage penalises large districts that are sparsely served by stops.
    All components are normalised by percentile rank across all served districts.
    """
    # Peak frequency (7–9 am)
    peak = (
        hourly[hourly["heure"].isin([7, 8])]
        .groupby("iris_code")["frequence_moy_min"]
        .mean()
        .rename("freq_pointe_min")
    )

    synth = (
        hourly
        .groupby(["iris_code", "iris_nom"])
        .agg(
            amplitude_h    = ("heure", lambda s: int(s.max() - s.min())),
            total_passages = ("nb_passages", "sum"),
            max_lignes     = ("nb_lignes",   "max"),
            max_arrets     = ("nb_arrets",   "max"),
        )
        .reset_index()
        .join(peak, on="iris_code")
    )

    # Spatial coverage (% of district within 500 m of a stop)
    cov = compute_coverage(iris, stops)
    synth = synth.merge(cov[["iris_code", "coverage_pct", "area_km2"]],
                        on="iris_code", how="left")
    synth["coverage_pct"] = synth["coverage_pct"].fillna(0)

    # Percentile-rank normalisation (0–100, relative to the full network)
    def pct_rank(s: pd.Series, ascending: bool = True) -> pd.Series:
        return s.rank(pct=True, ascending=ascending, na_option="keep") * 100

    amp_score  = pct_rank(synth["amplitude_h"])
    freq_score = pct_rank(synth["freq_pointe_min"], ascending=False)  # lower = better
    lig_score  = pct_rank(synth["max_lignes"])
    cov_score  = pct_rank(synth["coverage_pct"])

    raw = (
        0.25 * amp_score
      + 0.35 * freq_score
      + 0.20 * lig_score
      + 0.20 * cov_score
    )
    # Re-scale composite to 0–100 for a readable absolute scale
    synth["score_desserte"] = (
        (raw - raw.min()) / (raw.max() - raw.min()) * 100
    ).round(1)

    # Tercile categories
    synth["categorie"] = pd.cut(
        synth["score_desserte"],
        bins=[-0.1, 33.0, 67.0, 100.0],
        labels=["sous-desservi", "moyen", "bien desservi"],
    )

    # Attach IRIS geometries
    iris_id_col = _detect_col(iris, _IRIS_ID_CANDIDATES, "IRIS identifier")
    iris_geo = iris[[iris_id_col, "geometry"]].rename(columns={iris_id_col: "iris_code"})
    result = iris_geo.merge(synth, on="iris_code", how="left")

    if "coverage_pct" not in result.columns:
        result["coverage_pct"] = 0.0
    if "area_km2" not in result.columns:
        result["area_km2"] = 0.0
    result["coverage_pct"] = result["coverage_pct"].fillna(0)

    return gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:4326")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("1/4  Loading IRIS districts ...")
    iris = load_iris()
    print(f"     {len(iris)} IRIS districts loaded")

    print("2/4  Spatial join: stops x IRIS ...")
    stops_iris = join_stops_iris(iris)
    n_matched  = stops_iris["iris_code"].notna().sum()
    n_total    = len(stops_iris)
    print(f"     {n_matched}/{n_total} stops assigned to a district")

    print("3/4  Computing hourly indicators (DuckDB) ...")
    hourly = compute_hourly(stops_iris)
    print(f"     {len(hourly):,} rows (district x hour)")

    # Save hourly data for the Service Pulse feature
    HOURLY_FILE.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(HOURLY_FILE, index=False)
    print(f"     Hourly data -> {HOURLY_FILE}")

    print("4/4  Computing synthetic scores + spatial coverage ...")
    stops_all = gpd.read_parquet(STOPS_FILE)
    stops_pts = stops_all[stops_all["location_type"] == 0].copy()
    result    = compute_synthetic(hourly, iris, stops_pts)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_FILE, index=False)

    scored = result.dropna(subset=["score_desserte"])
    print(f"\nExported -> {OUTPUT_FILE}")
    print(f"   {len(result)} districts | mean score: {scored['score_desserte'].mean():.1f}/100")
    if "categorie" in result.columns:
        cats = result["categorie"].value_counts().sort_index(ascending=False)
        for cat, n in cats.items():
            print(f"   {str(cat):<20} {n:>4} districts")


if __name__ == "__main__":
    main()
