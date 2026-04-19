"""
tisseo_prepare_iris.py
======================
Download IRIS GE district boundaries from the IGN public WFS service
and filter them to the actual coverage area of the Tisséo network.

Source: https://data.geopf.fr/wfs  (layer STATISTICALUNITS.IRISGE:iris_ge)
No API key or account required.

Usage:
    python scripts/tisseo_prepare_iris.py

Output:
    data/geo/iris_toulouse.geojson   (used by tisseo_indicators.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GEO, PARQUET

import geopandas as gpd
import requests

# ── Configuration ─────────────────────────────────────────────────────────────

WFS_URL    = "https://data.geopf.fr/wfs/ows"
WFS_LAYER  = "STATISTICALUNITS.IRISGE:iris_ge"
WFS_PAGE   = 1000          # features per request (server limit)

STOPS_FILE  = PARQUET / "stops.geoparquet"
OUTPUT_FILE = GEO / "iris_toulouse.geojson"

BBOX_BUFFER = 0.02         # padding in degrees around stop bounding box (~1.5 km)


# ── Helpers ───────────────────────────────────────────────────────────────────

def stops_bbox() -> tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) in WGS-84 from stops.geoparquet."""
    stops = gpd.read_parquet(STOPS_FILE, columns=["geometry"])
    if stops.crs is None:
        stops = stops.set_crs("EPSG:4326")
    elif stops.crs.to_epsg() != 4326:
        stops = stops.to_crs("EPSG:4326")
    b = stops.total_bounds
    return (b[0] - BBOX_BUFFER, b[1] - BBOX_BUFFER,
            b[2] + BBOX_BUFFER, b[3] + BBOX_BUFFER)


def fetch_page(bbox: tuple, start: int, count: int) -> dict:
    minx, miny, maxx, maxy = bbox
    params = {
        "SERVICE"     : "WFS",
        "VERSION"     : "2.0.0",
        "REQUEST"     : "GetFeature",
        "TYPENAMES"   : WFS_LAYER,
        "OUTPUTFORMAT": "application/json",
        "BBOX"        : f"{minx},{miny},{maxx},{maxy},EPSG:4326",
        "COUNT"       : count,
        "STARTINDEX"  : start,
    }
    r = requests.get(WFS_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_all_iris(bbox: tuple) -> gpd.GeoDataFrame:
    """Download all IRIS districts within the bounding box via WFS pagination."""
    pages, start = [], 0
    while True:
        data     = fetch_page(bbox, start, WFS_PAGE)
        features = data.get("features", [])
        if not features:
            break
        pages.extend(features)
        print(f"  {len(pages)} IRIS received …", end="\r", flush=True)
        if len(features) < WFS_PAGE:
            break
        start += WFS_PAGE
    print()

    if not pages:
        raise RuntimeError("WFS returned no IRIS districts for this bounding box.")

    return gpd.GeoDataFrame.from_features(pages, crs="EPSG:4326")


# ── Main ──────────────────────────────────────────────────────────────────────

def prepare() -> None:
    GEO.mkdir(parents=True, exist_ok=True)

    print("1/3  Computing network bounding box …")
    bbox = stops_bbox()
    print(f"     WGS-84 bbox: lon [{bbox[0]:.4f} → {bbox[2]:.4f}]"
          f"  lat [{bbox[1]:.4f} → {bbox[3]:.4f}]")

    print("2/3  Downloading IRIS GE boundaries from IGN Géoplateforme WFS …")
    iris = fetch_all_iris(bbox)
    print(f"     {len(iris)} IRIS districts downloaded")
    print(f"     Columns: {iris.columns.tolist()}")

    print("3/3  Saving GeoJSON …")
    iris.to_file(OUTPUT_FILE, driver="GeoJSON")
    print(f"\nSaved → {OUTPUT_FILE}")
    print("   Next step: python scripts/tisseo_indicators.py")


if __name__ == "__main__":
    prepare()
