# toulouse-transit-explorer — path constants
from pathlib import Path

ROOT     = Path(__file__).parent
RAW      = ROOT / "data" / "raw"
PARQUET  = ROOT / "data" / "parquet"
GEO      = ROOT / "data" / "geo"
SCRIPTS  = ROOT / "scripts"

GTFS_ZIP = RAW / "tisseo_gtfs_v2.zip"
