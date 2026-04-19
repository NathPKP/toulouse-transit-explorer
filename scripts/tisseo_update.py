"""
tisseo_update.py
================
Check whether a newer Tisséo GTFS feed is available on Toulouse Open Data
and, if so, download it and re-run the data pipeline automatically.

Usage:
    python scripts/tisseo_update.py           # check + update if newer
    python scripts/tisseo_update.py --force   # force re-download
    python scripts/tisseo_update.py --check   # only report, do not download
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GEO, GTFS_ZIP, PARQUET, SCRIPTS

# ── Toulouse Open Data API ────────────────────────────────────────────────────

DATASET_ID   = "tisseo-gtfs"
API_INFO_URL = (
    f"https://data.toulouse-metropole.fr/api/explore/v2.1/"
    f"catalog/datasets/{DATASET_ID}/"
)
# The GTFS zip is exposed as a dataset attachment on the open data portal.
# The portal lists attachments at /files/ — we pick the first .zip found.
API_FILES_URL = (
    f"https://data.toulouse-metropole.fr/api/explore/v2.1/"
    f"catalog/datasets/{DATASET_ID}/attachments/"
)

META_KEY = "gtfs_modified"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 20) -> dict:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def remote_modified() -> str | None:
    """Return the dataset's last data-modified timestamp from the API."""
    try:
        data = _get(API_INFO_URL)
        return (
            data.get("metas", {})
                .get("default", {})
                .get("data_processed")
            or data.get("metas", {})
                   .get("default", {})
                   .get("modified")
        )
    except Exception as e:
        print(f"[warn] Could not reach Open Data API: {e}")
        return None


def local_modified() -> str | None:
    meta_file = PARQUET / "_meta.json"
    if not meta_file.exists():
        return None
    try:
        return json.loads(meta_file.read_text()).get(META_KEY)
    except Exception:
        return None


def save_modified(ts: str):
    meta_file = PARQUET / "_meta.json"
    meta = {}
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            pass
    meta[META_KEY] = ts
    meta_file.write_text(json.dumps(meta, indent=2))


def find_gtfs_download_url() -> str | None:
    """Find the download URL of the GTFS zip from the dataset's attachments."""
    try:
        data = _get(API_FILES_URL)
        attachments = data.get("results", []) or data if isinstance(data, list) else []
        for att in attachments:
            url = att.get("url", "")
            if url.lower().endswith(".zip"):
                return url
    except Exception as e:
        print(f"[warn] Could not fetch attachment list: {e}")
    return None


def download_gtfs(url: str):
    print(f"Downloading GTFS from:\n  {url}")
    GTFS_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        t0 = time.time()
        with open(GTFS_ZIP, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:5.1f}%  ({downloaded/1e6:.1f} MB)", end="", flush=True)
    elapsed = time.time() - t0
    print(f"\n  Done — {downloaded/1e6:.1f} MB in {elapsed:.1f} s")


def run_pipeline():
    steps = [
        (SCRIPTS / "tisseo_ingest.py",     "GTFS -> Parquet"),
        (SCRIPTS / "tisseo_indicators.py", "Coverage indicators"),
    ]
    for script, label in steps:
        print(f"\n[pipeline] {label}")
        result = subprocess.run([sys.executable, str(script)], cwd=str(SCRIPTS.parent))
        if result.returncode != 0:
            print(f"[error] Pipeline step failed: {script.name}")
            sys.exit(1)
    print("\n[pipeline] Complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Check for Tisséo GTFS updates and refresh the data pipeline"
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if already up to date")
    parser.add_argument("--check", action="store_true",
                        help="Only report update status, do not download")
    args = parser.parse_args()

    print("Checking for Tisséo GTFS updates...")
    remote = remote_modified()
    local  = local_modified()

    print(f"  Remote : {remote or 'unknown'}")
    print(f"  Local  : {local  or 'not downloaded yet'}")

    up_to_date = remote and local and remote == local

    if args.check:
        print("Up to date." if up_to_date else "Update available.")
        return

    if up_to_date and not args.force:
        print("Data is already up to date. Use --force to re-download.")
        return

    # Find download URL
    url = find_gtfs_download_url()
    if not url:
        print(
            "[error] Could not auto-detect the GTFS download URL.\n"
            f"Download manually from:\n"
            f"  https://data.toulouse-metropole.fr/explore/dataset/{DATASET_ID}/\n"
            f"Save as: {GTFS_ZIP}\n"
            "Then run: python scripts/tisseo_ingest.py && python scripts/tisseo_indicators.py"
        )
        sys.exit(1)

    download_gtfs(url)
    if remote:
        save_modified(remote)

    run_pipeline()


if __name__ == "__main__":
    main()
