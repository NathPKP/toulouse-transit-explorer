# toulouse-transit-explorer

> Interactive transit coverage and accessibility analysis for the **Tisséo** network in Toulouse, France.

Built with Python, Streamlit, GeoPandas, DuckDB, Folium and Plotly.

---

## Features

| Tab | Description |
|-----|-------------|
| **Network Coverage** | Choropleth map scoring each IRIS district (0–100) on 4 dimensions: service hours, peak frequency, line diversity, and spatial coverage. Address search to instantly locate any district |
| **My Commute** | Estimate transit travel time from every district to any destination — direct trips, 1 transfer, 2 transfers, with inter-stop walking (≤400 m) |
| **Multiple Destinations** | Weighted average commute from every district to several workplaces/places — helps choose where to live |
| **Service Pulse** | Animated hourly choropleth showing how service intensity evolves from 5 am to 11 pm, with a daily timeline chart |
| **Compare** | Side-by-side comparison of two IRIS districts across all coverage indicators and hourly activity |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Obtain data

**GTFS feed** — Download from Toulouse Open Data:  
https://data.toulouse-metropole.fr/explore/dataset/tisseo-gtfs/  
Save as `data/raw/tisseo_gtfs_v2.zip`.

**IRIS boundaries** — Fetched automatically by the pipeline from the IGN WFS service.

> **Note:** processed Parquet files (`data/parquet/`) and computed indicators (`data/geo/`) are included in the repository — you only need to re-run the pipeline if you want to update the data.

### 3. Run the data pipeline

```bash
# Step 1 — Ingest GTFS → Parquet
python scripts/tisseo_ingest.py

# Step 2 — (Optional) Validate data quality
python scripts/tisseo_check.py

# Step 3 — Download IRIS district boundaries
python scripts/tisseo_prepare_iris.py

# Step 4 — Compute coverage indicators
python scripts/tisseo_indicators.py
```

Or check for updates and re-run automatically:

```bash
python scripts/tisseo_update.py          # update if newer GTFS available
python scripts/tisseo_update.py --force  # force re-download
python scripts/tisseo_update.py --check  # check only, do not download
```

### 4. Launch the app

```bash
streamlit run app.py
```

---

## Project Structure

```
toulouse-transit-explorer/
├── app.py                      # Streamlit application (entry point)
├── config.py                   # Shared path constants
├── requirements.txt
├── LICENSE
├── .gitignore
├── .streamlit/
│   └── config.toml             # Streamlit theme & server config
├── data/
│   ├── README.md               # Data setup instructions
│   ├── raw/                    # GTFS zip (gitignored — download manually)
│   ├── parquet/                # Processed Parquet files (tracked in git)
│   └── geo/                    # IRIS boundaries + computed indicators
├── scripts/
│   ├── tisseo_ingest.py        # GTFS → Parquet pipeline
│   ├── tisseo_check.py         # Post-ingestion data quality checks
│   ├── tisseo_prepare_iris.py  # Download IRIS boundaries from IGN WFS
│   ├── tisseo_indicators.py    # Compute coverage scores + hourly data
│   ├── tisseo_update.py        # Auto-check and download newer GTFS
│   ├── tisseo_map.py           # Export standalone HTML coverage map
│   ├── tisseo_commute.py       # Commute routing engine + CLI map export
│   └── _map_index.py           # HTML index generator for outputs/
├── tests/
│   └── test_pipeline.py        # 26 sanity checks for pipeline outputs
└── outputs/                    # Generated HTML maps
```

---

## Architecture

### Data pipeline

```
tisseo_gtfs_v2.zip
    └── tisseo_ingest.py
            ├── stops.geoparquet
            ├── routes.parquet
            ├── trips.parquet
            └── stop_times.parquet  <- main analytical table

stops.geoparquet + stop_times.parquet
    └── tisseo_prepare_iris.py  ->  iris_toulouse.geojson
    └── tisseo_indicators.py
            ├── quartiers_indicators.parquet  (synthetic scores)
            └── hourly_indicators.parquet     (raw hourly counts, for Pulse)
```

### Routing algorithm (RAPTOR-style, 3 rounds)

1. **Round 0 — Direct**: query all origin stops reachable by a single vehicle trip to any destination stop.
2. **Walk expansion**: extend the reachable stop set to walking neighbours (≤400 m).
3. **Round 1 — 1 transfer**: for each stop reachable after expansion, find origin stops with a direct vehicle trip to it.
4. **Round 2 — 2 transfers**: repeat once more.

Total time = walk to stop + wait (½ headway) + vehicle ride + transfer penalties.

### Coverage score formula

```
score = 25% x amplitude_rank
      + 35% x peak_frequency_rank   (inverted: lower interval = better)
      + 20% x line_diversity_rank
      + 20% x spatial_coverage_rank
```

All components are normalised by **percentile rank** across all served districts, then rescaled to 0–100. Spatial coverage = % of district area within 500 m of an active stop (penalises large sparsely-served districts).

---

## Testing

```bash
pytest tests/
```

26 tests covering all pipeline outputs: file existence, schema validation, geometry integrity, data consistency and indicator ranges.

---

## Data Sources

| Source | Usage | License |
|--------|-------|---------|
| [Tisséo GTFS](https://data.toulouse-metropole.fr/explore/dataset/tisseo-gtfs/) | Transit timetable | Open Data (ODbL) |
| [IGN Géoplateforme WFS](https://data.geopf.fr/wfs) | IRIS district boundaries | Etalab Open License |
| [OpenStreetMap Nominatim](https://nominatim.openstreetmap.org/) | Address geocoding | ODbL |

---

## Contributing

Pull requests are welcome. Please open an issue first to discuss major changes.

```bash
# Development setup
pip install -r requirements.txt
streamlit run app.py
```

---

## License

[MIT](LICENSE) © toulouse-transit-explorer contributors
