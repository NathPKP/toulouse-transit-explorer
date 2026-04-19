"""
tisseo_map.py
=============
Generate an interactive HTML choropleth map of Tisséo transit coverage
by IRIS district, coloured by the composite coverage score (0–100).

Usage:
    python scripts/tisseo_map.py

Prerequisites:
    python scripts/tisseo_indicators.py  →  data/geo/quartiers_indicators.parquet

Output:
    outputs/carte_desserte.html
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GEO, ROOT

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from branca.colormap import LinearColormap, StepColormap

# ── Paths ─────────────────────────────────────────────────────────────────────

INDICATORS_FILE = GEO / "quartiers_indicators.parquet"
OUTPUT_FILE     = ROOT / "outputs" / "carte_desserte.html"

# ── Colour palettes ───────────────────────────────────────────────────────────

PALETTE_SCORE = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]
PALETTE_CAT   = {
    "bien desservi" : "#1a9850",
    "moyen"         : "#fc8d59",
    "sous-desservi" : "#d73027",
    "non desservi"  : "#cccccc",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> gpd.GeoDataFrame:
    if not INDICATORS_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {INDICATORS_FILE}\n"
            "Run first: python scripts/tisseo_indicators.py"
        )
    gdf = gpd.read_parquet(INDICATORS_FILE)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    if "categorie" in gdf.columns:
        gdf["categorie"] = gdf["categorie"].astype(str)
    gdf["categorie"] = gdf["categorie"].fillna("non desservi").replace("nan", "non desservi")

    for col in ["score_desserte", "freq_pointe_min"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].round(1)

    return gdf


# ── District popup HTML ───────────────────────────────────────────────────────

def popup_html(row: pd.Series) -> str:
    score     = row.get("score_desserte", np.nan)
    cat       = row.get("categorie",      "—")
    amplitude = row.get("amplitude_h",    np.nan)
    lignes    = row.get("max_lignes",     np.nan)
    arrets    = row.get("max_arrets",     np.nan)
    freq      = row.get("freq_pointe_min", np.nan)
    passages  = row.get("total_passages", np.nan)
    nom       = row.get("iris_nom",       row.get("iris_code", "—"))
    code      = row.get("iris_code",      "—")

    cat_color = PALETTE_CAT.get(cat, "#999")

    def fmt(v, unit="", decimals=0):
        if pd.isna(v):
            return "<em>n/a</em>"
        return f"{v:.{decimals}f} {unit}".strip()

    bar_w   = int(score) if not pd.isna(score) else 0
    bar_col = PALETTE_CAT.get(cat, "#ccc")

    return f"""
    <div style="font-family:Arial,sans-serif;font-size:13px;min-width:240px">
      <b style="font-size:15px">{nom}</b><br>
      <span style="color:#888;font-size:11px">{code}</span>
      <hr style="margin:6px 0;border-color:#eee">

      <div style="margin:6px 0">
        <span style="background:{cat_color};color:#fff;padding:2px 8px;
              border-radius:10px;font-size:12px">{cat}</span>
      </div>

      <div style="margin:8px 0">
        <div style="font-size:12px;color:#555;margin-bottom:2px">Coverage score</div>
        <div style="background:#eee;border-radius:4px;height:14px;width:100%">
          <div style="background:{bar_col};height:14px;border-radius:4px;
               width:{bar_w}%"></div>
        </div>
        <div style="text-align:right;font-size:12px;font-weight:bold">
          {fmt(score, "/100", 1)}
        </div>
      </div>

      <table style="width:100%;border-collapse:collapse;font-size:12px">
        <tr style="border-bottom:1px solid #f0f0f0">
          <td style="color:#666;padding:3px 0">Daily service span</td>
          <td style="text-align:right;font-weight:bold">{fmt(amplitude, "h")}</td>
        </tr>
        <tr style="border-bottom:1px solid #f0f0f0">
          <td style="color:#666;padding:3px 0">Lines (max/h)</td>
          <td style="text-align:right;font-weight:bold">{fmt(lignes)}</td>
        </tr>
        <tr style="border-bottom:1px solid #f0f0f0">
          <td style="color:#666;padding:3px 0">Active stops (max/h)</td>
          <td style="text-align:right;font-weight:bold">{fmt(arrets)}</td>
        </tr>
        <tr style="border-bottom:1px solid #f0f0f0">
          <td style="color:#666;padding:3px 0">Peak frequency (7–9 am)</td>
          <td style="text-align:right;font-weight:bold">{fmt(freq, "min", 1)}</td>
        </tr>
        <tr>
          <td style="color:#666;padding:3px 0">Daily passages (total)</td>
          <td style="text-align:right;font-weight:bold">{fmt(passages)}</td>
        </tr>
      </table>
    </div>
    """


# ── Score choropleth layer ─────────────────────────────────────────────────────

def layer_score(gdf: gpd.GeoDataFrame) -> tuple[folium.FeatureGroup, LinearColormap]:
    colormap = LinearColormap(
        colors=PALETTE_SCORE,
        vmin=0, vmax=100,
        caption="Coverage score (0 – 100)",
    )

    fg = folium.FeatureGroup(name="Coverage score", show=True)

    for _, row in gdf.iterrows():
        score = row.get("score_desserte", np.nan)
        color = colormap(score) if not pd.isna(score) else "#cccccc"

        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, c=color: {
                "fillColor"   : c,
                "color"       : "#555",
                "weight"      : 0.6,
                "fillOpacity" : 0.75,
            },
            tooltip=folium.Tooltip(
                f"{row.get('iris_nom', row.get('iris_code', ''))}"
                f" — {score:.1f}/100" if not pd.isna(score) else " — n/a",
                sticky=True,
            ),
            popup=folium.Popup(popup_html(row), max_width=280),
        ).add_to(fg)

    return fg, colormap


# ── Category layer ────────────────────────────────────────────────────────────

def layer_categorie(gdf: gpd.GeoDataFrame) -> folium.FeatureGroup:
    fg = folium.FeatureGroup(name="Coverage category", show=False)

    for _, row in gdf.iterrows():
        cat   = row.get("categorie", "non desservi")
        color = PALETTE_CAT.get(cat, "#cccccc")

        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, c=color: {
                "fillColor"   : c,
                "color"       : "#555",
                "weight"      : 0.6,
                "fillOpacity" : 0.70,
            },
            tooltip=folium.Tooltip(
                f"{row.get('iris_nom', row.get('iris_code', ''))} — {cat}",
                sticky=True,
            ),
            popup=folium.Popup(popup_html(row), max_width=280),
        ).add_to(fg)

    return fg


# ── Legend HTML ───────────────────────────────────────────────────────────────

LEGEND_HTML = """
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;
     background:white;padding:12px 16px;border-radius:8px;
     box-shadow:0 2px 8px rgba(0,0,0,.25);font-family:Arial,sans-serif;
     font-size:12px;line-height:1.8">
  <b style="font-size:13px">Categories</b><br>
  <span style="display:inline-block;width:14px;height:14px;border-radius:3px;
        background:#1a9850;margin-right:6px;vertical-align:middle"></span>Well served (≥ 67)<br>
  <span style="display:inline-block;width:14px;height:14px;border-radius:3px;
        background:#fc8d59;margin-right:6px;vertical-align:middle"></span>Average (33 – 67)<br>
  <span style="display:inline-block;width:14px;height:14px;border-radius:3px;
        background:#d73027;margin-right:6px;vertical-align:middle"></span>Under-served (< 33)<br>
  <span style="display:inline-block;width:14px;height:14px;border-radius:3px;
        background:#cccccc;margin-right:6px;vertical-align:middle"></span>No service
</div>
"""

TITLE_HTML = """
<div style="position:fixed;top:14px;left:50%;transform:translateX(-50%);
     z-index:1000;background:white;padding:8px 20px;border-radius:8px;
     box-shadow:0 2px 8px rgba(0,0,0,.25);font-family:Arial,sans-serif;
     font-size:15px;font-weight:bold;white-space:nowrap">
  Tisséo Network Coverage — by IRIS District
</div>
"""


# ── Map builder ───────────────────────────────────────────────────────────────

def build_map(gdf: gpd.GeoDataFrame) -> folium.Map:
    centroid = gdf.geometry.union_all().centroid
    m = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=12,
        tiles="CartoDB positron",
        attr="© OpenStreetMap contributors © CARTO",
        scrollWheelZoom=False,
    )

    fg_score, colormap = layer_score(gdf)
    fg_cat             = layer_categorie(gdf)
    fg_score.add_to(m)
    fg_cat.add_to(m)
    colormap.add_to(m)

    folium.TileLayer("OpenStreetMap", name="OSM basemap", show=False).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    m.get_root().html.add_child(folium.Element(TITLE_HTML))
    m.get_root().html.add_child(folium.Element(LEGEND_HTML))

    bounds = gdf.total_bounds
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    return m


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading indicators …")
    gdf    = load_data()
    scored = gdf["score_desserte"].notna().sum()
    print(f"  {len(gdf)} IRIS districts  ({scored} with score)")

    print("Building map …")
    m = build_map(gdf)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUTPUT_FILE))

    from _map_index import generate_index
    generate_index(OUTPUT_FILE.parent)

    print(f"\nMap   → {OUTPUT_FILE}")
    print(f"Index → {OUTPUT_FILE.parent / 'index.html'}")


if __name__ == "__main__":
    main()
