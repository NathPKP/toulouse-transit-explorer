"""
app.py — Toulouse Transit Explorer
Launch with: streamlit run app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

import json
import pickle

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from branca.colormap import LinearColormap

import config as cfg
from tisseo_commute import (
    COMMUTE_CAP, WALK_TRANSFER_MAX,
    _open_con, build_map, build_stop_times, compute_walk_pairs,
    compute_iris_commute, compute_wait, find_dest_stops, parse_dest,
)
from tisseo_map import build_map as build_desserte_map, load_data as load_desserte

# ── Translations ──────────────────────────────────────────────────────────────

_L: dict[str, dict[str, str]] = {
    "fr": {
        "app_title"        : "Réseau Tisséo — Toulouse",
        "app_sub"          : "Desserte et accessibilité par quartier IRIS",
        "lang_label"       : "Langue",
        "tab_network"      : "Desserte réseau",
        "tab_commute"      : "Mon trajet",
        "tab_multi"        : "Destinations multiples",
        "tab_pulse"        : "Pulsation réseau",
        # Tab 1
        "t1_title"         : "Score de desserte par quartier IRIS",
        "t1_caption"       : ("Composite pondéré de 4 indicateurs : amplitude horaire des "
                              "services, fréquence en heure de pointe (7h–9h), diversité des "
                              "lignes, et couverture spatiale (% du quartier à moins de 500 m "
                              "d'un arrêt). Score 0–100 normalisé par rang percentile."),
        "t1_iris"          : "Quartiers IRIS",
        "t1_well"          : "Bien desservis",
        "t1_medium"        : "Desserte moyenne",
        "t1_under"         : "Sous-desservis",
        "t1_noservice"     : "Non desservis",
        "t1_rebuild"       : "Reconstruire la carte",
        "t1_methodology"   : "Comment est calculé le score ?",
        "t1_method_body"   : (
            "Le score composite intègre quatre dimensions :\n\n"
            "- **Amplitude (25 %)** — nombre d'heures de fonctionnement du service dans la zone\n"
            "- **Fréquence pointe (35 %)** — intervalles moyens entre passages 7h–9h "
            "(plus c'est court, mieux c'est)\n"
            "- **Diversité lignes (20 %)** — nombre max de lignes desservant la zone sur une heure\n"
            "- **Couverture spatiale (20 %)** — part de la superficie du quartier à ≤500 m "
            "d'un arrêt actif (pénalise les grands quartiers peu couverts)\n\n"
            "Chaque composante est normalisée par rang percentile parmi l'ensemble des "
            "quartiers desservis, puis repondérée sur 0–100."
        ),
        # Tab 2
        "t2_title"         : "Temps de trajet depuis chaque quartier",
        "t2_caption"       : ("Estimez combien de temps il faut pour rejoindre votre lieu de "
                              "travail en transport en commun depuis n'importe quel quartier. "
                              "L'algorithme explore les trajets directs, à 1 et 2 correspondances, "
                              "avec marche possible entre arrêts voisins (≤400 m)."),
        "t2_step1"         : "Étape 1 — Destination",
        "t2_dest_ph"       : "Airbus Toulouse  |  CHU Purpan  |  43.5639, 1.3833",
        "t2_dest_hint"     : "Adresse, nom de lieu ou coordonnées GPS (lat, lon)",
        "t2_verify"        : "Vérifier",
        "t2_geo_ok"        : "Adresse validée",
        "t2_geo_pending"   : "Adresse non vérifiée — cliquez sur Vérifier.",
        "t2_step2"         : "Étape 2 — Paramètres",
        "t2_hour"          : "Heure de départ",
        "t2_walk"          : "Distance max à pied jusqu'à l'arrêt (m)",
        "t2_analyze"       : "Lancer l'analyse",
        "t2_stale"         : "Paramètres modifiés — relancez l'analyse pour mettre à jour.",
        "t2_info"          : ("Saisissez une destination, vérifiez-la, puis lancez l'analyse. "
                              "La carte colorise chaque quartier selon le temps de trajet estimé."),
        "t2_clear"         : "Réinitialiser",
        "t2_min"           : "Minimum",
        "t2_median"        : "Médiane",
        "t2_max"           : "Maximum",
        "t2_direct"        : "Directs",
        "t2_1corresp"      : "1 corresp.",
        "t2_2corresp"      : "2 corresp.",
        "t2_nolink"        : "Sans liaison",
        "t2_top"           : "Top 10 quartiers les plus proches",
        "t2_col_district"  : "Quartier",
        "t2_col_time"      : "Trajet total",
        "t2_col_mode"      : "Mode",
        "t2_col_stop"      : "Arrêt de départ",
        "t2_col_score"     : "Score desserte",
        "t2_col_cat"       : "Catégorie",
        # Tab 3
        "t3_title"         : "Trouver votre quartier idéal",
        "t3_caption"       : ("Renseignez vos lieux fréquents — travail, famille, loisirs — "
                              "et leur importance. La carte affiche la moyenne pondérée des "
                              "temps de trajet par quartier pour vous aider à choisir où vivre."),
        "t3_step1"         : "Étape 1 — Vos destinations",
        "t3_name_ph"       : "Mon travail",
        "t3_addr_ph"       : "ex : Airbus Toulouse  |  43.56, 1.38",
        "t3_verify"        : "Vérifier",
        "t3_weight_label"  : "Importance",
        "t3_add"           : "+ Ajouter une destination",
        "t3_step2"         : "Étape 2 — Paramètres",
        "t3_run"           : "Calculer les trajets",
        "t3_step3"         : "Étape 3 — Ajuster l'importance",
        "t3_w_caption"     : "Modifiez les curseurs puis cliquez sur Mettre à jour.",
        "t3_update"        : "Mettre à jour la carte",
        "t3_reset"         : "Réinitialiser tout",
        "t3_no_result"     : "Vérifiez au moins 2 adresses puis cliquez sur Calculer.",
        "t3_pending"       : "Non vérifiée",
        "t3_computed"      : "calculé",
        "t3_top"           : "Top 20 quartiers optimaux",
        "t3_avg"           : "Temps moyen",
        "t3_worst"         : "Pire trajet",
        "t3_all_conn"      : "Toutes dest.",
        "t3_best"          : "Meilleur score",
        "t3_iris_conn"     : "IRIS tous connectés",
        "t3_nolink"        : "Sans liaison",
        "t3_cat"           : "Catégorie",
        "t3_score"         : "Desserte",
        # Tab 4 — Service Pulse
        "tab_pulse"        : "Pulsation réseau",
        "t4_title"         : "Intensité du service heure par heure",
        "t4_caption"       : ("Observez comment le réseau Tisséo s'anime de 5h à 23h. "
                              "Chaque quartier est colorié selon le nombre de passages TC. "
                              "Appuyez sur ▶ pour lancer l'animation, ou faites glisser "
                              "le curseur pour explorer une heure précise."),
        "t4_active_iris"   : "Quartiers desservis",
        "t4_total_day"     : "Passages / journée",
        "t4_peak_hour"     : "Heure de pointe",
        "t4_peak_trips"    : "Passages (pointe)",
        "t4_nodata"        : ("Données horaires introuvables. Lancez d'abord :\n"
                              "`python scripts/tisseo_indicators.py`"),
        "t4_timeline"      : "Activité réseau sur la journée",
        "t4_trips"         : "Passages",
        "t4_lines"         : "Lignes",
        "t4_stops"         : "Arrêts",
        "t4_freq"          : "Intervalle",
        "t4_top"           : "Top 10 quartiers — heure de pointe",
        # Tab 5 — Compare
        "tab_compare"       : "Comparer",
        "t5_title"          : "Comparer deux quartiers",
        "t5_caption"        : "Sélectionnez deux quartiers IRIS pour comparer leurs indicateurs de desserte côte à côte.",
        "t5_d1"             : "Quartier A",
        "t5_d2"             : "Quartier B",
        "t5_same"           : "Sélectionnez deux quartiers différents.",
        "t5_hourly"         : "Activité horaire comparée",
        "t5_amplitude"      : "Amplitude",
        "t5_freq"           : "Fréq. pointe",
        "t5_lines"          : "Lignes max",
        "t5_coverage"       : "Couverture spatiale",
        "t5_area"           : "Surface",
        "t5_passages"       : "Passages / jour",
        # Address search (Tab 1)
        "t1_search_label"   : "Trouver mon quartier",
        "t1_search_ph"      : "Adresse ou nom de lieu...",
        "t1_search_btn"     : "Localiser",
        "t1_search_ok"      : "est dans le quartier",
        "t1_search_notfound": "Adresse hors du périmètre ou introuvable.",
        # Sidebar
        "sb_settings"      : "Paramètres",
        "sb_data"          : "Sources de données",
        "sb_gtfs"          : "Réseau TC : GTFS Tisséo (Toulouse)",
        "sb_iris"          : "Zones : contours IRIS IGN/INSEE",
        "sb_nominatim"     : "Géocodage : OpenStreetMap Nominatim",
        "sb_method"        : "Méthodologie",
        "sb_score_md"      : ("**Score desserte** : composite de 4 indicateurs normalisés par "
                              "rang percentile (amplitude, fréquence, lignes, couverture spatiale)."),
        "sb_commute_md"    : ("**Temps de trajet** : marche jusqu'à l'arrêt + attente + TC. "
                              "3 rounds : direct, 1 correspondance, 2 correspondances. "
                              "Marche inter-arrêts ≤400 m autorisée."),
        "sb_pulse_md"      : ("**Pulsation réseau** : nombre de passages TC par quartier IRIS "
                              "et par heure, calculé depuis le GTFS Tisséo."),
        "sb_iris_md"       : ("**IRIS** (Ilots Regroupés pour l'Information Statistique) : "
                              "découpage géographique INSEE d'environ 2 000 habitants par zone."),
        "sb_rebuild"       : "Reconstruire la carte desserte",
        "sb_version"       : "Toulouse Transit Explorer v3",
        # Errors / misc
        "err_no_addr"      : "Saisissez une adresse.",
        "err_no_stop"      : "Aucun arrêt Tisséo dans 500 m — essayez des coordonnées précises.",
        "err_min2"         : "Vérifiez au moins 2 adresses avant de lancer le calcul.",
        "computing"        : "Calcul en cours...",
        # Quick wins
        "t2_export"        : "Télécharger les résultats (CSV)",
        "t3_export"        : "Télécharger le classement (CSV)",
        "t5_radar"         : "Profil de desserte",
        "t5_score_lbl"     : "Score global",
    },
    "en": {
        "app_title"        : "Tisséo Network — Toulouse",
        "app_sub"          : "Transit coverage and accessibility by IRIS district",
        "lang_label"       : "Language",
        "tab_network"      : "Network coverage",
        "tab_commute"      : "My commute",
        "tab_multi"        : "Multiple destinations",
        "tab_pulse"        : "Service Pulse",
        # Tab 1
        "t1_title"         : "Transit coverage score by IRIS district",
        "t1_caption"       : ("Weighted composite of 4 indicators: service hours span, "
                              "peak-hour frequency (7–9 am), line diversity, and spatial "
                              "coverage (% of district within 500 m of a stop). "
                              "Score 0–100 normalised by percentile rank."),
        "t1_iris"          : "IRIS districts",
        "t1_well"          : "Well served",
        "t1_medium"        : "Average coverage",
        "t1_under"         : "Under-served",
        "t1_noservice"     : "No service",
        "t1_rebuild"       : "Rebuild map",
        "t1_methodology"   : "How is the score calculated?",
        "t1_method_body"   : (
            "The composite score integrates four dimensions:\n\n"
            "- **Span (25%)** — number of service hours active in the district\n"
            "- **Peak frequency (35%)** — average interval between trips 7–9 am "
            "(shorter is better)\n"
            "- **Line diversity (20%)** — max lines serving the district per hour\n"
            "- **Spatial coverage (20%)** — share of district area within ≤500 m "
            "of an active stop (penalises large districts with sparse coverage)\n\n"
            "Each component is normalised by percentile rank among all served districts, "
            "then rescaled to 0–100."
        ),
        # Tab 2
        "t2_title"         : "Travel time from each district",
        "t2_caption"       : ("Estimate how long it takes to reach your workplace "
                              "by public transit from any Toulouse district. "
                              "The algorithm covers direct trips, 1 transfer and 2 transfers, "
                              "with walking between nearby stops (≤400 m)."),
        "t2_step1"         : "Step 1 — Destination",
        "t2_dest_ph"       : "Airbus Toulouse  |  CHU Purpan  |  43.5639, 1.3833",
        "t2_dest_hint"     : "Address, place name or GPS coordinates (lat, lon)",
        "t2_verify"        : "Verify",
        "t2_geo_ok"        : "Address verified",
        "t2_geo_pending"   : "Address not verified — click Verify.",
        "t2_step2"         : "Step 2 — Parameters",
        "t2_hour"          : "Departure hour",
        "t2_walk"          : "Max walking distance to stop (m)",
        "t2_analyze"       : "Run analysis",
        "t2_stale"         : "Parameters changed — re-run to update the map.",
        "t2_info"          : ("Enter a destination, verify it, then run the analysis. "
                              "The map colours each district by estimated travel time."),
        "t2_clear"         : "Reset",
        "t2_min"           : "Minimum",
        "t2_median"        : "Median",
        "t2_max"           : "Maximum",
        "t2_direct"        : "Direct",
        "t2_1corresp"      : "1 transfer",
        "t2_2corresp"      : "2 transfers",
        "t2_nolink"        : "No link",
        "t2_top"           : "Top 10 best-connected districts",
        "t2_col_district"  : "District",
        "t2_col_time"      : "Total trip",
        "t2_col_mode"      : "Mode",
        "t2_col_stop"      : "Departure stop",
        "t2_col_score"     : "Coverage score",
        "t2_col_cat"       : "Category",
        # Tab 3
        "t3_title"         : "Find your ideal neighbourhood",
        "t3_caption"       : ("Enter your regular destinations — work, family, leisure — "
                              "and their relative importance. The map shows the weighted "
                              "average travel time per district to help you decide where to live."),
        "t3_step1"         : "Step 1 — Your destinations",
        "t3_name_ph"       : "My workplace",
        "t3_addr_ph"       : "e.g. Airbus Toulouse  |  43.56, 1.38",
        "t3_verify"        : "Verify",
        "t3_weight_label"  : "Importance",
        "t3_add"           : "+ Add destination",
        "t3_step2"         : "Step 2 — Parameters",
        "t3_run"           : "Calculate trips",
        "t3_step3"         : "Step 3 — Adjust importance",
        "t3_w_caption"     : "Adjust sliders then click Update.",
        "t3_update"        : "Update map",
        "t3_reset"         : "Reset all",
        "t3_no_result"     : "Verify at least 2 addresses then click Calculate.",
        "t3_pending"       : "Not verified",
        "t3_computed"      : "calculated",
        "t3_top"           : "Top 20 optimal districts",
        "t3_avg"           : "Avg. time",
        "t3_worst"         : "Worst trip",
        "t3_all_conn"      : "All dest.",
        "t3_best"          : "Best score",
        "t3_iris_conn"     : "IRIS all connected",
        "t3_nolink"        : "No link",
        "t3_cat"           : "Category",
        "t3_score"         : "Coverage",
        # Tab 4 — Service Pulse
        "tab_pulse"        : "Service Pulse",
        "t4_title"         : "Service intensity hour by hour",
        "t4_caption"       : ("Watch the Tisséo network come alive from 5 am to 11 pm. "
                              "Each district is coloured by the number of transit trips. "
                              "Press ▶ to play the animation, or drag the slider "
                              "to explore any specific hour."),
        "t4_active_iris"   : "Served districts",
        "t4_total_day"     : "Trips / day",
        "t4_peak_hour"     : "Peak hour",
        "t4_peak_trips"    : "Peak trips",
        "t4_nodata"        : ("Hourly data not found. Run first:\n"
                              "`python scripts/tisseo_indicators.py`"),
        "t4_timeline"      : "Network activity throughout the day",
        "t4_trips"         : "Trips",
        "t4_lines"         : "Lines",
        "t4_stops"         : "Stops",
        "t4_freq"          : "Interval",
        "t4_top"           : "Top 10 districts — peak hour",
        # Tab 5 — Compare
        "tab_compare"       : "Compare",
        "t5_title"          : "Compare two districts",
        "t5_caption"        : "Select two IRIS districts to compare their transit coverage indicators side by side.",
        "t5_d1"             : "District A",
        "t5_d2"             : "District B",
        "t5_same"           : "Please select two different districts.",
        "t5_hourly"         : "Hourly activity comparison",
        "t5_amplitude"      : "Amplitude",
        "t5_freq"           : "Peak frequency",
        "t5_lines"          : "Max lines",
        "t5_coverage"       : "Spatial coverage",
        "t5_area"           : "Area",
        "t5_passages"       : "Trips / day",
        # Address search (Tab 1)
        "t1_search_label"   : "Find my district",
        "t1_search_ph"      : "Address or place name...",
        "t1_search_btn"     : "Locate",
        "t1_search_ok"      : "is in district",
        "t1_search_notfound": "Address outside the Toulouse perimeter or not found.",
        # Sidebar
        "sb_settings"      : "Settings",
        "sb_data"          : "Data sources",
        "sb_gtfs"          : "Transit network: GTFS Tisséo (Toulouse)",
        "sb_iris"          : "Districts: IRIS boundaries IGN/INSEE",
        "sb_nominatim"     : "Geocoding: OpenStreetMap Nominatim",
        "sb_method"        : "Methodology",
        "sb_score_md"      : ("**Coverage score**: composite of 4 indicators normalised by "
                              "percentile rank (span, frequency, lines, spatial coverage)."),
        "sb_commute_md"    : ("**Travel time**: walk to stop + wait + transit. "
                              "3 rounds: direct, 1 transfer, 2 transfers. "
                              "Inter-stop walking ≤400 m allowed."),
        "sb_pulse_md"      : ("**Service Pulse**: transit trip count per IRIS district "
                              "per hour, computed from the Tisséo GTFS feed."),
        "sb_iris_md"       : ("**IRIS** (French statistical district): INSEE geographic unit "
                              "of roughly 2,000 inhabitants per zone."),
        "sb_rebuild"       : "Rebuild coverage map",
        "sb_version"       : "Toulouse Transit Explorer v3",
        # Errors / misc
        "err_no_addr"      : "Please enter an address.",
        "err_no_stop"      : "No Tisséo stop within 500 m — try precise coordinates.",
        "err_min2"         : "Verify at least 2 addresses before calculating.",
        "computing"        : "Computing...",
        # Quick wins
        "t2_export"        : "Download results (CSV)",
        "t3_export"        : "Download ranking (CSV)",
        "t5_radar"         : "Coverage profile",
        "t5_score_lbl"     : "Overall score",
    },
}


def t(key: str) -> str:
    lang = st.session_state.get("lang", "fr")
    return _L.get(lang, _L["fr"]).get(key, key)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Toulouse Transit Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* Hide Streamlit branding */
  [data-testid="stAppDeployButton"] {display:none !important}
  #MainMenu li:has(a[href*="streamlit"]),
  #MainMenu li:has(span[data-testid="stMainMenuReportBug"]),
  #MainMenu li:has(span[data-testid="stMainMenuAbout"]) {display:none !important}

  .block-container{padding-top:1rem;padding-bottom:1rem}

  .mcard{background:#f8fafc;border-radius:10px;padding:14px 12px;
         border:1px solid #e2e8f0;text-align:center;height:100%}
  .mcard .val{font-size:22px;font-weight:700;line-height:1.1}
  .mcard .lbl{font-size:11px;color:#64748b;margin-top:4px;font-weight:500}

  .step-row{display:flex;align-items:center;gap:10px;margin:18px 0 8px}
  .step-num{background:#1d4ed8;color:#fff;border-radius:50%;
            width:26px;height:26px;display:flex;align-items:center;
            justify-content:center;font-weight:700;font-size:13px;flex-shrink:0}
  .step-title{font-size:15px;font-weight:600;color:#1e293b}

  .dest-card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
             padding:11px 14px;margin-bottom:8px}
  .geo-ok{color:#10b981;font-size:12px;margin-top:3px}
  .geo-pending{color:#94a3b8;font-size:12px;margin-top:3px}
  .geo-err{color:#ef4444;font-size:12px;margin-top:3px}

  .stale-warn{background:#fffbeb;border:1px solid #fde68a;border-radius:6px;
              padding:8px 14px;font-size:12px;color:#92400e;margin:8px 0}

  div[data-testid="stForm"]{border:none;padding:0}
  section[data-testid="stSidebar"] .block-container{padding-top:0.5rem}

  @media (max-width: 768px) {
    .mcard .val{font-size:16px !important}
    .mcard{padding:8px 6px}
    .step-title{font-size:13px}
    .dest-card{padding:8px 10px}
    .block-container{padding-left:0.5rem !important;padding-right:0.5rem !important}
  }
</style>""", unsafe_allow_html=True)

PALETTE    = ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"]
DEST_COLS  = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]
DEST_ICONS = ["briefcase", "home", "university", "hospital-o", "star", "map-marker"]

HOURLY_FILE      = cfg.GEO / "hourly_indicators.parquet"
PULSE_CACHE_PATH = cfg.GEO / "_pulse_figure.pkl"

# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading network data…")
def get_desserte_data() -> gpd.GeoDataFrame:
    return load_desserte()

@st.cache_resource(show_spinner="Loading stops…")
def get_stop_points() -> gpd.GeoDataFrame:
    sp = gpd.read_parquet(cfg.PARQUET / "stops.geoparquet")
    sp = sp[sp["location_type"] == 0].copy()
    sp["lat"] = sp.geometry.y
    sp["lon"] = sp.geometry.x
    return sp

@st.cache_resource(show_spinner="Pre-computing transfer walk pairs…")
def get_walk_pairs() -> pd.DataFrame:
    return compute_walk_pairs(get_stop_points(), WALK_TRANSFER_MAX)

@st.cache_resource(show_spinner="Loading hourly data…")
def get_hourly_data() -> pd.DataFrame:
    if not HOURLY_FILE.exists():
        return pd.DataFrame()
    return pd.read_parquet(HOURLY_FILE)

@st.cache_data(
    show_spinner=False,
    hash_funcs={gpd.GeoDataFrame: lambda g: tuple(g["iris_code"].tolist())},
)
def run_commute_data(lat: float, lon: float, label: str, heure: int, marche: int):
    indicators = get_desserte_data()
    sp         = get_stop_points()
    wp         = get_walk_pairs()
    try:
        dest_stops = find_dest_stops(sp, lat, lon)
    except ValueError as e:
        return None, str(e)
    con    = _open_con()
    wait   = compute_wait(con, heure)
    st_df  = build_stop_times(con, dest_stops, sp, heure, walk_pairs=wp)
    con.close()
    result = compute_iris_commute(indicators, sp, st_df, wait, marche)
    return result, None


# ── Pulse disk cache ─────────────────────────────────────────────────────────

def _load_pulse_cache():
    if not PULSE_CACHE_PATH.exists():
        return None
    if HOURLY_FILE.exists() and PULSE_CACHE_PATH.stat().st_mtime < HOURLY_FILE.stat().st_mtime:
        return None
    try:
        with open(PULSE_CACHE_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def _save_pulse_cache(fig):
    try:
        with open(PULSE_CACHE_PATH, "wb") as f:
            pickle.dump(fig, f)
    except Exception:
        pass


# ── URL state ────────────────────────────────────────────────────────────────

def _init_from_query_params():
    if st.session_state.get("_qp_loaded"):
        return
    st.session_state["_qp_loaded"] = True
    params = st.query_params
    if "lat" not in params:
        return
    try:
        lat   = float(params["lat"])
        lon   = float(params["lon"])
        label = params.get("label", f"{lat:.5f}, {lon:.5f}")
        addr  = params.get("addr", label)
        h     = int(params.get("h", 8))
        m_val = int(params.get("m", 800))
        st.session_state.setdefault("c1_geo",  (lat, lon, label, addr))
        st.session_state.setdefault("c1_addr", addr)
        st.session_state.setdefault("c1_h",    h)
        st.session_state.setdefault("c1_m",    m_val)
    except (ValueError, KeyError):
        pass


# ── Radar chart ───────────────────────────────────────────────────────────────

def build_radar_chart(r1, r2, name1, name2, gdf):
    axes = [
        (t("t5_amplitude"), "amplitude_h",     False),
        (t("t5_freq"),      "freq_pointe_min",  True),
        (t("t5_lines"),     "max_lignes",       False),
        (t("t5_coverage"),  "coverage_pct",     False),
        (t("t5_score_lbl"), "score_desserte",   False),
    ]
    labels = [a[0] for a in axes]
    vals1, vals2 = [], []
    for _, col, invert in axes:
        col_min = float(gdf[col].min())
        col_max = float(gdf[col].max())
        rng = col_max - col_min or 1
        v1 = float(r1.get(col) or col_min)
        v2 = float(r2.get(col) or col_min)
        n1 = (v1 - col_min) / rng * 100
        n2 = (v2 - col_min) / rng * 100
        if invert:
            n1, n2 = 100 - n1, 100 - n2
        vals1.append(round(n1, 1))
        vals2.append(round(n2, 1))
    # close the polygon
    labels += [labels[0]]
    vals1  += [vals1[0]]
    vals2  += [vals2[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals1, theta=labels, fill="toself", name=name1,
        line_color="#3b82f6", fillcolor="rgba(59,130,246,0.15)"))
    fig.add_trace(go.Scatterpolar(
        r=vals2, theta=labels, fill="toself", name=name2,
        line_color="#10b981", fillcolor="rgba(16,185,129,0.15)"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True, height=380,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


# ── UI helpers ────────────────────────────────────────────────────────────────

def mcard(value: str, label: str, color: str = "#1e293b"):
    st.markdown(
        f'<div class="mcard"><div class="val" style="color:{color}">{value}</div>'
        f'<div class="lbl">{label}</div></div>',
        unsafe_allow_html=True)

def step_header(n: int, title: str):
    st.markdown(
        f'<div class="step-row"><div class="step-num">{n}</div>'
        f'<span class="step-title">{title}</span></div>',
        unsafe_allow_html=True)

def _fmt(v, unit="", dec=0):
    return f"{v:.{dec}f} {unit}".strip() if not pd.isna(v) else "—"


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        lang_options = {"Français": "fr", "English": "en"}
        cur_label    = "Français" if st.session_state.get("lang", "fr") == "fr" else "English"
        chosen = st.radio(t("lang_label"), list(lang_options.keys()),
                          index=list(lang_options.keys()).index(cur_label),
                          horizontal=True)
        if lang_options[chosen] != st.session_state.get("lang", "fr"):
            st.session_state.lang = lang_options[chosen]
            st.rerun()

        st.divider()
        st.markdown(f"**{t('sb_data')}**")
        st.caption(t("sb_gtfs"))
        st.caption(t("sb_iris"))
        st.caption(t("sb_nominatim"))

        st.divider()
        st.markdown(f"**{t('sb_method')}**")
        st.markdown(t("sb_score_md"))
        st.markdown(t("sb_commute_md"))
        st.markdown(t("sb_pulse_md"))
        st.markdown(t("sb_iris_md"))

        st.divider()
        if st.button(t("sb_rebuild"), use_container_width=True):
            st.session_state.pop("desserte_html", None)
            st.rerun()

        st.caption(t("sb_version"))


# ── Tab 1: Network coverage ───────────────────────────────────────────────────

def page_desserte():
    st.subheader(t("t1_title"))
    st.caption(t("t1_caption"))

    gdf = get_desserte_data()
    cat = gdf["categorie"].astype(str)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: mcard(str(len(gdf)),                           t("t1_iris"))
    with c2: mcard(str(int((cat == "bien desservi").sum())), t("t1_well"),     "#1a9850")
    with c3: mcard(str(int((cat == "moyen").sum())),         t("t1_medium"),   "#f59e0b")
    with c4: mcard(str(int((cat == "sous-desservi").sum())), t("t1_under"),    "#d73027")
    with c5: mcard(str(int((cat == "non desservi").sum())),  t("t1_noservice"),"#94a3b8")

    with st.expander(t("t1_methodology"), expanded=False):
        st.markdown(t("t1_method_body"))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Address search ────────────────────────────────────────────────────────
    st.markdown(f"**{t('t1_search_label')}**")
    col_addr, col_btn = st.columns([7, 1])
    with col_addr:
        search_addr = st.text_input(
            t("t1_search_label"), key="d_search_addr",
            placeholder=t("t1_search_ph"), label_visibility="collapsed")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button(t("t1_search_btn"), key="d_search_btn",
                               use_container_width=True)

    if search_btn and search_addr.strip():
        try:
            lat, lon, label = parse_dest(search_addr)
            from shapely.geometry import Point as _Point
            pt = gpd.GeoDataFrame({"geometry": [_Point(lon, lat)]}, crs="EPSG:4326")
            joined = gpd.sjoin(
                pt,
                gdf[["iris_nom", "iris_code", "score_desserte", "categorie", "geometry"]],
                how="left", predicate="within",
            )
            if not joined.empty and not pd.isna(joined.iloc[0].get("iris_code")):
                r = joined.iloc[0]
                st.session_state["d_found"] = {
                    "lat": lat, "lon": lon, "label": label, "addr": search_addr,
                    "iris_nom": r["iris_nom"], "iris_code": r["iris_code"],
                    "score": r["score_desserte"], "cat": str(r["categorie"]),
                }
            else:
                st.session_state["d_found"] = {
                    "lat": lat, "lon": lon, "label": label,
                    "addr": search_addr, "iris_nom": None,
                }
            st.session_state.pop("desserte_html", None)
        except ValueError as e:
            st.error(str(e))

    _CAT_COL = {"bien desservi": "#1a9850", "moyen": "#f59e0b",
                "sous-desservi": "#d73027"}
    found = st.session_state.get("d_found")
    if found and found.get("addr") == search_addr and search_addr.strip():
        if found.get("iris_nom"):
            cat    = found["cat"]
            cc     = _CAT_COL.get(cat, "#94a3b8")
            sc_str = f"{found['score']:.0f}/100" if not pd.isna(found["score"]) else "—"
            st.markdown(
                f'<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;'
                f'padding:10px 16px;margin-bottom:8px">'
                f'&#128205; <b>{found["label"]}</b> {t("t1_search_ok")} '
                f'<b>{found["iris_nom"]}</b> — Score : '
                f'<b style="color:{cc}">{sc_str}</b> '
                f'<span style="background:{cc};color:#fff;font-size:10px;'
                f'padding:2px 8px;border-radius:8px;margin-left:4px">{cat}</span>'
                f'</div>',
                unsafe_allow_html=True)
        else:
            st.warning(t("t1_search_notfound"))

    if "desserte_html" not in st.session_state:
        with st.spinner("Construction de la carte…"):
            m = build_desserte_map(gdf)
            if found and found.get("iris_nom"):
                folium.Marker(
                    [found["lat"], found["lon"]],
                    tooltip=found["label"],
                    popup=folium.Popup(
                        f"<b>{found['label']}</b><br>{found['iris_nom']}", max_width=200),
                    icon=folium.Icon(color="blue", icon="map-marker", prefix="fa"),
                ).add_to(m)
            st.session_state.desserte_html = m.get_root().render()
    components.html(st.session_state.desserte_html, height=740)


# ── Tab 2: Commute planner ────────────────────────────────────────────────────

def page_commute():
    st.subheader(t("t2_title"))
    st.caption(t("t2_caption"))

    step_header(1, t("t2_step1"))
    col_addr, col_btn, col_clear = st.columns([5, 1, 1])
    with col_addr:
        addr = st.text_input(
            t("t2_dest_hint"), key="c1_addr",
            placeholder=t("t2_dest_ph"), label_visibility="visible")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        verify = st.button(t("t2_verify"), key="c1_verify", use_container_width=True)
    with col_clear:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(t("t2_clear"), key="c1_clear", use_container_width=True):
            for k in ["c1_geo", "c1_result", "c1_map_html", "c1_dest", "c1_params"]:
                st.session_state.pop(k, None)
            st.rerun()

    if verify and addr.strip():
        with st.spinner("Vérification…"):
            try:
                lat, lon, label = parse_dest(addr)
                st.session_state["c1_geo"] = (lat, lon, label, addr)
            except ValueError as e:
                st.error(str(e))
                st.session_state.pop("c1_geo", None)

    geo = st.session_state.get("c1_geo")
    if geo and geo[3] == addr:
        st.markdown(
            f'<div class="geo-ok">&#10003; {geo[2]} &nbsp;—&nbsp; '
            f'{geo[0]:.5f}, {geo[1]:.5f}</div>',
            unsafe_allow_html=True)
    elif addr.strip():
        st.markdown(f'<div class="geo-pending">{t("t2_geo_pending")}</div>',
                    unsafe_allow_html=True)

    step_header(2, t("t2_step2"))
    col1, col2 = st.columns(2)
    with col1: heure  = st.slider(t("t2_hour"),  5, 23, 8,    key="c1_h")
    with col2: marche = st.slider(t("t2_walk"), 200, 1500, 800, 100, key="c1_m")

    run = st.button(t("t2_analyze"), type="primary", use_container_width=True, key="c1_run")

    if run:
        if not addr.strip():
            st.error(t("err_no_addr"))
        else:
            if geo and geo[3] == addr:
                lat, lon, label = geo[:3]
                coords_ok = True
            else:
                with st.spinner("Géocodage…"):
                    try:
                        lat, lon, label = parse_dest(addr)
                        st.session_state["c1_geo"] = (lat, lon, label, addr)
                        coords_ok = True
                    except ValueError as e:
                        st.error(str(e))
                        coords_ok = False

            if coords_ok:
                with st.spinner(t("computing")):
                    result, err = run_commute_data(lat, lon, label, heure, marche)
                if result is None:
                    st.error(err)
                    st.session_state.pop("c1_result", None)
                    st.session_state.pop("c1_map_html", None)
                else:
                    with st.spinner("Construction de la carte…"):
                        m = build_map(result, lat, lon, label, heure)
                    st.session_state["c1_result"]   = result
                    st.session_state["c1_map_html"] = m.get_root().render()
                    st.session_state["c1_dest"]     = (lat, lon, label)
                    st.session_state["c1_params"]   = (heure, marche)
                    st.query_params.update({
                        "lat": f"{lat:.6f}", "lon": f"{lon:.6f}",
                        "label": label, "addr": addr,
                        "h": str(heure), "m": str(marche),
                    })

    result   = st.session_state.get("c1_result")
    map_html = st.session_state.get("c1_map_html")

    if result is None or map_html is None:
        if not run:
            st.info(t("t2_info"))
        return

    cached_params = st.session_state.get("c1_params", (heure, marche))
    if (heure, marche) != cached_params:
        st.markdown(f'<div class="stale-warn">&#9888; {t("t2_stale")}</div>',
                    unsafe_allow_html=True)

    t_col = result["commute_total_min"]
    ntr   = result["commute_n_transfers"].fillna(0)

    st.markdown("---")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    with c1: mcard(f'{t_col.min():.0f} min',    t("t2_min"),     "#1a9850")
    with c2: mcard(f'{t_col.median():.0f} min', t("t2_median"))
    with c3: mcard(f'{t_col.max():.0f} min',    t("t2_max"),     "#d73027")
    with c4: mcard(str(int(((ntr == 0) & t_col.notna()).sum())), t("t2_direct"),   "#10b981")
    with c5: mcard(str(int((ntr == 1).sum())),                   t("t2_1corresp"), "#f59e0b")
    with c6: mcard(str(int((ntr == 2).sum())),                   t("t2_2corresp"), "#ef6820")
    with c7: mcard(str(int(t_col.isna().sum())),                 t("t2_nolink"),   "#94a3b8")

    st.markdown("<br>", unsafe_allow_html=True)
    components.html(map_html, height=740)

    st.markdown("---")
    st.markdown(f"**{t('t2_top')}**")
    top = (result.dropna(subset=["commute_total_min"])
           .nsmallest(10, "commute_total_min").copy())
    top["_time"]  = top["commute_total_min"].map(lambda x: f"{x:.0f} min")
    top["_mode"]  = top["commute_n_transfers"].fillna(0).map(
        lambda x: t("t2_direct") if x == 0
        else f"{int(x)} {t('t2_1corresp').split()[1] if int(x) == 1 else t('t2_2corresp').split()[1]}")
    top["_score"] = top["score_desserte"].map(
        lambda x: f"{x:.0f}/100" if not pd.isna(x) else "—")
    show_cols = {
        "iris_nom"          : t("t2_col_district"),
        "_time"             : t("t2_col_time"),
        "_mode"             : t("t2_col_mode"),
        "commute_stop_name" : t("t2_col_stop"),
        "_score"            : t("t2_col_score"),
        "categorie"         : t("t2_col_cat"),
    }
    df_export = top.rename(columns=show_cols)[list(show_cols.values())]
    st.dataframe(df_export, use_container_width=True, hide_index=True)
    st.download_button(t("t2_export"), df_export.to_csv(index=False),
                       "top10_trajets.csv", "text/csv")


# ── Tab 3: Multi-destination ──────────────────────────────────────────────────

def _init_multi():
    defaults = {
        "md_dests"     : [{"id": 0, "name": "Destination 1", "address": "", "weight": 5},
                          {"id": 1, "name": "Destination 2", "address": "", "weight": 5}],
        "md_next"      : 2,
        "md_geo"       : {},
        "md_data"      : {},
        "md_params"    : {},
        "md_combined"  : None,
        "md_dest_infos": [],
        "md_map"       : None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _invalidate_combined():
    st.session_state.md_combined = None
    st.session_state.md_map      = None


def combine_commutes(results, indicators):
    base    = indicators[["iris_code", "iris_nom", "score_desserte", "categorie", "geometry"]].copy()
    total_w = sum(w for *_, w in results)
    for gdf, label, *_, weight in results:
        sub = gdf[["iris_code", "commute_total_min", "commute_is_transfer"]].rename(
            columns={"commute_total_min": f"t_{label}", "commute_is_transfer": f"tr_{label}"})
        base = base.merge(sub, on="iris_code", how="left")
    t_cols = [f"t_{label}" for _, label, *_ in results]
    w_arr  = np.array([w for *_, w in results], dtype=float) / total_w
    t_mat  = base[t_cols].fillna(float(COMMUTE_CAP)).values
    base["commute_weighted"] = (t_mat * w_arr).sum(axis=1).round(1)
    base["commute_max"]      = t_mat.max(axis=1).round(1)
    base["all_connected"]    = base[t_cols].notna().all(axis=1)
    return gpd.GeoDataFrame(base, geometry="geometry", crs="EPSG:4326")


def _multi_popup(row, dest_infos):
    nom   = row.get("iris_nom", row.get("iris_code", "—"))
    wavg  = row.get("commute_weighted")
    wmax  = row.get("commute_max")
    score = row.get("score_desserte")
    cat   = str(row.get("categorie") or "non desservi")
    all_c = bool(row.get("all_connected", False))
    cc    = {"bien desservi": "#1a9850", "moyen": "#fc8d59",
             "sous-desservi": "#d73027", "non desservi": "#aaa"}
    cat_col = cc.get(cat, "#aaa")
    bar_col = ("#1a9850" if not pd.isna(wavg) and wavg < 20 else
               "#fee08b" if not pd.isna(wavg) and wavg < 35 else
               "#fc8d59" if not pd.isna(wavg) and wavg < 50 else "#d73027")
    bar_pct = int(min(wavg, COMMUTE_CAP) / COMMUTE_CAP * 100) if not pd.isna(wavg) else 0
    rows_html = ""
    for i, (label, *_) in enumerate(dest_infos):
        t_v = row.get(f"t_{label}")
        tr  = bool(row.get(f"tr_{label}", False))
        c   = DEST_COLS[i % len(DEST_COLS)]
        mode = ('<span style="color:#f59e0b;font-size:10px"> transfer</span>' if tr else "")
        val  = (f'<b style="color:{c}">{t_v:.0f} min</b>{mode}'
                if not pd.isna(t_v) else '<span style="color:#ccc">—</span>')
        rows_html += (f"<tr><td style='padding:3px 4px 3px 0;font-size:11px;color:#555'>"
                      f"<span style='color:{c}'>&#9679;</span> {label}</td>"
                      f"<td style='text-align:right;font-size:11px'>{val}</td></tr>")
    warn = ('' if all_c else
            '<div style="font-size:10px;color:#f59e0b;margin-top:3px">'
            '&#9888; Missing link</div>')
    sw = int(score) if not pd.isna(score) else 0
    return f"""<div style="font-family:Arial,sans-serif;min-width:230px">
      <b style="font-size:14px">{nom}</b>
      <hr style="margin:5px 0;border-color:#eee">
      <div style="font-size:20px;font-weight:700;color:{bar_col};text-align:center">
        {_fmt(wavg,"min")}
        <span style="font-size:11px;color:#888;font-weight:400">weighted avg.</span>
      </div>
      <div style="background:#eee;border-radius:3px;height:7px;margin:4px 0 7px">
        <div style="background:{bar_col};height:7px;border-radius:3px;width:{bar_pct}%"></div>
      </div>
      <table style="width:100%;border-collapse:collapse">{rows_html}</table>
      <div style="font-size:11px;color:#888;margin-top:3px">
        Worst trip: {_fmt(wmax,"min")}</div>
      {warn}
      <hr style="margin:6px 0;border-color:#eee">
      <span style="background:{cat_col};color:#fff;font-size:10px;
            padding:1px 6px;border-radius:8px">{cat}</span>
      <span style="float:right;font-size:11px;color:#666">{_fmt(score,"/100",1)}</span>
      <div style="background:#eee;border-radius:3px;height:5px;margin-top:16px">
        <div style="background:{cat_col};height:5px;border-radius:3px;width:{sw}%"></div>
      </div></div>"""


def build_multi_map(combined, dest_infos):
    valid = combined["commute_weighted"].dropna()
    vmin  = max(0.0, float(valid.min()) - 2) if not valid.empty else 0
    vmax  = float(min(COMMUTE_CAP, valid.quantile(0.95))) if not valid.empty else COMMUTE_CAP
    cmap  = LinearColormap(colors=PALETTE, vmin=vmin, vmax=vmax,
                           caption="Weighted avg. travel time (min)")
    c = combined.geometry.union_all().centroid
    m = folium.Map(location=[c.y, c.x], zoom_start=12, tiles="CartoDB positron",
                   attr="© OpenStreetMap contributors © CARTO",
                   scrollWheelZoom=False)
    fg = folium.FeatureGroup(name="Avg. commute", show=True)
    for _, row in combined.iterrows():
        w     = row.get("commute_weighted")
        color = cmap(min(w, COMMUTE_CAP)) if not pd.isna(w) else "#cccccc"
        nom   = row.get("iris_nom", "")
        tip   = f"{nom} — {w:.0f} min" if not pd.isna(w) else f"{nom} — incomplete link"
        folium.GeoJson(row.geometry.__geo_interface__,
            style_function=lambda _, c=color: {"fillColor": c, "color": "#555",
                                               "weight": 0.5, "fillOpacity": 0.78},
            tooltip=folium.Tooltip(tip, sticky=True),
            popup=folium.Popup(_multi_popup(row, dest_infos), max_width=280),
        ).add_to(fg)
    fg.add_to(m)
    cmap.add_to(m)
    for i, (label, lat, lon, w) in enumerate(dest_infos):
        total_w = sum(x for *_, x in dest_infos)
        pct = int(w / total_w * 100)
        folium.Marker(
            [lat, lon], tooltip=f"{label} ({pct}%)",
            popup=folium.Popup(
                f"<b>{label}</b><br>"
                f"<span style='font-size:11px;color:#888'>Weight: {pct}%</span>",
                max_width=160),
            icon=folium.Icon(
                color=["red", "blue", "green", "purple", "orange", "beige"][i % 6],
                icon=DEST_ICONS[i % len(DEST_ICONS)], prefix="fa"),
        ).add_to(m)
    b = combined.total_bounds
    m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    return m


def page_multi():
    _init_multi()
    indicators = get_desserte_data()
    dests      = st.session_state.md_dests

    st.subheader(t("t3_title"))
    st.caption(t("t3_caption"))

    step_header(1, t("t3_step1"))

    for i, d in enumerate(dests):
        color = DEST_COLS[i % len(DEST_COLS)]
        st.markdown(
            f'<div class="dest-card">'
            f'<span style="color:{color};font-weight:600;font-size:13px">'
            f'&#9679; {d["name"]}</span>',
            unsafe_allow_html=True)

        col_name, col_addr, col_btn, col_w, col_del = st.columns(
            [2, 4, 1, 2, 1], vertical_alignment="bottom")

        with col_name:
            d["name"] = st.text_input(
                "Name", value=d["name"], key=f"md_name_{d['id']}",
                label_visibility="collapsed", placeholder=t("t3_name_ph"))

        with col_addr:
            new_addr = st.text_input(
                "Address", value=d["address"], key=f"md_addr_{d['id']}",
                label_visibility="collapsed", placeholder=t("t3_addr_ph"))
            if new_addr != d["address"]:
                d["address"] = new_addr
                st.session_state.md_geo.pop(d["id"], None)
                st.session_state.md_data.pop(d["id"], None)
                _invalidate_combined()

        with col_btn:
            if st.button(t("t3_verify"), key=f"md_verify_{d['id']}",
                         use_container_width=True):
                if d["address"].strip():
                    with st.spinner(""):
                        try:
                            lat, lon, label = parse_dest(d["address"])
                            st.session_state.md_geo[d["id"]] = (lat, lon, label)
                            prev = st.session_state.md_params.get(d["id"])
                            if prev and prev[:2] != (lat, lon):
                                st.session_state.md_data.pop(d["id"], None)
                                _invalidate_combined()
                        except ValueError as e:
                            st.error(str(e))
                            st.session_state.md_geo.pop(d["id"], None)
                    st.rerun()
                else:
                    st.warning(t("err_no_addr"))

        with col_w:
            d["weight"] = st.slider(
                t("t3_weight_label"), 1, 10, d["weight"],
                key=f"md_w_{d['id']}", label_visibility="collapsed")

        with col_del:
            if len(dests) > 2 and st.button("✕", key=f"md_del_{d['id']}"):
                st.session_state.md_dests = [x for x in dests if x["id"] != d["id"]]
                st.session_state.md_geo.pop(d["id"], None)
                st.session_state.md_data.pop(d["id"], None)
                _invalidate_combined()
                st.rerun()

        geo  = st.session_state.md_geo.get(d["id"])
        data = st.session_state.md_data.get(d["id"])
        if geo:
            badge = f' &nbsp; &#10003; {t("t3_computed")}' if data is not None else ""
            st.markdown(
                f'<div class="geo-ok">&#10003; {geo[2]} &nbsp; '
                f'({geo[0]:.5f}, {geo[1]:.5f}){badge}</div>',
                unsafe_allow_html=True)
        elif d["address"].strip():
            st.markdown(f'<div class="geo-pending">{t("t3_pending")}</div>',
                        unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    col_add, col_reset, _ = st.columns([2, 2, 6])
    with col_add:
        if len(dests) < 6 and st.button(t("t3_add"), use_container_width=True):
            nid = st.session_state.md_next
            st.session_state.md_dests.append(
                {"id": nid, "name": f"Destination {len(dests)+1}",
                 "address": "", "weight": 5})
            st.session_state.md_next += 1
            st.rerun()
    with col_reset:
        if st.button(t("t3_reset"), use_container_width=True):
            for k in list(st.session_state.keys()):
                if k.startswith("md_"):
                    del st.session_state[k]
            st.rerun()

    step_header(2, t("t3_step2"))
    col1, col2 = st.columns(2)
    with col1: heure  = st.slider(t("t2_hour"), 5, 23, 8,    key="md_h")
    with col2: marche = st.slider(t("t2_walk"), 200, 1500, 800, 100, key="md_m")

    if (st.session_state.get("md_last_h") != heure or
            st.session_state.get("md_last_m") != marche):
        st.session_state.md_data.clear()
        _invalidate_combined()
    st.session_state["md_last_h"] = heure
    st.session_state["md_last_m"] = marche

    run_btn = st.button(t("t3_run"), type="primary", use_container_width=True, key="md_run")

    if run_btn:
        dests_ok = [d for d in dests
                    if st.session_state.md_geo.get(d["id"]) and d["address"].strip()]
        if len(dests_ok) < 2:
            st.error(t("err_min2"))
            st.stop()

        prog   = st.progress(0, t("computing"))
        errors = []
        for step, d in enumerate(dests_ok):
            geo = st.session_state.md_geo[d["id"]]
            lat, lon, label = geo
            params = (lat, lon, heure, marche)
            prog.progress(step / len(dests_ok), f"{d['name']} ({label})…")
            if (st.session_state.md_params.get(d["id"]) == params
                    and d["id"] in st.session_state.md_data):
                continue
            result, err = run_commute_data(lat, lon, label, heure, marche)
            if result is None:
                errors.append(f"{d['name']}: {err}")
                continue
            st.session_state.md_data[d["id"]]   = result
            st.session_state.md_params[d["id"]] = params
        prog.empty()
        for e in errors:
            st.warning(e)

        ready = [(st.session_state.md_data[d["id"]],
                  st.session_state.md_geo[d["id"]][2],
                  st.session_state.md_geo[d["id"]][0],
                  st.session_state.md_geo[d["id"]][1],
                  d["weight"])
                 for d in dests_ok if d["id"] in st.session_state.md_data]
        if len(ready) >= 2:
            st.session_state.md_combined   = combine_commutes(ready, indicators)
            st.session_state.md_dest_infos = [(lb, la, lo, w) for _, lb, la, lo, w in ready]
            st.session_state.md_map        = None

    dests_computed = [d for d in dests if d["id"] in st.session_state.md_data]
    if len(dests_computed) >= 2:
        step_header(3, t("t3_step3"))
        with st.expander(t("t3_step3"), expanded=False):
            st.caption(t("t3_w_caption"))
            with st.form("md_weights"):
                new_w = {}
                cols  = st.columns(len(dests_computed))
                for i, d in enumerate(dests_computed):
                    geo = st.session_state.md_geo.get(d["id"], ("", "", "—"))
                    with cols[i]:
                        new_w[d["id"]] = st.slider(
                            geo[2], 1, 10, d["weight"], key=f"md_fw_{d['id']}")
                if st.form_submit_button(t("t3_update"), use_container_width=True):
                    for d in dests_computed:
                        d["weight"] = new_w[d["id"]]
                    ready = [(st.session_state.md_data[d["id"]],
                              st.session_state.md_geo[d["id"]][2],
                              st.session_state.md_geo[d["id"]][0],
                              st.session_state.md_geo[d["id"]][1],
                              d["weight"])
                             for d in dests_computed]
                    st.session_state.md_combined   = combine_commutes(ready, indicators)
                    st.session_state.md_dest_infos = [(lb, la, lo, w)
                                                      for _, lb, la, lo, w in ready]
                    st.session_state.md_map        = None

    combined   = st.session_state.md_combined
    dest_infos = st.session_state.md_dest_infos
    if combined is None or not dest_infos:
        st.info(t("t3_no_result"))
        return

    wt = combined["commute_weighted"]
    st.markdown("---")
    n_extra = 3
    cols = st.columns(len(dest_infos) + n_extra)
    for i, (label, *_) in enumerate(dest_infos):
        t_v = combined[f"t_{label}"]
        with cols[i]:
            mcard(f"{t_v.min():.0f}–{t_v.median():.0f} min",
                  label, DEST_COLS[i % len(DEST_COLS)])
    with cols[-3]: mcard(f"{wt.min():.0f} min",                     t("t3_best"),     "#1a9850")
    with cols[-2]: mcard(str(int(combined["all_connected"].sum())),  t("t3_iris_conn"))
    with cols[-1]: mcard(str(int(wt.isna().sum())),                  t("t3_nolink"),   "#94a3b8")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.md_map is None:
        with st.spinner("Construction de la carte…"):
            m = build_multi_map(combined, dest_infos)
            st.session_state.md_map = m.get_root().render()
    components.html(st.session_state.md_map, height=740)

    st.markdown("---")
    st.markdown(f"**{t('t3_top')}**")
    t_cols = [f"t_{label}" for label, *_ in dest_infos]
    top = (combined.dropna(subset=["commute_weighted"])
           .nsmallest(20, "commute_weighted")
           [["iris_nom", "commute_weighted", "commute_max", "all_connected",
             "score_desserte", "categorie"] + t_cols].copy())
    top[t("t3_avg")]      = top["commute_weighted"].map(lambda x: f"{x:.0f} min")
    top[t("t3_worst")]    = top["commute_max"].map(lambda x: f"{x:.0f} min")
    _is_fr = st.session_state.get("lang", "fr") == "fr"
    top[t("t3_all_conn")] = top["all_connected"].map(
        lambda x: ("Oui" if x else "Partiel") if _is_fr else ("Yes" if x else "Partial"))
    top[t("t3_score")]    = top["score_desserte"].map(
        lambda x: f"{x:.0f}/100" if not pd.isna(x) else "—")
    for col in t_cols:
        top[col] = top[col].map(lambda x: f"{x:.0f} min" if not pd.isna(x) else "—")
    rename = {"iris_nom": t("t2_col_district"), "categorie": t("t3_cat")}
    rename.update({f"t_{lb}": lb for lb, *_ in dest_infos})
    show = ([t("t2_col_district"), t("t3_avg"), t("t3_worst"), t("t3_all_conn")]
            + [lb for lb, *_ in dest_infos] + [t("t3_score"), t("t3_cat")])
    df_export = top.rename(columns=rename)[show]
    st.dataframe(df_export, use_container_width=True, hide_index=True)
    st.download_button(t("t3_export"), df_export.to_csv(index=False),
                       "top20_quartiers.csv", "text/csv")


# ── Tab 4: Service Pulse ──────────────────────────────────────────────────────

def build_pulse_figure(gdf: gpd.GeoDataFrame,
                       hourly: pd.DataFrame) -> "px.Figure":
    """
    Build a Plotly animated choropleth covering every hour from 5 to 23.
    All 19 frames share the same colour scale so transitions are smooth.
    """
    # Simplify geometries (~50 % smaller GeoJSON, visually identical at city scale)
    gdf_s = gdf[["iris_code", "iris_nom", "geometry"]].copy()
    gdf_s["geometry"] = gdf_s.geometry.simplify(tolerance=0.0001, preserve_topology=True)
    geojson = json.loads(gdf_s[["iris_code", "geometry"]].to_json())

    # Full cross-product: every (iris_code × hour), zeros where no service
    base  = gdf_s[["iris_code", "iris_nom"]].drop_duplicates()
    hours = pd.DataFrame({"heure": range(5, 24)})
    full  = base.merge(hours, how="cross")
    full  = full.merge(
        hourly[["iris_code", "heure", "nb_passages",
                "nb_lignes", "nb_arrets", "frequence_moy_min"]],
        on=["iris_code", "heure"], how="left",
    )
    full["nb_passages"]       = full["nb_passages"].fillna(0).astype(int)
    full["nb_lignes"]         = full["nb_lignes"].fillna(0).astype(int)
    full["nb_arrets"]         = full["nb_arrets"].fillna(0).astype(int)
    full["frequence_moy_min"] = full["frequence_moy_min"].fillna(0).round(0).astype(int)
    full["heure_label"]       = full["heure"].astype(str).str.zfill(2) + ":00"
    full = full.sort_values(["heure", "iris_code"])

    vmax = max(1.0, float(hourly["nb_passages"].quantile(0.95)))

    fig = px.choropleth_mapbox(
        full,
        geojson=geojson,
        locations="iris_code",
        featureidkey="properties.iris_code",
        color="nb_passages",
        animation_frame="heure_label",
        hover_name="iris_nom",
        custom_data=["nb_lignes", "nb_arrets", "frequence_moy_min"],
        color_continuous_scale=["#e8f4fd", "#7dd3fc", "#0284c7", "#1e40af", "#1e3a5f"],
        range_color=[0, vmax],
        mapbox_style="carto-positron",
        zoom=10,
        center={"lat": 43.6047, "lon": 1.4442},
        opacity=0.82,
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Passages : <b>%{z}</b><br>"
            "Lignes : %{customdata[0]}<br>"
            "Arrêts : %{customdata[1]}<br>"
            "Fréquence : %{customdata[2]} min"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=820,
        coloraxis_colorbar=dict(
            title="Passages",
            thickness=14,
            len=0.55,
            y=0.5,
        ),
    )

    # Smooth animation: 700 ms per frame, 500 ms eased transition
    try:
        play = fig.layout.updatemenus[0].buttons[0]
        play.args[1]["frame"]["duration"]        = 700
        play.args[1]["transition"]["duration"]   = 500
        play.args[1]["transition"]["easing"]     = "cubic-in-out"
    except (IndexError, AttributeError):
        pass

    return fig


def page_pulse():
    st.subheader(t("t4_title"))
    st.caption(t("t4_caption"))

    hourly = get_hourly_data()
    if hourly.empty:
        st.warning(t("t4_nodata"))
        return

    gdf = get_desserte_data()

    # ── Global metrics ────────────────────────────────────────────────────────
    daily_by_hour = hourly.groupby("heure")["nb_passages"].sum()
    peak_hour_val = int(daily_by_hour.idxmax())
    peak_trips    = int(daily_by_hour.max())
    total_day     = int(daily_by_hour.sum())
    n_iris        = hourly["iris_code"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    with c1: mcard(str(n_iris),               t("t4_active_iris"))
    with c2: mcard(f"{total_day:,}",          t("t4_total_day"),  "#0284c7")
    with c3: mcard(f"{peak_hour_val:02d}:00", t("t4_peak_hour"),  "#1e40af")
    with c4: mcard(f"{peak_trips:,}",         t("t4_peak_trips"), "#1e3a5f")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Animated choropleth — disk-cached across sessions ────────────────────
    if "pulse_figure" not in st.session_state:
        fig = _load_pulse_cache()
        if fig is None:
            with st.spinner(t("computing")):
                fig = build_pulse_figure(gdf, hourly)
                _save_pulse_cache(fig)
        st.session_state.pulse_figure = fig

    st.plotly_chart(
        st.session_state.pulse_figure,
        use_container_width=True,
        config={"scrollZoom": False, "displayModeBar": False},
    )

    # ── Daily timeline bar chart ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"**{t('t4_timeline')}**")
    daily = (daily_by_hour
             .reset_index()
             .rename(columns={"nb_passages": t("t4_trips")}))
    daily.index = daily["heure"].astype(str).str.zfill(2) + ":00"
    st.bar_chart(daily[[t("t4_trips")]], use_container_width=True, height=140,
                 color="#0284c7")

    # ── Top-10 districts at peak hour ─────────────────────────────────────────
    peak_data = hourly[hourly["heure"] == peak_hour_val]
    if not peak_data.empty:
        st.markdown(f"**{t('t4_top')}**")
        top = peak_data.nlargest(10, "nb_passages")[
            ["iris_nom", "nb_passages", "nb_lignes", "nb_arrets", "frequence_moy_min"]
        ].copy()
        top.columns = [
            t("t2_col_district"),
            t("t4_trips"),
            t("t4_lines"),
            t("t4_stops"),
            t("t4_freq"),
        ]
        top[t("t4_freq")] = top[t("t4_freq")].map(
            lambda x: f"{x:.0f} min" if not pd.isna(x) else "—")
        st.dataframe(top, use_container_width=True, hide_index=True)


# ── Tab 5: District comparison ────────────────────────────────────────────────

def page_compare():
    st.subheader(t("t5_title"))
    st.caption(t("t5_caption"))

    gdf    = get_desserte_data()
    hourly = get_hourly_data()

    code_to_nom = gdf[["iris_code", "iris_nom"]].dropna().set_index("iris_code")["iris_nom"].to_dict()
    nom_to_code = {v: k for k, v in code_to_nom.items()}
    districts   = sorted(nom_to_code.keys())

    CAT_COL = {"bien desservi": "#1a9850", "moyen": "#f59e0b",
               "sous-desservi": "#d73027"}

    col1, col_vs, col2 = st.columns([5, 1, 5])
    with col1:
        d1_nom = st.selectbox(t("t5_d1"), districts, index=0, key="cmp_d1")
    with col_vs:
        st.markdown(
            "<div style='text-align:center;padding-top:28px;font-size:18px;"
            "font-weight:700;color:#94a3b8'>VS</div>",
            unsafe_allow_html=True)
    with col2:
        d2_nom = st.selectbox(t("t5_d2"), districts,
                              index=min(1, len(districts) - 1), key="cmp_d2")

    if d1_nom == d2_nom:
        st.info(t("t5_same"))
        return

    d1_code = nom_to_code[d1_nom]
    d2_code = nom_to_code[d2_nom]
    r1 = gdf[gdf["iris_code"] == d1_code].iloc[0]
    r2 = gdf[gdf["iris_code"] == d2_code].iloc[0]

    def score_card(row, name):
        sc  = row.get("score_desserte")
        cat = str(row.get("categorie", ""))
        cc  = CAT_COL.get(cat, "#94a3b8")
        sc_str = f"{sc:.0f}" if not pd.isna(sc) else "—"
        st.markdown(
            f'<div class="mcard" style="border-top:4px solid {cc}">'
            f'<div class="val" style="color:{cc};font-size:30px">{sc_str}'
            f'<span style="font-size:14px;color:#94a3b8;font-weight:400">/100</span></div>'
            f'<div class="lbl" style="font-size:13px;color:#1e293b;font-weight:600;'
            f'margin-top:6px">{name}</div>'
            f'<div style="font-size:11px;color:{cc};margin-top:3px">{cat}</div>'
            f'</div>',
            unsafe_allow_html=True)

    st.markdown("---")
    ca, _, cb = st.columns([5, 1, 5])
    with ca: score_card(r1, d1_nom)
    with cb: score_card(r2, d2_nom)

    st.markdown("<br>", unsafe_allow_html=True)

    metrics = [
        (t("t5_amplitude"), "amplitude_h",      "{:.0f} h"),
        (t("t5_freq"),      "freq_pointe_min",   "{:.1f} min"),
        (t("t5_lines"),     "max_lignes",        "{:.0f}"),
        (t("t5_coverage"),  "coverage_pct",      "{:.1f} %"),
        (t("t5_area"),      "area_km2",          "{:.2f} km²"),
        (t("t5_passages"),  "total_passages",    "{:.0f}"),
    ]
    for label, col_key, fmt in metrics:
        v1 = r1.get(col_key)
        v2 = r2.get(col_key)
        s1 = fmt.format(v1) if not pd.isna(v1) else "—"
        s2 = fmt.format(v2) if not pd.isna(v2) else "—"
        ca, _, cb = st.columns([5, 1, 5])
        with ca: mcard(s1, label)
        with cb: mcard(s2, label)

    st.markdown("---")
    st.markdown(f"**{t('t5_radar')}**")
    st.plotly_chart(
        build_radar_chart(r1, r2, d1_nom, d2_nom, gdf),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    if not hourly.empty:
        st.markdown("---")
        st.markdown(f"**{t('t5_hourly')}**")
        h1 = (hourly[hourly["iris_code"] == d1_code][["heure", "nb_passages"]]
              .rename(columns={"nb_passages": d1_nom}))
        h2 = (hourly[hourly["iris_code"] == d2_code][["heure", "nb_passages"]]
              .rename(columns={"nb_passages": d2_nom}))
        hm = h1.merge(h2, on="heure", how="outer").sort_values("heure")
        hm.index = hm["heure"].astype(str).str.zfill(2) + ":00"
        st.line_chart(hm[[d1_nom, d2_nom]], use_container_width=True, height=220)


# ── Main navigation ───────────────────────────────────────────────────────────

def main():
    if "lang" not in st.session_state:
        st.session_state.lang = "fr"

    _init_from_query_params()
    render_sidebar()

    st.markdown(
        f"<h1 style='margin-bottom:0;font-size:26px'>{t('app_title')}</h1>"
        f"<p style='color:#64748b;margin-top:4px;margin-bottom:16px;font-size:14px'>"
        f"{t('app_sub')}</p>",
        unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        t("tab_network"),
        t("tab_commute"),
        t("tab_multi"),
        t("tab_pulse"),
        t("tab_compare"),
    ])
    with tab1: page_desserte()
    with tab2: page_commute()
    with tab3: page_multi()
    with tab4: page_pulse()
    with tab5: page_compare()


if __name__ == "__main__":
    main()
