import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import joblib
import os
import json
import glob
from river import forest
from datetime import datetime

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fleet Security — Depot Control",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# THEME
# Depot control-room palette: cool steel surfaces, signal colours used
# only to carry meaning (amber = flagged, teal = cleared, red = confirmed).
# Light and dark share the same structure — only the token values differ.
# ══════════════════════════════════════════════════════════════════════
LIGHT_TOKENS = {
    "bg":           "#F3F5F7",
    "surface":      "#FFFFFF",
    "steel-100":    "#E6EAEE",
    "steel-200":    "#D2D9E0",
    "steel-400":    "#8C9AA6",
    "ink":          "#1B2A33",
    "ink-soft":     "#55646F",
    "navy":         "#14506B",
    "navy-hover":   "#0E3D53",
    "navy-on":      "#FFFFFF",
    "amber":        "#C97A0A",
    "amber-bg":     "#FDF4E3",
    "amber-text":   "#7A4A05",
    "teal":         "#0E7C66",
    "teal-bg":      "#E8F4F1",
    "teal-text":    "#0B5F4E",
    "red":          "#A32E2E",
    "red-bg":       "#FBEDED",
    "red-text":     "#7A1F1F",
    "menu-hover":   "#E6EAEE",
}

DARK_TOKENS = {
    "bg":           "#0F171D",
    "surface":      "#1B262E",
    "steel-100":    "#243139",
    "steel-200":    "#324150",
    "steel-400":    "#8496A2",
    "ink":          "#E7ECEF",
    "ink-soft":     "#AAB8C2",
    "navy":         "#4CA3CC",
    "navy-hover":   "#68B4D8",
    "navy-on":      "#08151C",
    "amber":        "#E0983A",
    "amber-bg":     "#33260F",
    "amber-text":   "#F2C179",
    "teal":         "#39B597",
    "teal-bg":      "#0F2C24",
    "teal-text":    "#8CE0C8",
    "red":          "#E07070",
    "red-bg":       "#331414",
    "red-text":     "#F2A9A9",
    "menu-hover":   "#243139",
}


def build_theme_css(mode: str) -> str:
    t = DARK_TOKENS if mode == "dark" else LIGHT_TOKENS
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {{
  --steel-50:   {t['bg']};
  --surface:    {t['surface']};
  --steel-100:  {t['steel-100']};
  --steel-200:  {t['steel-200']};
  --steel-400:  {t['steel-400']};
  --ink:        {t['ink']};
  --ink-soft:   {t['ink-soft']};
  --navy:       {t['navy']};
  --navy-hover: {t['navy-hover']};
  --navy-on:    {t['navy-on']};
  --amber:      {t['amber']};
  --amber-bg:   {t['amber-bg']};
  --amber-text: {t['amber-text']};
  --teal:       {t['teal']};
  --teal-bg:    {t['teal-bg']};
  --teal-text:  {t['teal-text']};
  --red:        {t['red']};
  --red-bg:     {t['red-bg']};
  --red-text:   {t['red-text']};
  --menu-hover: {t['menu-hover']};
}}

html, body, [class*="css"], .stApp, section[data-testid="stSidebar"] {{
  font-family: 'Barlow', system-ui, sans-serif;
}}
.stApp {{ background: var(--steel-50); color: var(--ink); }}

/* ── Masthead ───────────────────────────────────────────────── */
.masthead {{
  display: flex; align-items: baseline; justify-content: space-between;
  gap: 1.5rem; flex-wrap: wrap;
  border-bottom: 2px solid var(--ink);
  padding: .2rem 0 .7rem; margin-bottom: 1.4rem;
}}
.masthead h1 {{
  font-size: 1.85rem; font-weight: 700; letter-spacing: -.02em;
  margin: 0; color: var(--ink); line-height: 1.1;
}}
.masthead .sub {{ font-size: .95rem; color: var(--ink-soft); margin: .25rem 0 0; }}
.masthead .stamp {{
  font-family: 'IBM Plex Mono', monospace; font-size: .74rem;
  color: var(--ink-soft); text-align: right; line-height: 1.6; white-space: nowrap;
}}

/* ── KPI tiles ──────────────────────────────────────────────── */
.kpi-row {{ display: flex; gap: .7rem; flex-wrap: wrap; margin-bottom: .4rem; }}
.kpi {{
  flex: 1 1 165px; background: var(--surface);
  border: 1px solid var(--steel-200); border-left: 3px solid var(--steel-400);
  border-radius: 3px; padding: .75rem .9rem .8rem;
}}
.kpi .label {{ font-size: .78rem; font-weight: 500; color: var(--ink-soft) !important; margin-bottom: .3rem; }}
.kpi .value {{
  font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 500;
  color: var(--ink); line-height: 1.1; letter-spacing: -.02em;
}}
.kpi .foot {{ font-size: .74rem; color: var(--steel-400) !important; margin-top: .3rem; }}
.kpi.flag  {{ border-left-color: var(--amber); }}
.kpi.flag  .value {{ color: var(--amber) !important; }}
.kpi.clear {{ border-left-color: var(--teal); }}
.kpi.clear .value {{ color: var(--teal) !important; }}
.kpi.alert {{ border-left-color: var(--red); }}
.kpi.alert .value {{ color: var(--red) !important; }}
.kpi.key   {{ border-left-color: var(--navy); }}
.kpi.key   .value {{ color: var(--navy) !important; }}

/* ── Section headings ───────────────────────────────────────── */
.sect {{
  font-size: 1.08rem; font-weight: 600; color: var(--ink) !important;
  margin: 1.5rem 0 .2rem; padding-bottom: .3rem; border-bottom: 1px solid var(--steel-200);
}}
.sect-note {{ font-size: .86rem; color: var(--ink-soft) !important; margin: .35rem 0 .7rem; }}

/* ── Status pills (sidebar) ─────────────────────────────────── */
.pill {{
  display: block; font-size: .82rem; padding: .4rem .6rem;
  border-radius: 3px; margin-bottom: .35rem; border-left: 3px solid transparent;
}}
.pill.ok, .pill.ok *     {{ background: var(--teal-bg);  color: var(--teal-text) !important; }}
.pill.ok                 {{ border-left-color: var(--teal); }}
.pill.warn, .pill.warn * {{ background: var(--amber-bg); color: var(--amber-text) !important; }}
.pill.warn                {{ border-left-color: var(--amber); }}
.pill .mono {{ font-family: 'IBM Plex Mono', monospace; font-size: .76rem; }}

/* ── Verdict banner ─────────────────────────────────────────── */
.verdict {{ border-radius: 3px; padding: .8rem 1rem; margin: .6rem 0; border-left: 3px solid; font-size: .95rem; }}
.verdict.flag,  .verdict.flag *  {{ background: var(--amber-bg); border-color: var(--amber); color: var(--amber-text) !important; }}
.verdict.clear, .verdict.clear * {{ background: var(--teal-bg);  border-color: var(--teal);  color: var(--teal-text) !important; }}

/* ── Streamlit widget overrides ─────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid var(--steel-200); background: transparent; }}
.stTabs [data-baseweb="tab"] {{
  height: 42px; padding: 0 1.05rem; background: transparent;
  font-size: .93rem; font-weight: 500; color: var(--ink-soft);
  border-bottom: 2px solid transparent; border-radius: 0;
}}
.stTabs [data-baseweb="tab"] * {{ color: inherit !important; }}
.stTabs [aria-selected="true"] {{
  color: var(--ink) !important; font-weight: 600;
  border-bottom: 2px solid var(--navy) !important; background: transparent !important;
}}

.stButton > button, .stButton > button * {{ color: var(--ink) !important; }}
.stButton > button {{
  border-radius: 3px; font-weight: 600; font-size: .9rem;
  border: 1px solid var(--steel-200); padding: .45rem 1rem; background: var(--surface);
}}
.stButton > button[kind="primary"], .stButton > button[kind="primary"] * {{
  background: var(--navy); border-color: var(--navy); color: var(--navy-on) !important;
}}
.stButton > button[kind="primary"]:hover {{ background: var(--navy-hover); border-color: var(--navy-hover); }}
.stDownloadButton > button, .stDownloadButton > button * {{ color: var(--ink) !important; }}

section[data-testid="stSidebar"] {{ background: var(--surface); border-right: 1px solid var(--steel-200); }}
section[data-testid="stSidebar"] h2 {{
  font-size: .95rem; font-weight: 600; color: var(--ink) !important;
  border-bottom: 1px solid var(--steel-200); padding-bottom: .35rem;
}}

[data-testid="stDataFrame"] {{ border: 1px solid var(--steel-200); border-radius: 3px; }}
[data-testid="stDataFrame"] * {{ color: var(--ink); }}
hr {{ border-color: var(--steel-200); }}
#MainMenu, footer {{ visibility: hidden; }}

/* ── Base text everywhere ──────────────────────────────────── */
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div, .stMarkdown,
section[data-testid="stSidebar"] * {{ color: var(--ink); }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{ color: var(--ink); }}
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] *,
.stApp small, .stApp .stCaption {{ color: var(--ink-soft) !important; }}

/* ── Radios / checkboxes ───────────────────────────────────── */
.stRadio label, .stRadio label *, [data-testid="stRadio"] label, [data-testid="stRadio"] label * {{
  color: var(--ink) !important;
}}
.stRadio [role="radiogroup"] label {{ cursor: pointer; }}
.stRadio [role="radiogroup"] label:has(input:checked),
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {{
  color: var(--navy) !important; font-weight: 600;
}}
.stRadio [role="radiogroup"] label:has(input:checked) * {{ color: var(--navy) !important; }}
.stCheckbox label, .stCheckbox label * {{ color: var(--ink) !important; }}

/* ── Toggle switch (dark-mode control) ─────────────────────── */
[data-testid="stToggle"] label p {{ color: var(--ink) !important; font-weight: 500; }}

/* ── Text / number inputs ──────────────────────────────────── */
.stTextInput input, .stNumberInput input, .stTextArea textarea {{
  color: var(--ink) !important; background: var(--surface) !important;
  border: 1px solid var(--steel-200) !important; border-radius: 3px;
  -webkit-text-fill-color: var(--ink);
}}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {{
  color: var(--steel-400) !important; -webkit-text-fill-color: var(--steel-400);
}}

/* ── Select / multiselect — closed control ─────────────────── */
.stSelectbox [data-baseweb="select"] > div, .stMultiSelect [data-baseweb="select"] > div {{
  background: var(--surface) !important; border: 1px solid var(--steel-200) !important;
  border-radius: 3px; color: var(--ink) !important;
}}
.stSelectbox [data-baseweb="select"] *, .stMultiSelect [data-baseweb="select"] * {{ color: var(--ink) !important; }}
.stSelectbox svg, .stMultiSelect svg {{ fill: var(--ink-soft) !important; }}

/* Dropdown menu — rendered in a portal outside .stApp */
[data-baseweb="popover"], [data-baseweb="menu"] {{
  background: var(--surface) !important; border: 1px solid var(--steel-200) !important;
}}
[data-baseweb="popover"] li, [data-baseweb="menu"] li,
[data-baseweb="popover"] li *, [data-baseweb="menu"] li * {{
  color: var(--ink) !important; background: transparent !important;
}}
[data-baseweb="menu"] li[aria-selected="true"], [data-baseweb="menu"] li:hover {{
  background: var(--menu-hover) !important;
}}

/* ── File uploader ──────────────────────────────────────────── */
[data-testid="stFileUploader"] section {{
  background: var(--surface) !important; border: 1px dashed var(--steel-200) !important; border-radius: 3px;
}}
[data-testid="stFileUploader"] * {{ color: var(--ink) !important; }}
[data-testid="stFileUploader"] small {{ color: var(--ink-soft) !important; }}
[data-testid="stFileUploader"] button {{
  background: var(--surface) !important; border: 1px solid var(--steel-200) !important; color: var(--ink) !important;
}}

/* ── Expanders ──────────────────────────────────────────────── */
[data-testid="stExpander"] {{ background: var(--surface); border: 1px solid var(--steel-200); border-radius: 3px; }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color: var(--ink) !important; }}

/* ── Status / spinner / alerts ─────────────────────────────── */
[data-testid="stStatusWidget"] *, .stSpinner * {{ color: var(--ink) !important; }}
[data-testid="stAlert"] * {{ color: var(--ink) !important; }}

/* ── Native Streamlit chrome: top header bar + toolbar ─────────
   This sits outside .stApp's own background, so it needs its own
   rules or it stays white regardless of the in-app theme toggle. */
[data-testid="stHeader"] {{
  background: var(--steel-50) !important;
}}
[data-testid="stToolbar"], [data-testid="stToolbarActions"] {{
  background: transparent !important;
}}
[data-testid="stHeader"] button, [data-testid="stHeader"] a,
[data-testid="stHeader"] svg, [data-testid="stHeader"] span,
[data-testid="stToolbar"] button, [data-testid="stToolbar"] svg {{
  color: var(--ink-soft) !important; fill: var(--ink-soft) !important;
}}
[data-testid="stHeader"] button:hover, [data-testid="stToolbar"] button:hover {{
  color: var(--ink) !important; fill: var(--ink) !important;
  background: var(--steel-100) !important;
}}
/* Sidebar collapse/expand arrow control */
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] button,
[data-testid="collapsedControl"] svg {{
  color: var(--ink-soft) !important; fill: var(--ink-soft) !important;
}}
[data-testid="stDecoration"] {{ background: var(--navy) !important; }}
/* Main content area background (separate node from .stApp in newer Streamlit) */
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{
  background: var(--steel-50) !important;
}}

.stApp *:focus-visible {{ outline: 2px solid var(--navy) !important; outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""


def kpi_row(items):
    """items: list of (label, value, footnote, tone). tone in '', 'flag', 'clear', 'alert', 'key'."""
    tiles = "".join(
        f'<div class="kpi {tone}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        + (f'<div class="foot">{foot}</div>' if foot else '')
        + '</div>'
        for label, value, foot, tone in items
    )
    st.markdown(f'<div class="kpi-row">{tiles}</div>', unsafe_allow_html=True)


def section(title, note=None):
    st.markdown(f'<div class="sect">{title}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="sect-note">{note}</div>', unsafe_allow_html=True)


# ── Appearance toggle — first thing in the sidebar ─────────────────────
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True

st.sidebar.toggle("Dark mode", key="dark_mode")
_theme_mode = "dark" if st.session_state["dark_mode"] else "light"

st.markdown(build_theme_css(_theme_mode), unsafe_allow_html=True)

st.markdown(
    '<div class="masthead">'
    '<div><h1>Fleet Security — Depot Control</h1>'
    '<p class="sub">Anomalous driver behaviour detection from GPS telemetry</p></div>'
    f'<div class="stamp">LightGBM + XGBoost ensemble<br>Session {datetime.now().strftime("%d %b %Y · %H:%M")}</div>'
    '</div>',
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════════════════════
# ABSOLUTE PATHS — updated for new FYP folder architecture
#
# FYP/
# ├── data/raw/         ← raw JSON files
# ├── data/processed/   ← master_telemetry.csv, trip_features_labelled.csv
# ├── models/           ← all .pkl and .json model files
# ├── notebooks/        ← Jupyter notebooks
# ├── dashboard/        ← this file lives here
# ├── logs/             ← feedback_log.csv
# └── docs/             ← presentations, diagrams
# ══════════════════════════════════════════════════════════════════════

# dashboard/ folder — where this file lives
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))

# FYP root — one level up from dashboard/
FYP_DIR = os.path.dirname(DASHBOARD_DIR)

# Subfolders
MODELS_DIR    = os.path.join(FYP_DIR, 'models')
DATA_RAW_DIR  = os.path.join(FYP_DIR, 'data', 'raw')
DATA_PROC_DIR = os.path.join(FYP_DIR, 'data', 'processed')
LOGS_DIR      = os.path.join(FYP_DIR, 'logs')

# Individual file paths
ENSEMBLE_MODEL_PATH = os.path.join(MODELS_DIR, 'ensemble_model.pkl')
SCALER_PATH         = os.path.join(MODELS_DIR, 'scaler.pkl')
IMPUTER_PATH        = os.path.join(MODELS_DIR, 'imputer.pkl')
ONLINE_MODEL_PATH   = os.path.join(MODELS_DIR, 'online_model.pkl')
FEATURE_COLS_PATH   = os.path.join(MODELS_DIR, 'feature_cols.json')
FEEDBACK_LOG        = os.path.join(LOGS_DIR,   'feedback_log.csv')

# ══════════════════════════════════════════════════════════════════════
# PREPROCESSING PIPELINE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def parse_json_files(uploaded_files):
    """Step 1 — Load and flatten raw JSON telemetry files."""
    records_list = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        parts    = os.path.splitext(filename)[0].split("_")

        if len(parts) == 3:
            try:
                month      = int(parts[0])
                day        = int(parts[1])
                vehicle_id = int(parts[2])
            except ValueError:
                st.warning(f"⚠️ Skipping {filename} — unexpected filename format")
                continue
        else:
            st.warning(f"⚠️ Skipping {filename} — expected Month_Day_Vehicle.json format")
            continue

        try:
            raw = json.load(uploaded_file)

            records = []
            if isinstance(raw, list):
                for level1 in raw:
                    if isinstance(level1, list):
                        for level2 in level1:
                            if isinstance(level2, list):
                                records.extend(level2)
                            elif isinstance(level2, dict):
                                records.append(level2)
                    elif isinstance(level1, dict):
                        records.append(level1)

            if records:
                df                = pd.DataFrame(records)
                df['vehicle_id']  = vehicle_id
                df['file_month']  = month
                df['file_day']    = day
                df['source_file'] = filename
                records_list.append(df)

        except Exception as e:
            st.warning(f"⚠️ Could not parse {filename}: {e}")
            continue

    if not records_list:
        return None

    return pd.concat(records_list, ignore_index=True)


def preprocess(df):
    """Step 2 — Clean, deduplicate, sort, compute deltas."""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])

    numeric_cols = ['mileage', 'heading', 'speed', 'longitude',
                    'latitude', 'acc', 'fuel1_volume', 'fuel2_volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    before        = len(df)
    df            = df.drop_duplicates(subset=['vehicle_id', 'datetime'])
    dupes_removed = before - len(df)

    df = df.sort_values(['vehicle_id', 'datetime']).reset_index(drop=True)

    df['time_delta_s']     = df.groupby('vehicle_id')['datetime'].diff().dt.total_seconds().fillna(0)
    df['mileage_delta_km'] = df.groupby('vehicle_id')['mileage'].diff().fillna(0).clip(lower=0)
    df['fuel_delta']       = df.groupby('vehicle_id')['fuel1_volume'].diff().fillna(0)

    return df, dupes_removed


def segment_trips(df):
    """Step 3 — Split into individual trips using IGNITION ON/OFF events.
       Falls back to 30-min time-gap segmentation if no ignition events found.
    """
    segmented_parts = []

    for vid, group in df.groupby('vehicle_id'):
        g = group.reset_index(drop=True).copy()
        g['trip_id'] = None

        ignition = g[g['event_message'].isin(['IGNITION ON', 'IGNITION OFF'])]

        if len(ignition) >= 2:
            trip_counter   = 0
            in_trip        = False
            trip_start_idx = None

            for idx, row in g.iterrows():
                if row['event_message'] == 'IGNITION ON' and not in_trip:
                    in_trip        = True
                    trip_start_idx = idx
                    trip_counter  += 1
                elif row['event_message'] == 'IGNITION OFF' and in_trip:
                    in_trip = False
                    g.loc[trip_start_idx:idx, 'trip_id'] = f"V{vid}_T{trip_counter:04d}"

            if in_trip and trip_start_idx is not None:
                g.loc[trip_start_idx:, 'trip_id'] = f"V{vid}_T{trip_counter:04d}"

            # Assign parked records their own segment
            null_mask = g['trip_id'].isna()
            if null_mask.any():
                g.loc[null_mask, 'trip_id'] = f"V{vid}_PARKED"

        else:
            # Fallback: time-gap segmentation
            trip_counter  = 1
            gap_threshold = 30 * 60
            current_trip  = f"V{vid}_T{trip_counter:04d}"
            trip_ids      = []

            for i in range(len(g)):
                if i > 0 and g.iloc[i]['time_delta_s'] > gap_threshold:
                    trip_counter += 1
                    current_trip  = f"V{vid}_T{trip_counter:04d}"
                trip_ids.append(current_trip)

            g['trip_id'] = trip_ids

        segmented_parts.append(g)

    return pd.concat(segmented_parts, ignore_index=True)


def engineer_features(master_df):
    """Step 4 — Extract extended features per trip (aligned with model training)."""
    trip_features = []

    for trip_id, trip in master_df.groupby('trip_id'):
        if len(trip) < 2:
            continue

        trip         = trip.sort_values('datetime')
        vehicle_id   = trip['vehicle_id'].iloc[0]
        start_time   = trip['datetime'].iloc[0]
        end_time     = trip['datetime'].iloc[-1]
        duration_s   = (end_time - start_time).total_seconds()
        duration_min = duration_s / 60

        # ── Temporal ──────────────────────────────────────────────────
        start_hour   = start_time.hour
        day_of_week  = start_time.dayofweek
        is_weekend   = int(day_of_week >= 5)
        is_afterhours = int(start_hour < 6 or start_hour >= 22)
        is_night     = int(start_hour >= 22 or start_hour < 5)
        is_earlyam   = int(5 <= start_hour < 7)
        is_lunchtime = int(11 <= start_hour < 14)

        # ── Kinematic ─────────────────────────────────────────────────
        speeds     = trip['speed'].dropna()
        speed_mean = speeds.mean() if len(speeds) > 0 else 0
        speed_max  = speeds.max()  if len(speeds) > 0 else 0
        speed_std  = speeds.std()  if len(speeds) > 1 else 0
        speed_cv   = (speed_std / speed_mean) if speed_mean > 0 else 0
        speed_above_80_ratio  = float((speeds > 80).sum()  / len(speeds)) if len(speeds) > 0 else 0
        speed_above_100_ratio = float((speeds > 100).sum() / len(speeds)) if len(speeds) > 0 else 0

        speed_vals        = trip['speed'].values
        time_vals         = trip['time_delta_s'].values
        accel             = np.where(time_vals[1:] > 0,
                                     np.diff(speed_vals) / time_vals[1:], 0)
        harsh_accel_count = int(np.sum(accel >  3.0))
        harsh_brake_count = int(np.sum(accel < -3.0))
        decel_events      = int(np.sum(accel < -2.0))

        idle_mask   = (trip['speed'] == 0) & (trip['acc'] == 1)
        idle_time_s = trip.loc[idle_mask, 'time_delta_s'].sum()
        idle_ratio  = idle_time_s / duration_s if duration_s > 0 else 0

        # ── Spatial ───────────────────────────────────────────────────
        total_km     = trip['mileage_delta_km'].sum()
        stop_count   = int(((trip['speed'] == 0) & (trip['speed'].shift(1) > 0)).sum())
        km_per_hour  = (total_km / (duration_s / 3600)) if duration_s > 0 else 0
        stops_per_km = (stop_count / total_km) if total_km > 0.5 else 0

        if 'heading' in trip.columns:
            headings        = trip['heading'].dropna()
            heading_std     = headings.std() if len(headings) > 1 else 0
            heading_changes = int((headings.diff().abs() > 45).sum())
        else:
            heading_std     = 0
            heading_changes = 0

        # ── Fuel ──────────────────────────────────────────────────────
        fuel_start    = trip['fuel1_volume'].iloc[0]
        fuel_end      = trip['fuel1_volume'].iloc[-1]
        fuel_consumed = fuel_start - fuel_end
        fuel_per_100km = (fuel_consumed / total_km * 100) if total_km > 0.5 else np.nan
        fuel_drop_rate = (fuel_consumed / duration_min) if duration_min > 0 else 0

        stationary            = trip[trip['speed'] == 0]
        fuel_drops_stationary = stationary['fuel_delta'].clip(upper=0).sum()
        suspicious_fuel_drop  = float(abs(fuel_drops_stationary))

        trip_features.append({
            'trip_id':                 trip_id,
            'vehicle_id':              vehicle_id,
            'start_time':              start_time,
            'end_time':                end_time,
            'start_hour':              start_hour,
            'day_of_week':             day_of_week,
            'is_weekend':              is_weekend,
            'is_afterhours':           is_afterhours,
            'is_night':                is_night,
            'is_earlyam':              is_earlyam,
            'is_lunchtime':            is_lunchtime,
            'speed_mean':              round(float(speed_mean), 2),
            'speed_max':               round(float(speed_max), 2),
            'speed_std':               round(float(speed_std), 4),
            'speed_cv':                round(speed_cv, 4),
            'speed_above_80_ratio':    round(speed_above_80_ratio, 4),
            'speed_above_100_ratio':   round(speed_above_100_ratio, 4),
            'harsh_accel_count':       harsh_accel_count,
            'harsh_brake_count':       harsh_brake_count,
            'decel_events':            decel_events,
            'idle_time_s':             round(idle_time_s, 2),
            'idle_ratio':              round(idle_ratio, 4),
            'total_km':                round(total_km, 4),
            'stop_count':              stop_count,
            'km_per_hour':             round(km_per_hour, 4),
            'stops_per_km':            round(stops_per_km, 4),
            'heading_std':             round(heading_std, 4),
            'heading_changes':         heading_changes,
            'fuel_start_L':            round(fuel_start, 2),
            'fuel_end_L':              round(fuel_end, 2),
            'fuel_consumed_L':         round(fuel_consumed, 2),
            'fuel_per_100km':          round(fuel_per_100km, 2) if not np.isnan(fuel_per_100km) else np.nan,
            'fuel_drop_rate':          round(fuel_drop_rate, 4),
            'suspicious_fuel_drop_L':  round(suspicious_fuel_drop, 2),
            'duration_min':            round(duration_min, 2),
        })

    return pd.DataFrame(trip_features)


def run_full_pipeline(uploaded_files):
    """Runs all 4 preprocessing steps, returns feature dataframe."""
    with st.status("⚙️ Running preprocessing pipeline...", expanded=True) as status:

        st.write("📥 Step 1 — Parsing JSON files...")
        raw_df = parse_json_files(uploaded_files)
        if raw_df is None:
            st.error("❌ No valid records parsed. Check your JSON file format.")
            return None
        st.write(f"   ✅ {len(raw_df):,} raw records from {len(uploaded_files)} file(s)")

        raw_df['speed'] = pd.to_numeric(raw_df['speed'], errors='coerce')
        moving = raw_df[raw_df['speed'] > 0]
        parked = raw_df[raw_df['speed'] == 0]
        st.write(f"   📊 Moving records: {len(moving):,} | Parked records: {len(parked):,}")

        ignition_count = raw_df[raw_df['event_message'].isin(['IGNITION ON', 'IGNITION OFF'])].shape[0]
        st.write(f"   🔑 Ignition events found: {ignition_count}")

        st.write("🔧 Step 2 — Cleaning & deduplicating...")
        clean_df, dupes = preprocess(raw_df)
        st.write(f"   ✅ {dupes} duplicates removed → {len(clean_df):,} clean records")

        st.write("🚗 Step 3 — Segmenting trips...")
        segmented_df = segment_trips(clean_df)
        trip_count   = segmented_df['trip_id'].nunique()
        st.write(f"   ✅ {trip_count} segments across {segmented_df['vehicle_id'].nunique()} vehicle(s)")

        st.write("⚙️ Step 4 — Engineering features...")
        features_df = engineer_features(segmented_df)
        st.write(f"   ✅ {len(features_df)} trip segments × {len(features_df.columns)} features")

        if len(features_df) == 0:
            st.error("❌ No usable trips found in these files.")
            st.info("💡 Use the 🗂️ File Scanner tab to find files with real vehicle movement.")
            return None

        status.update(label="✅ Pipeline complete!", state="complete")

    return features_df


# ══════════════════════════════════════════════════════════════════════
# LOAD ALL MODELS
# ══════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## System status")

# Ensemble model (Module 3)
required_files = [ENSEMBLE_MODEL_PATH, SCALER_PATH, IMPUTER_PATH]
missing_files  = [os.path.basename(f) for f in required_files if not os.path.exists(f)]

if not missing_files:
    model       = joblib.load(ENSEMBLE_MODEL_PATH)
    scaler      = joblib.load(SCALER_PATH)
    imputer     = joblib.load(IMPUTER_PATH)
    model_ready = True
    st.sidebar.markdown(
        '<div class="pill ok">Detection model ready<br>'
        '<span class="mono">LightGBM + XGBoost</span></div>',
        unsafe_allow_html=True)
else:
    model_ready = False
    st.sidebar.markdown(
        '<div class="pill warn">Detection model missing<br>'
        f'<span class="mono">{", ".join(missing_files)}</span></div>',
        unsafe_allow_html=True)
    st.sidebar.caption("Run module3_model_training.ipynb to generate these files.")

# Online model (Module 4)
if os.path.exists(ONLINE_MODEL_PATH):
    online_model = joblib.load(ONLINE_MODEL_PATH)
    st.sidebar.markdown(
        '<div class="pill ok">Online learning active<br>'
        '<span class="mono">updates on each verdict</span></div>',
        unsafe_allow_html=True)
else:
    online_model = None
    st.sidebar.markdown(
        '<div class="pill warn">Online learning unavailable<br>'
        '<span class="mono">run Module 4 to enable</span></div>',
        unsafe_allow_html=True)

# Feature columns
if os.path.exists(FEATURE_COLS_PATH):
    with open(FEATURE_COLS_PATH, 'r') as f:
        feature_cols = json.load(f)
else:
    feature_cols = None

# Show active paths in sidebar (helps debug)
with st.sidebar.expander("File locations"):
    st.caption(f"FYP Root: {FYP_DIR}")
    st.caption(f"Models:   {MODELS_DIR}")
    st.caption(f"Raw Data: {DATA_RAW_DIR}")
    st.caption(f"Logs:     {LOGS_DIR}")

# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "Audit",
    "Operational impact",
    "Verdict log",
    "File scanner",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Real-Time Audit
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.sidebar.markdown("## Data input")
    input_mode = st.sidebar.radio(
        "Source",
        ["Raw JSON telemetry", "Preprocessed CSV"],
        label_visibility="collapsed",
    )

    features_df = None

    # ── MODE A: Raw JSON ──────────────────────────────────────────────
    if "JSON" in input_mode:
        uploaded_jsons = st.sidebar.file_uploader(
            "Upload GPS telemetry JSON files",
            type=['json'],
            accept_multiple_files=True,
            help="Upload Month_Day_Vehicle.json files from data/raw/"
        )

        if uploaded_jsons:
            st.sidebar.caption(f"{len(uploaded_jsons)} file(s) staged")
            section("Staged telemetry",
                    f"{len(uploaded_jsons)} file(s) ready. Processing segments them into "
                    f"trips and derives the behavioural features the model scores.")

            if st.button("Process files", type="primary"):
                features_df = run_full_pipeline(uploaded_jsons)
                if features_df is not None:
                    st.session_state['features_df'] = features_df
                    st.session_state.pop('audit_results', None)
                    st.session_state.pop('reviewed_trips', None)
                    st.success(f"✅ Done! {len(features_df)} trip segments ready for audit.")
        else:
            section("No telemetry loaded",
                    "Upload Month_Day_Vehicle.json files from the sidebar to begin. "
                    "The File scanner tab tells you which files contain actual vehicle "
                    "movement before you commit to processing them.")

    # ── MODE B: Preprocessed CSV ──────────────────────────────────────
    else:
        uploaded_csv = st.sidebar.file_uploader(
            "Upload trip_features_labelled.csv",
            type=['csv'],
            help="Found in data/processed/trip_features_labelled.csv"
        )

        if uploaded_csv:
            features_df = pd.read_csv(uploaded_csv)
            st.session_state['features_df'] = features_df
            st.session_state.pop('audit_results', None)
            st.session_state.pop('reviewed_trips', None)
            st.caption(f"{len(features_df)} trips loaded.")
        else:
            section("No telemetry loaded",
                    "Upload trip_features_labelled.csv from your data/processed folder.")

    # Restore from session state
    if features_df is None and 'features_df' in st.session_state:
        features_df = st.session_state['features_df']

    # ── Summary + Audit ───────────────────────────────────────────────
    if features_df is not None:
        fuel_total = features_df['fuel_consumed_L'].sum() if 'fuel_consumed_L' in features_df.columns else 0
        section("Fleet summary")
        kpi_row([
            ("Trips segmented", f"{len(features_df):,}", None, ""),
            ("Vehicles",        f"{features_df['vehicle_id'].nunique()}", None, ""),
            ("Distance",        f"{features_df['total_km'].sum():,.0f}", "km", ""),
            ("Fuel tracked",    f"{fuel_total:,.1f}", "litres", ""),
        ])

        if not model_ready:
            st.markdown(
                '<div class="verdict flag">Detection model not loaded. Run '
                'module3_model_training.ipynb, then copy the .pkl files into your '
                'models folder.</div>', unsafe_allow_html=True)
        else:
            if st.button("Run security audit", type="primary"):
                with st.spinner("Scoring trips..."):
                    drop_cols = [
                        'trip_id', 'vehicle_id', 'start_time', 'end_time',
                        'is_anomaly', 'anomaly_score',
                        'flag_fuel_theft', 'flag_aggressive', 'flag_afterhours',
                        'flag_abnormal_fuel', 'flag_long_trip',
                        # Direct label-defining features — excluded from model training
                        'suspicious_fuel_drop_L',
                        'is_afterhours',
                        'fuel_per_100km',
                        'duration_min',
                    ]
                    X_new = features_df.drop(columns=[c for c in drop_cols if c in features_df.columns])

                    # Fill any missing columns with 0 (handles old CSVs missing new features)
                    for col in feature_cols:
                        if col not in X_new.columns:
                            X_new[col] = 0

                    X_new = X_new[feature_cols]  # enforce exact column order
                    X_imputed = imputer.transform(X_new)
                    X_scaled  = scaler.transform(X_imputed)
                    preds     = model.predict(X_scaled)

                    features_df['Detection'] = preds
                    st.session_state['audit_results'] = features_df

    # ── Results Table ─────────────────────────────────────────────────
    if 'audit_results' in st.session_state:
        res_df    = st.session_state['audit_results']
        anomalies = res_df[res_df['Detection'] == 1]
        normal    = res_df[res_df['Detection'] == 0]

        if 'reviewed_trips' not in st.session_state:
            st.session_state['reviewed_trips'] = {}   # trip_id -> verdict label
        reviewed = st.session_state['reviewed_trips']

        pending = anomalies[~anomalies['trip_id'].isin(reviewed.keys())]

        rate = len(anomalies) / len(res_df) * 100

        section("Audit result")
        kpi_row([
            ("Awaiting review", f"{len(pending):,}", "trips", "flag" if len(pending) else "clear"),
            ("Reviewed",        f"{len(reviewed):,}", "trips", ""),
            ("Cleared",         f"{len(normal):,}",   "trips", "clear"),
            ("Flag rate",       f"{rate:.1f}%",       "of all trips", ""),
        ])

        if len(anomalies) == 0:
            st.markdown('<div class="verdict clear">No anomalous trips in this batch. '
                        'Nothing requires operator review.</div>', unsafe_allow_html=True)
        elif len(pending) == 0:
            st.markdown(
                f'<div class="verdict clear">All {len(anomalies)} flagged trip(s) have '
                'been reviewed. Nothing outstanding.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="verdict flag">{len(pending)} trip(s) still need operator review. '
                'Verify each one below — every verdict updates the model.</div>',
                unsafe_allow_html=True)

        section("All trips", "Flagged trips are listed first.")

        display_cols = ['trip_id', 'vehicle_id', 'start_time',
                        'total_km', 'speed_max', 'suspicious_fuel_drop_L', 'Detection']
        disp = res_df[[c for c in display_cols if c in res_df.columns]].copy()
        disp = disp.sort_values('Detection', ascending=False)
        disp['Detection'] = disp['Detection'].map({0: "Cleared", 1: "Flagged"})
        # Reviewed trips show their actual verdict instead of the raw model flag
        disp['Detection'] = disp.apply(
            lambda r: ("Confirmed" if reviewed.get(r['trip_id']) == "Confirmed incident"
                       else "Dismissed") if r['trip_id'] in reviewed else r['Detection'],
            axis=1
        )
        disp = disp.rename(columns={
            'trip_id':                'Trip',
            'vehicle_id':             'Vehicle',
            'start_time':             'Started',
            'total_km':               'Distance (km)',
            'speed_max':              'Peak speed (km/h)',
            'suspicious_fuel_drop_L': 'Stationary fuel drop (L)',
            'Detection':              'Status',
        })

        def mark_status(col):
            colors = {
                'Flagged':   '#C97A0A',
                'Confirmed': '#A32E2E',
                'Dismissed': '#0E7C66',
                'Cleared':   '#0E7C66',
            }
            return [f'color: {colors.get(v, "inherit")}; font-weight: 600'
                    if v in ('Flagged', 'Confirmed') else f'color: {colors.get(v, "inherit")}'
                    for v in col]

        st.dataframe(
            disp.style.apply(mark_status, subset=['Status']),
            use_container_width=True, hide_index=True,
        )

        # ── Module 4: Operator Feedback ───────────────────────────────
        if not pending.empty:
            section("Operator verification",
                    "Confirm or dismiss each flagged trip. Verdicts are logged and fed "
                    "straight into the online model — no retraining required. Once you "
                    "record a verdict, that trip drops off this list.")

            selected_trip = st.selectbox(
                "Trip under review",
                pending['trip_id'].tolist()
            )

            details = pending[pending['trip_id'] == selected_trip].iloc[0]

            start_hr = int(details.get('start_hour', 0))
            kpi_row([
                ("Stationary fuel drop", f"{details.get('suspicious_fuel_drop_L', 0):.2f}", "litres", "flag"),
                ("Peak speed",           f"{details.get('speed_max', 0):.0f}", "km/h", ""),
                ("Idle time",            f"{details.get('idle_time_s', 0)/60:.0f}", "minutes", ""),
                ("Departed",             f"{start_hr:02d}:00",
                 "outside operating hours" if (start_hr < 6 or start_hr >= 22) else "within operating hours",
                 "flag" if (start_hr < 6 or start_hr >= 22) else ""),
            ])

            action = st.radio(
                "Verdict",
                ["Not yet reviewed",
                 "Confirmed incident",
                 "Authorised — dismiss flag"],
                horizontal=True
            )

            if st.button("Record verdict", type="primary"):
                if "Not yet" in action:
                    st.warning("Choose a verdict before recording.")
                else:
                    true_label = 1 if "Confirmed incident" in action else 0

                    # Ensure logs/ folder exists
                    os.makedirs(LOGS_DIR, exist_ok=True)

                    feedback_entry = {
                        'timestamp' : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'trip_id'   : selected_trip,
                        'vehicle_id': details.get('vehicle_id', ''),
                        'decision'  : action,
                        'true_label': true_label,
                        'fuel_drop' : details.get('suspicious_fuel_drop_L', 0),
                        'speed_max' : details.get('speed_max', 0),
                    }
                    fb_new = pd.DataFrame([feedback_entry])
                    if os.path.exists(FEEDBACK_LOG):
                        fb_new.to_csv(FEEDBACK_LOG, mode='a', header=False, index=False)
                    else:
                        fb_new.to_csv(FEEDBACK_LOG, index=False)

                    if online_model is not None and feature_cols is not None:
                        trip_row = pending[pending['trip_id'] == selected_trip]
                        xi = trip_row[feature_cols].fillna(0).iloc[0].to_dict()
                        online_model.learn_one(xi, true_label)
                        joblib.dump(online_model, ONLINE_MODEL_PATH)
                        online_updated = True
                    else:
                        online_updated = False

                    # Mark this trip as reviewed so it drops out of the pending list
                    st.session_state['reviewed_trips'][selected_trip] = action

                    if online_updated:
                        st.toast(f"Verdict recorded for {selected_trip} — online model updated.",
                                 icon="✅")
                    else:
                        st.toast(f"Verdict recorded for {selected_trip} — online model not "
                                 "loaded, saved to log only.", icon="⚠️")

                    st.rerun()

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Security Analytics
# ══════════════════════════════════════════════════════════════════════
with tab2:
    section("Programme targets",
            "Precision and false-positive rate are measured from real operator verdicts. "
            "Diesel reduction is a pilot simulation, not observed data.")

    if os.path.exists(FEEDBACK_LOG):
        fb_df      = pd.read_csv(FEEDBACK_LOG)
        confirmed  = fb_df[fb_df['true_label'] == 1]
        false_alms = fb_df[fb_df['true_label'] == 0]
        total_fb   = len(fb_df)
        precision  = len(confirmed) / total_fb * 100 if total_fb > 0 else 0
        fpr        = len(false_alms) / total_fb * 100 if total_fb > 0 else 0
    else:
        confirmed = false_alms = pd.DataFrame()
        precision = fpr = total_fb = 0

    if total_fb > 0:
        prec_tone = "clear" if precision >= 90 else "flag"
        kpi_row([
            ("Detection precision", f"{precision:.1f}%", "target ≥90%", prec_tone),
            ("Confirmed incidents", f"{len(confirmed)}", f"from {total_fb} verdicts", "alert"),
            ("False positive rate", f"{fpr:.1f}%", "dismissed flags", ""),
            ("Diesel loss cut",     "24.5%", "simulated · target ≥20%", "key"),
        ])
        if precision < 90:
            st.markdown(
                f'<div class="verdict flag">Precision is {precision:.1f}%, below the 90% '
                f'target. With only {total_fb} verdict(s) recorded this figure is still '
                'volatile — review more flagged trips before drawing conclusions.</div>',
                unsafe_allow_html=True)
    else:
        kpi_row([
            ("Detection precision", "—", "target ≥90%", ""),
            ("Confirmed incidents", "—", "no verdicts yet", ""),
            ("False positive rate", "—", "no verdicts yet", ""),
            ("Diesel loss cut",     "24.5%", "simulated · target ≥20%", "key"),
        ])
        st.markdown('<div class="verdict flag">No operator verdicts recorded yet. '
                    'Run an audit and verify flagged trips to populate these figures.</div>',
                    unsafe_allow_html=True)

    section("Diesel loss, 12-month pilot simulation",
            "Modelled baseline against modelled post-deployment loss. Deployment begins in May.")

    months        = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    loss_baseline = [450, 430, 460, 440, 480, 450, 400, 390, 380, 370, 360, 350]
    loss_with_ai  = [450, 430, 460, 440, 320, 280, 250, 210, 190, 180, 160, 150]

    trend_df = pd.DataFrame({
        "Month":    months * 2,
        "Litres":   loss_baseline + loss_with_ai,
        "Scenario": ["Baseline"] * 12 + ["With detection system"] * 12,
        "Order":    list(range(12)) * 2,
    })

    line = alt.Chart(trend_df).mark_line(point=alt.OverlayMarkDef(size=38), strokeWidth=2.4).encode(
        x=alt.X("Month:N", sort=months, title=None,
                axis=alt.Axis(labelAngle=0, labelColor="#55646F", domainColor="#D2D9E0", tickColor="#D2D9E0")),
        y=alt.Y("Litres:Q", title="Diesel lost (litres)",
                axis=alt.Axis(labelColor="#55646F", titleColor="#55646F",
                              gridColor="#E6EAEE", domainOpacity=0)),
        color=alt.Color("Scenario:N",
                        scale=alt.Scale(domain=["Baseline", "With detection system"],
                                        range=["#8C9AA6", "#14506B"]),
                        legend=alt.Legend(title=None, orient="top", labelColor="#1B2A33")),
        strokeDash=alt.StrokeDash("Scenario:N",
                                  scale=alt.Scale(domain=["Baseline", "With detection system"],
                                                  range=[[5, 4], [1, 0]]),
                                  legend=None),
        tooltip=["Month", "Scenario", "Litres"],
    )

    deploy = alt.Chart(pd.DataFrame({"Month": ["May"]})).mark_rule(
        color="#C97A0A", strokeWidth=1.5, strokeDash=[3, 3]
    ).encode(x=alt.X("Month:N", sort=months))

    st.altair_chart(
        (line + deploy).properties(height=320).configure_view(strokeOpacity=0)
                       .configure(font="Barlow", background="#FFFFFF", padding=18),
        use_container_width=True,
    )

    reduction_pct = (sum(loss_baseline[4:]) - sum(loss_with_ai[4:])) / sum(loss_baseline[4:]) * 100
    st.markdown(
        f'<div class="verdict clear">Simulated reduction from deployment onward: '
        f'<strong>{reduction_pct:.1f}%</strong> against the ≥20% target. '
        'These are modelled figures for the pilot design, not measured results.</div>',
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — Feedback Log
# ══════════════════════════════════════════════════════════════════════
with tab3:
    section("Operator verdict log",
            "Every recorded verdict, in order. Each one was also passed to the online "
            "model at the moment it was submitted.")

    if os.path.exists(FEEDBACK_LOG):
        fb_df = pd.read_csv(FEEDBACK_LOG)
        n_conf = len(fb_df[fb_df['true_label'] == 1])
        n_dism = len(fb_df[fb_df['true_label'] == 0])

        kpi_row([
            ("Verdicts recorded",  f"{len(fb_df):,}", None, ""),
            ("Confirmed incidents", f"{n_conf:,}", "true positives", "alert"),
            ("Dismissed flags",     f"{n_dism:,}", "false positives", ""),
        ])

        st.dataframe(fb_df, use_container_width=True, hide_index=True)

        with st.expander("Clear the log"):
            st.caption("This permanently deletes every recorded verdict. The online model "
                       "keeps what it has already learned — clearing the log does not undo it.")
            if st.button("Delete all verdicts"):
                os.remove(FEEDBACK_LOG)
                st.rerun()
    else:
        st.markdown('<div class="verdict flag">No verdicts recorded yet. Run an audit, '
                    'then verify flagged trips to start building the log.</div>',
                    unsafe_allow_html=True)
        st.caption(f"The log will be created at {FEEDBACK_LOG}")

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — File Scanner
# ══════════════════════════════════════════════════════════════════════
with tab4:
    section("File scanner",
            "Check which telemetry files contain actual vehicle movement before you "
            "process them. Parked-vehicle files produce no usable trips.")

    folder_path = st.text_input(
        "Folder to scan",
        value=DATA_RAW_DIR,
    )

    if st.button("Scan folder", type="primary"):
        if not os.path.exists(folder_path):
            st.error(f"❌ Folder not found: {folder_path}")
        else:
            json_files = glob.glob(os.path.join(folder_path, "*.json"))

            if not json_files:
                st.warning("No JSON files found in that folder.")
            else:
                st.caption(f"Scanning {len(json_files)} files...")

                scan_results = []
                progress     = st.progress(0)

                for i, file_path in enumerate(sorted(json_files)):
                    filename = os.path.basename(file_path)
                    progress.progress((i + 1) / len(json_files))

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            raw = json.load(f)

                        records = []
                        for l1 in raw:
                            for l2 in l1:
                                if isinstance(l2, list):
                                    records.extend(l2)
                                elif isinstance(l2, dict):
                                    records.append(l2)

                        df_scan                 = pd.DataFrame(records)
                        df_scan['speed']        = pd.to_numeric(df_scan['speed'], errors='coerce')
                        df_scan['fuel1_volume'] = pd.to_numeric(df_scan['fuel1_volume'], errors='coerce')

                        max_speed     = df_scan['speed'].max()
                        ignition_ons  = (df_scan['event_message'] == 'IGNITION ON').sum()
                        ignition_offs = (df_scan['event_message'] == 'IGNITION OFF').sum()
                        total_records = len(df_scan)
                        fuel_range    = df_scan['fuel1_volume'].max() - df_scan['fuel1_volume'].min()
                        has_movement  = max_speed > 0 and ignition_ons >= 1

                        scan_results.append({
                            'File':           filename,
                            'Records':        total_records,
                            'Max Speed':      round(max_speed, 1),
                            'IGNITION ON':    ignition_ons,
                            'IGNITION OFF':   ignition_offs,
                            'Fuel Range (L)': round(fuel_range, 2),
                            'Has Movement':   'Moving' if has_movement else 'Parked',
                        })

                    except Exception as e:
                        scan_results.append({
                            'File':           filename,
                            'Records':        0,
                            'Max Speed':      0,
                            'IGNITION ON':    0,
                            'IGNITION OFF':   0,
                            'Fuel Range (L)': 0,
                            'Has Movement':   f'Unreadable: {e}',
                        })

                progress.empty()
                scan_df = pd.DataFrame(scan_results)

                good = scan_df[scan_df['Has Movement'] == 'Moving']
                park = scan_df[scan_df['Has Movement'] == 'Parked']

                kpi_row([
                    ("Files scanned",  f"{len(scan_df)}", None, ""),
                    ("With movement",  f"{len(good)}", "usable", "clear"),
                    ("Parked only",    f"{len(park)}", "no trips", ""),
                ])

                if not good.empty:
                    section("Files with vehicle movement",
                            f"Upload these {len(good)} file(s) in the Audit tab.")
                    st.dataframe(good, use_container_width=True, hide_index=True)

                if not park.empty:
                    with st.expander(f"Parked files ({len(park)})"):
                        st.caption("No ignition events or no recorded speed — these "
                                   "produce no trip segments.")
                        st.dataframe(park, use_container_width=True, hide_index=True)
                        st.caption("These files only contain parked vehicle data.")