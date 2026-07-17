import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import glob
from river import forest
from datetime import datetime

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Logistics Fleet Security System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Logistics Fleet Security System")
st.markdown("**Predictive Modelling of Anomalous Driver Behaviour**")

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
st.sidebar.header("⚙️ System Status")

# Ensemble model (Module 3)
required_files = [ENSEMBLE_MODEL_PATH, SCALER_PATH, IMPUTER_PATH]
missing_files  = [os.path.basename(f) for f in required_files if not os.path.exists(f)]

if not missing_files:
    model       = joblib.load(ENSEMBLE_MODEL_PATH)
    scaler      = joblib.load(SCALER_PATH)
    imputer     = joblib.load(IMPUTER_PATH)
    model_ready = True
    st.sidebar.success("✅ Ensemble Model Loaded")
else:
    model_ready = False
    st.sidebar.warning(f"⚠️ Missing: {', '.join(missing_files)}")
    st.sidebar.info("Run notebooks/module3_model_training.ipynb first.")

# Online model (Module 4)
if os.path.exists(ONLINE_MODEL_PATH):
    online_model = joblib.load(ONLINE_MODEL_PATH)
    st.sidebar.success("✅ Online Learning Model Loaded")
else:
    online_model = None
    st.sidebar.warning("⚠️ Online model not found — run Module 4 first")

# Feature columns
if os.path.exists(FEATURE_COLS_PATH):
    with open(FEATURE_COLS_PATH, 'r') as f:
        feature_cols = json.load(f)
else:
    feature_cols = None

# Show active paths in sidebar (helps debug)
with st.sidebar.expander("📁 Active Paths"):
    st.caption(f"FYP Root: {FYP_DIR}")
    st.caption(f"Models:   {MODELS_DIR}")
    st.caption(f"Raw Data: {DATA_RAW_DIR}")
    st.caption(f"Logs:     {LOGS_DIR}")

# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Real-Time Audit",
    "📊 Security Analytics",
    "🔄 Feedback Log",
    "🗂️ File Scanner"
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Real-Time Audit
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Trip Risk Assessment")

    st.sidebar.header("📂 Data Input")
    input_mode = st.sidebar.radio(
        "Choose input type:",
        ["📁 Raw JSON Files (recommended)", "📄 Preprocessed CSV"]
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
            st.sidebar.success(f"✅ {len(uploaded_jsons)} file(s) ready")
            st.info(f"📁 {len(uploaded_jsons)} JSON file(s) uploaded — click below to process")

            if st.button("⚙️ Process JSON Files", type="primary"):
                features_df = run_full_pipeline(uploaded_jsons)
                if features_df is not None:
                    st.session_state['features_df'] = features_df
                    st.session_state.pop('audit_results', None)
                    st.success(f"✅ Done! {len(features_df)} trip segments ready for audit.")
        else:
            st.info("👈 Upload your Month_Day_Vehicle.json files from the sidebar.\n\n"
                    "💡 Use the **🗂️ File Scanner** tab to identify which files have real vehicle movement.")

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
            st.success(f"✅ CSV loaded — {len(features_df)} trips ready.")
        else:
            st.info("👈 Upload trip_features_labelled.csv from data/processed/ folder.")

    # Restore from session state
    if features_df is None and 'features_df' in st.session_state:
        features_df = st.session_state['features_df']

    # ── Summary + Audit ───────────────────────────────────────────────
    if features_df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Trips",     len(features_df))
        c2.metric("Vehicles Active", features_df['vehicle_id'].nunique())
        c3.metric("Total Distance",  f"{features_df['total_km'].sum():.0f} km")
        fuel_total = features_df['fuel_consumed_L'].sum() if 'fuel_consumed_L' in features_df.columns else 0
        c4.metric("Fuel Tracked", f"{fuel_total:.1f} L")

        st.divider()

        if not model_ready:
            st.warning("⚠️ Model files not found. Run notebooks/module3_model_training.ipynb first.")
        else:
            if st.button("🚀 Run Security Audit", type="primary"):
                with st.spinner("Analysing trips with AI model..."):
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
                    st.success(f"✅ Audit complete! {int(preds.sum())} anomalies out of {len(preds)} trips.")

    # ── Results Table ─────────────────────────────────────────────────
    if 'audit_results' in st.session_state:
        res_df    = st.session_state['audit_results']
        anomalies = res_df[res_df['Detection'] == 1]
        normal    = res_df[res_df['Detection'] == 0]

        r1, r2, r3 = st.columns(3)
        r1.metric("🚨 Anomalies Detected", len(anomalies))
        r2.metric("✅ Normal Trips",        len(normal))
        r3.metric("⚠️ Anomaly Rate",        f"{len(anomalies)/len(res_df)*100:.1f}%")

        st.write("### 📜 Full Audit Results")

        display_cols = ['trip_id', 'vehicle_id', 'start_time',
                        'total_km', 'speed_max', 'suspicious_fuel_drop_L', 'Detection']
        disp = res_df[[c for c in display_cols if c in res_df.columns]].copy()
        disp['Detection'] = disp['Detection'].map({0: "✅ Normal", 1: "🚨 ANOMALY"})

        def highlight_anomaly(row):
            return ['background-color: #ffcccc' if row['Detection'] == "🚨 ANOMALY"
                    else '' for _ in row]

        st.dataframe(
            disp.style.apply(highlight_anomaly, axis=1),
            use_container_width=True
        )

        # ── Module 4: Operator Feedback ───────────────────────────────
        if not anomalies.empty:
            st.divider()
            st.subheader("🛠️ Module 4 — Operator Incident Verification")
            st.caption("Confirm each flagged trip — your decision updates the AI model instantly.")

            selected_trip = st.selectbox(
                "Select Anomalous Trip to Investigate:",
                anomalies['trip_id'].tolist()
            )

            details = anomalies[anomalies['trip_id'] == selected_trip].iloc[0]

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("⛽ Fuel Drop",  f"{details.get('suspicious_fuel_drop_L', 0):.2f} L")
            col_b.metric("🚗 Max Speed",  f"{details.get('speed_max', 0):.1f} km/h")
            col_c.metric("⏱️ Idle Time",  f"{details.get('idle_time_s', 0):.0f} s")
            col_d.metric("🕐 Start Hour", f"{int(details.get('start_hour', 0))}:00")

            action = st.radio(
                "Fleet Manager Decision:",
                ["⏳ Pending Review",
                 "✅ Confirmed Theft / Incident",
                 "❌ False Alarm (Authorised)"],
                horizontal=True
            )

            if st.button("📨 Submit Feedback", type="primary"):
                if "Pending" in action:
                    st.warning("Please make a decision before submitting.")
                else:
                    true_label = 1 if "Confirmed" in action else 0

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
                        trip_row = anomalies[anomalies['trip_id'] == selected_trip]
                        xi = trip_row[feature_cols].fillna(0).iloc[0].to_dict()
                        online_model.learn_one(xi, true_label)
                        joblib.dump(online_model, ONLINE_MODEL_PATH)
                        st.success(f"✅ Feedback submitted for **{selected_trip}**!")
                        st.info("🔄 Online model updated instantly — no retraining needed.")
                    else:
                        st.success(f"✅ Feedback saved for **{selected_trip}**!")
                        st.warning("⚠️ Online model not loaded — only saved to logs/feedback_log.csv.")

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Security Analytics
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Operational Impact — KPIs")
    st.markdown("**Target:** ≥90% Precision | ≥20% Reduction in Diesel Loss")

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

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Anomaly Precision",
              f"{precision:.1f}%" if total_fb > 0 else "—", "Target: ≥90%")
    k2.metric("Diesel Loss Reduction", "24.5%", "Target: ≥20% (Simulated)")
    k3.metric("Confirmed Incidents",   len(confirmed) if total_fb > 0 else "—")
    k4.metric("False Positive Rate",
              f"{fpr:.1f}%" if total_fb > 0 else "—", "Target: Low")

    st.caption("Precision & FPR calculated from real operator feedback. Diesel reduction is simulated for pilot.")
    st.divider()

    st.subheader("📉 12-Month Diesel Loss Trend (Pilot Simulation)")
    months        = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    loss_baseline = [450, 430, 460, 440, 480, 450, 400, 390, 380, 370, 360, 350]
    loss_with_ai  = [450, 430, 460, 440, 320, 280, 250, 210, 190, 180, 160, 150]

    chart_df = pd.DataFrame({
        "Month":              months,
        "Baseline Loss (L)":  loss_baseline,
        "With AI System (L)": loss_with_ai,
    }).set_index("Month")

    st.line_chart(chart_df)
    st.caption("Figure 1: Simulated diesel loss before vs after AI deployment. Pilot started May.")

    reduction_pct = (sum(loss_baseline[4:]) - sum(loss_with_ai[4:])) / sum(loss_baseline[4:]) * 100
    st.success(f"📊 Simulated reduction: **{reduction_pct:.1f}%** — Target ≥20% ✅")

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — Feedback Log
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🔄 Operator Feedback Log")
    st.caption(f"Saved to: logs/feedback_log.csv — each entry triggers an online model update.")

    if os.path.exists(FEEDBACK_LOG):
        fb_df = pd.read_csv(FEEDBACK_LOG)
        st.dataframe(fb_df, use_container_width=True)

        f1, f2, f3 = st.columns(3)
        f1.metric("Total Entries",       len(fb_df))
        f2.metric("Confirmed Incidents", len(fb_df[fb_df['true_label'] == 1]))
        f3.metric("False Alarms",        len(fb_df[fb_df['true_label'] == 0]))

        if st.button("🗑️ Clear Feedback Log"):
            os.remove(FEEDBACK_LOG)
            st.success("Feedback log cleared.")
            st.rerun()
    else:
        st.info("No feedback submitted yet. Verify anomalies in Tab 1 to start building the log.")
        st.caption(f"Log will be created at: {FEEDBACK_LOG}")

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — File Scanner
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🗂️ File Scanner")
    st.markdown("Scan your JSON files to find which ones have **real vehicle movement** "
                "before uploading to the audit.")

    # Default to data/raw/ folder
    folder_path = st.text_input(
        "Folder path to scan:",
        value=DATA_RAW_DIR,
        help="Default points to your FYP/data/raw/ folder"
    )

    if st.button("🔍 Scan Folder", type="primary"):
        if not os.path.exists(folder_path):
            st.error(f"❌ Folder not found: {folder_path}")
        else:
            json_files = glob.glob(os.path.join(folder_path, "*.json"))

            if not json_files:
                st.warning("No JSON files found in that folder.")
            else:
                st.info(f"Found {len(json_files)} JSON files. Scanning...")

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
                            'Has Movement':   '✅ Yes' if has_movement else '💤 Parked',
                        })

                    except Exception as e:
                        scan_results.append({
                            'File':           filename,
                            'Records':        0,
                            'Max Speed':      0,
                            'IGNITION ON':    0,
                            'IGNITION OFF':   0,
                            'Fuel Range (L)': 0,
                            'Has Movement':   f'❌ Error: {e}',
                        })

                progress.empty()
                scan_df = pd.DataFrame(scan_results)

                good = scan_df[scan_df['Has Movement'] == '✅ Yes']
                park = scan_df[scan_df['Has Movement'] == '💤 Parked']

                s1, s2, s3 = st.columns(3)
                s1.metric("Total Files",        len(scan_df))
                s2.metric("✅ Files With Trips", len(good))
                s3.metric("💤 Parked Files",     len(park))

                st.divider()

                if not good.empty:
                    st.subheader("✅ Files With Real Vehicle Movement — Use These!")
                    st.dataframe(good, use_container_width=True)
                    st.info(f"💡 Upload these {len(good)} files to the Real-Time Audit tab "
                            f"for proper anomaly detection.")

                if not park.empty:
                    with st.expander(f"💤 Parked / No Movement Files ({len(park)} files)"):
                        st.dataframe(park, use_container_width=True)
                        st.caption("These files only contain parked vehicle data.")