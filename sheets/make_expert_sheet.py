"""
make_expert_sheet.py
Builds the blinded review sheet that the domain expert fills in.

Run:   python make_expert_sheet.py
Needs: pandas, numpy, joblib, openpyxl  (pip install openpyxl if it complains)

Produces (in OUT_DIR):
  expert_review_sheet.xlsx      <- send THIS to the collaborator
  expert_answer_key_PRIVATE.csv <- keep this; it holds the system's flags
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import joblib

# ── Paths (edit only if your folders differ) ─────────────────────────
DATA_PATH  = r"C:\Users\User\Music\FYP\data\processed\trip_features_labelled.csv"
MODELS_DIR = r"C:\Users\User\Music\FYP\models"
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))

N_FLAGGED         = 40   # trips the system flagged
N_UNFLAGGED_RAND  = 5    # random trips it did not flag
N_UNFLAGGED_NEAR  = 5    # unflagged trips that scored closest to the threshold
MAX_PER_VEHICLE   = 8    # stops one vehicle dominating the flagged sample
SEED              = 42
SHOW_REAL_VEHICLE = True # False -> "Vehicle A, B, C..." instead of the real number

# A trip must be long enough for a person to judge
MIN_KM, MAX_KM   = 1.0, 1500.0
MIN_MIN, MAX_MIN = 5.0, 1440.0

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=DATA_PATH)
ap.add_argument("--models", default=MODELS_DIR)
ap.add_argument("--out", default=OUT_DIR)
args = ap.parse_args()
DATA_PATH, MODELS_DIR, OUT_DIR = args.data, args.models, args.out
os.makedirs(OUT_DIR, exist_ok=True)

try:
    import openpyxl  # noqa: F401
except ImportError:
    sys.exit("openpyxl is missing. Run:  pip install openpyxl   then run this script again.")

# ── Load data and the SAME model the dashboard uses ──────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["start_time", "end_time"])
model   = joblib.load(os.path.join(MODELS_DIR, "ensemble_model.pkl"))
scaler  = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
imputer = joblib.load(os.path.join(MODELS_DIR, "imputer.pkl"))
with open(os.path.join(MODELS_DIR, "feature_cols.json")) as f:
    feature_cols = json.load(f)
tpath = os.path.join(MODELS_DIR, "threshold.json")
if not os.path.exists(tpath):
    sys.exit("threshold.json not found. Run the save cell in the notebook first.")
with open(tpath) as f:
    thr = float(json.load(f)["threshold"])

X = scaler.transform(imputer.transform(df[feature_cols]))
df["proba"]   = model.predict_proba(X)[:, 1]
df["flagged"] = (df["proba"] >= thr).astype(int)
df["dur_min"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60

print(f"Decision threshold: {thr:.4f}")
print(f"Trips scored: {len(df)}, flagged: {int(df.flagged.sum())}")

reviewable = df["total_km"].between(MIN_KM, MAX_KM) & df["dur_min"].between(MIN_MIN, MAX_MIN)
print(f"Reviewable (>= {MIN_KM} km and >= {MIN_MIN:.0f} min): {int(reviewable.sum())} of {len(df)}")

rng = np.random.default_rng(SEED)
pool_f = df[reviewable & (df.flagged == 1)].copy()
pool_u = df[reviewable & (df.flagged == 0)].copy()

# ── Group flagged trips by what looks unusual, so the sample is varied ─
fuel_cut = pool_f["fuel_drop_rate"].quantile(0.75) if len(pool_f) else 0
dur_cut  = pool_f["dur_min"].quantile(0.75) if len(pool_f) else 0
def group_of(r):
    if r.is_night == 1 or r.is_earlyam == 1: return "night / early morning"
    if r.fuel_drop_rate >= fuel_cut and r.fuel_drop_rate > 0: return "high fuel use"
    if r.speed_above_80_ratio > 0: return "high speed"
    if r.dur_min >= dur_cut: return "long trip"
    return "other"
pool_f["group"] = pool_f.apply(group_of, axis=1)

# round-robin across groups, spreading vehicles
picked, vcount = [], {}
groups = {g: list(rng.permutation(sub.index)) for g, sub in pool_f.groupby("group")}
order = list(groups)
n_target = min(N_FLAGGED, len(pool_f))
while len(picked) < n_target and any(groups.values()):
    progressed = False
    for g in order:
        if len(picked) >= n_target: break
        q = groups[g]
        for k, idx in enumerate(q):
            v = pool_f.at[idx, "vehicle_id"]
            if vcount.get(v, 0) < MAX_PER_VEHICLE:
                picked.append(idx); vcount[v] = vcount.get(v, 0) + 1
                q.pop(k); progressed = True
                break
    if not progressed: break
if len(picked) < n_target:                       # relax the per-vehicle cap
    left = [i for q in groups.values() for i in q]
    picked += list(rng.permutation(left))[: n_target - len(picked)]
sel_f = pool_f.loc[picked].copy(); sel_f["source"] = "flagged"

# ── Unflagged: some random, some near misses ─────────────────────────
near = pool_u.sort_values("proba", ascending=False).head(N_UNFLAGGED_NEAR).copy()
near["source"] = "unflagged (near miss)"
rest = pool_u.drop(near.index)
rand = rest.loc[rng.permutation(rest.index)[:N_UNFLAGGED_RAND]].copy()
rand["source"] = "unflagged (random)"
sel_u = pd.concat([near, rand]); sel_u["group"] = "n/a"

sel = pd.concat([sel_f, sel_u]).sample(frac=1, random_state=SEED).reset_index(drop=True)
sel["Trip"] = [f"T{i+1:02d}" for i in range(len(sel))]

vmap = {v: chr(65 + i) for i, v in enumerate(sorted(sel.vehicle_id.unique()))}
sel["Vehicle"] = sel.vehicle_id.map(lambda v: f"Vehicle {v}" if SHOW_REAL_VEHICLE else f"Vehicle {vmap[v]}")

# ── Private answer key ───────────────────────────────────────────────
key = sel[["Trip", "trip_id", "vehicle_id", "flagged", "proba", "source", "group", "is_anomaly"]].copy()
key = key.rename(columns={"flagged": "system_flagged", "is_anomaly": "rule_label"})
key.to_csv(os.path.join(OUT_DIR, "expert_answer_key_PRIVATE.csv"), index=False)

# ── Blinded sheet for the expert ─────────────────────────────────────
dow = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
sheet = pd.DataFrame({
    "Trip": sel["Trip"],
    "Vehicle": sel["Vehicle"],
    "Date": sel["start_time"].dt.strftime("%Y-%m-%d") + " (" + sel["day_of_week"].map(lambda d: dow[int(d)]) + ")",
    "Start": sel["start_time"].dt.strftime("%H:%M"),
    "End": sel["end_time"].dt.strftime("%H:%M"),
    "Duration (min)": sel["dur_min"].round(0).astype(int),
    "Distance (km)": sel["total_km"].round(1),
    "Average speed (km/h)": sel["speed_mean"].round(0).astype(int),
    "Time above 80 km/h (%)": (sel["speed_above_80_ratio"] * 100).round(0).astype(int),
    "Fuel used (L)": (sel["fuel_drop_rate"] * sel["dur_min"]).round(1),
    "Verdict": "",
    "Type / reason": "",
    "Notes": "",
})

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter

wb = Workbook()
ws0 = wb.active; ws0.title = "Instructions"
lines = [
    ("Trip review: how to fill this in", True),
    ("", False),
    ("Purpose: our system flags trips that may be incidents, using rules we defined ourselves. "
     "We need your judgement on whether these trips look like real incidents. Your answers are the only independent check we have.", False),
    ("", False),
    ("What to do (about 30 to 45 minutes):", True),
    ("1. Go to the 'Trips' sheet. Each row is one trip.", False),
    ("2. Read the details for the trip (date, time, distance, speed, fuel used).", False),
    ("3. In 'Verdict' choose one: Real incident / Normal operation / Not sure.", False),
    ("4. In 'Type / reason' pick the type if it is an incident, or the likely reason if it is normal.", False),
    ("5. Use 'Notes' for anything else. This is optional.", False),
    ("6. Save the file and send it back as it is.", False),
    ("", False),
    ("Meaning of the verdicts:", True),
    ("Real incident: this should be investigated (for example fuel theft, unauthorised use, harsh driving).", False),
    ("Normal operation: a legitimate trip, even if it looks unusual in the numbers.", False),
    ("Not sure: you cannot tell from this information. This is a perfectly good answer.", False),
    ("", False),
    ("Notes on the data:", True),
    ("Fuel used is the fuel level change over the trip. "
     "You are not shown the system's decision on purpose, so please judge each trip on its own.", False),
    ("", False),
    ("Example row (not a real trip): Trip T00, 2024-03-05 (Tue), 22:10 to 23:40, 12 km, fuel used 9.0 L "
     "-> Verdict: Real incident, Type: Fuel theft, Notes: vehicle should have been parked.", False),
]
for i, (t, bold) in enumerate(lines, 1):
    c = ws0.cell(row=i, column=1, value=t)
    c.font = Font(bold=bold, size=13 if i == 1 else 11)
    c.alignment = Alignment(wrap_text=True, vertical="top")
ws0.column_dimensions["A"].width = 110

ws = wb.create_sheet("Trips")
cols = list(sheet.columns)
ws.append(cols)
for row in sheet.itertuples(index=False):
    ws.append(list(row))

hdr_fill = PatternFill("solid", fgColor="1F3A4D")
ans_fill = PatternFill("solid", fgColor="FFF4CC")
thin = Side(style="thin", color="BBBBBB")
for j, name in enumerate(cols, 1):
    c = ws.cell(row=1, column=j)
    c.font = Font(bold=True, color="FFFFFF"); c.fill = hdr_fill
    c.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
    ws.column_dimensions[get_column_letter(j)].width = 16 if j < 11 else (18 if j == 11 else 28)
ws.column_dimensions["C"].width = 18
ws.column_dimensions["M"].width = 40
ws.row_dimensions[1].height = 34
for r in range(2, len(sheet) + 2):
    for j in range(1, len(cols) + 1):
        c = ws.cell(row=r, column=j)
        c.border = Border(top=thin, bottom=thin, left=thin, right=thin)
        c.alignment = Alignment(vertical="center", wrap_text=(j >= 11))
        if j >= 11: c.fill = ans_fill
ws.freeze_panes = "C2"

last = len(sheet) + 1
dv1 = DataValidation(type="list", formula1='"Real incident,Normal operation,Not sure"', allow_blank=True)
dv2 = DataValidation(type="list", allow_blank=True,
        formula1='"Fuel theft,Unauthorised use,Harsh driving,Other incident,Refuelling,Normal delivery,Sensor or GPS error,Other normal"')
ws.add_data_validation(dv1); ws.add_data_validation(dv2)
dv1.add(f"K2:K{last}"); dv2.add(f"L2:L{last}")

out_xlsx = os.path.join(OUT_DIR, "expert_review_sheet.xlsx")
wb.save(out_xlsx)

print(f"\nSelected {len(sel)} trips:")
print(sel["source"].value_counts().to_string())
print("\nFlagged trips by group:"); print(sel[sel.source == "flagged"]["group"].value_counts().to_string())
print("\nFlagged trips by vehicle:"); print(sel[sel.source == "flagged"]["vehicle_id"].value_counts().sort_index().to_string())
print(f"\nSend this file  : {out_xlsx}")
print(f"Keep this PRIVATE: {os.path.join(OUT_DIR, 'expert_answer_key_PRIVATE.csv')}")