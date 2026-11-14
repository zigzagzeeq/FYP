"""
score_expert_results.py
Run AFTER the expert returns the sheet.

Run:   python score_expert_results.py
Put the returned file next to this script as  expert_review_sheet_RETURNED.xlsx
(or pass --sheet "path\\to\\file.xlsx"). The private answer key must be in the same folder.
"""
import os, sys, argparse
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--sheet", default=os.path.join(here, "expert_review_sheet_RETURNED.xlsx"))
ap.add_argument("--key",   default=os.path.join(here, "expert_answer_key_PRIVATE.csv"))
args = ap.parse_args()

for p in (args.sheet, args.key):
    if not os.path.exists(p):
        sys.exit(f"File not found: {p}")

ans = pd.read_excel(args.sheet, sheet_name="Trips")
key = pd.read_csv(args.key)
m = key.merge(ans[["Trip", "Verdict", "Type / reason", "Notes"]], on="Trip", how="left")

m["Verdict"] = m["Verdict"].fillna("").astype(str).str.strip()
unanswered = (m["Verdict"] == "").sum()
if unanswered:
    print(f"Note: {unanswered} trips have no verdict and are left out.\n")
m = m[m["Verdict"] != ""]

def count(df, v): return int((df["Verdict"] == v).sum())

fl = m[m["system_flagged"] == 1]
un = m[m["system_flagged"] == 0]
real, normal, unsure = count(fl, "Real incident"), count(fl, "Normal operation"), count(fl, "Not sure")
n = len(fl)

print("=== FLAGGED trips (system said: anomaly) ===")
print(f"Reviewed         : {n}")
print(f"Real incident    : {real}")
print(f"Normal operation : {normal}")
print(f"Not sure         : {unsure}")
if real + normal:
    print(f"\nPrecision (excluding 'Not sure'): {real}/{real+normal} = {real/(real+normal)*100:.1f}%")
if n:
    print(f"Precision (counting 'Not sure' as not confirmed): {real}/{n} = {real/n*100:.1f}%")

print("\n=== UNFLAGGED trips (system said: normal) ===")
print(f"Reviewed         : {len(un)}")
print(f"Expert says incident : {count(un, 'Real incident')}")
print(f"Expert says normal   : {count(un, 'Normal operation')}")
print(f"Not sure             : {count(un, 'Not sure')}")
print("(Small sample, includes near-miss trips on purpose. Do not report a recall from this.)")

print("\n=== Confirmed rate by trip group (flagged only) ===")
if n:
    g = fl.groupby("group")["Verdict"].apply(lambda s: f"{(s=='Real incident').sum()}/{len(s)}")
    print(g.to_string())

print("\n=== Agreement with your rule labels (flagged trips) ===")
if n:
    print(pd.crosstab(fl["rule_label"], fl["Verdict"]).to_string())

out = os.path.join(here, "expert_results_scored.csv")
m.to_csv(out, index=False)
print(f"\nFull table saved: {out}")
