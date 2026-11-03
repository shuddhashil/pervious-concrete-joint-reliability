"""Step 0: CA mass-field semantics audit (36 studies). Read-only; no repairs."""
import numpy as np, pandas as pd, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
RUN = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\research\Stage1_CA_Forward_Refit_20260913\run_01"
TAB = os.path.join(RUN, "tables"); os.makedirs(TAB, exist_ok=True)
COH = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\data\reconciled_phase2v\phase1_reconciled_eligible.csv"
df = pd.read_csv(COH)
for c in ["CA (kg/m3)", "sand (kg/m3)", "cem (kg/m3)", "CA/Cem"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

rows = []
for st, d in df.groupby("_src_ffill"):
    ca, sand, cem = d["CA (kg/m3)"], d["sand (kg/m3)"], d["cem (kg/m3)"]
    tot = ca + sand
    # check: does CA+Sand stay constant while sand varies (would hint CA=complement of total)? corr(ca, sand)
    sand_varies = sand.std(ddof=0) > 1e-9
    corr = float(ca.corr(sand)) if (sand_varies and ca.std(ddof=0) > 1e-9) else np.nan
    tot_const = bool(tot.std(ddof=0) < 1e-6) if len(d) > 1 else True
    flags, cls = [], "CONSISTENT_BUT_NOT_SOURCE_VERIFIED"
    if (ca <= 0).any():
        flags.append("CA<=0 present"); cls = "CONTRADICTORY"
    if len(d) > 1 and ca.nunique() == 1 and (sand > 0).any():
        flags.append("single CA value with sand>0 present (design constant; source label accepted, mass-field alone cannot separate coarse-vs-total)")
    if sand_varies and tot_const and len(d) > 2:
        flags.append("CA+sand constant while sand varies: CA behaves as complement of a fixed total"); cls = "AMBIGUOUS"
    if np.isfinite(corr) and corr < -0.95 and len(d) > 3:
        flags.append(f"CA-sand corr {corr:.3f}: near-perfect trade-off, ambiguous total-vs-coarse split")
        cls = "AMBIGUOUS"
    if ca.min() < 800 or ca.max() > 2100:
        flags.append("CA outside 800-2100 plausibility band")
        if cls == "CONSISTENT_BUT_NOT_SOURCE_VERIFIED": cls = "AMBIGUOUS"
    rows.append(dict(study=st, n=len(d), ca_min=round(float(ca.min()),2), ca_max=round(float(ca.max()),2),
                     ca_mean=round(float(ca.mean()),1), n_unique_ca=int(ca.nunique()),
                     frac_sand_pos=round(float((sand>0).mean()),3), ca_sand_corr=round(corr,3) if np.isfinite(corr) else "",
                     total_const=tot_const, classification=cls, flags=" | ".join(flags)))
out = pd.DataFrame(rows).sort_values("study")
out.to_csv(os.path.join(TAB, "CA_SEMANTICS_AUDIT.csv"), index=False)
print(out[["study","n","n_unique_ca","frac_sand_pos","classification"]].to_string())
print("class counts:", out["classification"].value_counts().to_dict())
# manuscript spot check
man = open(r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\source\manuscript.txt", encoding="utf-8", errors="replace").read()
import re
for m in ["1522", "1404"]:
    i = man.find(m)
    print(f"--- manuscript '{m}':", man[max(0,i-200):i+200].replace("\n"," ")[:400] if i>=0 else "NOT FOUND")
print("STEP0 DONE")
