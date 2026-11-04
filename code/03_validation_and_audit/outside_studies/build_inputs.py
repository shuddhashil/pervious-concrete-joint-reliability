"""Book1.xlsx (Mahalingam 2016, Sonebi & Bassuoni 2013, Meininger 1988) -> model inputs. No prediction is made here.
Rule fixed before any prediction: every source's stated aggregate range is spread uniformly in log size, exactly as for the two own papers.
usage: python -I build_inputs.py   -> book1_inputs.csv"""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
KIT = os.path.join(REPO, "code", "02_forward_model")
DATA = os.path.join(REPO, "data", "outside_studies")
sys.path.insert(0, KIT); sys.dont_write_bytecode = True
import numpy as np, pandas as pd, openpyxl, torch
import robust_fits as RF
R, L = RF.R, RF.L
CK = torch.load(os.path.join(REPO, "results", "forward_model_folds", "lab_effect", "fulldata_Q5a3_lab.pt"), weights_only=False)
ASC = np.log2(np.array([2.36, 4.75, 6.3, 9.5, 12.5, 19.0, 25.0]))
EDGES = [2.36, 4.75, 6.3, 9.5, 12.5, 19.0, 25.0]     # six bins, coarse -> fine: 19-25, 12.5-19, 9.5-12.5, 6.3-9.5, 4.75-6.3, 2.36-4.75


def qs(bins_pct, tie="mid"):
    """same helper as analysis/own_papers_validation_2026-10-26/predict_own.py (checked there against the 631 training rows)"""
    m = np.asarray(bins_pct, float); m = m / m.sum()
    cum = np.r_[0.0, np.cumsum(m[::-1])]
    out = []
    for p in (0.10, 0.25, 0.50, 0.75, 0.90):
        j = int(np.searchsorted(cum, p - 1e-9, side="left"))
        lo = float(ASC[0]) if j == 0 else float(ASC[j - 1] + (p - cum[j - 1]) / (cum[j] - cum[j - 1]) * (ASC[j] - ASC[j - 1]))
        k = j
        while k + 1 < len(cum) and cum[k + 1] <= p + 1e-9: k += 1
        hi = lo if k == j else float(ASC[k])
        out.append(lo if tie == "left" else hi if tie == "right" else 0.5 * (lo + hi))
    return out


def uniform_bins(lo, hi):
    """percent of the material in each of the six bins, mass uniform in log size between lo and hi (mm); material outside 2.36-25 is not allowed"""
    tot = math.log(hi) - math.log(lo); b = []
    for k in range(5, -1, -1):
        a, c = EDGES[k], EDGES[k + 1]
        top, bot = min(hi, c), max(lo, a)
        b.append(100 * (math.log(top) - math.log(bot)) / tot if top > bot else 0.0)
    return b


RANGE = {"Mahalingam": {"S1": (4.75, 9.5), "S2": (9.5, 12.5)}, "Sonebi": (10.0, 20.0), "Meininger": (2.36, 9.5)}

ws = openpyxl.load_workbook(os.path.join(DATA, "Book1.xlsx"), data_only=True)["Sheet1"]
rows, lab = [], None
for r in range(4, 40):
    idv, agg = ws.cell(r, 3).value, ws.cell(r, 4).value
    if idv is None: continue
    if ws.cell(r, 1).value: lab = ws.cell(r, 1).value.split()[-1]     # Mahalingam / Sonebi / Meininger
    cem, ca, wat, sand, sp, por, perm, cs = [float(ws.cell(r, c).value) for c in (6, 7, 8, 9, 10, 11, 12, 13)]
    if lab == "Mahalingam":
        lo, hi = RANGE[lab]["S1" if "S1" in str(idv) else "S2"]
    else:
        lo, hi = RANGE[lab]
    b = uniform_bins(lo, hi)
    rows.append(dict(lab=lab, id=idv, porosity=por, wc=wat / cem, cem=cem, CA=ca, SP=sp, sand=sand, water=wat,
                     bins=" ".join(f"{x:.2f}" for x in b), **dict(zip(["q10", "q25", "q50", "q75", "q90"], qs(b))),
                     cs=cs, k=perm))
D = pd.DataFrame(rows)

RF.frame()      # activates the 11-input feature set
assert list(CK["features"]) == list(L.FEATURES)
# domain check against the 631 training rows (raw units)
feats = list(CK["features"]); Xt = np.asarray(CK["X_model"])
X = D[["porosity", "wc", "cem", "CA", "SP", "sand", "q10", "q25", "q50", "q75", "q90"]].copy()
X.columns = ["porosity", "w/c", "cem (kg/m3)", "CA (kg/m3)", "SP (kg/m3)", "sand (kg/m3)", "q10", "q25", "q50", "q75", "q90"]
Xm = L._apply_feature_transforms(X[L.FEATURES]).to_numpy(float)
lo_t, hi_t = Xt.min(0), Xt.max(0)
D["outside_training_range"] = [", ".join(f for f, v, a, b in zip(feats, row, lo_t, hi_t) if v < a - 1e-9 or v > b + 1e-9) for row in Xm]
D["cs_ok_range"] = (D.cs >= 1.06) & (D.cs <= 68.8)
D.to_csv(os.path.join(DATA, "book1_inputs.csv"), index=False)
pd.set_option("display.width", 250, "display.max_columns", 40)
print(D[["lab", "id", "porosity", "wc", "cem", "CA", "SP", "q10", "q50", "q90", "cs", "k", "outside_training_range"]].round(3).to_string())
print(D.groupby("lab").size().to_dict())
