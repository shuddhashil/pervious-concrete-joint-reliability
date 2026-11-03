"""Out-of-corpus check of the lab-effect forward model on the author's own two papers (2026-10-26).
Model = the saved full-data lab-effect fit (631 records), untouched. Two ways to predict each of the 41 mixtures, as in the held-out audit of the paper:
  newlab  the lab is unseen (zero offset, no data from it)
  near3   the model is conditioned, hyperparameters fixed, on the 3 OTHER records of the same lab nearest to the predicted record in standardised input space
usage: python -I predict_own.py [--scale closure|density|mould] [--volume 1.0] [--sand-reading coarse|total] [--dgrade renorm|lastbin] [--onelab] [--tag NAME]
writes pred_<tag>.csv and out_<tag>.json next to this file."""
import sys, os, json, math, argparse
KIT = "C:/Users/Shuddhashil/Desktop/Mix Design Paper Rebuild/research/RobustnessTests_20261021/colab_kit"
sys.path.insert(0, KIT); sys.dont_write_bytecode = True
import numpy as np, pandas as pd, torch
import robust_fits as RF
import lab_fits as LF
R, L = RF.R, RF.L
ap = argparse.ArgumentParser()
ap.add_argument("--scale", default="closure"); ap.add_argument("--volume", type=float, default=1.0); ap.add_argument("--sand-reading", default="coarse")
ap.add_argument("--dgrade", default="renorm"); ap.add_argument("--tie", default="mid"); ap.add_argument("--flip-tide", action="store_true"); ap.add_argument("--onelab", action="store_true"); ap.add_argument("--tag", default="")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
CK = torch.load("C:/Users/Shuddhashil/Desktop/Mix Design Paper Rebuild/research/RobustnessTests_20261021/colab_lab/out_lab/fulldata_Q5a3_lab.pt", weights_only=False)
feats, groups, X631, yS, yP, st = RF.frame()      # activates Q5a3 (11 inputs) and loads the 631 training rows
assert list(CK["features"]) == list(L.FEATURES), (CK["features"], L.FEATURES)
ASC = np.log2(np.array([2.36, 4.75, 6.3, 9.5, 12.5, 19.0, 25.0]))


def qs(bins_pct, tie=None):
    """passing sizes q10..q90 (log2 mm) from six bin masses (coarse -> fine), mass uniform in log size within each bin.
    When the cumulative curve is flat at p (an empty bin between two occupied ones) the quantile is a whole interval; the training sheet resolves such exact ties
    inconsistently (left end for some rows, right end for others), so the choice is a parameter: left, right or the midpoint."""
    tie = tie or a.tie
    m = np.asarray(bins_pct, float)
    m = m / m.sum()
    cum = np.r_[0.0, np.cumsum(m[::-1])]          # cumulative passing at the ascending edges 2.36 ... 25 mm
    out = []
    for p in (0.10, 0.25, 0.50, 0.75, 0.90):
        j = int(np.searchsorted(cum, p - 1e-9, side="left"))
        lo = float(ASC[0]) if j == 0 else float(ASC[j - 1] + (p - cum[j - 1]) / (cum[j] - cum[j - 1]) * (ASC[j] - ASC[j - 1]))
        k = j
        while k + 1 < len(cum) and cum[k + 1] <= p + 1e-9: k += 1    # extend over the plateau
        hi = lo if k == j else float(ASC[k])
        out.append(lo if tie == "left" else hi if tie == "right" else 0.5 * (lo + hi))
    return out


# sanity: the helper reproduces the q columns of the training sheet (left or right end at the exact-tie rows)
tr = pd.read_excel(os.path.join(KIT, "data_with_passing_sizes.xlsx"))
bad = 0
for i in range(631):
    y = tr.loc[i, ["q10", "q25", "q50", "q75", "q90"]].astype(float).values
    b = tr.loc[i, R.BINS].astype(float).values
    if not (np.abs(np.array(qs(b, "left")) - y).max() < 1e-6 or np.abs(np.array(qs(b, "right")) - y).max() < 1e-6): bad += 1
print("training rows whose q is neither the left nor the right end of the plateau:", bad)
assert bad == 0


def LNO(hi, lo, a_, b_):
    """overlap in log size of the interval (lo, hi) with (b_, a_)"""
    top, bot = min(hi, a_), max(lo, b_)
    return math.log(top) - math.log(bot) if top > bot else 0.0


def six_bins(lab, r):
    if lab == "TiDE":
        s = 1.0 - r.large_share if a.flip_tide else r.large_share
        fl = LNO(19, 12.5, 19, 9.5) / LNO(19, 9.5, 19, 9.5)
        fs = LNO(9.5, 6.3, 9.5, 4.75) / LNO(9.5, 4.75, 9.5, 4.75)
        return [0, s * fl * 100, s * (1 - fl) * 100, (1 - s) * fs * 100, (1 - s) * (1 - fs) * 100, 0]
    gA, gB, gC, gD = [float(x) for x in r.groups.split()]
    cA = LNO(9.5, 6.3, 9.5, 4.75) / LNO(9.5, 4.75, 9.5, 4.75)
    dfrac = LNO(4.75, 2.36, 4.75, 1.18) / LNO(4.75, 1.18, 4.75, 1.18)
    last = gD * (dfrac if a.dgrade == "renorm" else 1.0)
    return [0, gA, gB, gC * cA, gC * (1 - cA), last]      # renormalised inside qs()


SG = dict(TiDE=2.824, A=2.901, B=2.912, C=2.917, D=2.918)
raw = pd.read_csv(os.path.join(HERE, "own_papers_raw.csv"))
rows = []
for _, r in raw.iterrows():
    lab = r.lab
    bins = six_bins(lab, r)
    p = r.porosity / 100.0
    if lab == "TiDE":
        C, W, CA, S, SP, rho = r.cement_g, r.water_g, r.aggregate_g, 0.0, 0.0, SG["TiDE"]
    else:
        g = np.array([float(x) for x in r.groups.split()]) / 100
        rho = 1.0 / (g / np.array([SG["A"], SG["B"], SG["C"], SG["D"]])).sum()
        C, W = 1.0, 0.27
        CA = 4.0 if a.sand_reading == "coarse" else 3.8
        S, SP = 0.2, 0.0
    if a.scale == "mould" and lab == "TiDE":
        s = 1000.0 / (math.pi / 4 * 11.0 ** 2 * 20.0)             # Table 2 grams are per 110 x 200 mm mould (1900.7 cm3) -> kg per m3
    elif a.scale in ("closure", "mould"):
        vb = C / 3.15 + W + CA / rho + S / 2.65 + SP / 1.07      # absolute volume of one batch (cm3 if g, litres if kg)
        s = (a.volume - p) * 1e6 / vb / 1000.0 if lab == "TiDE" else (a.volume - p) * 1000.0 / vb     # -> kg per m3 per unit batch quantity
    elif lab == "RIENGG":
        s = r.density_g_cm3 * 1000.0 / (C + CA + S)              # cement + aggregate + sand = measured dry density (RIENGG reports it)
    else:                                                        # TiDE reports no density: absolute-volume closure at V = 1 m3
        vb = C / 3.15 + W + CA / rho
        s = (1.0 - p) * 1e6 / vb / 1000.0
    rows.append(dict(lab=lab, run=int(r.run), porosity=r.porosity, wc=W / C, cem=C * s, CA=CA * s, SP=SP * s, sand=S * s, water=W * s,
                     **dict(zip(["q10", "q25", "q50", "q75", "q90"], qs(bins))), cs=r.cs_mpa, k=r.perm_mm_s))
D = pd.DataFrame(rows)
Xraw = D[["porosity", "wc", "cem", "CA", "SP", "sand", "q10", "q25", "q50", "q75", "q90"]].copy()
Xraw.columns = ["porosity", "w/c", "cem (kg/m3)", "CA (kg/m3)", "SP (kg/m3)", "sand (kg/m3)", "q10", "q25", "q50", "q75", "q90"]
Xraw = Xraw[L.FEATURES]
Xm = L._apply_feature_transforms(Xraw).to_numpy(float)
xsc = L.XStandardizer(); xsc.mean, xsc.std = np.asarray(CK["x_mean"]), np.asarray(CK["x_std"])
ysc = L.YStandardizer(); ysc.mean, ysc.std = np.asarray(CK["y_mean"]), np.asarray(CK["y_std"])
Xs = xsc.transform(Xm)
Y = np.stack([D.cs.values, np.log(D.k.values)], 1)
labid = D.lab.map({"TiDE": 0, "RIENGG": 0 if a.onelab else 1}).values.astype(float) + LF.NEW0 + 50
Ztr = np.column_stack([xsc.transform(CK["X_model"]), np.asarray(CK["study_index"], float)])
Ytr = ysc.transform(CK["Y"])
lik = L._build_likelihood()
model = LF.LabLCMGP(LF.T_(Ztr), LF.T_(Ytr), lik, R.CFG).to(device=L.DEVICE, dtype=L.DTYPE)
model.load_state_dict(CK["model"]); lik.load_state_dict(CK["lik"]); model.eval(); lik.eval()
mu, sg = L.predict(model, lik, LF.T_(Ztr[:40]), fast_mode=R.FAST)
m, s = LF.unstd(mu, sg, ysc)
print("self-test, RMSE of CS for 40 training rows at their own study index:", round(float(np.sqrt(np.mean((m[:, 0] - CK["Y"][:40, 0]) ** 2))), 3))


def Z(i):
    return np.column_stack([Xs[i], labid[i]])


mu, sg = L.predict(model, lik, LF.T_(Z(np.arange(len(D)))), fast_mode=R.FAST)
m0, s0 = LF.unstd(mu, sg, ysc)
m3, s3, pick = np.zeros_like(m0), np.zeros_like(s0), []
for i in range(len(D)):
    pool = np.flatnonzero((labid == labid[i]) & (np.arange(len(D)) != i))
    j = pool[np.argsort(((Xs[pool] - Xs[i]) ** 2).sum(1), kind="stable")[:3]]
    pick.append(j.tolist())
    model.set_train_data(inputs=LF.T_(np.vstack([Ztr, Z(j)])), targets=LF.T_(np.vstack([Ytr, ysc.transform(Y[j])])), strict=False)
    mu, sg = L.predict(model, lik, LF.T_(Z(np.array([i]))), fast_mode=R.FAST)
    mm, ss = LF.unstd(mu, sg, ysc)
    m3[i], s3[i] = mm[0], ss[0]


def met(y, p, sd):
    e = p - y
    sst = ((y - y.mean()) ** 2).sum()
    return dict(n=int(len(y)), rmse=float(np.sqrt((e ** 2).mean())), mae=float(np.abs(e).mean()), bias=float(e.mean()), r2=float(1 - (e ** 2).sum() / sst),
                corr=float(np.corrcoef(y, p)[0, 1]), cov90=float((np.abs(e) <= 1.645 * sd).mean()))


out = dict(args=vars(a), n=int(len(D)), cement_per_m3=[float(D.cem.min()), float(D.cem.median()), float(D.cem.max())], CA_per_m3=[float(D.CA.min()), float(D.CA.median()), float(D.CA.max())])
for name, M, S_ in (("newlab", m0, s0), ("near3", m3, s3)):
    out[name] = {}
    for grp, mask in (("all", np.ones(len(D), bool)), ("TiDE", D.lab.values == "TiDE"), ("RIENGG", D.lab.values == "RIENGG")):
        out[name][grp] = dict(CS=met(Y[mask, 0], M[mask, 0], S_[mask, 0]), lnk=met(Y[mask, 1], M[mask, 1], S_[mask, 1]))
D["cs_pred_newlab"], D["lnk_pred_newlab"] = m0[:, 0], m0[:, 1]
D["cs_pred_near3"], D["lnk_pred_near3"], D["sd_cs_near3"], D["sd_lnk_near3"] = m3[:, 0], m3[:, 1], s3[:, 0], s3[:, 1]
D["near3_runs"] = [",".join(str(int(D.run[j])) for j in p) for p in pick]
tag = a.tag or f"{a.scale}_V{a.volume}_{a.sand_reading}_{a.dgrade}_{a.tie}{'_onelab' if a.onelab else ''}{'_fliptide' if a.flip_tide else ''}"
D.to_csv(os.path.join(HERE, f"pred_{tag}.csv"), index=False)
json.dump(out, open(os.path.join(HERE, f"out_{tag}.json"), "w"), indent=1)
print(tag)
print(json.dumps({k: out[k]["all"] for k in ("newlab", "near3")}, indent=1))
print("cement/m3 (min, median, max)", [round(x) for x in out["cement_per_m3"]], "CA/m3", [round(x) for x in out["CA_per_m3"]])
