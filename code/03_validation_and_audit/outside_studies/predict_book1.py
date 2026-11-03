"""Out-of-corpus check of the saved lab-effect model on Book1 (Mahalingam 2016, Sonebi 2013, Meininger 1988). Same two modes as predict_own.py:
newlab = unseen lab, no data from it; near3 = conditioned (hyperparameters fixed) on the 3 other mixtures of the same source nearest in standardised input space.
usage: python -I predict_book1.py  (reads book1_inputs.csv; writes pred_book1.csv, out_book1.json)"""
import sys, os, json
KIT = "C:/Users/Shuddhashil/Desktop/Mix Design Paper Rebuild/research/RobustnessTests_20261021/colab_kit"
sys.path.insert(0, KIT); sys.dont_write_bytecode = True
import numpy as np, pandas as pd, torch
import robust_fits as RF
import lab_fits as LF
R, L = RF.R, RF.L
HERE = os.path.dirname(os.path.abspath(__file__))
CK = torch.load("C:/Users/Shuddhashil/Desktop/Mix Design Paper Rebuild/research/RobustnessTests_20261021/colab_lab/out_lab/fulldata_Q5a3_lab.pt", weights_only=False)
RF.frame()
assert list(CK["features"]) == list(L.FEATURES)
D = pd.read_csv(os.path.join(HERE, "book1_inputs.csv"))
Xraw = D[["porosity", "wc", "cem", "CA", "SP", "sand", "q10", "q25", "q50", "q75", "q90"]].copy()
Xraw.columns = ["porosity", "w/c", "cem (kg/m3)", "CA (kg/m3)", "SP (kg/m3)", "sand (kg/m3)", "q10", "q25", "q50", "q75", "q90"]
Xm = L._apply_feature_transforms(Xraw[L.FEATURES]).to_numpy(float)
xsc = L.XStandardizer(); xsc.mean, xsc.std = np.asarray(CK["x_mean"]), np.asarray(CK["x_std"])
ysc = L.YStandardizer(); ysc.mean, ysc.std = np.asarray(CK["y_mean"]), np.asarray(CK["y_std"])
Xs = xsc.transform(Xm)
Y = np.stack([D.cs.values, np.log(D.k.values)], 1)
labid = D.lab.map({"Mahalingam": 0, "Sonebi": 1, "Meininger": 2}).values.astype(float) + LF.NEW0 + 50
Ztr = np.column_stack([xsc.transform(CK["X_model"]), np.asarray(CK["study_index"], float)])
Ytr = ysc.transform(CK["Y"])
lik = L._build_likelihood()
model = LF.LabLCMGP(LF.T_(Ztr), LF.T_(Ytr), lik, R.CFG).to(device=L.DEVICE, dtype=L.DTYPE)
model.load_state_dict(CK["model"]); lik.load_state_dict(CK["lik"]); model.eval(); lik.eval()
mu, sg = L.predict(model, lik, LF.T_(Ztr[:40]), fast_mode=R.FAST)
m, s = LF.unstd(mu, sg, ysc)
print("self-test, RMSE of CS for 40 training rows:", round(float(np.sqrt(np.mean((m[:, 0] - CK["Y"][:40, 0]) ** 2))), 3))


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


out = dict(n=int(len(D)))
for name, M, S_ in (("newlab", m0, s0), ("near3", m3, s3)):
    out[name] = {}
    for grp in ("all", "Mahalingam", "Sonebi", "Meininger"):
        mask = np.ones(len(D), bool) if grp == "all" else D.lab.values == grp
        out[name][grp] = dict(CS=met(Y[mask, 0], M[mask, 0], S_[mask, 0]), lnk=met(Y[mask, 1], M[mask, 1], S_[mask, 1]))
D["cs_pred_newlab"], D["lnk_pred_newlab"] = m0[:, 0], m0[:, 1]
D["cs_pred_near3"], D["lnk_pred_near3"], D["sd_cs_near3"], D["sd_lnk_near3"] = m3[:, 0], m3[:, 1], s3[:, 0], s3[:, 1]
D["near3_ids"] = [";".join(str(D.id[j]) for j in p) for p in pick]
D.to_csv(os.path.join(HERE, "pred_book1.csv"), index=False)
json.dump(out, open(os.path.join(HERE, "out_book1.json"), "w"), indent=1)
for mode in ("newlab", "near3"):
    print("\n==", mode)
    for grp, v in out[mode].items():
        print(f"{grp:11s} n={v['CS']['n']:2d} | CS rmse {v['CS']['rmse']:.2f} mae {v['CS']['mae']:.2f} bias {v['CS']['bias']:+.2f} R2 {v['CS']['r2']:.2f} r {v['CS']['corr']:.2f} cov90 {v['CS']['cov90']:.2f}"
              f" | lnk rmse {v['lnk']['rmse']:.2f} mae {v['lnk']['mae']:.2f} bias {v['lnk']['bias']:+.2f} R2 {v['lnk']['r2']:.2f} r {v['lnk']['corr']:.2f} cov90 {v['lnk']['cov90']:.2f}")
