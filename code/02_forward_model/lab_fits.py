"""Lab-effect forward model, feasibility fits. Sandbox; deployed model untouched.
Model = the deployed A3-1 LCM-GP (same 11 inputs, kernels, priors, bounds, likelihood, Adam lr 0.02, 700 iters x 2 restarts, seeds) PLUS one covariance term
for the laboratory: k_lab = B (x) 1[same study], B a free 2x2 task covariance (rank 2), so every study has its own offset in CS and ln k, correlated across the
two outputs, with between-study spread learned from the data. The study index rides as a 12th, unstandardised input column that only k_lab reads.
A study absent from training (a new laboratory) gets an unseen index: zero offset in the mean, B added to the predictive variance.
  labrand K   the paper's random split K (test studies are known to the model)
  labgrp K    study-grouped fold K: (a) held-out studies as new labs; (b) per held-out record, the model conditioned on M=3 OTHER records of its study
              (trial batches; same draw rule and seed as variant A), hyperparameters NOT refitted
  labfull     full-data fit (631)
usage: python lab_fits.py all | labrand K | labgrp K | labfull | summary"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.dont_write_bytecode = True
import numpy as np, torch, gpytorch
import robust_fits as RF
R, L = RF.R, RF.L
OUT = os.path.join(HERE, "out_lab"); os.makedirs(OUT, exist_ok=True)
LAB = 11; M_TRIAL = 3; NEW0 = 1000


class SameLab(gpytorch.kernels.Kernel):
    is_stationary = False

    def forward(self, x1, x2, diag=False, **params):
        a, b = x1[..., LAB], x2[..., LAB]
        if diag: return (a == b).to(x1.dtype)
        return (a.unsqueeze(-1) == b.unsqueeze(-2)).to(x1.dtype)


Base = L.LCMGPModel


class LabLCMGP(Base):
    def __init__(self, train_x, train_y, likelihood, cfg):
        super().__init__(train_x, train_y, likelihood, cfg)
        self.lab_covar_module = gpytorch.kernels.MultitaskKernel(SameLab(), num_tasks=L.NUM_TASKS, rank=2)
        with torch.no_grad():
            self.lab_covar_module.task_covar_module.covar_factor.fill_(0.3); self.lab_covar_module.task_covar_module.raw_var.fill_(-3.0)

    def forward(self, x):
        return gpytorch.distributions.MultitaskMultivariateNormal(self.mean_module(x), self.shared_covar_module(x) + self.lab_covar_module(x))


L.LCMGPModel = LabLCMGP
T_ = RF.T_


def lab_B(model, ysd):
    B = model.lab_covar_module.task_covar_module.covar_matrix.to_dense().detach().double().cpu().numpy()
    sd = np.sqrt(np.diag(B)) * ysd
    return dict(B_std=B.tolist(), sd_CS_mpa=float(sd[0]), sd_lnk=float(sd[1]), corr=float(B[0, 1] / np.sqrt(B[0, 0] * B[1, 1])))


def fit(tr_X, tr_Y, seed_k, label, iters=None):
    L.set_seed(20260913 + seed_k)
    return L.train_lcmgp(T_(tr_X), T_(tr_Y), R.CFG, iters or R.ITERS, R.LR, R.RESTARTS, 12345 + seed_k, fast_mode=R.FAST, verbosity=0, label=label)


def unstd(mu_s, sig_s, ysc):
    return mu_s * ysc.std + ysc.mean, L.apply_sigma_floors(sig_s * ysc.std)


def metrics(y, m, s):
    return {'rmse_strength': L.rmse(y[:, 0], m[:, 0]), 'r2_strength': L.r2_score(y[:, 0], m[:, 0]), 'rmse_logperm': L.rmse(y[:, 1], m[:, 1]), 'r2_logperm': L.r2_score(y[:, 1], m[:, 1]),
            'cov90_strength_raw_gaussian': L.coverage(y[:, 0], m[:, 0], s[:, 0], 1.6449), 'cov90_logperm_raw_gaussian': L.coverage(y[:, 1], m[:, 1], s[:, 1], 1.6449)}


def record(job, y, m, s, te, extra):
    s = np.maximum(s, 1e-12); mm = metrics(y, m, s)
    json.dump(dict(job=job, config="Q5a3+lab", features=list(L.FEATURES), n_test=len(te), metrics={k: float(v) for k, v in mm.items()},
                   y_true_CS=y[:, 0].tolist(), y_pred_CS=m[:, 0].tolist(), y_std_raw_CS=s[:, 0].tolist(), y_true_logQ=y[:, 1].tolist(), y_pred_logQ=m[:, 1].tolist(),
                   y_std_raw_logQ=s[:, 1].tolist(), original_row_index=np.asarray(te).tolist(), **extra), open(os.path.join(OUT, f"{job}.json"), "w"))
    print(f"DONE {job} | CS RMSE {mm['rmse_strength']:.3f} R2 {mm['r2_strength']:.3f} cov90 {mm['cov90_strength_raw_gaussian']:.3f} | lnk RMSE {mm['rmse_logperm']:.3f} R2 {mm['r2_logperm']:.3f} cov90 {mm['cov90_logperm_raw_gaussian']:.3f}", flush=True)


def prep(X, Y, sid, tr):
    xsc = L.XStandardizer().fit(X[tr]); ysc = L.YStandardizer().fit(Y[tr])
    Z = lambda idx, ids: np.column_stack([xsc.transform(X[idx]), ids]); return xsc, ysc, Z


def run(job, k=None):
    feats, groups, X, yS, yP, st = RF.frame(); Y = np.stack([yS, yP], 1)
    names = sorted(set(st)); sid = np.array([names.index(s) for s in st], float)
    if job == "labrand":
        if os.path.exists(os.path.join(OUT, f"labrand_split{k}.json")): return
        sp = json.load(open(R.SPLITS))[f"split_{k}"]; tr, te = np.asarray(sp["train_idx"]), np.asarray(sp["test_idx"]); t0 = time.time()
        xsc, ysc, Z = prep(X, Y, sid, tr); model, lik = fit(Z(tr, sid[tr]), ysc.transform(Y[tr]), k, f"labrand{k}")
        mu, sg = L.predict(model, lik, T_(Z(te, sid[te])), fast_mode=R.FAST); m, s = unstd(mu, sg, ysc)
        record(f"labrand_split{k}", Y[te], m, s, te, dict(minutes=(time.time() - t0) / 60, lab=lab_B(model, ysc.std), hyper=R.hyper(model, lik)))
    elif job == "labgrp":
        if os.path.exists(os.path.join(OUT, f"labgrp_trial{M_TRIAL}_split{k}.json")): return
        sp = RF.grouped_splits(st)[f"split_{k}"]; tr, te = np.asarray(sp["train_idx"]), np.asarray(sp["test_idx"]); t0 = time.time()
        xsc, ysc, Z = prep(X, Y, sid, tr); new = np.array([NEW0 + names.index(s) for s in st[te]], float)
        model, lik = fit(Z(tr, sid[tr]), ysc.transform(Y[tr]), k, f"labgrp{k}")
        mu, sg = L.predict(model, lik, T_(Z(te, new)), fast_mode=R.FAST); m, s = unstd(mu, sg, ysc)
        meta = dict(minutes=(time.time() - t0) / 60, lab=lab_B(model, ysc.std), studies=sp["studies"], hyper=R.hyper(model, lik))
        record(f"labgrp_newlab_split{k}", Y[te], m, s, te, meta)
        # trial batches: same draw rule/seed as build_variant.py mode A (one rng over folds 0..4 in order; replay earlier folds' draws for alignment)
        rng = np.random.default_rng(20261021); allsp = RF.grouped_splits(st)
        for kk in range(k + 1):
            te_k = np.asarray(allsp[f"split_{kk}"]["test_idx"]); st_k = st[te_k]; picks = []
            for i in range(len(te_k)):
                pool = np.flatnonzero((st_k == st_k[i]) & (np.arange(len(te_k)) != i)); picks.append(rng.choice(pool, size=min(M_TRIAL, len(pool)), replace=False))
        mt, stt = np.zeros((len(te), 2)), np.zeros((len(te), 2)); Xtr0, Ytr0 = Z(tr, sid[tr]), ysc.transform(Y[tr])
        for i in range(len(te)):
            aug = te[picks[i]]; Xa = np.vstack([Xtr0, Z(aug, new[picks[i]])]); Ya = np.vstack([Ytr0, ysc.transform(Y[aug])])
            model.set_train_data(inputs=T_(Xa), targets=T_(Ya), strict=False)
            mu, sg = L.predict(model, lik, T_(Z(te[i:i + 1], new[i:i + 1])), fast_mode=R.FAST); mt[i], stt[i] = unstd(mu, sg, ysc)[0][0], unstd(mu, sg, ysc)[1][0]
        model.set_train_data(inputs=T_(Xtr0), targets=T_(Ytr0), strict=False)
        record(f"labgrp_trial{M_TRIAL}_split{k}", Y[te], mt, stt, te, dict(meta, trial_rows=[te[p].tolist() for p in picks]))
    elif job == "labfull":
        p = os.path.join(OUT, "fulldata_Q5a3_lab.pt")
        if os.path.exists(p): return
        tr = np.arange(631); xsc, ysc, Z = prep(X, Y, sid, tr); t0 = time.time(); L.set_seed(20260913)
        model, lik = L.train_lcmgp(T_(Z(tr, sid)), T_(ysc.transform(Y)), R.CFG, 900, 0.02, 2, 12345 + 999999, fast_mode=True, verbosity=0, label="full-lab")
        torch.save({'model': {a: v.detach().cpu() for a, v in model.state_dict().items()}, 'lik': {a: v.detach().cpu() for a, v in lik.state_dict().items()},
                    'features': feats, 'groups': groups, 'transform_recipe': L.TRANSFORM_RECIPE, 'cfg': R.CFG, 'x_mean': np.asarray(xsc.mean), 'x_std': np.asarray(xsc.std),
                    'y_mean': np.asarray(ysc.mean), 'y_std': np.asarray(ysc.std), 'X_model': X, 'Y': Y, 'study_index': sid, 'study_names': names, 'lab': lab_B(model, ysc.std)}, p)
        json.dump(dict(lab=lab_B(model, ysc.std), hyper=R.hyper(model, lik)), open(os.path.join(OUT, "fulldata_Q5a3_lab_hyper.json"), "w"), indent=1)
        print(f"DONE labfull {(time.time()-t0)/60:.1f} min | lab spread {lab_B(model, ysc.std)}", flush=True)


def summary():
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".json") and "hyper" not in f:
            d = json.load(open(os.path.join(OUT, f))); m = d["metrics"]
            print(f"{f:30s} n {d['n_test']:4d} | CS RMSE {m['rmse_strength']:.3f} R2 {m['r2_strength']:.3f} cov90 {m['cov90_strength_raw_gaussian']:.3f} | lnk RMSE {m['rmse_logperm']:.3f} cov90 {m['cov90_logperm_raw_gaussian']:.3f} | lab {d.get('lab', {}).get('sd_CS_mpa', float('nan')):.2f} MPa")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "all":
        for k in range(5): run("labgrp", k)
        for k in range(5): run("labrand", k)
        run("labfull"); summary()
    elif a[0] == "summary": summary()
    else: run(a[0], int(a[1]) if len(a) > 1 else None)
