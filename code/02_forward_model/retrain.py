"""LCM-GP retrain. Production engine/model untouched.
Follows the authoritative refit (research\\Stage1_CA_Forward_Refit_20260913\\run_01\\src\\refit_lcmgp.py and step5_production.py):
same module, same 5 fixed 80/20 splits, kernel rbf+rq + grouped RBF, rank 1, Adam lr 0.02, 700 iters x 2 restarts per split,
900 iters x 2 restarts for the full-data fit, same seeds. The ONLY change per config is the gradation-input contract.

Usage: python retrain.py --config E|Q5|Q5n|M4 --split 0..4      (one split fit, checkpointed)
       python retrain.py --config E|Q5|Q5n|M4 --full            (full-data fit)
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import LCMGP_without_head as L

HERE = os.path.dirname(os.path.abspath(__file__))
SF = MOD = HERE
XLSX0 = None
XLSX = os.path.join(HERE, "data_with_passing_sizes.xlsx")
SPLITS = os.path.join(HERE, "splits_A3-1_remapped.json")  # A3-1: original fixed splits with the 19 removed rows dropped and indices remapped
BINS = ['R25-19mm', 'R19-12.5mm', 'R12.5-9.5mm', 'R9.5-6.3mm', 'R6.3-4.75mm', 'R4.75-2.36mm']
ASC = np.log2(np.array([2.36, 4.75, 6.3, 9.5, 12.5, 19.0, 25.0]))
RECIPE = ['porosity', 'w/c', 'cem (kg/m3)', 'CA (kg/m3)', 'SP (kg/m3)', 'sand (kg/m3)']
Q = ['q10', 'q25', 'q50', 'q75', 'q90']
CONFIGS = {
    # name: (gradation inputs, log cols, log1p cols, raw cols, keep the gradation-only kernel group?)
    "E":   (['WAS', 'I_gap', 'Cu', 'Aag(mm^-1)'], ['Cu'], ['Aag(mm^-1)'], ['WAS', 'I_gap'], True),      # deployed M1global contract
    "Q5":  (Q, [], [], Q, True),                                                                        # five passing sizes
    "Q5n": (Q, [], [], Q, False),                                                                       # five passing sizes, no gradation-only group
    "M4":  (['WAS', 'Aag(mm^-1)', 'q50', 'iqr'], [], ['Aag(mm^-1)'], ['WAS', 'q50', 'iqr'], True),      # WAS, Aag, median size, spread
    "Q5a3": (Q, [], [], Q, False),                                                                      # model A3: recipe-only global RBF; all-input tied RQ
    "Q5t": (Q, [], [], Q, False),                                                                       # model A2: as Q5n, ONE shared lengthscale for the five passing sizes
}
TIED = {"Q5t", "Q5a3"}   # configs whose global kernels share one lengthscale across the gradation inputs


def _tied_kernels(rbf_recipe_only=False):
    """Model A2: global RBF and RQ kernels with ARD over the recipe inputs but ONE shared lengthscale for the five passing sizes.
    Same bounds (0.2-4), prior (LogNormal(0, 0.45) on each distinct lengthscale), initial values and RQ alpha settings as the original builder."""
    import gpytorch
    from gpytorch.constraints import Interval
    from gpytorch.priors import LogNormalPrior, GammaPrior
    n_rec = len(RECIPE); n_all = len(L.FEATURES)
    idx = list(range(n_rec)) + [n_rec] * (n_all - n_rec)          # map 11 inputs -> 7 distinct lengthscales

    def tie(cls):
        class Tied(cls):
            def __init__(self, **kw):
                super().__init__(ard_num_dims=n_rec + 1, **kw)
                self.register_buffer("tie_idx", torch.tensor(idx, dtype=torch.long))
                self.ard_num_dims = n_all            # inputs are 11-dimensional; the 7 distinct lengthscales are expanded to 11 below

            @property
            def lengthscale(self):
                return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)[..., self.tie_idx]

            @lengthscale.setter
            def lengthscale(self, value):
                self._set_lengthscale(value)
        Tied.__name__ = "Tied" + cls.__name__
        return Tied

    def reg(k):
        k.register_constraint("raw_lengthscale", Interval(0.2, 4.0))
        k.register_prior("lengthscale_prior", LogNormalPrior(0.0, 0.45), lambda m: m.raw_lengthscale_constraint.transform(m.raw_lengthscale))
        with torch.no_grad():
            k.initialize(raw_lengthscale=k.raw_lengthscale_constraint.inverse_transform(torch.ones_like(k.raw_lengthscale)))
        return k

    def build(kernel_name, use_additive, add_kernel_name):
        cont = list(L.CONT_IDXS)
        k1 = (reg(gpytorch.kernels.RBFKernel(ard_num_dims=n_rec, active_dims=list(range(n_rec))))
              if rbf_recipe_only else reg(tie(gpytorch.kernels.RBFKernel)(active_dims=cont)))
        k2 = reg(tie(gpytorch.kernels.RQKernel)(active_dims=cont))
        k2.register_constraint("raw_alpha", Interval(0.25, 10.0)); k2.register_prior("alpha_prior", GammaPrior(2.0, 1.0), "alpha")
        with torch.no_grad(): k2.alpha = 2.0
        ks = k1 + k2
        if use_additive:
            for g in L._additive_groups().values():
                kg = gpytorch.kernels.RBFKernel(ard_num_dims=len(g), active_dims=g); L._reg_ls(kg); ks = ks + kg
        return ks
    return build
CFG = {'kernel': 'rbf+rq', 'rank': 1, 'use_additive': True, 'add_kernel': 'rbf'}
ITERS, LR, RESTARTS, FAST = 700, 0.02, 2, True


def quant(G, p):
    """log2 size at passing fraction p; mass uniform in log size within each bin (same rule as the model-free battery)"""
    f = G[:, ::-1]; F = np.concatenate([np.zeros((len(G), 1)), np.cumsum(f, 1)], 1); F[:, -1] = 1.0
    out = np.empty(len(G))
    for r in range(len(G)):
        k = int(np.searchsorted(F[r], p, side="left")); k = min(max(k, 1), 6)
        while k < 6 and F[r, k] - F[r, k - 1] <= 0: k += 1
        lo, hi = F[r, k - 1], F[r, k]
        out[r] = ASC[k - 1] + (p - lo) / max(hi - lo, 1e-12) * (ASC[k] - ASC[k - 1])
    return out


def ensure_data():
    if os.path.exists(XLSX): return
    df = pd.read_excel(XLSX0)
    G = df[BINS].apply(pd.to_numeric, errors="coerce").to_numpy(float); s = G.sum(1, keepdims=True)
    ok = np.isfinite(G).all(1) & (s[:, 0] > 0); Gn = np.where(ok[:, None], G / np.where(s > 0, s, 1), np.nan)
    for p in (10, 25, 50, 75, 90):
        col = np.full(len(df), np.nan); col[ok] = quant(Gn[ok], p / 100); df[f"q{p}"] = col
    df["iqr"] = df["q75"] - df["q25"]
    df.to_excel(XLSX, index=False)


def activate(config):
    grad, logc, log1pc, rawc, keep_group = CONFIGS[config]
    feats = RECIPE + grad
    L.FEATURES = list(feats); L.CONT_IDXS = list(range(len(feats)))
    L.TRANSFORM_RECIPE = {'log_cols': ['cem (kg/m3)', 'CA (kg/m3)'] + logc, 'log1p_cols': ['SP (kg/m3)', 'sand (kg/m3)'] + log1pc,
                          'raw_cols': ['porosity', 'w/c'] + rawc, 'categorical_cols': []}
    ix = lambda f: feats.index(f)
    groups = {'Binder': [ix('w/c'), ix('cem (kg/m3)'), ix('SP (kg/m3)')], 'Filler': [ix('porosity'), ix('sand (kg/m3)')]}
    if keep_group: groups['Aggregate'] = [ix(g) for g in grad]
    L._additive_groups = lambda: groups
    if config in TIED:
        assert RT_CFG_OK(), "unexpected kernel config"
        L._make_cont_kernel = _tied_kernels(rbf_recipe_only=(config == "Q5a3"))
    return feats, groups


def RT_CFG_OK():
    return CFG == {'kernel': 'rbf+rq', 'rank': 1, 'use_additive': True, 'add_kernel': 'rbf'}


def hyper(model, lik):
    sd = {k: v.detach().cpu().to(torch.float64).numpy() for k, v in {**model.state_dict(), **lik.state_dict()}.items()}
    ls = {}
    for n_, m_ in model.named_modules():
        if hasattr(m_, "lengthscale") and getattr(m_, "raw_lengthscale", None) is not None:
            ad = m_.active_dims.tolist() if m_.active_dims is not None else list(range(len(L.FEATURES)))
            ls[n_] = {"dims": [L.FEATURES[i] for i in ad], "lengthscale": np.round(m_.lengthscale.detach().cpu().numpy().ravel(), 4).tolist()}
    raw = {k: np.asarray(v).ravel().tolist() for k, v in sd.items() if any(s in k.lower() for s in ['alpha', 'noise', 'outputscale', 'task', 'mean'])}
    return {"lengthscales": ls, "other": raw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS)); ap.add_argument("--split", type=int); ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    ensure_data(); feats, groups = activate(a.config)
    X, yS, yP, n0 = L.load_and_prepare_data(XLSX)
    assert X.shape == (631, len(feats)), X.shape
    T_ = lambda z: torch.as_tensor(z, dtype=L.DTYPE, device=L.DEVICE).contiguous()
    t0 = time.time()
    if a.full:
        out_pt = os.path.join(MOD, f"fulldata_{a.config}.pt")
        if os.path.exists(out_pt): print("exists", out_pt); return
        L.set_seed(20260913)
        xsc = L.XStandardizer().fit(X); Yall = np.stack([yS, yP], 1); ysc = L.YStandardizer().fit(Yall)
        model, lik = L.train_lcmgp(T_(xsc.transform(X)), T_(ysc.transform(Yall)), CFG, 900, 0.02, 2, 12345 + 999999, fast_mode=True, verbosity=0, label=f"full-{a.config}")
        torch.save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'lik': {k: v.detach().cpu() for k, v in lik.state_dict().items()},
                    'features': feats, 'groups': groups, 'transform_recipe': L.TRANSFORM_RECIPE, 'cfg': CFG,
                    'x_mean': np.asarray(xsc.mean), 'x_std': np.asarray(xsc.std), 'y_mean': np.asarray(ysc.mean), 'y_std': np.asarray(ysc.std),
                    'X_model': X, 'Y': Yall}, out_pt)
        json.dump(hyper(model, lik), open(os.path.join(MOD, f"fulldata_{a.config}_hyper.json"), "w"), indent=1)
        print(f"DONE {a.config} full-data fit in {(time.time() - t0) / 60:.1f} min", flush=True); return
    p = os.path.join(SF, f"{a.config}_split{a.split}.json")
    if os.path.exists(p): print("exists", p); return
    L.set_seed(20260913 + a.split)
    sp = json.load(open(SPLITS))[f"split_{a.split}"]
    tr = np.asarray(sp["train_idx"], dtype=int); te = np.asarray(sp["test_idx"], dtype=int)
    rec = L._prepare_split(X, yS, yP, tr, te)
    model, lik = L.train_lcmgp(T_(rec['Xtr_s']), T_(rec['Ytr_s']), CFG, ITERS, LR, RESTARTS, 12345 + a.split, fast_mode=FAST, verbosity=0, label=f"{a.config}_split{a.split}")
    mu_s, sig_s = L.predict(model, lik, T_(rec['Xte_s']), fast_mode=FAST)
    mean = mu_s * rec['y_std'] + rec['y_mean']; std_raw = L.apply_sigma_floors(sig_s * rec['y_std']); Yte = rec['Yte']
    yt_s, yt_p, yp_s, yp_p = Yte[:, 0], Yte[:, 1], mean[:, 0], mean[:, 1]
    ss = np.maximum(std_raw[:, 0], 1e-12); sq = np.maximum(std_raw[:, 1], 1e-12)
    m = {'rmse_strength': L.rmse(yt_s, yp_s), 'mae_strength': L.mae(yt_s, yp_s), 'r2_strength': L.r2_score(yt_s, yp_s),
         'cov90_strength_raw_gaussian': L.coverage(yt_s, yp_s, ss, 1.6449), 'nlpd_strength_raw': L.nlpd_gaussian(yt_s, yp_s, ss),
         'rmse_logperm': L.rmse(yt_p, yp_p), 'mae_logperm': L.mae(yt_p, yp_p), 'r2_logperm': L.r2_score(yt_p, yp_p),
         'cov90_logperm_raw_gaussian': L.coverage(yt_p, yp_p, sq, 1.6449), 'nlpd_logperm_raw': L.nlpd_gaussian(yt_p, yp_p, sq)}
    torch.save({'model_state_dict': model.state_dict(), 'likelihood_state_dict': lik.state_dict()}, os.path.join(SF, f"{a.config}_split{a.split}_state.pt"))
    dt = time.time() - t0
    json.dump(dict(config=a.config, split=a.split, features=feats, groups=groups, n_train=len(tr), n_test=len(te), iters=ITERS, lr=LR, restarts=RESTARTS,
                   minutes=dt / 60, metrics={k: float(v) for k, v in m.items()}, y_true_CS=yt_s.tolist(), y_pred_CS=yp_s.tolist(), y_std_raw_CS=ss.tolist(),
                   y_true_logQ=yt_p.tolist(), y_pred_logQ=yp_p.tolist(), y_std_raw_logQ=sq.tolist(), original_row_index=te.tolist(), hyper=hyper(model, lik)), open(p, "w"))
    print(f"DONE {a.config} split {a.split} in {dt / 60:.1f} min | RMSE_CS={m['rmse_strength']:.3f} R2_CS={m['r2_strength']:.3f} RMSE_lnk={m['rmse_logperm']:.3f} R2_lnk={m['r2_logperm']:.3f}", flush=True)


if __name__ == "__main__":
    main()
