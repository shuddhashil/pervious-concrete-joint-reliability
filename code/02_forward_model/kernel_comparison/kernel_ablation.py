"""Kernel-structure ablation around the deployed A3-1 forward model (sandbox; deployed model/engine untouched).
Same 631 records, same 11 inputs/transforms/priors/bounds, same five fixed splits, same Adam lr 0.02, 700 iters x 2 restarts, same seeds as
research/LCMGP_Retrain_A3-1_20261003/retrain.py (imported, not edited). Only the kernel structure (or task rank) changes per variant.
usage: python kernel_ablation.py VARIANT SPLIT   |   python kernel_ablation.py --queue
Output: fits/<variant>_split<k>.json (predictions + metrics). Reference (deployed) split fits already exist in the A3-1 retrain folder."""
import json, os, sys, time
sys.dont_write_bytecode = True
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "code", "02_forward_model"))
import numpy as np, torch, gpytorch
import retrain as R
L = R.L
from gpytorch.constraints import Interval
from gpytorch.priors import LogNormalPrior, GammaPrior
HERE = os.path.dirname(os.path.abspath(__file__)); FITS = os.path.join(HERE, "fits")
N_REC, N_ALL = 6, 11
VARIANTS = {  # rbf: recipe-only | all11 | None ; rq: tied | untied | None ; mat: Matern-5/2 over all 11 ; groups ; rank
    "deployed":           dict(rbf="recipe", rq="tied",   mat=False, groups=("Binder", "Filler"), rank=1),
    "q_untied":           dict(rbf="recipe", rq="untied", mat=False, groups=("Binder", "Filler"), rank=1),
    "no_binder":          dict(rbf="recipe", rq="tied",   mat=False, groups=("Filler",), rank=1),
    "no_filler":          dict(rbf="recipe", rq="tied",   mat=False, groups=("Binder",), rank=1),
    "no_groups":          dict(rbf="recipe", rq="tied",   mat=False, groups=(), rank=1),
    "no_recipe_rbf":      dict(rbf=None,     rq="tied",   mat=False, groups=("Binder", "Filler"), rank=1),
    "rbf_all_for_rq":     dict(rbf="all11",  rq=None,     mat=False, groups=("Binder", "Filler"), rank=1),
    "rbf_only":           dict(rbf="all11",  rq=None,     mat=False, groups=(), rank=1),
    "matern52_only":      dict(rbf=None,     rq=None,     mat=True,  groups=(), rank=1),
    "rank2":              dict(rbf="recipe", rq="tied",   mat=False, groups=("Binder", "Filler"), rank=2),
}

def make_builder(v):
    idx = list(range(N_REC)) + [N_REC] * (N_ALL - N_REC)
    def tie(cls):
        class Tied(cls):
            def __init__(self, **kw):
                super().__init__(ard_num_dims=N_REC + 1, **kw)
                self.register_buffer("tie_idx", torch.tensor(idx, dtype=torch.long)); self.ard_num_dims = N_ALL
            @property
            def lengthscale(self): return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)[..., self.tie_idx]
            @lengthscale.setter
            def lengthscale(self, value): self._set_lengthscale(value)
        return Tied
    def reg(k):
        k.register_constraint("raw_lengthscale", Interval(0.2, 4.0))
        k.register_prior("lengthscale_prior", LogNormalPrior(0.0, 0.45), lambda m: m.raw_lengthscale_constraint.transform(m.raw_lengthscale))
        with torch.no_grad(): k.initialize(raw_lengthscale=k.raw_lengthscale_constraint.inverse_transform(torch.ones_like(k.raw_lengthscale)))
        return k
    def alpha(k):
        k.register_constraint("raw_alpha", Interval(0.25, 10.0)); k.register_prior("alpha_prior", GammaPrior(2.0, 1.0), "alpha")
        with torch.no_grad(): k.alpha = 2.0
        return k
    def build(kernel_name, use_additive, add_kernel_name):
        cont = list(L.CONT_IDXS); ks = []
        if v["rbf"] == "recipe": ks.append(reg(gpytorch.kernels.RBFKernel(ard_num_dims=N_REC, active_dims=list(range(N_REC)))))
        if v["rbf"] == "all11":  ks.append(reg(gpytorch.kernels.RBFKernel(ard_num_dims=N_ALL, active_dims=cont)))
        if v["rq"] == "tied":    ks.append(alpha(reg(tie(gpytorch.kernels.RQKernel)(active_dims=cont))))
        if v["rq"] == "untied":  ks.append(alpha(reg(gpytorch.kernels.RQKernel(ard_num_dims=N_ALL, active_dims=cont))))
        if v["mat"]:             ks.append(reg(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=N_ALL, active_dims=cont)))
        k = ks[0]
        for kk in ks[1:]: k = k + kk
        for name, g in L._additive_groups().items():   # groups were restricted to v["groups"] in activate below
            kg = gpytorch.kernels.RBFKernel(ard_num_dims=len(g), active_dims=g); L._reg_ls(kg); k = k + kg
        return k
    return build

def fit_split(name, split):
    v = VARIANTS[name]; p = os.path.join(FITS, f"{name}_split{split}.json")
    if os.path.exists(p): print("exists", p); return
    R.ensure_data(); feats, groups = R.activate("Q5a3")                      # deployed feature/transform contract
    keep = {g: groups[g] for g in v["groups"]}; L._additive_groups = lambda: keep
    L._make_cont_kernel = make_builder(v)
    X, yS, yP, n0 = L.load_and_prepare_data(R.XLSX); assert X.shape == (631, 11)
    T_ = lambda z: torch.as_tensor(z, dtype=L.DTYPE, device=L.DEVICE).contiguous()
    cfg = dict(R.CFG); cfg["rank"] = v["rank"]; cfg["use_additive"] = bool(v["groups"]); t0 = time.time()
    L.set_seed(20260913 + split)
    sp = json.load(open(R.SPLITS))[f"split_{split}"]; tr = np.asarray(sp["train_idx"], dtype=int); te = np.asarray(sp["test_idx"], dtype=int)
    rec = L._prepare_split(X, yS, yP, tr, te)
    model, lik = L.train_lcmgp(T_(rec['Xtr_s']), T_(rec['Ytr_s']), cfg, R.ITERS, R.LR, R.RESTARTS, 12345 + split, fast_mode=R.FAST, verbosity=0, label=f"{name}_split{split}")
    mu_s, sig_s = L.predict(model, lik, T_(rec['Xte_s']), fast_mode=R.FAST)
    mean = mu_s * rec['y_std'] + rec['y_mean']; std_raw = L.apply_sigma_floors(sig_s * rec['y_std']); Yte = rec['Yte']
    yt_s, yt_p, yp_s, yp_p = Yte[:, 0], Yte[:, 1], mean[:, 0], mean[:, 1]; ss = np.maximum(std_raw[:, 0], 1e-12); sq = np.maximum(std_raw[:, 1], 1e-12)
    m = {'rmse_strength': L.rmse(yt_s, yp_s), 'r2_strength': L.r2_score(yt_s, yp_s), 'cov90_strength_raw_gaussian': L.coverage(yt_s, yp_s, ss, 1.6449), 'nlpd_strength_raw': L.nlpd_gaussian(yt_s, yp_s, ss),
         'rmse_logperm': L.rmse(yt_p, yp_p), 'r2_logperm': L.r2_score(yt_p, yp_p), 'cov90_logperm_raw_gaussian': L.coverage(yt_p, yp_p, sq, 1.6449), 'nlpd_logperm_raw': L.nlpd_gaussian(yt_p, yp_p, sq)}
    json.dump(dict(variant=name, spec={k: (list(x) if isinstance(x, tuple) else x) for k, x in v.items()}, split=split, n_train=len(tr), n_test=len(te), minutes=(time.time() - t0) / 60,
                   metrics={k: float(x) for k, x in m.items()}, y_true_CS=yt_s.tolist(), y_pred_CS=yp_s.tolist(), y_std_raw_CS=ss.tolist(), y_true_logQ=yt_p.tolist(), y_pred_logQ=yp_p.tolist(),
                   y_std_raw_logQ=sq.tolist(), original_row_index=te.tolist(), hyper=R.hyper(model, lik)), open(p, "w"))
    print(f"DONE {name} split {split} {(time.time()-t0)/60:.1f} min | CS RMSE {m['rmse_strength']:.3f} lnk RMSE {m['rmse_logperm']:.3f}", flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "--queue":
        order = [n for n in VARIANTS if n != "deployed"]
        for n in order:
            for s in range(5): fit_split(n, s)
    else: fit_split(sys.argv[1], int(sys.argv[2]))
