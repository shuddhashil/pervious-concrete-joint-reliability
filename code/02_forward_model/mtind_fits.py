"""multi-task lab-effect GP (the paper's model, MT) versus two independent lab-effect GPs (IND),
on the paper's 5 study-grouped folds, under three adaptation protocols. Sandbox; the deployed engine and paper are untouched.

MT  = lab_fits.LabLCMGP unchanged (ICM over CS and ln k, rank-1 + diagonal task matrix, plus a 2x2 laboratory term B_lab).
IND = two GPs with NOTHING shared: strength and ln k each get their own copy of the same continuous kernel (own lengthscales, own outputscale,
      same priors and bounds), their own laboratory variance (diagonal B_lab) and their own noise. Written in the same two-task container with
      the task matrices FIXED to e1e1' and e2e2' (no cross-task covariance anywhere), so the training loop, restarts, seeds, priors, likelihood and
      predict function are byte-for-byte the paper's; the joint marginal likelihood then factorises into the two single-output likelihoods.
Fit = lab_fits.fit (700 iters x 2 restarts, Adam 0.02, seeds 20260913+k / 12345+k); full-data fit = lab_fits labfull settings (900 x 2).

Adaptation protocols per held-out (new) laboratory, hyperparameters never refitted:
  newlab   no trial batches (zero offset, B_lab in the predictive variance)
  near3    the 3 records of the same study nearest to the predicted record (standardised inputs) = the paper's protocol (lab_fits2.near3)
  fixed3_r for r = 0..4: 3 records of each held-out study drawn ONCE per study (seed from r and the study name, identical for MT and IND),
           the model conditioned on them, evaluated on the study's OTHER records only (studies with < 4 records are not evaluated)
Two SDs are saved per prediction: 'raw' = the model's own predictive SD (lik(model(x)), as lab_fits) and 'eng' = the engine convention
(lab-free posterior on the fold's training data + noise, as icm_std_folds.py) that the residual bank and the certificate use.
usage: python mtind_fits.py fold MT|IND K | full IND | all"""
import json, os, sys, time, math, copy, zlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.dont_write_bytecode = True
import numpy as np, torch, gpytorch
import lab_fits as LF
RF, R, L = LF.RF, LF.R, LF.L
OUT = os.path.join(HERE, "out_mtind"); os.makedirs(OUT, exist_ok=True)
N_SEL = 5


def _fix_task(idx_kernel, t):
    """task matrix fixed to e_t e_t' (covar_factor = e_t, var -> 0), not trained."""
    with torch.no_grad():
        idx_kernel.covar_factor.zero_(); idx_kernel.covar_factor[t, 0] = 1.0; idx_kernel.raw_var.fill_(-40.0)
    idx_kernel.covar_factor.requires_grad_(False); idx_kernel.raw_var.requires_grad_(False)


class IndLab(LF.Base):
    def __init__(self, train_x, train_y, likelihood, cfg):
        gpytorch.models.ExactGP.__init__(self, train_x, train_y, likelihood)
        self.cfg = dict(cfg); self.mean_module = L.TwoTaskConstantMean(); comps = []
        for t in range(L.NUM_TASKS):
            cont = L._make_cont_kernel(cfg['kernel'], cfg['use_additive'], cfg['add_kernel'])
            sk = L._scaled_kernel(cont, init=0.8, lo=0.05, hi=3.0, prior_log_mean=math.log(0.8), prior_sd=0.4)
            mk = gpytorch.kernels.MultitaskKernel(sk, num_tasks=L.NUM_TASKS, rank=1); _fix_task(mk.task_covar_module, t); comps.append(mk)
        self.cs_covar_module, self.q_covar_module = comps
        self.lab_covar_module = gpytorch.kernels.MultitaskKernel(LF.SameLab(), num_tasks=L.NUM_TASKS, rank=1)
        with torch.no_grad():
            self.lab_covar_module.task_covar_module.covar_factor.zero_(); self.lab_covar_module.task_covar_module.raw_var.fill_(-3.0)
        self.lab_covar_module.task_covar_module.covar_factor.requires_grad_(False)       # diagonal B_lab: no lab correlation between outputs

    def shared_cov(self, x1, x2):
        return self.cs_covar_module(x1, x2) + self.q_covar_module(x1, x2)

    def forward(self, x):
        return gpytorch.distributions.MultitaskMultivariateNormal(self.mean_module(x), self.shared_cov(x, x) + self.lab_covar_module(x))


def shared_cov(model, x1, x2):
    return model.shared_cov(x1, x2) if isinstance(model, IndLab) else model.shared_covar_module(x1, x2)


MODELS = {"MT": LF.LabLCMGP, "IND": IndLab}


def fit(kind, Xtr, Ytr, seed_k, label, iters=None, restarts=None, full=False):
    L.LCMGPModel = MODELS[kind]
    if full:
        L.set_seed(20260913)
        return L.train_lcmgp(LF.T_(Xtr), LF.T_(Ytr), R.CFG, 900, 0.02, 2, 12345 + 999999, fast_mode=True, verbosity=0, label=label)
    return LF.fit(Xtr, Ytr, seed_k, label)


class CF:
    """Exact closed form of a fitted two-task lab model on its training set (float64): new-laboratory mean (lab term in the training covariance,
    zero offset at the query) and the engine-convention variance (lab-free training covariance + noise). Same algebra as fast_engine for the ICM."""
    def __init__(self, model, lik, X12, Ys):
        self.m = copy.deepcopy(model).double().eval(); lk = copy.deepcopy(lik).double().eval(); dev = next(self.m.parameters()).device
        self.X = torch.as_tensor(X12, dtype=torch.float64, device=dev); n = self.X.shape[0]; self.dev = dev
        with torch.no_grad():
            Ks = shared_cov(self.m, self.X, self.X).to_dense(); Kl = self.m.lab_covar_module(self.X, self.X).to_dense()
            self.noise = lk.task_noises.detach().double().reshape(-1); nz = self.noise.repeat(n)
            self.m0 = torch.stack([self.m.mean_module.strength_const.detach(), self.m.mean_module.perm_const.detach()]).double()
            r = (torch.as_tensor(Ys, dtype=torch.float64, device=dev) - self.m0[None, :]).reshape(-1)
            Lf = torch.linalg.cholesky(Ks + Kl + torch.diag(nz)); self.alpha = torch.cholesky_solve(r[:, None], Lf)[:, 0]
            L0 = torch.linalg.cholesky(Ks + torch.diag(nz)); self.K0inv = torch.cholesky_inverse(L0)
            x0 = self.X[:1]; self.kdiag = torch.diagonal(shared_cov(self.m, x0, x0).to_dense()).double()          # stationary: k(x,x) constant per task
            B_lab = self.m.lab_covar_module.task_covar_module.covar_matrix.to_dense().detach().double(); self.B_lab = B_lab

    def predict(self, Xq12, chunk=4096):
        mu, var = [], []
        with torch.no_grad():
            for a in range(0, len(Xq12), chunk):
                xq = torch.as_tensor(np.asarray(Xq12[a:a + chunk]), dtype=torch.float64, device=self.dev); nq = xq.shape[0]
                kq = shared_cov(self.m, xq, self.X).to_dense()
                mu.append((self.m0[None, :] + (kq @ self.alpha).reshape(nq, 2)).cpu().numpy())
                q = ((kq @ self.K0inv) * kq).sum(1).reshape(nq, 2)
                var.append(torch.clamp(self.kdiag[None, :] - q + self.noise[None, :], min=1e-12).cpu().numpy())
        return np.concatenate(mu), np.concatenate(var)


def fixed_sets(st_te, r):
    """3 adaptation records per held-out study with >= 4 records; seed from (r, study name) only -> identical for both models and every run."""
    out = {}
    for s in sorted(set(st_te)):
        idx = np.flatnonzero(st_te == s)
        if len(idx) < 4: continue
        rng = np.random.default_rng(20261029 + 1000 * r + zlib.crc32(s.encode()) % 100000); out[s] = np.sort(rng.choice(idx, size=3, replace=False))
    return out


def run_fold(kind, k):
    p = os.path.join(OUT, f"{kind}_fold{k}.json")
    if os.path.exists(p): print("exists", p, flush=True); return
    feats, groups, X, yS, yP, st = RF.frame(); Y = np.stack([yS, yP], 1); names = sorted(set(st)); sid = np.array([names.index(s) for s in st], float)
    sp = RF.grouped_splits(st)[f"split_{k}"]; tr, te = np.asarray(sp["train_idx"]), np.asarray(sp["test_idx"]); t0 = time.time()
    xsc, ysc, Z = LF.prep(X, Y, sid, tr); new = np.array([LF.NEW0 + names.index(s) for s in st[te]], float)
    Xtr0, Ytr0 = Z(tr, sid[tr]), ysc.transform(Y[tr])
    model, lik = fit(kind, Xtr0, Ytr0, k, f"{kind}grp{k}")
    torch.save({'model': model.state_dict(), 'lik': lik.state_dict(), 'x_mean': xsc.mean, 'x_std': xsc.std, 'y_mean': ysc.mean, 'y_std': ysc.std}, os.path.join(OUT, f"{kind}_state_fold{k}.pt"))
    cf = CF(model, lik, Xtr0, Ytr0)
    _, v_eng = cf.predict(Z(te, new)); sd_eng = L.apply_sigma_floors(np.sqrt(v_eng) * ysc.std)          # engine SD (same for every protocol)
    Xs = xsc.transform(X[te]); st_te = st[te]; res = dict(job=f"{kind}_fold{k}", kind=kind, fold=k, studies=sp["studies"], rows=te.tolist(), study=st_te.tolist(),
                                                           y_CS=Y[te, 0].tolist(), y_lnk=Y[te, 1].tolist(), sd_eng=sd_eng.tolist(), protocols={})

    def pred(Xa, Ya, q_rows, q_ids):
        model.set_train_data(inputs=LF.T_(Xa), targets=LF.T_(Ya), strict=False)
        mu, sg = L.predict(model, lik, LF.T_(Z(te[q_rows], q_ids)), fast_mode=R.FAST); return LF.unstd(mu, sg, ysc)

    m, s = pred(Xtr0, Ytr0, np.arange(len(te)), new)                        # newlab
    mcf, _ = cf.predict(Z(te, new)); mcf = mcf * ysc.std + ysc.mean
    res["protocols"]["newlab"] = dict(mu=m.tolist(), sd_raw=s.tolist(), eval=list(range(len(te))), check_cf_vs_gpytorch_mean_max_abs=float(np.abs(mcf - m).max()))
    mt, stt, rows = np.zeros((len(te), 2)), np.zeros((len(te), 2)), []
    for i in range(len(te)):                                                 # near3 (lab_fits2 rule)
        pool = np.flatnonzero((st_te == st_te[i]) & (np.arange(len(te)) != i)); pick = pool[np.argsort(((Xs[pool] - Xs[i]) ** 2).sum(1), kind="stable")[:3]]
        aug = te[pick]; mm, ss = pred(np.vstack([Xtr0, Z(aug, new[pick])]), np.vstack([Ytr0, ysc.transform(Y[aug])]), np.array([i]), new[i:i + 1])
        mt[i], stt[i] = mm[0], ss[0]; rows.append(aug.tolist())
    res["protocols"]["near3"] = dict(mu=mt.tolist(), sd_raw=stt.tolist(), eval=list(range(len(te))), trial_rows=rows)
    for r in range(N_SEL):                                                   # fixed3_r
        sets = fixed_sets(st_te, r); mu = np.full((len(te), 2), np.nan); sd = np.full((len(te), 2), np.nan); ev = []; trial = {}
        for s_, pick in sets.items():
            q = np.setdiff1d(np.flatnonzero(st_te == s_), pick); aug = te[pick]; trial[s_] = aug.tolist()
            mm, ss = pred(np.vstack([Xtr0, Z(aug, new[pick])]), np.vstack([Ytr0, ysc.transform(Y[aug])]), q, new[q]); mu[q], sd[q] = mm, ss; ev += q.tolist()
        res["protocols"][f"fixed3_{r}"] = dict(mu=np.nan_to_num(mu, nan=-999).tolist(), sd_raw=np.nan_to_num(sd, nan=-999).tolist(), eval=sorted(ev), trial_rows=trial)
    model.set_train_data(inputs=LF.T_(Xtr0), targets=LF.T_(Ytr0), strict=False)
    res.update(minutes=(time.time() - t0) / 60, hyper=dict(noise_std=cf.noise.tolist(), B_lab_std=cf.B_lab.cpu().numpy().tolist(), kdiag=cf.kdiag.cpu().numpy().tolist(),
                                                           y_std=np.asarray(ysc.std).tolist()))
    json.dump(res, open(p, "w")); print(f"DONE {kind} fold {k} {res['minutes']:.1f} min | CF check {res['protocols']['newlab']['check_cf_vs_gpytorch_mean_max_abs']:.2e}", flush=True)


def run_full(kind):
    p = os.path.join(OUT, f"{kind}_full.pt")
    if os.path.exists(p): print("exists", p, flush=True); return
    feats, groups, X, yS, yP, st = RF.frame(); Y = np.stack([yS, yP], 1); names = sorted(set(st)); sid = np.array([names.index(s) for s in st], float)
    tr = np.arange(len(X)); xsc, ysc, Z = LF.prep(X, Y, sid, tr); t0 = time.time()
    model, lik = fit(kind, Z(tr, sid), ysc.transform(Y), 0, f"full-{kind}", full=True)
    torch.save({'model': {a: v.detach().cpu() for a, v in model.state_dict().items()}, 'lik': {a: v.detach().cpu() for a, v in lik.state_dict().items()}, 'kind': kind,
                'x_mean': np.asarray(xsc.mean), 'x_std': np.asarray(xsc.std), 'y_mean': np.asarray(ysc.mean), 'y_std': np.asarray(ysc.std), 'X_model': X, 'Y': Y,
                'study_index': sid, 'study_names': names}, p)
    print(f"DONE full {kind} {(time.time() - t0) / 60:.1f} min", flush=True)


def export_ind():
    """Engine oracle of the full-data IND model: per output t, kernel spec (indkern), weights a_t, lab-free inverse K0_t^-1, mean, noise, k(x,x),
    on the full-data standardised inputs. Outputs are independent, so the interleaved two-task algebra splits exactly into the t::2 blocks.
    Checks: plain-torch kernel = gpytorch kernel; per-output closed form = CF (two-task) = gpytorch predict mean, on 300 probe points."""
    import indkern
    p = os.path.join(OUT, "IND_oracle.pt")
    if os.path.exists(p): print("exists", p, flush=True); return
    R.activate("Q5a3")                                                       # kernel builder of the fits (RF.frame() does this in the fit jobs)
    S = torch.load(os.path.join(OUT, "IND_full.pt"), map_location="cpu", weights_only=False); L.LCMGPModel = IndLab
    X, Y, sid = S["X_model"], S["Y"], S["study_index"]; Xs = (X - S["x_mean"]) / S["x_std"]; Ys = (Y - S["y_mean"]) / S["y_std"]; X12 = np.column_stack([Xs, sid])
    lik = L._build_likelihood(); model = IndLab(LF.T_(X12), LF.T_(Ys), lik, R.CFG).to(device=L.DEVICE, dtype=L.DTYPE)
    model.load_state_dict(S["model"]); lik.load_state_dict(S["lik"]); model.eval(); lik.eval()
    cf = CF(model, lik, X12, Ys); dev = cf.dev; Xt = torch.as_tensor(Xs, dtype=torch.float64, device=dev)
    rng = np.random.default_rng(1); probe = Xs[rng.choice(len(Xs), 300)] + 0.3 * rng.standard_normal((300, Xs.shape[1])); Pt = torch.as_tensor(probe, dtype=torch.float64, device=dev)
    mu_cf, var_cf = cf.predict(np.column_stack([probe, np.full(300, 99999.0)])); out = dict(x_mean=S["x_mean"], x_std=S["x_std"], y_mean=S["y_mean"], y_std=S["y_std"], X=Xs, tasks=[], checks={})
    md = cf.m
    for t, comp in enumerate((md.cs_covar_module, md.q_covar_module)):
        spec = indkern.leaves_of(comp.data_covar_module)
        with torch.no_grad():
            Kg = comp.data_covar_module(Pt, Xt).to_dense(); Kp = indkern.K(spec, Pt, Xt); out["checks"][f"kernel_max_abs_t{t}"] = float((Kg - Kp).abs().max())
            a_t = cf.alpha[t::2]; K0i = cf.K0inv[t::2, t::2]; mu = cf.m0[t] + Kp @ a_t; q = ((Kp @ K0i) * Kp).sum(1)
            kd = float(indkern.K(spec, Pt[:1], Pt[:1])[0, 0]); var = kd - q + cf.noise[t]
        out["checks"][f"mean_vs_CF_max_abs_t{t}"] = float(np.abs(mu.cpu().numpy() - mu_cf[:, t]).max()); out["checks"][f"var_vs_CF_max_abs_t{t}"] = float(np.abs(var.cpu().numpy() - var_cf[:, t]).max())
        out["checks"][f"cross_block_K0inv_max_abs_t{t}"] = float(cf.K0inv[t::2, (1 - t)::2].abs().max())
        out["tasks"].append(dict(spec=spec, a=a_t.cpu().numpy(), K0inv=K0i.cpu().numpy(), m0=float(cf.m0[t]), noise=float(cf.noise[t]), kdiag=kd))
    mg, _ = L.predict(model, lik, LF.T_(np.column_stack([probe, np.full(300, 99999.0)])), fast_mode=False); out["checks"]["CF_vs_gpytorch_mean_max_abs"] = float(np.abs(mg - mu_cf).max())
    out["probe"] = probe; out["probe_mu_std"] = mu_cf; out["probe_var_std"] = var_cf
    ok = all(v < 1e-6 for k, v in out["checks"].items() if k.startswith(("kernel", "mean_vs", "var_vs", "cross"))) and out["checks"]["CF_vs_gpytorch_mean_max_abs"] < 1e-3
    print("IND oracle checks", json.dumps(out["checks"]), "OK" if ok else "FAILED", flush=True); assert ok
    torch.save(out, p)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "all":
        for k in range(5):
            for kind in ("MT", "IND"): run_fold(kind, k)
        run_full("IND"); export_ind()
    elif a[0] == "fold": run_fold(a[1], int(a[2]))
    elif a[0] == "full": run_full(a[1])
    elif a[0] == "export": export_ind()
