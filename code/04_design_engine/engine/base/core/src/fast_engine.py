"""Process-local fast scenario-prediction engine for the frozen 10-D LCM-GP.

Computes, for a batch of manufacturing scenarios, exactly the five arrays the
robust evaluator consumes -- ``muS, muQ, sS, sQ, support`` -- without changing
any model weight, formula, threshold or frozen source file.  Every stage is a
re-expression of the frozen computation that removes redundant work:

* Scenario features: one batched descriptor call and one NumPy transform /
  standardisation instead of a per-scenario Python loop + pandas round trips.
  The standardised matrix is computed once and shared by the GP, the
  applicability-domain (AD) kNN, the local-residual kNN and the support model
  (the frozen path computed it twice).
* GP predictive mean/variance: the frozen model is an intrinsic coregionalisation
  model, ``K = K_x (x) B`` with task-specific noise ``I (x) D``.  GPyTorch
  re-diagonalises the 650x650 training kernel and re-applies it to a 1300 x 2N
  right-hand side on every call.  Here the training-side eigendecomposition is
  done once (float64) and the exact predictive mean/variance are evaluated from
  the same kernel cross-covariances via that closed form (see ``FastGP``).
* AD and local-residual kNN: SciPy ``cdist`` instead of an (N, 650, 10)
  broadcast tensor; the selected neighbour *sets* and reductions are unchanged.
* KDE support score: exact brute-force log-sum-exp of the Gaussian kernel
  instead of sklearn's tree walk (same quantity; rtol/atol of the frozen
  estimator are 0).

Each replacement is self-checked against the frozen implementation on a small
random batch the first time it is used; on any disagreement beyond tolerance the
engine disables itself (``available`` becomes False) and callers fall back to the
frozen path.  ``DISABLE_FAST_ENGINE=1`` opts out entirely.

Equivalence tests: ``tests/test_fast_engine_equivalence.py``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from a3_quantiles import quantiles_batch  # sandbox A3 engine

_CHUNK = 8192  # rows per GP / kNN chunk (bounds memory; results are row-independent)


def _to_numpy(x, dtype=float):
    """Safe array coercion: works for a plain array/list OR a torch tensor on
    any device (e.g. a checkpoint dict entry moved to CUDA by torch.load's
    map_location). Never assumes CPU; never silently drops data."""
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy().astype(dtype)
    return __import__("numpy").asarray(x, dtype=dtype)


class _Engine:
    def __init__(self, E):
        import torch
        import fast_backend as FB
        import fast_descriptors as FD

        self.E = E
        self.L = E.L
        self.FD = FD
        self.torch = torch
        b = E.get_bundle()
        self.obj = b["obj"]
        self.support = b["support"]
        self.save_obj, self.model, self.likelihood = FB._load(E.ORACLE_PT)
        so = self.save_obj
        L = self.L

        # ---- feature transform / standardisation constants ----
        self.features = list(so["features"])
        recipe = dict(L.TRANSFORM_RECIPE)
        self.log_idx = [self.features.index(c) for c in recipe["log_cols"]]
        self.log1p_idx = [self.features.index(c) for c in recipe["log1p_cols"]]
        self.x_mean = _to_numpy(so["x_cont_mean"])
        x_std = _to_numpy(so["x_cont_std"])
        self.x_std = np.where(x_std == 0, 1.0, x_std)
        self.cont_idxs = list(L.CONT_IDXS)
        if self.cont_idxs != list(range(len(self.features))):
            raise RuntimeError("fast_engine assumes every feature is standardised")
        # support model standardisation (frozen path uses obj mean/std)
        obj_mean = _to_numpy(self.obj["x_cont_mean"])
        obj_std = _to_numpy(self.obj["x_cont_std"])
        if not (np.array_equal(obj_mean, self.x_mean) and np.array_equal(obj_std, x_std)):
            raise RuntimeError("support and GP standardisation differ; fast_engine cannot share features")

        # ---- output scaling ----
        self.y_mean = _to_numpy(so["y_mean"])
        self.y_std = _to_numpy(so["y_std"])
        self.combine_rule = so.get("uncertainty_combine_rule", L.UNCERTAINTY_COMBINE_RULE)
        self.local_q = max(2, int(so.get("local_residual_default_q_neighbors", L.LOCAL_RESIDUAL_K_NEIGHBORS)))
        self.ad_eta = float(so.get("ad_inflation_eta", L.AD_INFLATION_ETA))

        # ---- AD reference ----
        X_train = _to_numpy(so["train_x"])
        X_train = X_train[np.isfinite(X_train).all(axis=1)]
        ad_ref = so.get("ad_reference", {})
        self.ad_X = X_train
        self.ad_k = int(max(1, min(int(ad_ref.get("ad_k_neighbors", min(L.AD_K_NEIGHBORS, len(X_train)))),
                                   len(X_train))))
        self.ad_tau = float(ad_ref.get("tau_AD", np.inf))
        self.has_ad = "ad_reference" in so

        # ---- local-residual reference ----
        Xr = _to_numpy(so["oos_residual_features_std"])
        rs = _to_numpy(so["oos_residual_strength"]).ravel()
        rp = _to_numpy(so["oos_residual_logperm"]).ravel()
        fin = np.isfinite(Xr).all(axis=1) & np.isfinite(rs) & np.isfinite(rp)
        self.res_X, self.res_s, self.res_p = Xr[fin], rs[fin], rp[fin]
        if len(self.res_X) < self.local_q:
            raise RuntimeError("too few local-residual references for fast_engine")

        # ---- support model ----
        self.ocsvm = self.support["ocsvm"]
        kde = self.support["kde"]
        self.kde = kde
        self.kde_X = np.asarray(kde.tree_.data, dtype=float)
        self.kde_h = float(kde.bandwidth_)
        d = self.kde_X.shape[1]
        self.kde_log_norm = -0.5 * d * np.log(2.0 * np.pi) - d * np.log(self.kde_h) - np.log(self.kde_X.shape[0])
        if getattr(kde, "kernel", "gaussian") != "gaussian":
            raise RuntimeError("fast_engine KDE path implemented for the gaussian kernel only")

        self.gp = FastGP(self.model, self.likelihood, so, L, torch)

        self.available = True
        self._self_check()

    # ------------------------------------------------------------------ features
    def standardised_features(self, F):
        """(N, 10) raw features in frozen order -> standardised GP/support matrix."""
        X = np.array(F, dtype=float, copy=True)
        if not np.isfinite(X).all():
            raise ValueError("Prediction input contains missing/invalid values")
        for j in self.log_idx:
            if np.any(X[:, j] <= 0):
                raise ValueError(f"Column '{self.features[j]}' must be positive for log transform.")
            X[:, j] = np.log(X[:, j])
        for j in self.log1p_idx:
            if np.any(X[:, j] < 0):
                raise ValueError(f"Column '{self.features[j]}' must be non-negative for log1p transform.")
            X[:, j] = np.log1p(X[:, j])
        return (X - self.x_mean) / self.x_std

    def scenario_features(self, mfg):
        """Manufacturing-scenario dict -> (N, 10) raw features (frozen column order)."""
        G = np.round(np.asarray(mfg["g_actual"], dtype=float), 12)  # frozen cache key rounding
        d = descriptors_batch(self.FD, G, self.E.DIAM)
        cols = {
            "porosity": np.asarray(mfg["phi_actual"], dtype=float),
            "w/c": np.asarray(mfg["wc_actual"], dtype=float),
            "cem (kg/m3)": np.asarray(mfg["C_actual"], dtype=float),
            "CA (kg/m3)": np.asarray(mfg["A_actual"], dtype=float),
            "SP (kg/m3)": np.asarray(mfg["SP_actual"], dtype=float),
            "sand (kg/m3)": np.asarray(mfg["S_actual"], dtype=float),
            "WAS": d["WAS"], "I_gap": d["I_gap"], "Cu": d["Cu"], "Aag(mm^-1)": d["Aag"],
            "q10": d["q10"], "q25": d["q25"], "q50": d["q50"], "q75": d["q75"], "q90": d["q90"],
        }
        return np.column_stack([cols[c] for c in self.features])

    # -------------------------------------------------------------------- pieces
    def _knn_mean(self, Xs, ref, k):
        d = cdist(Xs, ref)
        return np.partition(d, k - 1, axis=1)[:, :k], d

    def ad_ratio(self, Xs):
        nn, _ = self._knn_mean(Xs, self.ad_X, self.ad_k)
        D = np.mean(nn, axis=1)
        ratio = D / max(self.ad_tau, 1e-12) if np.isfinite(self.ad_tau) else np.zeros_like(D)
        return ratio

    def local_std(self, Xs):
        q = self.local_q
        d = cdist(Xs, self.res_X)
        nn_idx = np.argpartition(d, q - 1, axis=1)[:, :q]
        s_std = np.std(self.res_s[nn_idx], axis=1, ddof=1)
        p_std = np.std(self.res_p[nn_idx], axis=1, ddof=1)
        return s_std, p_std

    def kde_log_density(self, Xs):
        out = np.empty(Xs.shape[0], dtype=float)
        h2 = 2.0 * self.kde_h * self.kde_h
        for a in range(0, Xs.shape[0], 2048):
            d2 = cdist(Xs[a:a + 2048], self.kde_X, "sqeuclidean")
            out[a:a + 2048] = logsumexp(-d2 / h2, axis=1) + self.kde_log_norm
        return out

    def support_scores(self, Xs):
        """Combined support score (same combination as the frozen scorer)."""
        F = self.E.F  # frozen feasibility module (constants / stable sigmoid)
        art = self.support
        n = Xs.shape[0]
        finite = np.isfinite(Xs).all(axis=1)
        comb = np.full(n, np.nan, dtype=float)
        if finite.any():
            Xf = Xs[finite]
            dec = self.ocsvm.decision_function(Xf).reshape(-1)
            kld = self.kde_log_density(Xf)
            scale = float(art.get("ocsvm_scale", 1.0))
            if not np.isfinite(scale) or scale <= F.EPS:
                scale = 1.0
            oc = F._sigmoid_stable(dec / scale)
            lr = np.minimum(kld - float(art["kde_log_max"]), 0.0)
            lr = np.maximum(lr, -745.0)
            kd = np.clip(np.exp(lr), 0.0, 1.0)
            comb[finite] = np.clip(0.5 * oc + 0.5 * kd, 0.0, 1.0)
        return comb

    # --------------------------------------------------------------- main entry
    def arrays_from_features(self, F):
        """Raw (N,10) features -> (muS, muQ, sS, sQ, support)."""
        L = self.L
        n = F.shape[0]
        muS = np.empty(n); muQ = np.empty(n); sS = np.empty(n); sQ = np.empty(n); sup = np.empty(n)
        for a in range(0, n, _CHUNK):
            Xs = self.standardised_features(F[a:a + _CHUNK])
            mu_s, sig_s = self.gp.predict(Xs)
            mean = mu_s * self.y_std + self.y_mean
            std_raw = L.apply_sigma_floors(sig_s * self.y_std)
            ls, lp = self.local_std(Xs)
            pre_s = L._combine_std(std_raw[:, 0], ls, self.combine_rule)
            pre_p = L._combine_std(std_raw[:, 1], lp, self.combine_rule)
            if self.has_ad:
                infl = L._ad_inflation_from_ratio(self.ad_ratio(Xs), eta=self.ad_eta)
            else:
                infl = L._ad_inflation_from_ratio(np.full(len(Xs), np.nan), eta=self.ad_eta)
            sl = slice(a, a + len(Xs))
            muS[sl] = mean[:, 0]; muQ[sl] = mean[:, 1]
            sS[sl] = pre_s * infl; sQ[sl] = pre_p * infl
            sup[sl] = self.support_scores(Xs)
        return muS, muQ, sS, sQ, sup

    def scenario_arrays(self, mfg):
        return self.arrays_from_features(self.scenario_features(mfg))

    # ---------------------------------------------------------------- self-check
    def _self_check(self):
        """Compare against the frozen path on a small deterministic batch."""
        import pandas as pd

        rng = np.random.default_rng(12345)
        Xtr = _to_numpy(self.save_obj["train_x"])
        # raw-feature batch: invert the frozen transform on perturbed training rows
        Z = Xtr[rng.choice(len(Xtr), 48, replace=False)] + rng.normal(0.0, 0.35, (48, Xtr.shape[1]))
        Xraw = Z * self.x_std + self.x_mean
        for j in self.log1p_idx:
            Xraw[:, j] = np.maximum(Xraw[:, j], 0.0)  # log1p domain is non-negative
        for j in self.log_idx:
            Xraw[:, j] = np.exp(Xraw[:, j])
        for j in self.log1p_idx:
            Xraw[:, j] = np.expm1(Xraw[:, j])
        frame = pd.DataFrame(Xraw, columns=self.features)
        try:
            ref = self.E.predict_fast(self.E.ORACLE_PT, frame)  # frozen path (fast_backend)
            ref_sup = np.asarray(self.E._support_scores(frame), dtype=float)
            got = self.arrays_from_features(Xraw)
            want = (ref["strength_pred_mpa"].to_numpy(float), ref["logperm_pred"].to_numpy(float),
                    ref["strength_std_effective_mpa"].to_numpy(float),
                    ref["logperm_std_effective"].to_numpy(float), ref_sup)
            tol = (1e-4, 1e-4, 1e-4, 1e-4, 1e-8)
            names = ("muS", "muQ", "sS", "sQ", "support")
            for nm, g, w, t in zip(names, got, want, tol):
                err = float(np.max(np.abs(g - w) / (1.0 + np.abs(w))))
                if not err <= t:
                    print(f"[perf] fast_engine self-check FAILED for {nm}: rel err {err:.3e} > {t}; "
                          f"using frozen path", flush=True)
                    self.available = False
                    return
        except Exception as ex:  # noqa: BLE001
            print(f"[perf] fast_engine self-check raised {ex!r}; using frozen path", flush=True)
            self.available = False


def dpp_grid_batch_vec(FD, G, percents=(10.0, 60.0)):
    """Fully vectorised ``fast_descriptors.dpp_grid_batch``.

    For each row the passing curve ``p`` (500 grid thresholds) is non-decreasing,
    so equal-value groups are contiguous runs.  The frozen ``_dpp_grid`` builds the
    (unique value, mean log-threshold of the run) polyline and linearly
    interpolates at ``percent``.  Here the run boundaries come from counting
    (p < value) / (p <= value) along the row and the run means from one shared
    cumulative sum of the (row-independent) log-threshold grid.
    """
    G = np.asarray(G, dtype=float)
    n = G.shape[0]
    p = FD.grid_passing_batch(G)  # (N, 500) non-decreasing
    x = FD._LOG_TH
    cumx = np.concatenate([[0.0], np.cumsum(x)])
    m = p.shape[1]
    ok = np.isfinite(p).all(axis=1)
    out = {}
    rows = np.arange(n)
    for percent in percents:
        res = np.full(n, np.nan, dtype=float)
        i1 = (p < percent).sum(axis=1)  # first index with p >= percent
        i1c = np.minimum(i1, m - 1)
        pv = p[rows, i1c]
        valid = ok & (i1 < m) & (percent >= p[:, 0])
        exact = valid & (pv == percent)
        # exact knot: value == percent, mean of its run
        i2 = (p <= percent).sum(axis=1)
        cnt = np.maximum(i2 - i1, 1)
        mean_exact = (cumx[np.minimum(i2, m)] - cumx[np.minimum(i1, m)]) / cnt
        res[exact] = np.exp(mean_exact[exact])
        # between knots: run of the value just below (lo) and the value just above (hi)
        inter = valid & ~exact & (i1 > 0)
        if inter.any():
            r = rows[inter]
            lo_v = p[r, i1[inter] - 1]
            hi_v = pv[inter]
            j1 = (p[r] < lo_v[:, None]).sum(axis=1)
            k2 = (p[r] <= hi_v[:, None]).sum(axis=1)
            ii = i1[inter]
            ux_lo = (cumx[ii] - cumx[j1]) / (ii - j1)
            ux_hi = (cumx[k2] - cumx[ii]) / (k2 - ii)
            res[inter] = np.exp(ux_lo + (percent - lo_v) / (hi_v - lo_v) * (ux_hi - ux_lo))
        out[percent] = res
    return out


def descriptors_batch(FD, G, diam):
    """``fast_descriptors.descriptors_precise_batch`` with the vectorised D10/D60."""
    G = np.asarray(G, dtype=float)
    diam = np.asarray(diam, dtype=float)
    was = G @ diam
    aag = np.sum(6.0 * G / diam[None, :], axis=1)
    ig = np.einsum("ni,nj,ij->n", G, G, FD._build_logratio_ut(diam))
    dpp = dpp_grid_batch_vec(FD, G, (10.0, 60.0))
    d10, d60 = dpp[10.0], dpp[60.0]
    with np.errstate(invalid="ignore", divide="ignore"):
        cu = np.where(d10 > 0, d60 / d10, np.nan)
    q = quantiles_batch(G)
    return {"WAS": was, "I_gap": ig, "Cu": cu, "Aag": aag, "q10": q[:, 0], "q25": q[:, 1], "q50": q[:, 2], "q75": q[:, 3], "q90": q[:, 4]}


class FastGP:
    """Exact predictive mean / variance of the frozen ICM multitask GP.

    Frozen model: prior mean ``m`` (per task), kernel ``K_x (x) B`` (interleaved
    data-major layout), Gaussian task noise ``I (x) D``.  With
    ``B' = D^{-1/2} B D^{-1/2} = Q_B diag(mu) Q_B^T`` and ``K_x = Q_K diag(lam) Q_K^T``:

        (K_x (x) B + I (x) D)^{-1}
            = (I (x) D^{-1/2})(Q_K (x) Q_B) diag(lam_m mu_t + 1)^{-1} (Q_K (x) Q_B)^T (I (x) D^{-1/2})

    For a test point with data cross-covariance ``c`` (length n), the covariance
    correction for task a is ``sum_t W[t,a]^2 s_t`` with ``W = Q_B^T D^{-1/2} B`` and
    ``s_t = sum_m (c^T Q_K)_m^2 / (lam_m mu_t + 1)``; predictive variance is
    ``k(x,x) B_aa - correction + D_aa`` (likelihood noise included, as in
    ``likelihood(model(x)).variance``).  The mean is
    ``m + (c^T A) B`` with ``A`` the frozen ``mean_cache`` reshaped (n, tasks).
    All training-side factors are formed once, in float64 (GPyTorch also solves in
    float64), from the operators GPyTorch itself built for the frozen model.
    """

    def __init__(self, model, likelihood, save_obj, L, torch):
        self.torch = torch
        self.model = model
        self.L = L
        model.eval(); likelihood.eval()
        fast_mode = bool(save_obj.get("fast_mode", True))
        Xtr = model.train_inputs[0]
        # warm GPyTorch's own caches with the frozen predict (builds mean_cache)
        L.predict(model, likelihood, Xtr[:2].clone(), fast_mode=fast_mode)
        ps = model.prediction_strategy
        lt = ps.lik_train_train_covar
        lin = lt.linear_op.linear_ops
        Kx = lin[0].to_dense().detach().double().cpu().numpy()
        B = lin[1].to_dense().detach().double().cpu().numpy()
        nT = B.shape[0]
        n = Kx.shape[0]
        dvec = lt.diag_tensor._diagonal().detach().double().cpu().numpy().reshape(n, nT)
        if not (dvec == dvec[0]).all():
            raise RuntimeError("non-constant task noise; fast GP path not applicable")
        d = dvec[0]
        dh = 1.0 / np.sqrt(d)
        Bp = dh[:, None] * B * dh[None, :]
        mu, QB = np.linalg.eigh(0.5 * (Bp + Bp.T))
        lam, QK = np.linalg.eigh(0.5 * (Kx + Kx.T))
        self.QK = QK
        self.g = 1.0 / (lam[:, None] * mu[None, :] + 1.0)  # (n, T)
        self.W2 = (QB.T @ (dh[:, None] * B)) ** 2  # (T_t, T_a)
        self.B = B
        self.Bdiag = np.diag(B).copy()
        self.d = d
        self.alpha = ps.mean_cache.detach().double().cpu().numpy().reshape(n, nT)
        self.m0 = np.array([float(model.mean_module.strength_const), float(model.mean_module.perm_const)])
        self.Xtr = Xtr.detach()
        self.dk = model.shared_covar_module.data_covar_module
        self.n = n
        self.nT = nT

    def predict(self, Xs):
        torch = self.torch
        with torch.no_grad():
            Xt = torch.as_tensor(np.ascontiguousarray(Xs), dtype=self.L.DTYPE, device=self.Xtr.device).contiguous()
            C = self.dk.forward(Xt, self.Xtr)
            if hasattr(C, "to_dense"):
                C = C.to_dense()
            kd = self.dk.forward(Xt, Xt, diag=True)
            if hasattr(kd, "to_dense"):
                kd = kd.to_dense()
            C = C.double().cpu().numpy()
            kd = kd.double().cpu().numpy()
        mean = self.m0[None, :] + (C @ self.alpha) @ self.B.T
        U = C @ self.QK
        s = (U * U) @ self.g  # (N, T)
        var = kd[:, None] * self.Bdiag[None, :] - s @ self.W2 + self.d[None, :]
        sig = np.sqrt(np.maximum(var, 1e-12))
        return mean, sig


_ENGINE = None
_DISABLED = os.environ.get("DISABLE_FAST_ENGINE") == "1"


def get_engine(E):
    """Return the process-wide engine or None if disabled / failed its self-check."""
    global _ENGINE, _DISABLED
    if _DISABLED:
        return None
    if _ENGINE is None:
        try:
            eng = _Engine(E)
        except Exception as ex:  # noqa: BLE001
            print(f"[perf] fast_engine unavailable ({ex!r}); using frozen path", flush=True)
            _DISABLED = True
            return None
        if not eng.available:
            _DISABLED = True
            return None
        _ENGINE = eng
        print("[perf] fast_engine active (exact GP/kNN/KDE fast path, self-check passed)", flush=True)
    return _ENGINE
