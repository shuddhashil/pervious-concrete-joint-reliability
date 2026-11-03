"""Direct local conditional-density estimators of p(phi | x).

B0 global KDE baseline; B1 kNN empirical; B2 NNKCDE Euclidean; B3 NNKCDE
shrinkage-Mahalanobis; B4 adaptive NW-KCDE; B5 joint-KDE ratio; B6 joint GMM
with analytic conditionals; B7 local NNK-GMM hybrid; B8 stacked ensemble.
S1/S2/S3 joint-state anomaly sections (secondary diagnostics only).

FORBIDDEN here: CatBoost/RF/XGB/NN/GP point predictors of phi, mediated or
hierarchical models. Nothing in this file predicts phi-hat; every primary
method estimates a full conditional density. Tuning is ALWAYS by grouped
held-out conditional log-score (density quality), NEVER by region width.

Conventions
-----------
Xt : transformed but unscaled recipe matrix (B3 metric space).
Xs : standardized recipe matrix (kernel spaces; scaler fit on FIT only).
groups : nominal recipe-group ids; neighbor search ALWAYS excludes
    same-group rows so near-duplicate recipes cannot leak.
"""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

FLOOR = 1e-12


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def silverman(x):
    x = np.asarray(x, float)
    n = len(x)
    sd = float(np.std(x, ddof=1)) if n > 1 else 1.0
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float((q75 - q25) / 1.34) if n > 1 else sd
    s = min(sd, iqr) if min(sd, iqr) > 0 else max(sd, 1e-6)
    return 0.9 * s * n ** (-0.2)


def gauss1d(x, mu, h):
    h = max(float(h), 1e-9)
    z = (np.asarray(x, float) - np.asarray(mu, float)) / h
    return np.exp(-0.5 * z ** 2) / (h * np.sqrt(2.0 * np.pi))


def euclid2(A, B):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)


def apply_group_exclusion(D, g_tr, g_q):
    """Set distances to +inf where train/query share a recipe group."""
    if g_q is None:
        return D
    D = np.array(D, float, copy=True)
    D[:, :] = np.where(np.asarray(g_tr)[None, :]
                       == np.asarray(g_q)[:, None], np.inf, D)
    return D


def knn(X_tr, g_tr, X_q, g_q, k, D=None):
    """k nearest train rows per query with group exclusion.

    Returns (idx, dist), rows with no eligible neighbor get idx=-1, dist=inf.
    """
    if D is None:
        D = euclid2(X_q, X_tr)
    D = apply_group_exclusion(D, g_tr, g_q)
    k = int(min(k, D.shape[1]))
    idx = np.argsort(D, axis=1, kind='stable')[:, :k]
    dist = np.take_along_axis(D, idx, axis=1)
    bad = ~np.isfinite(dist)
    idx = np.where(bad, -1, idx)
    return idx, dist


def inner_folds(groups, n_splits=3):
    return list(GroupKFold(n_splits=n_splits).split(
        np.zeros(len(groups)), groups=np.asarray(groups)))


def mean_logscore(scores):
    scores = np.asarray(scores, float)
    scores = scores[np.isfinite(scores)]
    return float(scores.mean()) if len(scores) else -np.inf


# --------------------------------------------------------------------------
# base class for density methods
# --------------------------------------------------------------------------

class CDEMethod:
    name = 'base'

    def tune(self, Xt, Xs, phi, groups, seed=0):
        """Return (best_params, oof_logscores_of_best)."""
        raise NotImplementedError

    def fit(self, Xt, Xs, phi, groups, params, seed=0):
        raise NotImplementedError

    def point_logpdf(self, Xt_q, Xs_q, g_q, phi_q):
        raise NotImplementedError

    def grid_density(self, Xt_q, Xs_q, g_q, grid):
        raise NotImplementedError


# --------------------------------------------------------------------------
# B0 -- global porosity KDE baseline
# --------------------------------------------------------------------------

class B0GlobalKDE(CDEMethod):
    name = 'B0'

    def __init__(self, scales=(0.5, 0.75, 1.0, 1.5)):
        self.scales = list(scales)

    def _fit_h(self, phi_tr, scale):
        return max(silverman(phi_tr) * scale, 1e-6)

    def tune(self, Xt, Xs, phi, groups, seed=0):
        phi = np.asarray(phi, float)
        best, best_sc, oof_best = -np.inf, None, None
        for s in self.scales:
            oof = []
            for tri, vai in inner_folds(groups):
                h = self._fit_h(phi[tri], s)
                oof.append(np.log(gauss1d(phi[vai, None],
                                          phi[tri][None, :], h).mean(1)
                                  + FLOOR))
            oof = np.concatenate(oof)
            sc = mean_logscore(oof)
            if sc > best:
                best, best_sc, oof_best = sc, {'bw_scale': s}, oof
        return best_sc, oof_best

    def fit(self, Xt, Xs, phi, groups, params, seed=0):
        self.phi_tr = np.asarray(phi, float)
        self.h = self._fit_h(self.phi_tr, params['bw_scale'])
        self.info = {'h': self.h, 'n': len(self.phi_tr)}
        return self

    def _dens(self, pts):
        K = gauss1d(np.asarray(pts, float)[:, None],
                    self.phi_tr[None, :], self.h)
        return K.mean(1) + FLOOR

    def point_logpdf(self, Xt_q, Xs_q, g_q, phi_q):
        return np.log(self._dens(phi_q))

    def grid_density(self, Xt_q, Xs_q, g_q, grid):
        nq = np.asarray(Xs_q).shape[0]
        d = self._dens(grid)
        return np.repeat(d[None, :], nq, axis=0)


# --------------------------------------------------------------------------
# B1 -- kNN empirical local distribution (fixed Silverman smoothing)
# --------------------------------------------------------------------------

class B1KNN(CDEMethod):
    name = 'B1'

    def __init__(self, ks=(15, 25, 40, 60, 100, 150)):
        self.ks = list(ks)

    def tune(self, Xt, Xs, phi, groups, seed=0):
        phi = np.asarray(phi, float)
        best, best_p, oof_best = -np.inf, None, None
        for k in self.ks:
            oof = []
            for tri, vai in inner_folds(groups):
                h = max(silverman(phi[tri]), 1e-6)
                idx, _ = knn(Xs[tri], groups[tri], Xs[vai], groups[vai], k)
                for r, ii in enumerate(idx):
                    ii = ii[ii >= 0]
                    if len(ii) == 0:
                        oof.append(np.log(FLOOR))
                        continue
                    oof.append(np.log(gauss1d(phi[vai[r]],
                                              phi[tri][ii], h).mean()
                                      + FLOOR))
            oof = np.array(oof)
            sc = mean_logscore(oof)
            if sc > best:
                best, best_p, oof_best = sc, {'k': k}, oof
        return best_p, oof_best

    def fit(self, Xt, Xs, phi, groups, params, seed=0):
        self.Xs = np.asarray(Xs, float)
        self.phi_tr = np.asarray(phi, float)
        self.g_tr = np.asarray(groups)
        self.k = int(params['k'])
        self.h = max(silverman(self.phi_tr), 1e-6)
        self.info = {'k': self.k, 'h': self.h}
        return self

    def _nbr_phi(self, Xs_q, g_q):
        idx, dist = knn(self.Xs, self.g_tr, Xs_q, g_q, self.k)
        return idx, dist

    def point_logpdf(self, Xt_q, Xs_q, g_q, phi_q):
        idx, _ = self._nbr_phi(Xs_q, g_q)
        out = np.empty(len(phi_q))
        for r, ii in enumerate(idx):
            ii = ii[ii >= 0]
            if len(ii) == 0:
                out[r] = np.log(FLOOR)
                continue
            out[r] = np.log(gauss1d(phi_q[r], self.phi_tr[ii],
                                    self.h).mean() + FLOOR)
        return out

    def grid_density(self, Xt_q, Xs_q, g_q, grid):
        idx, _ = self._nbr_phi(Xs_q, g_q)
        G = np.asarray(grid, float)
        out = np.empty((len(idx), len(G)))
        for r, ii in enumerate(idx):
            ii = ii[ii >= 0]
            if len(ii) == 0:
                out[r] = np.full(len(G), 1.0 / (G[-1] - G[0]))
                continue
            out[r] = gauss1d(G[None, :], self.phi_tr[ii][:, None],
                             self.h).mean(0) + FLOOR
        return out


# --------------------------------------------------------------------------
# B2 / B3 -- NNKCDE (Euclidean / shrinkage-Mahalanobis)
# --------------------------------------------------------------------------

class _NNKCDE(CDEMethod):
    metric = 'euclidean'

    def __init__(self, ks=(15, 25, 40, 60, 100, 150),
                 hphis=(0.75, 1.25, 2.0, 3.0)):
        self.ks = list(ks)
        self.hphis = list(hphis)

    # -- distance space --
    def _space_fit(self, Xt_tr, Xs_tr):
        return Xs_tr  # overridden by B3

    def _space_q(self, Xt_q, Xs_q):
        return Xs_q  # overridden by B3

    def _point_scores(self, S_tr, g_tr, phi_tr, S_q, g_q, phi_q, k, h):
        idx, _ = knn(S_tr, g_tr, S_q, g_q, k)
        out = np.empty(len(phi_q))
        for r, ii in enumerate(idx):
            ii = ii[ii >= 0]
            if len(ii) == 0:
                out[r] = np.log(FLOOR)
                continue
            out[r] = np.log(gauss1d(phi_q[r], phi_tr[ii], h).mean()
                            + FLOOR)
        return out

    def tune(self, Xt, Xs, phi, groups, seed=0):
        Xt = np.asarray(Xt, float)
        Xs = np.asarray(Xs, float)
        phi = np.asarray(phi, float)
        groups = np.asarray(groups)
        best, best_p, oof_best = -np.inf, None, None
        for k in self.ks:
            for h in self.hphis:
                oof = []
                for tri, vai in inner_folds(groups):
                    S_tr = self._space_fit(Xt[tri], Xs[tri])
                    S_q = self._space_q(Xt[vai], Xs[vai])
                    oof.append(self._point_scores(
                        S_tr, groups[tri], phi[tri],
                        S_q, groups[vai], phi[vai], k, h))
                oof = np.concatenate(oof)
                sc = mean_logscore(oof)
                if sc > best:
                    best, best_p = sc, {'k': k, 'h_phi': h}
                    oof_best = oof
        return best_p, oof_best

    def fit(self, Xt, Xs, phi, groups, params, seed=0):
        self.S_tr = self._space_fit(np.asarray(Xt, float),
                                    np.asarray(Xs, float))
        self.phi_tr = np.asarray(phi, float)
        self.g_tr = np.asarray(groups)
        self.k = int(params['k'])
        self.h = float(params['h_phi'])
        self.info = {'k': self.k, 'h_phi': self.h}
        return self

    def point_logpdf(self, Xt_q, Xs_q, g_q, phi_q):
        S_q = self._space_q(np.asarray(Xt_q, float),
                            np.asarray(Xs_q, float))
        return self._point_scores(self.S_tr, self.g_tr, self.phi_tr,
                                  S_q, g_q, np.asarray(phi_q, float),
                                  self.k, self.h)

    def grid_density(self, Xt_q, Xs_q, g_q, grid):
        S_q = self._space_q(np.asarray(Xt_q, float),
                            np.asarray(Xs_q, float))
        idx, _ = knn(self.S_tr, self.g_tr, S_q, g_q, self.k)
        G = np.asarray(grid, float)
        out = np.empty((len(idx), len(G)))
        for r, ii in enumerate(idx):
            ii = ii[ii >= 0]
            if len(ii) == 0:
                out[r] = np.full(len(G), 1.0 / (G[-1] - G[0]))
                continue
            out[r] = gauss1d(G[None, :], self.phi_tr[ii][:, None],
                             self.h).mean(0) + FLOOR
        return out


class B2NNKCDE(_NNKCDE):
    name = 'B2'


class B3NNKCDEMaha(_NNKCDE):
    name = 'B3'

    def _whiten(self, Xt_tr):
        lw = LedoitWolf().fit(Xt_tr)
        prec = lw.precision_
        # stable symmetric whitening via eigendecomposition
        w, V = np.linalg.eigh((prec + prec.T) / 2.0)
        w = np.clip(w, 1e-12, None)
        W = V @ np.diag(np.sqrt(w)) @ V.T
        mu = Xt_tr.mean(0)
        return {'W': W, 'mu': mu,
                'cond': float(np.max(w) / np.min(w))}

    def _space_fit(self, Xt_tr, Xs_tr):
        self._wit = self._whiten(np.asarray(Xt_tr, float))
        return (np.asarray(Xt_tr, float) - self._wit['mu']) @ self._wit['W']

    def _space_q(self, Xt_q, Xs_q):
        return (np.asarray(Xt_q, float) - self._wit['mu']) @ self._wit['W']

    def fit(self, Xt, Xs, phi, groups, params, seed=0):
        super().fit(Xt, Xs, phi, groups, params, seed=seed)
        self.info['lw_cond'] = float(self._wit['cond'])
        return self


# --------------------------------------------------------------------------
# B4 -- adaptive NW kernel conditional density estimator
# --------------------------------------------------------------------------

class B4AdaptiveKCDE(CDEMethod):
    name = 'B4'

    def __init__(self, hxs=(0.5, 1.0, 2.0, 4.0),
                 hphis=(0.75, 1.5, 2.5),
                 adks=(25, 50, 100), adss=(0.75, 1.0, 1.5)):
        self.hxs = list(hxs)
        self.hphis = list(hphis)
        self.adks = list(adks)
        self.adss = list(adss)

    def configs(self):
        cfgs = [{'mode': 'fixed', 'h_x': hx, 'h_phi': hp}
                for hx in self.hxs for hp in self.hphis]
        cfgs += [{'mode': 'adaptive', 'k_ad': k, 'scale': s, 'h_phi': hp}
                 for k in self.adks for s in self.adss for hp in self.hphis]
        return cfgs

    @staticmethod
    def _weights(D, g_tr, g_q, cfg):
        D = apply_group_exclusion(np.array(D, float, copy=True), g_tr, g_q)
        if cfg['mode'] == 'fixed':
            hx = max(float(cfg['h_x']), 1e-9)
            W = np.exp(-0.5 * (D / hx) ** 2)
        else:
            k = int(cfg['k_ad'])
            s = float(cfg['scale'])
            Dk = np.sort(D, axis=1, kind='stable')[:, min(k, D.shape[1] - 1)]
            Dk = np.where(np.isfinite(Dk), Dk, np.nanmax(
                np.where(np.isfinite(D), D, 0.0), axis=1) + 1.0)
            hx = np.maximum(Dk * s, 1e-9)[:, None]
            W = np.exp(-0.5 * (D / hx) ** 2)
        W[~np.isfinite(D)] = 0.0
        return W

    def _point_scores(self, Xs_tr, g_tr, phi_tr, Xs_q, g_q, phi_q, cfg):
        D = euclid2(Xs_q, Xs_tr)
        W = self._weights(D, g_tr, g_q, cfg)
        h = float(cfg['h_phi'])
        out = np.empty(len(phi_q))
        for r in range(len(phi_q)):
            w = W[r]
            sw = w.sum()
            if sw < 1e-8:
                out[r] = np.log(FLOOR)
                continue
            out[r] = np.log((w * gauss1d(phi_q[r], phi_tr, h)).sum()
                            / sw + FLOOR)
        return out

    def tune(self, Xt, Xs, phi, groups, seed=0):
        Xs = np.asarray(Xs, float)
        phi = np.asarray(phi, float)
        groups = np.asarray(groups)
        best, best_p, oof_best = -np.inf, None, None
        for cfg in self.configs():
            oof = []
            for tri, vai in inner_folds(groups):
                oof.append(self._point_scores(
                    Xs[tri], groups[tri], phi[tri],
                    Xs[vai], groups[vai], phi[vai], cfg))
            oof = np.concatenate(oof)
            sc = mean_logscore(oof)
            if sc > best:
                best, best_p, oof_best = sc, dict(cfg), oof
        return best_p, oof_best

    def fit(self, Xt, Xs, phi, groups, params, seed=0):
        self.Xs = np.asarray(Xs, float)
        self.phi_tr = np.asarray(phi, float)
        self.g_tr = np.asarray(groups)
        self.cfg = dict(params)
        self.info = dict(params)
        self.n_unstable = 0
        return self

    def _fit_weights(self, Xs_q, g_q):
        D = euclid2(np.asarray(Xs_q, float), self.Xs)
        return self._weights(D, self.g_tr, g_q, self.cfg)

    def point_logpdf(self, Xt_q, Xs_q, g_q, phi_q):
        W = self._fit_weights(Xs_q, g_q)
        h = float(self.cfg['h_phi'])
        phi_q = np.asarray(phi_q, float)
        out = np.empty(len(phi_q))
        for r in range(len(phi_q)):
            w = W[r]
            sw = w.sum()
            if sw < 1e-8:
                out[r] = np.log(FLOOR)
                continue
            out[r] = np.log((w * gauss1d(phi_q[r], self.phi_tr, h)).sum()
                            / sw + FLOOR)
        return out

    def grid_density(self, Xt_q, Xs_q, g_q, grid):
        W = self._fit_weights(Xs_q, g_q)
        h = float(self.cfg['h_phi'])
        G = np.asarray(grid, float)
        K = gauss1d(G[None, :], self.phi_tr[:, None], h)  # (ntr, G)
        sw = W.sum(1)
        bad = sw < 1e-8
        self.n_unstable += int(bad.sum())
        sw = np.where(bad, 1.0, sw)
        out = (W @ K) / sw[:, None] + FLOOR
        if bad.any():
            glob = K.mean(0) + FLOOR
            out[bad] = glob[None, :]
        return out


# --------------------------------------------------------------------------
# B5 -- joint KDE converted to conditional (direct nonparametric benchmark)
# --------------------------------------------------------------------------

class B5JointKDE(CDEMethod):
    name = 'B5'

    def __init__(self, bws=(0.5, 0.75, 1.0, 1.5, 2.0)):
        self.bws = list(bws)

    def _joint_ll(self, Xv, pv_std, X_tr, p_tr_std, h):
        D2 = euclid2(Xv, X_tr) + (pv_std[:, None] - p_tr_std[None, :]) ** 2
        d = Xv.shape[1] + 1
        ll = logsumexp(-0.5 * D2 / h ** 2, axis=1) - np.log(X_tr.shape[0]) \
            - 0.5 * d * np.log(2 * np.pi * h ** 2)
        return ll

    def tune(self, Xt, Xs, phi, groups, seed=0):
        Xs = np.asarray(Xs, float)
        phi = np.asarray(phi, float)
        groups = np.asarray(groups)
        best, best_p, oof_best = -np.inf, None, None
        for h in self.bws:
            oof_joint, oof_cond = [], []
            for tri, vai in inner_folds(groups):
                mu, sd = phi[tri].mean(), phi[tri].std() + 1e-12
                ptr = (phi[tri] - mu) / sd
                pv = (phi[vai] - mu) / sd
                oof_joint.append(self._joint_ll(Xs[vai], pv, Xs[tri],
                                                ptr, h))
                # conditional score of the derived conditional
                w = np.exp(-0.5 * apply_group_exclusion(
                    euclid2(Xs[vai], Xs[tri]), groups[tri],
                    groups[vai]) / h ** 2)
                sw = w.sum(1)
                hp = h * sd
                for r in range(len(vai)):
                    if sw[r] < 1e-8:
                        oof_cond.append(np.log(FLOOR))
                        continue
                    oof_cond.append(np.log(
                        (w[r] * gauss1d(phi[vai[r]], phi[tri], hp)).sum()
                        / sw[r] + FLOOR))
            oof_joint = np.concatenate(oof_joint)
            oof_cond = np.array(oof_cond)
            sc = mean_logscore(oof_joint)  # selection by JOINT likelihood
            if sc > best:
                best, best_p = sc, {'h': h}
                oof_best = oof_cond  # stacking uses conditional scores
        return best_p, oof_best

    def fit(self, Xt, Xs, phi, groups, params, seed=0):
        self.Xs = np.asarray(Xs, float)
        self.phi_tr = np.asarray(phi, float)
        self.g_tr = np.asarray(groups)
        self.h = float(params['h'])
        self.mu = float(self.phi_tr.mean())
        self.sd = float(self.phi_tr.std() + 1e-12)
        self.h_phi = self.h * self.sd
        self.info = {'h': self.h, 'h_phi_orig_units': self.h_phi}
        self.n_unstable = 0
        return self

    def _weights(self, Xs_q, g_q):
        D = apply_group_exclusion(euclid2(np.asarray(Xs_q, float), self.Xs),
                                  self.g_tr, g_q)
        W = np.exp(-0.5 * D / self.h ** 2)
        W[~np.isfinite(D)] = 0.0
        return W

    def point_logpdf(self, Xt_q, Xs_q, g_q, phi_q):
        W = self._weights(Xs_q, g_q)
        phi_q = np.asarray(phi_q, float)
        out = np.empty(len(phi_q))
        for r in range(len(phi_q)):
            sw = W[r].sum()
            if sw < 1e-8:
                out[r] = np.log(FLOOR)
                continue
            out[r] = np.log((W[r] * gauss1d(phi_q[r], self.phi_tr,
                                            self.h_phi)).sum() / sw
                            + FLOOR)
        return out

    def grid_density(self, Xt_q, Xs_q, g_q, grid):
        W = self._weights(Xs_q, g_q)
        G = np.asarray(grid, float)
        K = gauss1d(G[None, :], self.phi_tr[:, None], self.h_phi)
        sw = W.sum(1)
        bad = sw < 1e-8
        self.n_unstable += int(bad.sum())
        sw = np.where(bad, 1.0, sw)
        out = (W @ K) / sw[:, None] + FLOOR
        if bad.any():
            out[bad] = K.mean(0)[None, :] + FLOOR
        return out


# --------------------------------------------------------------------------
# B6 -- joint GMM with analytic conditionals
# --------------------------------------------------------------------------

class B6JointGMM(CDEMethod):
    name = 'B6'

    def __init__(self, Ks=(1, 2, 3, 4, 6, 8), reg_covar=1e-3, n_init=3):
        self.Ks = list(Ks)
        self.reg_covar = float(reg_covar)
        self.n_init = int(n_init)

    def _fit_gmm(self, Z, K, seed):
        gm = GaussianMixture(n_components=int(K), covariance_type='full',
                             reg_covar=self.reg_covar, n_init=self.n_init,
                             random_state=int(seed), max_iter=500)
        with np.errstate(all='ignore'):
            gm.fit(Z)
        return gm

    def tune(self, Xt, Xs, phi, groups, seed=0):
        Xs = np.asarray(Xs, float)
        phi = np.asarray(phi, float)
        groups = np.asarray(groups)
        best, best_p, oof_best = -np.inf, None, None
        bic_table = {}
        for K in self.Ks:
            oof, bics, ok = [], [], True
            for fi, (tri, vai) in enumerate(inner_folds(groups)):
                mu, sd = phi[tri].mean(), phi[tri].std() + 1e-12
                Ztr = np.hstack([Xs[tri], ((phi[tri] - mu) / sd)[:, None]])
                Zva = np.hstack([Xs[vai], ((phi[vai] - mu) / sd)[:, None]])
                try:
                    gm = self._fit_gmm(Ztr, K, seed + fi)
                    if not np.isfinite(gm.lower_bound_):
                        ok = False
                        break
                    ll = gm.score_samples(Zva) - np.log(sd)
                    oof.append(ll)
                    bics.append(gm.bic(Ztr))
                except Exception:
                    ok = False
                    break
            if not ok:
                bic_table[int(K)] = {'bic': None, 'heldout_ll': None,
                                     'failed': True}
                continue
            oof = np.concatenate(oof)
            sc = mean_logscore(oof)
            bic_table[int(K)] = {'bic': float(np.mean(bics)),
                                 'heldout_ll': sc, 'failed': False}
            if sc > best:
                best, best_p, oof_best = sc, {'K': int(K)}, oof
        self._last_bic = bic_table
        return best_p, oof_best

    def fit(self, Xt, Xs, phi, groups, params, seed=0):
        Xs = np.asarray(Xs, float)
        self.phi_tr = np.asarray(phi, float)
        self.mu = float(self.phi_tr.mean())
        self.sd = float(self.phi_tr.std() + 1e-12)
        Z = np.hstack([Xs, ((self.phi_tr - self.mu) / self.sd)[:, None]])
        K = int(params['K'])
        self.gm = self._fit_gmm(Z, K, seed)
        assert np.isfinite(self.gm.lower_bound_), 'GMM did not converge'
        d = Xs.shape[1]
        self.d = d
        self.pi = self.gm.weights_
        self.mux = self.gm.means_[:, :d]
        self.mup = self.gm.means_[:, d]
        S = self.gm.covariances_
        assert np.isfinite(S).all(), 'degenerate GMM covariance'
        self.Sxx_inv = np.linalg.inv(S[:, :d, :d])
        self.Sxp = S[:, :d, d]
        self.Spp = S[:, d, d]
        self.info = {'K': K, 'bic': float(self.gm.bic(Z)),
                     'converged': bool(self.gm.converged_),
                     'n_iter': int(self.gm.n_iter_)}
        return self

    def _cond_params(self, Xq):
        Xq = np.asarray(Xq, float)
        nq = Xq.shape[0]
        K = len(self.pi)
        logw = np.empty((nq, K))
        mc = np.empty((nq, K))
        vc = np.empty((nq, K))
        for c in range(K):
            logw[:, c] = (np.log(self.pi[c])
                           + multivariate_normal.logpdf(
                               Xq, mean=self.mux[c],
                               cov=self.gm.covariances_[c][:self.d, :self.d],
                               allow_singular=False))
            dx = Xq - self.mux[c]
            A = dx @ self.Sxx_inv[c]
            mc[:, c] = self.mup[c] + A @ self.Sxp[c]
            vc[:, c] = max(float(self.Spp[c] - self.Sxp[c] @
                                 self.Sxx_inv[c] @ self.Sxp[c]), 1e-6)
        logw -= logsumexp(logw, axis=1, keepdims=True)
        W = np.exp(logw)
        # back to original phi units
        mc = mc * self.sd + self.mu
        vc = vc * self.sd ** 2
        return W, mc, vc

    def point_logpdf(self, Xt_q, Xs_q, g_q, phi_q):
        W, mc, vc = self._cond_params(Xs_q)
        phi_q = np.asarray(phi_q, float)
        sd = np.sqrt(vc)
        z = (phi_q[:, None] - mc) / sd
        comp = np.exp(-0.5 * z ** 2) / (sd * np.sqrt(2 * np.pi))
        return np.log((W * comp).sum(1) + FLOOR)

    def grid_density(self, Xt_q, Xs_q, g_q, grid):
        W, mc, vc = self._cond_params(Xs_q)
        G = np.asarray(grid, float)
        sd = np.sqrt(vc)
        z = (G[None, None, :] - mc[:, :, None]) / sd[:, :, None]
        comp = np.exp(-0.5 * z ** 2) / (sd[:, :, None] * np.sqrt(2 * np.pi))
        return (W[:, :, None] * comp).sum(1) + FLOOR

    def cond_weights(self, Xs_q):
        W, _, _ = self._cond_params(Xs_q)
        return W


# --------------------------------------------------------------------------
# B7 -- local GMM / NNK-GMM hybrid (1-D GMM on neighbor porosities)
# --------------------------------------------------------------------------

class B7LocalGMM(CDEMethod):
    name = 'B7'

    def __init__(self, ks=(30, 50, 80, 120), local_Ks=(1, 2, 3),
                 min_n_per_mode=15, n_init=3):
        self.ks = list(ks)
        self.local_Ks = list(local_Ks)
        self.min_n = int(min_n_per_mode)
        self.n_init = int(n_init)

    def _fit_local(self, vals, seed, n_init=None):
        vals = np.asarray(vals, float).reshape(-1, 1)
        elig = [K for K in self.local_Ks if len(vals) >= self.min_n * K]
        if not elig:
            elig = [1]
        best, best_gm = np.inf, None
        for K in elig:
            gm = GaussianMixture(n_components=K, covariance_type='full',
                                 reg_covar=1e-6,
                                 n_init=self.n_init if n_init is None
                                 else n_init,
                                 random_state=int(seed), max_iter=500)
            with np.errstate(all='ignore'):
                gm.fit(vals)
            bic = gm.bic(vals)
            if np.isfinite(bic) and bic < best:
                best, best_gm = bic, gm
        return best_gm

    def _point_scores(self, Xs_tr, g_tr, phi_tr, Xs_q, g_q, phi_q, k,
                      seed):
        idx, _ = knn(Xs_tr, g_tr, Xs_q, g_q, k)
        out = np.empty(len(phi_q))
        for r, ii in enumerate(idx):
            ii = ii[ii >= 0]
            if len(ii) < self.min_n:
                out[r] = np.log(FLOOR)
                continue
            # tuning only: single restart (final fit uses full n_init)
            gm = self._fit_local(phi_tr[ii], seed + r, n_init=1)
            out[r] = float(gm.score_samples(np.array([[phi_q[r]]]))[0])
        return out

    def tune(self, Xt, Xs, phi, groups, seed=0):
        Xs = np.asarray(Xs, float)
        phi = np.asarray(phi, float)
        groups = np.asarray(groups)
        best, best_p, oof_best, pos_best = -np.inf, None, None, None
        for k in self.ks:
            oof, pos = [], []
            for tri, vai in inner_folds(groups):
                # deterministic every-2nd subsample: tuning cost control;
                # still grouped held-out log-score, documented
                sub = vai[::2]
                oof.append(self._point_scores(
                    Xs[tri], groups[tri], phi[tri],
                    Xs[sub], groups[sub], phi[sub], k, seed))
                pos.append(sub)
            oof = np.concatenate(oof)
            pos = np.concatenate(pos)
            sc = mean_logscore(oof)
            if sc > best:
                best, best_p, oof_best, pos_best = sc, {'k': k}, oof, pos
        return best_p, oof_best, pos_best

    def fit(self, Xt, Xs, phi, groups, params, seed=0):
        self.Xs = np.asarray(Xs, float)
        self.phi_tr = np.asarray(phi, float)
        self.g_tr = np.asarray(groups)
        self.k = int(params['k'])
        self.seed = int(seed)
        self.info = {'k': self.k}
        return self

    def _mixtures(self, Xs_q, g_q):
        idx, _ = knn(self.Xs, self.g_tr, np.asarray(Xs_q, float),
                     g_q, self.k)
        gms = []
        for r, ii in enumerate(idx):
            ii = ii[ii >= 0]
            if len(ii) < self.min_n:
                gms.append(None)
                continue
            gms.append(self._fit_local(self.phi_tr[ii], self.seed + r))
        return gms

    def point_logpdf(self, Xt_q, Xs_q, g_q, phi_q):
        gms = self._mixtures(Xs_q, g_q)
        phi_q = np.asarray(phi_q, float)
        out = np.empty(len(phi_q))
        for r, gm in enumerate(gms):
            if gm is None:
                out[r] = np.log(FLOOR)
                continue
            out[r] = float(gm.score_samples(np.array([[phi_q[r]]]))[0])
        return out

    def grid_density(self, Xt_q, Xs_q, g_q, grid):
        gms = self._mixtures(Xs_q, g_q)
        G = np.asarray(grid, float)
        out = np.empty((len(gms), len(G)))
        for r, gm in enumerate(gms):
            if gm is None:
                out[r] = np.full(len(G), 1.0 / (G[-1] - G[0]))
                continue
            out[r] = np.exp(gm.score_samples(G.reshape(-1, 1))) + FLOOR
        return out


# --------------------------------------------------------------------------
# B8 -- stacked conditional-density ensemble (density averaging only)
# --------------------------------------------------------------------------

class B8Stack(CDEMethod):
    name = 'B8'

    def __init__(self, members):
        self.members = list(members)  # fitted CDEMethod instances

    def tune(self, *a, **k):
        raise RuntimeError('B8 weights come from members inner-OOF scores')

    def fit_weights(self, oof_dict, pos_dict=None):
        """Simplex weights maximizing pooled inner-OOF log-score.

        oof_dict: {member_name: (n,) pointwise log-scores}.
        pos_dict: optional {member_name: FIT-relative positions}; weights
            are optimized on the intersection of evaluated points.
        ONLY inner out-of-fold predictions are used (never CAL/test).
        """
        from scipy.optimize import minimize
        names = [m.name for m in self.members]
        if pos_dict is None:
            common = None
            S = np.vstack([np.asarray(oof_dict[nm], float) for nm in names])
        else:
            maps = []
            for nm in names:
                pp = np.asarray(pos_dict[nm]).tolist()
                ss = np.asarray(oof_dict[nm], float).tolist()
                maps.append({p: s for p, s in zip(pp, ss)})
            common = sorted(set.intersection(
                *[set(mm.keys()) for mm in maps]))
            assert len(common) >= 30, 'too few common OOF points'
            S = np.vstack([[mm[p] for p in common] for mm in maps])
        S = np.where(np.isfinite(S), S, np.log(FLOOR))
        best_single = int(np.argmax(S.mean(1)))
        m = len(names)

        def neg_ll(v):
            w = np.exp(v - logsumexp(v))
            mix = logsumexp(np.log(np.maximum(w, 1e-300))[:, None] + S,
                            axis=0)
            return -float(mix.mean())

        v0 = np.zeros(m)
        try:
            res = minimize(neg_ll, v0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-4,
                                    'fatol': 1e-6})
            w = np.exp(res.x - logsumexp(res.x))
        except Exception:
            w = np.zeros(m)
            w[best_single] = 1.0
        # verify improvement over best single constituent
        mix = logsumexp(np.log(np.maximum(w, 1e-300))[:, None] + S,
                        axis=0)
        gain = float(mix.mean() - S[best_single].mean())
        if gain <= 0:
            w = np.zeros(m)
            w[best_single] = 1.0
            gain = 0.0
        self.weights_ = w
        self.member_names_ = names
        self.oof_gain_ = gain
        self.best_single_ = names[best_single]
        self.info = {'weights': {n: float(v) for n, v in zip(names, w)},
                     'oof_gain_over_best': gain,
                     'best_single': names[best_single]}
        assert abs(w.sum() - 1.0) < 1e-9 and (w >= 0).all()
        return self

    def fit(self, *a, **k):
        return self  # members already fit; nothing else to fit

    def _stack(self, mats):
        w = self.weights_
        out = sum(wi * np.asarray(mt, float)
                  for wi, mt in zip(w, mats))
        return out

    def point_logpdf(self, Xt_q, Xs_q, g_q, phi_q):
        mats = [np.exp(m.point_logpdf(Xt_q, Xs_q, g_q, phi_q))
                for m in self.members]
        return np.log(self._stack(mats) + FLOOR)

    def grid_density(self, Xt_q, Xs_q, g_q, grid):
        mats = [m.grid_density(Xt_q, Xs_q, g_q, grid)
                for m in self.members]
        return self._stack(mats) + FLOOR


# --------------------------------------------------------------------------
# S1 / S2 / S3 -- secondary joint-state anomaly sections
# --------------------------------------------------------------------------

class _SectionMethod:
    name = 'S?'

    def tune(self, Xs, phi, groups, seed=0):
        raise NotImplementedError

    def fit(self, Xs, phi, groups, params, seed=0):
        raise NotImplementedError

    def obs_scores(self, Xs_q, phi_q):
        """Anomaly score (higher = more supported) at observed states."""
        raise NotImplementedError

    def grid_scores(self, Xs_q, grid):
        """(nq, G) scores scanning phi at fixed x."""
        raise NotImplementedError


def _retention_pick(results):
    """Pick (retention closest to 0.90, tie-break decoy rejection)."""
    best = None
    for r in results:
        key = (abs(r['ret'] - 0.90), -r['decoy'])
        if best is None or key < best[0]:
            best = (key, r)
    return best[1]['params']


def _tune_sections(factory, Xs, phi, groups, seed, shifts=(5.0, 8.0)):
    Xs = np.asarray(Xs, float)
    phi = np.asarray(phi, float)
    groups = np.asarray(groups)
    results = []
    for params in factory.grid():
        rets, decs = [], []
        for tri, vai in inner_folds(groups):
            m = factory.build(params, seed).fit_inner(Xs[tri], phi[tri])
            s_tr = m.obs_scores_inner(Xs[tri], phi[tri])
            t = float(np.quantile(s_tr, 0.10))
            s_va = m.obs_scores_inner(Xs[vai], phi[vai])
            rets.append(float((s_va >= t).mean()))
            dj = []
            for s in shifts:
                for dp in (phi[vai] + s, phi[vai] - s):
                    dj.append(float((m.obs_scores_inner(Xs[vai], dp)
                                     >= t).mean()))
            decs.append(1.0 - float(np.mean(dj)))
        results.append({'params': params, 'ret': float(np.mean(rets)),
                        'decoy': float(np.mean(decs))})
    return _retention_pick(results), results


class S1OCSVM(_SectionMethod):
    name = 'S1'

    def __init__(self, nus=(0.05, 0.1, 0.2),
                 gammas=('scale/10', 'scale', 'scale*10')):
        self.nus = list(nus)
        self.gammas = list(gammas)

    def grid(self):
        return [{'nu': nu, 'gamma': g} for nu in self.nus
                for g in self.gammas]

    def build(self, params, seed):
        return _S1Fit(params)

    def tune(self, Xs, phi, groups, seed=0):
        return _tune_sections(self, Xs, phi, groups, seed)

    def fit(self, Xs, phi, groups, params, seed=0):
        Xs = np.asarray(Xs, float)
        phi = np.asarray(phi, float)
        mu, sd = phi.mean(), phi.std() + 1e-12
        Z = np.hstack([Xs, ((phi - mu) / sd)[:, None]])
        g = params['gamma']
        gamma = g if isinstance(g, str) else float(g)
        if isinstance(gamma, str):
            base = 1.0 / (Z.shape[1] * Z.var())
            gamma = {'scale/10': base / 10.0, 'scale': base,
                     'scale*10': base * 10.0}[gamma]
        self.clf = OneClassSVM(nu=float(params['nu']), gamma=float(gamma))
        self.clf.fit(Z)
        self.mu, self.sd = float(mu), float(sd)
        self.info = dict(params)
        return self

    def _Z(self, Xs_q, phi_v):
        Xs_q = np.asarray(Xs_q, float)
        phi_v = np.asarray(phi_v, float)
        n = Xs_q.shape[0]
        if phi_v.ndim == 0:
            P = np.full(n, float(phi_v))
        elif phi_v.shape == (n,):
            P = phi_v
        else:
            raise ValueError('phi shape')
        return np.hstack([Xs_q, ((P - self.mu) / self.sd)[:, None]])

    def obs_scores(self, Xs_q, phi_q):
        return self.clf.decision_function(self._Z(Xs_q, phi_q))

    def grid_scores(self, Xs_q, grid):
        G = np.asarray(grid, float)
        Xs_q = np.asarray(Xs_q, float)
        out = np.empty((Xs_q.shape[0], len(G)))
        for r in range(Xs_q.shape[0]):
            Z = np.hstack([np.repeat(Xs_q[[r]], len(G), axis=0),
                           ((G - self.mu) / self.sd)[:, None]])
            out[r] = self.clf.decision_function(Z)
        return out


class _S1Fit:
    """Inner-fold OCSVM twin for tuning (same construction)."""

    def __init__(self, params):
        self.params = params

    def fit_inner(self, Xtr, ptr):
        m = S1OCSVM()
        m.fit(Xtr, ptr, np.zeros(len(ptr), dtype=int), self.params,
              seed=0)
        self._m = m
        return self

    def obs_scores_inner(self, Xq, pq):
        return self._m.obs_scores(Xq, pq)


class S2IForest(_SectionMethod):
    name = 'S2'

    def __init__(self, max_samples=(0.5, 1.0),
                 contaminations=(0.05, 0.1, 0.2)):
        self.max_samples = list(max_samples)
        self.contaminations = list(contaminations)

    def grid(self):
        return [{'max_samples': ms, 'contamination': c}
                for ms in self.max_samples for c in self.contaminations]

    def build(self, params, seed):
        return _S2Fit(params, seed)

    def tune(self, Xs, phi, groups, seed=0):
        return _tune_sections(self, Xs, phi, groups, seed)

    def fit(self, Xs, phi, groups, params, seed=0):
        Xs = np.asarray(Xs, float)
        phi = np.asarray(phi, float)
        mu, sd = phi.mean(), phi.std() + 1e-12
        Z = np.hstack([Xs, ((phi - mu) / sd)[:, None]])
        ms = params['max_samples']
        self.clf = IsolationForest(
            max_samples=ms, contamination=float(params['contamination']),
            random_state=int(seed))
        self.clf.fit(Z)
        self.mu, self.sd = float(mu), float(sd)
        self.info = dict(params)
        return self

    def obs_scores(self, Xs_q, phi_q):
        Xs_q = np.asarray(Xs_q, float)
        phi_q = np.asarray(phi_q, float)
        Z = np.hstack([Xs_q, ((phi_q - self.mu) / self.sd)[:, None]])
        return self.clf.score_samples(Z)

    def grid_scores(self, Xs_q, grid):
        G = np.asarray(grid, float)
        Xs_q = np.asarray(Xs_q, float)
        out = np.empty((Xs_q.shape[0], len(G)))
        for r in range(Xs_q.shape[0]):
            Z = np.hstack([np.repeat(Xs_q[[r]], len(G), axis=0),
                           ((G - self.mu) / self.sd)[:, None]])
            out[r] = self.clf.score_samples(Z)
        return out


class _S2Fit:
    def __init__(self, params, seed):
        self.params = params
        self.seed = seed

    def fit_inner(self, Xtr, ptr):
        m = S2IForest()
        m.fit(Xtr, ptr, np.zeros(len(ptr), dtype=int), self.params,
              seed=self.seed)
        self._m = m
        return self

    def obs_scores_inner(self, Xq, pq):
        return self._m.obs_scores(Xq, pq)


class S3JointKNN(_SectionMethod):
    name = 'S3'

    def __init__(self, ks=(20, 50, 100)):
        self.ks = list(ks)

    def grid(self):
        return [{'k': k} for k in self.ks]

    def build(self, params, seed):
        return _S3Fit(params)

    def tune(self, Xs, phi, groups, seed=0):
        return _tune_sections(self, Xs, phi, groups, seed)

    def fit(self, Xs, phi, groups, params, seed=0):
        Xs = np.asarray(Xs, float)
        phi = np.asarray(phi, float)
        mu, sd = phi.mean(), phi.std() + 1e-12
        self.Ztr = np.hstack([Xs, ((phi - mu) / sd)[:, None]])
        self.k = int(params['k'])
        self.mu, self.sd = float(mu), float(sd)
        self.info = dict(params)
        self.nn = NearestNeighbors(
            n_neighbors=min(self.k, len(self.Ztr))).fit(self.Ztr)
        return self

    def _score(self, Zq):
        d, _ = self.nn.kneighbors(np.asarray(Zq, float))
        return -d.mean(1)

    def obs_scores(self, Xs_q, phi_q):
        Xs_q = np.asarray(Xs_q, float)
        phi_q = np.asarray(phi_q, float)
        Zq = np.hstack([Xs_q, ((phi_q - self.mu) / self.sd)[:, None]])
        return self._score(Zq)

    def grid_scores(self, Xs_q, grid):
        G = np.asarray(grid, float)
        Xs_q = np.asarray(Xs_q, float)
        out = np.empty((Xs_q.shape[0], len(G)))
        for r in range(Xs_q.shape[0]):
            Zq = np.hstack([np.repeat(Xs_q[[r]], len(G), axis=0),
                            ((G - self.mu) / self.sd)[:, None]])
            out[r] = self._score(Zq)
        return out


class _S3Fit:
    def __init__(self, params):
        self.params = params

    def fit_inner(self, Xtr, ptr):
        m = S3JointKNN()
        m.fit(Xtr, ptr, np.zeros(len(ptr), dtype=int), self.params,
              seed=0)
        self._m = m
        return self

    def obs_scores_inner(self, Xq, pq):
        return self._m.obs_scores(Xq, pq)


def make_density_method(name, config_grids):
    """Factory honoring config.json grids."""
    g = config_grids
    if name == 'B0':
        return B0GlobalKDE(scales=g['B0_bw_scale'])
    if name == 'B1':
        return B1KNN(ks=g['B1_k'])
    if name == 'B2':
        return B2NNKCDE(ks=g['B2_k'], hphis=g['B2_phi_bw'])
    if name == 'B3':
        return B3NNKCDEMaha(ks=g['B3_k'], hphis=g['B3_phi_bw'])
    if name == 'B4':
        return B4AdaptiveKCDE(hxs=g['B4_fixed_recipe_bw'],
                              hphis=g['B4_phi_bw'],
                              adks=g['B4_adaptive_k'],
                              adss=g['B4_adaptive_scale'])
    if name == 'B5':
        return B5JointKDE(bws=g['B5_joint_bw'])
    if name == 'B6':
        return B6JointGMM(Ks=g['B6_K'], reg_covar=g['B6_reg_covar'])
    if name == 'B7':
        return B7LocalGMM(ks=g['B7_k'], local_Ks=g['B7_local_K'],
                          min_n_per_mode=g['B7_min_n_per_mode'])
    raise ValueError(name)


def make_section_method(name, config_grids):
    g = config_grids
    if name == 'S1':
        return S1OCSVM(nus=g['S1_nu'], gammas=g['S1_gamma'])
    if name == 'S2':
        return S2IForest(max_samples=g['S2_max_samples'],
                         contaminations=g['S2_contamination'])
    if name == 'S3':
        return S3JointKNN(ks=g['S3_k'])
    raise ValueError(name)
