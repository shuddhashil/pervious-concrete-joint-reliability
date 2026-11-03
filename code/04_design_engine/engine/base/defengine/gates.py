"""Support gates for the new engine.

EXACT, cheap, vectorised (identical decisions to the frozen engine): recipe gate, deployment window, W1 gradation gate.
NEW, cheap SCREEN for the porosity-plausibility rule: a locally adaptive split-conformal interval (grouped by study):
   interval(x) = median(neighbour porosity) +/- t * (MAD(neighbour porosity) + eps)
   neighbours = k nearest training recipes in the frozen standardised recipe space; t = conformal quantile of the normalised absolute
   residuals computed with SAME-STUDY neighbours excluded (so calibration is held-out by study, the strictest grouping available).
The frozen six-estimator porosity density (stage17) stays the FINAL authority: every finalist is re-checked with it (verify.py); the screen
only prunes the search. Its agreement with the frozen rule is measured in tests/test_gates_agreement.py (decision E-04)."""
import os, numpy as np, pandas as pd
from . import corelib as C

S17, GS = C.S17, C.GS


def _train_recipe_space():
    d = pd.read_csv(os.path.join(str(C.CORE), "data_inputs", "phase1_reconciled_eligible.csv"))
    B = S17._bundle()
    bins = d[GS.BIN_COLS].to_numpy(float); bins = bins / bins.sum(1, keepdims=True)
    A = d["CA (kg/m3)"].to_numpy(float)
    Xraw = np.column_stack([d["w/c"], d["cem (kg/m3)"], d["SP (kg/m3)"], d["sand (kg/m3)"], bins * A[:, None]])
    Xs = (S17.transform_xraw(Xraw) - B["scaler_mean"]) / B["scaler_sd"]
    return Xs, d["porosity"].to_numpy(float), d["_src_ffill"].to_numpy()


class CheapPorosityGate:
    def __init__(self, k=30, alpha=0.10, eps=1.0):
        self.k, self.alpha, self.eps = k, alpha, eps
        self.Xs, self.phi, self.grp = _train_recipe_space()
        from sklearn.neighbors import NearestNeighbors
        self.nn = NearestNeighbors(n_neighbors=min(len(self.Xs), k + 80)).fit(self.Xs)   # extra neighbours so same-study exclusion still leaves k
        self.t = self._calibrate()

    def _local(self, idx_row):
        v = self.phi[idx_row]; med = np.median(v); mad = np.median(np.abs(v - med)); return med, mad

    def _calibrate(self):
        _, I = self.nn.kneighbors(self.Xs)
        r = []
        for i in range(len(self.Xs)):
            keep = [j for j in I[i] if self.grp[j] != self.grp[i]][: self.k]          # held out by study
            med, mad = self._local(keep); r.append(abs(self.phi[i] - med) / (mad + self.eps))
        r = np.sort(np.array(r)); n = len(r); kq = int(np.ceil((n + 1) * (1 - self.alpha)))
        return float(r[min(kq, n) - 1])

    def intervals(self, Xs_query):
        _, I = self.nn.kneighbors(Xs_query)
        out = np.empty((len(Xs_query), 2))
        for i, row in enumerate(I):
            med, mad = self._local(row[: self.k]); w = self.t * (mad + self.eps); out[i] = (med - w, med + w)
        return out

    def check(self, Xraw, phi):
        """Xraw: (n,10) frozen recipe coordinates; phi: (n,) target porosity. Returns (inside_bool, distance_pp_to_interval)."""
        B = S17._bundle()
        Xs = (S17.transform_xraw(Xraw) - B["scaler_mean"]) / B["scaler_sd"]
        iv = self.intervals(Xs); phi = np.asarray(phi, float)
        dist = np.maximum(np.maximum(iv[:, 0] - phi, phi - iv[:, 1]), 0.0)
        return dist <= 0.0, dist


def nominal_gates_exact(nominals):
    """Exact vectorised gates: returns dict of arrays (recipe_pass, recipe_excess, deployment_ok, w1_violation, w1_pass)."""
    Xraw = np.array([S17.candidate_to_xraw(c)[0] for c in nominals])
    rs = S17.recipe_support_batch(Xraw)
    phi = np.array([float(c["phi"]) for c in nominals])
    G = np.array([c["g"] for c in nominals], float)
    gc = GS.load_contract(); eps = gc.get("epsilon_95")
    T, _ = GS.load_bundle(); w1m = GS.w1_pairwise(G, T).min(axis=1)
    v_w = np.maximum(0.0, (w1m - eps) / eps)
    return dict(Xraw=Xraw, recipe_pass=np.array([r["recipe_pass"] for r in rs]), recipe_excess=np.array([r["recipe_excess"] for r in rs]),
                deployment_ok=(phi >= S17.DEPLOY_LO) & (phi <= S17.DEPLOY_HI), phi=phi, w1_violation=v_w, w1_pass=v_w <= 0.0, w1_min=w1m)


# ----------------------------------------------------------------------------------------------------------------------
# Search gate = frozen rule with the slow local-mixture member replaced by a single Gaussian (decision E-04). Frozen gate = reference.
class SearchGate:
    """Vectorised penalty for a population. Penalty components mirror full_de_search.state_support_penalty: v_r + v_phi/10 + v_deploy/10 + v_w1."""
    def __init__(self):
        S17._bundle()
        import sys
        self.CD = sys.modules["conditional_density"]
        B = S17._bundle(); self.B = B
        self.names = [n for n in B["member_names"] if n not in ("B3", "B4", "B7")]
        self.w = {n: float(x) for n, x in zip(B["member_names"], B["weights"])}
        self.m7 = B["fitted_members"]["B7"]; self.grid = np.asarray(B["phi_grid"], float); self.thr = float(B["conformal_thresholds"][0.1])
        self.FLOOR = self.CD.FLOOR

    def penalties(self, nominals):
        ex = nominal_gates_exact(nominals); n = len(nominals)
        Xt = S17.transform_xraw(ex["Xraw"]); Xs = (Xt - self.B["scaler_mean"]) / self.B["scaler_sd"]
        D = np.zeros((n, len(self.grid)))
        for nm in self.names: D += self.w[nm] * np.asarray(self.B["fitted_members"][nm].grid_density(Xt, Xs, None, self.grid), float)
        idx, _ = self.CD.knn(self.m7.Xs, self.m7.g_tr, Xs, None, self.m7.k)
        for i, ii in enumerate(idx):
            ii = ii[ii >= 0]
            if len(ii) < self.m7.min_n: d7 = np.full(len(self.grid), 1.0 / (self.grid[-1] - self.grid[0]))
            else:
                v = self.m7.phi_tr[ii]; sd = max(v.std(), 0.3); d7 = np.exp(-0.5 * ((self.grid - v.mean()) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)) + self.FLOOR
            D[i] += self.w["B7"] * d7
        v_phi = np.zeros(n); inside = np.zeros(n, bool)
        for i in range(n):
            if not ex["recipe_pass"][i]: continue                       # no HDR is fabricated for a recipe outside the evidence domain
            ivs, _ = S17._intervals_from_mask(D[i] >= self.thr, self.grid)
            ins, dist, _ = S17._dist_to_intervals(float(ex["phi"][i]), ivs); inside[i] = ins; v_phi[i] = 0.0 if ins else dist
        lo, hi = S17.DEPLOY_LO, S17.DEPLOY_HI
        v_d = np.where(ex["deployment_ok"], 0.0, np.minimum(np.abs(ex["phi"] - lo), np.abs(ex["phi"] - hi)) / 10.0)
        pen = ex["recipe_excess"] + v_phi / 10.0 + v_d + ex["w1_violation"]
        return dict(penalty=pen, recipe_pass=ex["recipe_pass"], phi_inside=inside, deployment_ok=ex["deployment_ok"], w1_pass=ex["w1_pass"], v_phi=v_phi, recipe_excess=ex["recipe_excess"])


def frozen_gate(nominal):
    """The UNMODIFIED frozen verdict for one design (what final acceptance uses). Mirrors full_de_search.state_support_penalty."""
    ss = S17.state_support(nominal)
    gc = GS.load_contract(); v_w, w1m = GS.violation(np.asarray(nominal["g"], float), gc.get("epsilon_95"))
    v_p_raw = float(ss["phi_violation"]); v_p = 0.0 if (v_p_raw != v_p_raw or ss["phi_in_conf90"]) else v_p_raw / 10.0
    lo, hi = S17.DEPLOY_LO, S17.DEPLOY_HI
    v_d = 0.0 if ss["deployment_ok"] else min(abs(float(nominal["phi"]) - lo), abs(float(nominal["phi"]) - hi)) / 10.0
    pen = float(ss["recipe_excess"]) + v_p + v_d + float(v_w)
    return dict(penalty=pen, recipe_pass=bool(ss["recipe_pass"]), porosity_hdr=bool(ss["phi_in_conf90"]), deployment_ok=bool(ss["deployment_ok"]),
                w1_gate=bool(v_w <= 0.0), all_pass=bool(pen <= 0.0 and ss["empirical_pass"] and ss["deployment_ok"]), w1_min=float(w1m))
