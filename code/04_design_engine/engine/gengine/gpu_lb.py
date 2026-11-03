"""GPU R_LCB95.
Same estimand as the frozen CPU certificate (core/src/efficient_verification.decisive_verify):
    R_LCB95 = min( Wilson 95% lower bound of the joint pass fraction (porosity-conditional residual draw per scenario),
                   5th percentile of the cluster bootstrap over the residual-bank rows of the candidate's porosity band )
with the bootstrap unit selectable: "study" (the frozen certificate), "recipe_group" or "row" (no-lab variants, reported beside it).
Nothing is fitted. Forward model = defengine.gpu.GPUEngine (already tested equal to the frozen CPU engine; A5).
Use with the GPU python; acceptance-grade runs call eng.set_lowprec(False)."""
import math
import numpy as np, torch
import pandas as pd
from defengine import corelib as C

Z95 = 1.6448536269514722
UNITS = ("study", "recipe_group", "row")


def wilson_lower_bound(k, n, z=Z95):
    """Copy of efficient_verification.wilson_lower_bound."""
    if n == 0:
        return 0.0
    phat = k / n; denom = 1 + z * z / n; center = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return max(0.0, (center - margin) / denom)


def sample_counts(n_units, n_reps, seed):
    """Identical to cluster_bootstrap.sample_study_bootstrap_counts(n_units, n_units, n_reps, default_rng(seed)) (numpy, CPU)."""
    import cluster_bootstrap as CB
    return CB.sample_study_bootstrap_counts(n_units, n_units, n_reps, np.random.default_rng(seed))


class LB:
    def __init__(self, eng, recipe_groups_csv=None, n_boot=100000, boot_seed=20261003, device="cuda"):
        import porosity_conditional_residual as PCR
        self.eng, self.dev, self.n_boot = eng, torch.device(device), n_boot
        merged, edges, _ = PCR._load_binned_bank(C.BANK_CSV, 3); self.edges = np.asarray(edges, float); self.merged = merged
        self.edges_inner = torch.as_tensor(self.edges[1:-1], dtype=torch.float64, device=self.dev)
        if recipe_groups_csv is not None:
            rg = pd.read_csv(recipe_groups_csv).set_index("original_row_index")["recipe_group"]; merged["recipe_group"] = merged["original_row_index"].map(rg).to_numpy()
            assert merged["recipe_group"].notna().all()
        t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64, device=self.dev)
        self.band = []
        for b in range(3):
            sub = merged[merged["_bin"] == b]; d = dict(n=len(sub), Es=t(sub["signed_standardised_strength_residual"]), Eq=t(sub["signed_standardised_logk_residual"]), units={})
            ids = {"study": sub["study_id"].to_numpy(), "row": np.arange(len(sub))}
            if "recipe_group" in sub: ids["recipe_group"] = sub["recipe_group"].to_numpy()
            for u, a in ids.items():
                uq, inv = np.unique(a, return_inverse=True)
                d["units"][u] = dict(K=len(uq), idx=torch.as_tensor(inv, dtype=torch.long, device=self.dev), nrows=t(np.bincount(inv, minlength=len(uq))))
            self.band.append(d)
        self.counts = {}; self.boot_seed = boot_seed
        self.n_units = {(b, u): self.band[b]["units"][u]["K"] for b in range(3) for u in self.band[b]["units"]}

    def counts_for(self, b, u):
        """Cached (n_boot, K) count matrix for band b / unit u; drawn once from a fixed seed (production use)."""
        if (b, u) not in self.counts:
            self.counts[(b, u)] = torch.as_tensor(sample_counts(self.n_units[(b, u)], self.n_boot, self.boot_seed + 10 * b + UNITS.index(u)), dtype=torch.float64, device=self.dev)
        return self.counts[(b, u)]

    def lcb95(self, nom, bank, u=None, e_draw=None, counts=None, units=("study",), chunk_designs=8, arrays=None):
        """nom: dict of float64 GPU tensors (C,r,A,P,S,phi (p,), g (p,6)); bank: manufacturing bank (numpy object, shared by all designs).
        Wilson residuals: e_draw = (eS, eQ) tensors (N,) or (P,N) (equivalence tests), else drawn per scenario from its porosity band with u ((N,) uniforms).
        counts: optional dict {(band, unit): (n_boot,K) tensor} (equivalence tests); default cached fixed-seed counts.
        arrays: optional precomputed (muS, muQ, sS, sQ, support, phi_actual) tensors (P,N) - used to test everything except the forward model against the CPU arrays. Returns dict of numpy arrays (P,)."""
        P = nom["C"].shape[0]; out = {k: [] for k in ("rhat", "wilson", "k", "support_fraction", "band", "phi_med")}
        for uu in units: out["boot_" + uu] = []
        out["p_sum"] = {uu: [] for uu in units}
        for a in range(0, P, chunk_designs):
            sl = {k: v[a:a + chunk_designs] for k, v in nom.items()}
            muS, muQ, sS, sQ, sup, phi = (self.eng.arrays_t(sl, bank, with_phi=True) if arrays is None else [x[a:a + chunk_designs] for x in arrays]); insup = sup >= C.SUPPORT_THR; p = muS.shape[0]; N = muS.shape[1]
            if e_draw is not None: eS, eQ = e_draw[0].expand(p, N), e_draw[1].expand(p, N)
            else: eS, eQ = self.eng.conditional_draws(phi, u)
            ok = insup & (muS + sS * eS >= C.S_STAR) & (muQ + sQ * eQ >= C.LOGK_STAR); kk = ok.sum(1)
            med = torch.quantile(phi, 0.5, dim=1); bd = torch.bucketize(med.contiguous(), self.edges_inner, right=False).clamp(0, 2)
            for i in range(p):
                b = int(bd[i]); B = self.band[b]; out["k"].append(int(kk[i])); out["rhat"].append(float(kk[i]) / N); out["wilson"].append(wilson_lower_bound(int(kk[i]), N))
                out["support_fraction"].append(float(insup[i].double().mean())); out["band"].append(b); out["phi_med"].append(float(med[i]))
                S_r = muS[i][None, :] + sS[i][None, :] * B["Es"][:, None]; Q_r = muQ[i][None, :] + sQ[i][None, :] * B["Eq"][:, None]
                p_r = (insup[i][None, :] & (S_r >= C.S_STAR) & (Q_r >= C.LOGK_STAR)).double().mean(1)          # per residual row pass probability (n_b,)
                for uu in units:
                    U = B["units"][uu]; psum = torch.zeros(U["K"], dtype=torch.float64, device=self.dev).index_add_(0, U["idx"], p_r)
                    cn = counts[(b, uu)] if counts is not None else self.counts_for(b, uu)
                    boot = (cn @ psum) / (cn @ U["nrows"]); out["boot_" + uu].append(float(torch.quantile(boot, 0.05))); out["p_sum"][uu].append(psum.cpu().numpy())
        res = {k: (np.asarray(v) if k != "p_sum" else v) for k, v in out.items()}
        res["LB"] = np.minimum(res["wilson"], res["boot_study"]) if "study" in units else None
        for uu in units:
            if uu != "study": res["LB_" + uu] = np.minimum(res["wilson"], res["boot_" + uu])
        return res


def lcb95_tie_bracket(lb, nom, bank, **kw):
    """Run lcb95 under the two extreme tie rules of the 10-nearest-residual selection (lowest-index = torch.topk default, and highest-index = reference rows reversed).
    The CPU engine breaks exact ties arbitrarily (np.argpartition); its result lies between the two (test B2). Returns (res_lowest, res_highest). Costs 2x the forward pass."""
    r1 = lb.lcb95(nom, bank, **kw); e = lb.eng
    try:
        e.res_X, e.res_s, e.res_p = e.res_X.flip(0), e.res_s.flip(0), e.res_p.flip(0)
        r2 = lb.lcb95(nom, bank, **kw)
    finally:
        e.res_X, e.res_s, e.res_p = e.res_X.flip(0), e.res_s.flip(0), e.res_p.flip(0)
    return r1, r2
