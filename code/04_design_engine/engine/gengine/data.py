"""Recorded units, data-only evidence counts and the conflict flag (plan section 5). RobustEngine 2026-10-22: the study column (_src_ffill) IS now read, ONLY to count
how many distinct studies supply the near recorded passers. Recommendation eligibility (ev["n_near_pass"] > 0, which every Tier-1 rule tests) now requires
near passers from >= MIN_STUDIES studies AND absolute-volume closure |V - 1| <= VOL_TOL; the unit count is kept as n_near_pass_units.
Unit = de-duplicated recipe (C, w/c, CA, SP, sand, six bin fractions rounded to 3 dp, porosity excluded); outcome = mean over its rows. Same construction as gengine/families_radii.py (Phase 1)."""
import numpy as np, pandas as pd, torch
from gengine.common import C, T, DEV, ROOT, BASE, W_LOG
import os
MIN_STUDIES, VOL_TOL = 2, 0.05
RHO = dict(C=3150.0, W=1000.0, A=2650.0, S=2650.0, P=1070.0)     # kg/m3; volume of solids + water + target porosity per batched m3


def volume(C_, r, A, S, P, phi):
    return C_ / RHO["C"] + r * C_ / RHO["W"] + A / RHO["A"] + S / RHO["S"] + P / RHO["P"] + phi / 100.0


BINS = ['R25-19mm', 'R19-12.5mm', 'R12.5-9.5mm', 'R9.5-6.3mm', 'R6.3-4.75mm', 'R4.75-2.36mm']
REC = ["cem (kg/m3)", "w/c", "CA (kg/m3)", "SP (kg/m3)", "sand (kg/m3)", "porosity"]


class Data:
    def __init__(self, cfg):
        self.cfg = cfg; d = pd.read_csv(os.path.join(BASE, "core", "data_inputs", "phase1_reconciled_eligible.csv")); assert len(d) == 631
        G = d[BINS].to_numpy(float); G = G / G.sum(1, keepdims=True)
        key = pd.DataFrame(np.round(np.column_stack([d[REC[:5]].to_numpy(float), G]), 3)).astype(str).agg("|".join, axis=1); uid, inv = np.unique(key.to_numpy(), return_inverse=True); nU = len(uid)
        d["unit"] = inv; d["blend_key"] = pd.DataFrame(np.round(G, 6)).astype(str).agg("|".join, axis=1); bl, binv = np.unique(d["blend_key"].to_numpy(), return_inverse=True); d["blend"] = binv
        U = d.groupby("unit").agg(CS=("comp_converted", "mean"), K=("perm (mm/s)", "mean"), phi=("porosity", "mean"), nrows=("unit", "size"), blend=("blend", "first")).reset_index()
        first = d.groupby("unit").first(); Ru = first[REC].to_numpy(float); Ru[:, 5] = U["phi"].to_numpy()
        Gu = np.array([G[d.unit.to_numpy() == u][0] for u in range(nU)]); Bl = np.array([G[d.blend.to_numpy() == b][0] for b in range(len(bl))])
        self.unit_studies = [set(d.loc[d.unit == u, "_src_ffill"].astype(str)) for u in range(nU)]
        U["pass"] = (U.CS >= C.S_STAR) & (U.K >= C.K_STAR); self.U, self.Ru, self.Gu, self.Bl, self.nU, self.d = U, Ru, Gu, Bl, nU, d
        self.scale = Ru.std(0, ddof=0); self.mg, self.mr = cfg["radii"]["m_g"], cfg["radii"]["m_r"]; self.lam = cfg["radii"]["lambda"]; self.lams = cfg["radii"]["lambda_extra"]
        self.Rg, self.Rr = self.lam * self.mg, self.lam * self.mr; self.blend_of = U["blend"].to_numpy()
        # G-3 (author-approved correction 3): a recorded passer whose porosity lies outside the 20-30 % deployment window is NOT a passer for the engine (the walk would clamp it into the window and test a different mix); it is neither
        # flagged recorded_passer nor counted as near-passer evidence. passU_raw = measured CS >= S* and k >= K* regardless of porosity (logged only); passU = passU_raw and porosity inside the window.
        lo, hi = C.S17.DEPLOY_LO, C.S17.DEPLOY_HI; self.phi_in_window = (Ru[:, 5] >= lo) & (Ru[:, 5] <= hi); self.passU_raw = U["pass"].to_numpy(); self.passU = self.passU_raw & self.phi_in_window
        # residuals of the SAVED cross-fitted bank (no conditioning of the GP); standardised by the A3-1 split-mean RMSEs
        bank = pd.read_csv(C.BANK_CSV); r = cfg["conflict"]["rmse_S"], cfg["conflict"]["rmse_Q"]
        bank["unit"] = bank["original_row_index"].map(dict(zip(range(len(d)), d["unit"]))); bank["e_S"] = (bank["y_strength"] - bank["mu_strength"]) / r[0]; bank["e_Q"] = (bank["y_logk"] - bank["mu_logk"]) / r[1]
        eu = bank.groupby("unit")[["e_S", "e_Q"]].mean().reindex(range(nU)); self.e = eu.to_numpy(); self.has = ~np.isnan(self.e[:, 0])
        self.Fu = T(np.cumsum(Gu[:, :5], 1)); self.Ru_t = T(Ru); self.sc_t = T(self.scale); self.W = T(W_LOG)

    def dists(self, rec, g):
        """rec (n,6) [C, w/c, CA, SP, sand, phi] tensor, g (n,6) tensor -> d_grad (n,nU), d_rec (n,nU) tensors."""
        Fg = torch.cumsum(g[:, :5], 1); dg = ((Fg[:, None, :] - self.Fu[None]).abs() * self.W).sum(-1)
        dr = torch.sqrt((((rec[:, None, :] - self.Ru_t[None]) / self.sc_t) ** 2).sum(-1)); return dg, dr

    def conflict(self, idx):
        """Plan 5.2 clauses (i) and (ii) only (the locked rule): (i) mean e <= -m1 in CS or ln k; (ii) >= 2 residual-carrying near units whose e spread > m2. e = (observed - model)/RMSE from the cross-fitted bank."""
        m1, m2 = self.cfg["conflict"]["mult_optimistic"], self.cfg["conflict"]["mult_disagree"]; idx = [i for i in idx if self.has[i]]; n = len(idx)
        if n == 0: return dict(n_res=0, flag=False, clause_i=False, clause_ii=False)
        e = self.e[idx]; mean = e.mean(0); spread = e.max(0) - e.min(0); c1 = bool((mean <= -m1).any()); c2 = bool(n >= 2 and (spread > m2).any())
        return dict(n_res=n, mean_eS=float(mean[0]), mean_eQ=float(mean[1]), spread_S=float(spread[0]), spread_Q=float(spread[1]), clause_i=c1, clause_ii=c2, flag=c1 or c2)

    def evidence(self, nom, max_ids=30):
        """Per design (nominal only): counts of near de-duplicated units at the frozen lambda and at the extra lambdas, near-passer detail, conflict flag, porosity range of the near units."""
        rec = torch.stack([nom["C"], nom["r"], nom["A"], nom["P"], nom["S"], nom["phi"]], 1); dg, dr = self.dists(rec, nom["g"]); out = []
        for i in range(rec.shape[0]):
            a, b = dg[i].cpu().numpy(), dr[i].cpu().numpy(); near = np.flatnonzero((a <= self.Rg) & (b <= self.Rr)); npass = near[self.passU[near]]
            st_pass = set().union(*[self.unit_studies[u] for u in npass]) if len(npass) else set()
            V = volume(*(float(nom[k][i]) for k in ("C", "r", "A", "S", "P", "phi"))); ok = len(st_pass) >= MIN_STUDIES and abs(V - 1.0) <= VOL_TOL
            ev = dict(n_near=int(len(near)), n_near_pass=int(len(npass)) if ok else 0, n_near_pass_units=int(len(npass)), n_near_pass_studies=int(len(st_pass)),
                      near_pass_studies=sorted(st_pass), volume_m3=V, volume_ok=bool(abs(V - 1.0) <= VOL_TOL), n_near_pass_blends=int(len(set(self.blend_of[npass]))), near_ids=near[:max_ids].tolist(), near_pass_ids=npass.tolist())
            ev["by_lambda_n_near_pass"] = {str(l): int(self.passU[np.flatnonzero((a <= l * self.mg) & (b <= l * self.mr))].sum()) for l in self.lams}
            ev["by_lambda_n_near"] = {str(l): int(((a <= l * self.mg) & (b <= l * self.mr)).sum()) for l in self.lams}
            if len(npass): j = npass[np.argmin(np.maximum(a[npass] / self.Rg, b[npass] / self.Rr))]; ev["nearest_pass"] = dict(unit=int(j), C=float(self.Ru[j, 0]), phi=float(self.Ru[j, 5]), d_grad=float(a[j]), d_rec=float(b[j]))
            if len(near):
                ph = self.Ru[near, 5]; ev.update(phi_near_min=float(ph.min()), phi_near_max=float(ph.max()), phi_near_median=float(np.median(ph)), phi_in_neighbour_range=bool(ph.min() <= float(nom["phi"][i]) <= ph.max()))
            ev["conflict"] = self.conflict(near.tolist()); out.append(ev)
        return out

    def evidence_bruteforce(self, nomd, lam=None):
        """CPU reference for ONE design dict (loops over every unit with scalar maths; used by test C3)."""
        lam = lam or self.lam; rec = np.array([nomd["C"], nomd["r"], nomd["A"], nomd["P"], nomd["S"], nomd["phi"]], float); g = np.asarray(nomd["g"], float); near = []
        for u in range(self.nU):
            dg = float((np.abs(np.cumsum(g[:5]) - np.cumsum(self.Gu[u][:5])) * W_LOG).sum()); dr = float(np.sqrt((((rec - self.Ru[u]) / self.scale) ** 2).sum()))
            if dg <= lam * self.mg and dr <= lam * self.mr: near.append(u)
        return near
