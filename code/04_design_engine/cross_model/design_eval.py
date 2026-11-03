"""design decisions of the multi-task model (MT, the paper's engine, unchanged) and the independent-GP model
(IND) on the SAME candidate pool (the 2026-10-22 six-target runs; candidates.json = the set sep_eval.py re-scored).
IND engine = the paper engine with (a) the GP mean / SD of GPUEngine.model replaced by the two independent full-data GPs (IND_oracle.pt, exact
closed form, new-laboratory mean, lab-free SD + noise: the engine's own convention) and (b) its own residual bank, local-residual reference and
calibration bundles (variant IND_near3, built by build_bank.py from the IND near-3 study-grouped folds). Gates, family rule, support model,
manufacturing banks, porosity bands, bootstrap and acceptance rule are the engine's, untouched.
Step 1 re-score: every candidate at all 11 gradation neighbours, both tie rules, N_accept scenarios, one fixed bank per target (common random numbers);
accepted = feasible & joint lower bound >= R_ACC (= R* + walk margin). Step 2 final evaluation (GPU, same estimand as the CPU certificate): the lowest
accepted design, N = 100000 scenarios, 2 independent replicate banks, all 11 neighbours, both tie rules; reported per component (Monte Carlo
joint-pass estimate, Wilson 95 % lower bound, study-bootstrap 5th percentile, final = min).
usage (env GENGINE_BASE / GENGINE_CONFIG / GENGINE_ROOT set to the target copy):
  python design_eval.py MT|IND <candidates.json> <out_json> [<ind_oracle.pt>] [<extra_designs.json>]"""
import os, sys, json, time, types
sys.dont_write_bytecode = True
ROOT = os.environ["GENGINE_ROOT"]; sys.path.insert(0, ROOT); KIT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, KIT)
import numpy as np, torch
from gengine.common import C, T, decode, nom_dicts, SHIFT_LABELS, gpu_lb, nbhd
from gengine import config
from gengine.walk import Ctx

KIND, CANDS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]; IND_PT = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None
EXTRA = sys.argv[5] if len(sys.argv) > 5 else None
cfg, H = config.load(); ctx = Ctx(cfg, H); R_ACC = ctx.R_ACC; tau = ctx.tau; eng = ctx.eng
name = f"S{C.S_STAR:g}_K{C.K_STAR:g}_R{C.REL_TARGET:.2f}"; print(KIND, name, "R_ACC", R_ACC, "bank rows", len(ctx.LB.merged), flush=True)
REP = dict(kind=KIND, target=name, R_ACC=R_ACC, bank_csv_rows=int(len(ctx.LB.merged)))

if KIND == "IND":
    import indkern
    O = torch.load(IND_PT, map_location="cpu", weights_only=False)
    assert np.allclose(O["x_mean"], eng.xm.cpu().numpy()) and np.allclose(O["x_std"], eng.xs.cpu().numpy()), "IND scaler differs from the engine's"
    assert np.allclose(O["y_mean"], eng.ym.cpu().numpy()) and np.allclose(O["y_std"], eng.ys.cpu().numpy())
    d64 = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64, device=eng.dev)
    IX = d64(O["X"]); TK = [dict(spec=t["spec"], a=d64(t["a"]), K0i=d64(t["K0inv"]), m0=t["m0"], noise=t["noise"], kd=t["kdiag"]) for t in O["tasks"]]

    def ind_gp(Xs64):
        mu, var = [], []
        for t in TK:
            Kq = indkern.K(t["spec"], Xs64, IX); mu.append(t["m0"] + Kq @ t["a"]); var.append(t["kd"] - ((Kq @ t["K0i"]) * Kq).sum(1) + t["noise"])
        return torch.stack(mu, 1), torch.stack(var, 1)

    pm, pv = ind_gp(d64(O["probe"])); REP["probe_check_mean_max_abs"] = float((pm - d64(O["probe_mu_std"])).abs().max()); REP["probe_check_var_max_abs"] = float((pv - d64(O["probe_var_std"])).abs().max())
    assert REP["probe_check_mean_max_abs"] < 1e-8 and REP["probe_check_var_max_abs"] < 1e-8, REP
    from defengine.gpu import SIGMA_FLOOR

    def model_ind(self, Xs64, Xs):
        """GPUEngine.model with the ICM block replaced by ind_gp; every line from the local-residual spread on is copied unchanged."""
        with torch.no_grad():
            mean, var = ind_gp(Xs64); sig = torch.sqrt(torch.clamp(var, min=1e-12))
            mu = mean * self.ys + self.ym; sr = torch.clamp(sig * self.ys, min=SIGMA_FLOOR)
            Xq = Xs64.float() if self.lowprec else Xs64
            dl = torch.cdist(Xq, self.res_X); il = dl.topk(self.local_q, dim=1, largest=False).indices
            ls = self.res_s[il].std(dim=1, unbiased=True).double(); lp = self.res_p[il].std(dim=1, unbiased=True).double()
            da = torch.cdist(Xq, self.ad_X).topk(self.ad_k, dim=1, largest=False).values.mean(dim=1).double()
            ratio = da / max(self.ad_tau, 1e-12); infl = torch.where(ratio > 0.75, 1.0 + self.ad_eta * (ratio - 0.75) / 0.25, torch.ones_like(ratio)).clamp(1.0, 3.0)
            sS = torch.maximum(torch.clamp(sr[:, 0], min=1e-12), torch.clamp(ls, min=1e-12)) * infl
            sQ = torch.maximum(torch.clamp(sr[:, 1], min=1e-12), torch.clamp(lp, min=1e-12)) * infl
            dv = torch.cdist(Xq, self.sv) ** 2; dec = ((torch.exp(-self.gamma * dv) @ self.dual) + self.icpt).double()
            oc = torch.sigmoid(torch.clamp(dec / self.scale, -60.0, 60.0))
            kld = (torch.logsumexp(-(torch.cdist(Xq, self.kde_X) ** 2) / self.kde_h2, dim=1) + self.kde_norm).double()
            kd_ = torch.clamp(torch.exp(torch.clamp(torch.clamp(kld - self.kde_max, max=0.0), min=-745.0)), 0.0, 1.0)
            sup = torch.clamp(0.5 * oc + 0.5 * kd_, 0.0, 1.0)
        return mu[:, 0], mu[:, 1], sS, sQ, sup

    orig_model = eng.model
    xchk = (eng.Xtr[:64].double()); a_old = orig_model(xchk, xchk.float()); eng.model = types.MethodType(model_ind, eng); a_new = eng.model(xchk, xchk.float())
    REP["support_identical_after_patch"] = bool(torch.equal(a_old[4], a_new[4])); assert REP["support_identical_after_patch"]
    REP["mean_shift_vs_MT_at_training_points_CS_mpa"] = float((a_old[0] - a_new[0]).abs().mean()); REP["mean_shift_vs_MT_lnk"] = float((a_old[1] - a_new[1]).abs().mean())
    print("IND patch active", json.dumps({k: v for k, v in REP.items() if "check" in k or "shift" in k}), flush=True)

# ---- step 1: re-score the shared candidate pool (joint rule, the paper's acceptance) ----
keys = [] if EXTRA else [(int(f), float(c), tuple(z)) for f, c, z in json.load(open(CANDS))[name]]
bank, u = ctx.bank(cfg["walk"]["N_accept"], seed=20261027); eng.set_lowprec(False)
LBj = np.full(len(keys), np.nan); RH = np.full(len(keys), np.nan); feas = np.zeros(len(keys), bool); worst = [None] * len(keys); t0 = time.time(); B = 24
for f in ([] if EXTRA else range(3)):
    idx = [i for i, k in enumerate(keys) if k[0] == f]
    for a in range(0, len(idx), B):
        ii = idx[a:a + B]; Z = T([keys[i][2] for i in ii]); Cv = T([keys[i][1] for i in ii]); nom = decode(Z, Cv)
        nc, owner, kk, pen, fok, fd = ctx.exact_feasible(nom, f); own = owner.cpu().numpy(); kkn = kk.cpu().numpy()
        r1, r2 = gpu_lb.lcb95_tie_bracket(ctx.LB, nc, bank, u=u, units=("study",)); LB = np.minimum(r1["LB"], r2["LB"]); SF = r1["support_fraction"]; rh = np.minimum(r1["rhat"], r2["rhat"])
        pn, fo = pen.cpu().numpy(), fok.cpu().numpy()
        for j, i in enumerate(ii):
            ix = np.flatnonzero(own == j); LBj[i] = float(LB[ix].min()); RH[i] = float(rh[ix].min()); worst[i] = SHIFT_LABELS[int(kkn[ix[int(np.argmin(LB[ix]))]])]
            feas[i] = bool((pn[ix] <= 0).all() and fo[ix].all() and (SF[ix] >= 1 - 1e-12).all())
    print(f"family {f} scored, {time.time() - t0:.0f}s", flush=True)
acc = feas & (LBj >= R_ACC); ilow = int(np.flatnonzero(acc)[0]) if acc.any() else None          # candidates are sorted by cement
REP.update(n_candidates=len(keys), n_feasible=int(feas.sum()), n_accepted=int(acc.sum()), rescore_minutes=(time.time() - t0) / 60,
           lowest_accepted=None if ilow is None else dict(C=keys[ilow][1], family=keys[ilow][0], Z=list(keys[ilow][2]), LB_rescore=LBj[ilow], Rhat_rescore=RH[ilow], worst=worst[ilow]))
print(json.dumps({k: REP[k] for k in ("n_candidates", "n_feasible", "n_accepted", "lowest_accepted")}, default=float), flush=True)


# ---- step 2: final evaluation (GPU), N = 100000, 2 replicate banks, all neighbours, both tie rules, components reported ----
def final_eval(fam, Cc, Z, tag):
    nom = decode(T([Z]), T([Cc])); nc, owner, kk, pen, fok, fd = ctx.exact_feasible(nom, fam); labs = [SHIFT_LABELS[int(x)] for x in kk.cpu().numpy()]
    out = dict(tag=tag, C=Cc, family=fam, design=nom_dicts(nom)[0], gates_pass=bool((pen.cpu().numpy() <= 0).all()), family_rule_pass=bool(fok.cpu().numpy().all()), reps=[])
    for rep, seed in enumerate((20261029, 20261129)):
        bk, uu = ctx.bank(100000, seed=seed); r1, r2 = gpu_lb.lcb95_tie_bracket(ctx.LB, nc, bk, u=uu, units=("study",))
        rows = [dict(shift=labs[j], Rhat=float(min(r1["rhat"][j], r2["rhat"][j])), wilson=float(min(r1["wilson"][j], r2["wilson"][j])), boot_study=float(min(r1["boot_study"][j], r2["boot_study"][j])),
                     support=float(min(r1["support_fraction"][j], r2["support_fraction"][j]))) for j in range(len(labs))]
        for r in rows: r["R_LCB95"] = min(r["wilson"], r["boot_study"])
        out["reps"].append(rows)
    allr = [r for rows in out["reps"] for r in rows]; w = min(allr, key=lambda r: r["R_LCB95"]); nomr = [r for r in allr if r["shift"] == "nominal"]
    out.update(final_R_LCB95=w["R_LCB95"], final_worst_shift=w["shift"], final_components_at_worst=dict(Rhat=w["Rhat"], wilson=w["wilson"], boot_study=w["boot_study"]),
               nominal_Rhat=min(r["Rhat"] for r in nomr), nominal_R_LCB95=min(r["R_LCB95"] for r in nomr), min_Rhat_all=min(r["Rhat"] for r in allr),
               min_support=min(r["support"] for r in allr), accepted_final=bool(w["R_LCB95"] >= C.REL_TARGET and out["gates_pass"] and out["family_rule_pass"] and min(r["support"] for r in allr) >= 1 - 1e-12))
    print(tag, Cc, "final R_LCB95", round(out["final_R_LCB95"], 4), "at", out["final_worst_shift"], "| Rhat", round(w["Rhat"], 4), "wilson", round(w["wilson"], 4), "boot", round(w["boot_study"], 4), flush=True)
    return out


REP["final"] = []
if ilow is not None: REP["final"].append(final_eval(keys[ilow][0], keys[ilow][1], list(keys[ilow][2]), f"lowest_accepted_{KIND}"))
if EXTRA and os.path.exists(EXTRA):
    for e in json.load(open(EXTRA)).get(name, []): REP["final"].append(final_eval(int(e["family"]), float(e["C"]), list(e["Z"]), e["tag"]))
REP["table"] = [dict(C=k[1], family=k[0], feasible=bool(feas[i]), LB=LBj[i], Rhat=RH[i]) for i, k in enumerate(keys)]
json.dump(REP, open(OUT, "w"), indent=1, default=float); print("saved", OUT, flush=True)
