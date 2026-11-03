"""CPU certificate with the porosity scatter SD as a parameter and the no-lab bootstrap variants printed beside the frozen study-cluster bound. The body of decisive_verify_cfg is a line-for-line copy of core/src/efficient_verification.decisive_verify
with exactly two additions: (1) UncertaintySpec(..., porosity_sd_override_pp=sd_pp), (2) extra bootstrap units. With sd_pp=None and units=("study",) the numbers equal the frozen function
(test T-cert in tests/test_cert_params.py). Frozen core is imported, never edited. Needs `defengine.corelib` already imported (sys.path to the base)."""
import time, hashlib
import numpy as np, pandas as pd
from defengine import corelib as C, gates as G

EV, U, E, RT = C.EV, C.U, C.E, C.RT
import cluster_bootstrap as CB            # noqa: E402
import porosity_conditional_residual as PCR   # noqa: E402

RG_CSV = None            # set by the caller: base/../stages/00_base/recipe_groups_631.csv (for the recipe_group variant)


def _units(df):
    ids = {"study": df["study_id"].to_numpy(), "row": np.arange(len(df))}
    if RG_CSV:
        rg = pd.read_csv(RG_CSV).set_index("original_row_index")["recipe_group"]; ids["recipe_group"] = df["original_row_index"].map(rg).to_numpy()
    return ids


def decisive_verify_cfg(cid, nominal, n_scenarios, n_boot, seed, sd_pp=None, bias_pp=None, units=("study",)):
    t0 = time.time()
    spec = U.UncertaintySpec(C.CONTRACT, porosity_sd_override_pp=sd_pp, porosity_bias_override_pp=bias_pp)
    mfg_bank = U.build_manufacturing_bank(spec, n=n_scenarios, seed=seed)
    muS, muQ, sS, sQ, support, phi_actual = EV.compute_scenario_predictions(nominal, mfg_bank)
    thr = E.SUPPORT_THRESHOLD
    res_bank = PCR.build_residual_bank_porosity_conditional(C.BANK_CSV, phi_actual, seed=seed + 1)
    joint_pass, s_margin, q_margin, in_support = EV._joint_pass(muS, muQ, sS, sQ, support, res_bank.e_strength, res_bank.e_logk, thr)
    k = int(joint_pass.sum()); rhat = k / n_scenarios; wilson_lb = EV.wilson_lower_bound(k, n_scenarios)
    df, _bin, _ = PCR.filter_residual_bank_to_scenario_bin(C.BANK_CSV, phi_actual)
    cb = CB.run_cluster_bootstrap(muS, muQ, sS, sQ, support, thr, df, C.S_STAR, C.LOGK_STAR, n_reps=n_boot, seed=seed + 2)
    calib_lb = cb["R_LCB95_cluster"]; r_lcb95 = min(wilson_lb, calib_lb)
    out = {"cid": cid, "n_scenarios": n_scenarios, "n_bootstrap_reps": n_boot, "joint_pass_count": k, "Rhat_joint": rhat, "mc_wilson_lower_95": wilson_lb,
           "calib_bootstrap_R_LCB95": calib_lb, "R_LCB95_joint": r_lcb95, "support_fraction": float(in_support.mean()), "porosity_sd_pp": spec.porosity_sd_pp,
           "achieved_phi_p50": float(np.percentile(phi_actual, 50)), "mfg_bank_hash": mfg_bank.bank_hash(), "res_bank_hash": res_bank.bank_hash(),
           "n_studies_cluster_bootstrap": cb["n_studies"], "per_study_mean_p_r": cb["per_study_table"]["mean_p_r"], "p_r": cb["p_r"].tolist(), "boot_rhats_study": cb["boot_rhats"].tolist(), "elapsed_s": time.time() - t0}
    if len(units) > 1 or units[0] != "study":
        ids = _units(df)
        for uu in units:
            if uu == "study": continue
            studies, psum, nrows = CB.study_aggregates(ids[uu], cb["p_r"]); counts = CB.sample_study_bootstrap_counts(len(studies), len(studies), n_boot, np.random.default_rng(seed + 2))
            b = CB.cluster_bootstrap_reliability_fast(psum, nrows, counts); out["boot_" + uu] = float(np.percentile(b, 5)); out["n_units_" + uu] = int(len(studies)); out["R_LCB95_" + uu] = min(wilson_lb, out["boot_" + uu])
    return out


def sequential_cfg(cid, nominal, seed, n_boot=100000, sd_pp=None, bias_pp=None, units=("study",)):
    """Same escalation as efficient_verification.sequential_decisive_verify (N=20000 -> 50000 -> 100000)."""
    r = decisive_verify_cfg(cid, nominal, 20000, n_boot, seed, sd_pp, bias_pp, units)
    if abs(r["R_LCB95_joint"] - RT.RELIABILITY) > 0.005: return r, "N=20000 decisive (gap > 0.005)"
    r2 = decisive_verify_cfg(cid, nominal, 50000, n_boot, seed + 100, sd_pp, bias_pp, units)
    if abs(r2["R_LCB95_joint"] - RT.RELIABILITY) > 0.002: return r2, "N=50000 decisive (gap > 0.002)"
    return decisive_verify_cfg(cid, nominal, 100000, n_boot, seed + 200, sd_pp, bias_pp, units), "N=100000 (hard cap reached, still within 0.002 of threshold)"


def verify_design_cfg(cid, nominal, replicate=0, sd_pp=None, n_boot=100000, units=("study",)):
    """Counterpart of defengine.verify.verify_design; same seeds (seed_for(cid, replicate)) so results are comparable across SD values."""
    from defengine import verify as V
    nom = {k: (np.asarray(v, float) if k == "g" else float(v)) for k, v in nominal.items()}; seed = V.seed_for(cid, replicate)
    res, note = sequential_cfg(cid, nom, seed, n_boot, sd_pp, None, units); fg = G.frozen_gate(nom)
    res.update(candidate_id=cid, replicate=replicate, seed=seed, escalation_note=note, frozen_gate=fg, model_hash=C.model_hash(),
               PASS=bool(res["R_LCB95_joint"] >= C.REL_TARGET and res["support_fraction"] >= 1.0 - 1e-12 and fg["all_pass"]))
    return res
