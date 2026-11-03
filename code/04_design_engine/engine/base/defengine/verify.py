"""Independent verification (acceptance authority). Calls the FROZEN estimator (EV.sequential_decisive_verify: N=20k -> 50k -> 100k escalation,
Wilson bound + 100,000-replicate study-cluster bootstrap, porosity-conditional residual sampling) with seeds that are disjoint from every
search seed, and applies the FROZEN nominal gates. PASS requires: R_LCB95 >= target AND all scenarios model-supported AND frozen nominal gates pass.
Nothing is written inside the hashed core folder. Decision E-01 / E-02 (model hash computed)."""
import hashlib, time
import numpy as np
from . import corelib as C, gates as G

VERIFY_SEED_FLOOR = 5_000_000_000          # search seeds are < 1e9; verification streams live far above


def seed_for(candidate_id, replicate=0):
    d = hashlib.sha256(f"{candidate_id}|rep{replicate}".encode()).digest()
    return VERIFY_SEED_FLOOR + int.from_bytes(d[:6], "big") % 1_000_000_000_000


def verify_design(candidate_id, nominal, replicate=0, n_boot=100000, verbose=False):
    nom = {k: (np.asarray(v, float) if k == "g" else float(v)) for k, v in nominal.items()}
    t0 = time.time(); seed = seed_for(candidate_id, replicate)
    res, note = C.EV.sequential_decisive_verify(candidate_id, nom, seed=seed, n_boot=n_boot, verbose=verbose)
    fg = G.frozen_gate(nom)
    ok = bool(res["R_LCB95_joint"] >= C.REL_TARGET and res["support_fraction"] >= 1.0 - 1e-12 and fg["all_pass"])
    keep = ("Rhat_joint", "mc_wilson_lower_95", "calib_bootstrap_R_LCB95", "R_LCB95_joint", "support_fraction", "n_scenarios", "n_bootstrap_reps", "joint_pass_count",
            "strength_margin_p05", "logk_margin_p05", "achieved_phi_p05", "achieved_phi_p50", "achieved_phi_p95", "probability_strength_fail_only", "probability_perm_fail_only",
            "probability_both_fail", "probability_support_fail", "n_studies_cluster_bootstrap", "mfg_bank_hash", "res_bank_hash", "uncertainty_contract_hash")
    out = {k: res[k] for k in keep if k in res}
    out.update(candidate_id=candidate_id, replicate=replicate, seed=seed, escalation_note=note, frozen_gate=fg, PASS=ok, model_hash=C.model_hash(),
               nominal={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in nom.items()}, seconds=time.time() - t0)
    return out
