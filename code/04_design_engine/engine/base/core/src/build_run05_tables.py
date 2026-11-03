"""Build DE_CONTINUOUS_SEARCH.csv, CONTINUOUS_DECISIVE_VERIFICATION.csv,
INTEGER_CEMENT_PROFILE.csv, INTEGER_LEVEL_NOMAD_RUNS.csv,
FINAL_DECISIVE_VERIFICATION.csv, FINAL_STAGE1_CANDIDATE.csv,
PER_STUDY_RELIABILITY.csv directly from trace JSONs (no hand-transcription,
same discipline as run_04)."""
import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
TAB = os.path.join(RUN, "tables")
TRACES = os.path.join(RUN, "traces")
MODEL_HASH = "1bf27004724bd311a5cfec98b4084d1063fe739704be2344616bf52698851a69"


def de_continuous_search():
    rows = []
    for path in sorted(glob.glob(os.path.join(TRACES, "de_shortlist_seed*.csv"))):
        rows.append(pd.read_csv(path))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    df.to_csv(os.path.join(TAB, "DE_CONTINUOUS_SEARCH.csv"), index=False)
    return df


def _load_decisive(tag):
    path = os.path.join(TRACES, f"decisive_verification_{tag}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def continuous_decisive_verification(tags):
    rows = []
    for tag in tags:
        d = _load_decisive(tag)
        if d is None:
            continue
        cand = d["cand"]
        rows.append({
            "tag": tag, "cid": d["cid"], "C": cand["C"], "r": cand["r"], "A": cand["A"], "P": cand["P"],
            "S": cand["S"], "phi": cand["phi"], **{f"g{i+1}": cand["g"][i] for i in range(6)},
            "n_scenarios": d["n_scenarios"], "n_bootstrap_reps": d["n_bootstrap_reps"],
            "Rhat_joint": d["Rhat_joint"], "mc_wilson_lower_95": d["mc_wilson_lower_95"],
            "calib_bootstrap_R_LCB95": d["calib_bootstrap_R_LCB95"], "R_LCB95_joint": d["R_LCB95_joint"],
            "support_fraction": d["support_fraction"], "PASS_FAIL": d["PASS_FAIL"],
            "escalation_note": d["escalation_note"], "model_hash": MODEL_HASH,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "CONTINUOUS_DECISIVE_VERIFICATION.csv"), index=False)
    return df


def integer_level_nomad_runs():
    path = os.path.join(TRACES, "integer_cement_profile_results.json")
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path) as f:
        r = json.load(f)
    rows = []
    for level, starts in r["results"].items():
        for tag, s in starts.items():
            rows.append({"cement_level_kg_m3": float(level), "start_tag": tag,
                         "final_Rhat_search_bank": s["final_Rhat_search_bank"],
                         "n_evals_used": s["n_evals_used"], "run_flag": s["run_flag"],
                         "stop_reason": s["stop_reason"], "elapsed_s": s["elapsed_s"],
                         "moved_from_start": s["x0"] != s["x_best"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "INTEGER_LEVEL_NOMAD_RUNS.csv"), index=False)
    return df


def integer_cement_profile(levels):
    rows = []
    for level in levels:
        tag = f"{int(level)}"
        d = _load_decisive(tag)
        row = {"cement_level_kg_m3": level}
        if d:
            row.update({"cid": d["cid"], "decisive_Rhat": d["Rhat_joint"], "decisive_R_LCB95": d["R_LCB95_joint"],
                        "support_fraction": d.get("support_fraction", 0.0),
                        "nominal_state_support_pass": bool(d.get("nominal_support_recheck", {}).get("nominal_state_support_pass", False)),
                        "PASS_FAIL": d["PASS_FAIL"], "n_scenarios": d["n_scenarios"],
                        "n_bootstrap_reps": d["n_bootstrap_reps"], "escalation_note": d["escalation_note"]})
        else:
            row.update({"cid": None, "decisive_Rhat": None, "decisive_R_LCB95": None, "support_fraction": None,
                        "nominal_state_support_pass": None, "PASS_FAIL": "NOT_TESTED",
                        "n_scenarios": None, "n_bootstrap_reps": None, "escalation_note": None})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "INTEGER_CEMENT_PROFILE.csv"), index=False)
    return df


def final_decisive_verification(levels):
    rows = []
    for level in levels:
        tag = f"{int(level)}"
        d = _load_decisive(tag)
        if d is None:
            continue
        cand = d["cand"]
        rows.append({
            "cid": d["cid"], "cement_level_kg_m3": level, "wc": cand["r"], "CA_kg_m3": cand["A"],
            "SP_kg_m3": cand["P"], "sand_kg_m3": cand["S"], "porosity_nominal_pct": cand["phi"],
            **{f"g{i+1}": cand["g"][i] for i in range(6)},
            "n_scenarios": d["n_scenarios"], "n_bootstrap_reps": d["n_bootstrap_reps"],
            "Rhat_joint": d["Rhat_joint"], "mc_wilson_lower_95": d["mc_wilson_lower_95"],
            "calib_bootstrap_R_LCB95": d["calib_bootstrap_R_LCB95"], "R_LCB95_joint": d["R_LCB95_joint"],
            "support_fraction": d["support_fraction"], "strength_margin_p05": d["strength_margin_p05"],
            "logk_margin_p05": d["logk_margin_p05"], "achieved_phi_p05": d["achieved_phi_p05"],
            "achieved_phi_p50": d["achieved_phi_p50"], "achieved_phi_p95": d["achieved_phi_p95"],
            "probability_strength_fail_only": d["probability_strength_fail_only"],
            "probability_perm_fail_only": d["probability_perm_fail_only"],
            "probability_both_fail": d["probability_both_fail"],
            "probability_support_fail": d["probability_support_fail"],
            "n_studies_cluster_bootstrap": d["n_studies_cluster_bootstrap"],
            "escalation_note": d["escalation_note"], "PASS_FAIL": d["PASS_FAIL"], "model_hash": MODEL_HASH,
        })
    df = pd.DataFrame(rows).sort_values("cement_level_kg_m3") if rows else pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "FINAL_VERIFICATION.csv"), index=False)
    df.to_csv(os.path.join(TAB, "FINAL_DECISIVE_VERIFICATION.csv"), index=False)
    return df


def final_stage1_candidate(selected_level):
    tag = f"{int(selected_level)}"
    d = _load_decisive(tag)
    if d is None:
        raise FileNotFoundError(f"no decisive verification record for selected level {selected_level}")
    cand = d["cand"]
    row = {
        "selected": True, "cid": d["cid"], "cement_level_kg_m3": selected_level,
        "wc": cand["r"], "water_kg_m3": cand["r"] * cand["C"], "CA_kg_m3": cand["A"], "SP_kg_m3": cand["P"],
        "sand_kg_m3": cand["S"], "porosity_nominal_pct": cand["phi"],
        **{f"g{i+1}": cand["g"][i] for i in range(6)},
        "Rhat_joint": d["Rhat_joint"], "R_LCB95_joint": d["R_LCB95_joint"],
        "n_scenarios": d["n_scenarios"], "n_bootstrap_reps": d["n_bootstrap_reps"],
        "support_fraction": d["support_fraction"], "model_hash": MODEL_HASH,
        "PHYSICAL_INPUTS_COMPLETE": False, "VOLUME_BASIS_CHECKED": False, "EXPERIMENTALLY_VERIFIED": False,
        "batch_ready": "NOT_YET_DECLARED", "proposal_kind": "manufacturing-robust Stage-One computational proposal",
    }
    pd.DataFrame([row]).to_csv(os.path.join(TAB, "FINAL_STAGE1_CANDIDATE.csv"), index=False)
    return row


def per_study_reliability(tags_with_levels):
    """tags_with_levels: list of (tag, cement_level) for decisive candidates."""
    rows = []
    for tag, level in tags_with_levels:
        d = _load_decisive(tag)
        if d is None:
            continue
        pst = d["per_study_table"]
        for sid, n_rows, mean_p in zip(pst["study_id"], pst["n_rows"], pst["mean_p_r"]):
            rows.append({"cement_level_kg_m3": level, "cid": d["cid"], "study_id": sid,
                         "n_rows": n_rows, "mean_p_r": mean_p})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "PER_STUDY_RELIABILITY.csv"), index=False, encoding="utf-8")
    return df
