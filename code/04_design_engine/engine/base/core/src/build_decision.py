"""Apply the frozen lexicographic selection logic PLUS the new adjacent-
integer bracket check to the run_05 integer-profile decisive verification
results, and write decision.json."""
import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import build_final_cement_table as LEX  # noqa: E402
import bracket_check as BC  # noqa: E402


def main(continuous_frontier_C=None):
    with open(os.path.join(RUN, "TARGET_CONTRACT.json"), encoding="utf-8") as f:
        targets = json.load(f)
    strength_target = float(targets["strength_target_mpa"])
    permeability_target = float(targets["permeability_target_mm_s"])
    reliability_target = float(targets["required_joint_reliability"])

    df = pd.read_csv(os.path.join(RUN, "tables", "INTEGER_CEMENT_PROFILE.csv")).dropna(subset=["PASS_FAIL"])
    df = df[df["PASS_FAIL"] != "NOT_TESTED"]

    all_candidates = [
        {"cement_level_kg_m3": float(row["cement_level_kg_m3"]),
        "R_LCB95_joint": float(row["decisive_R_LCB95"]) if pd.notna(row["decisive_R_LCB95"]) else -1.0,
         "support_fraction": float(row.get("support_fraction", 0.0) or 0.0),
         "decisive_pass": str(row.get("PASS_FAIL", "")) == "PASS",
         "cid": row.get("cid")}
        for _, row in df.iterrows()
    ]
    result = LEX.lexicographic_select(all_candidates)

    tested_levels_status = {float(row["cement_level_kg_m3"]): row["PASS_FAIL"] for _, row in df.iterrows()}

    decision = {
        "run_id": f"{os.path.basename(os.path.dirname(RUN))}/{os.path.basename(RUN)}",
        "stage1_research_question": (
            "Among mixtures inside the frozen model/domain/support contract, what is the lowest nominal "
            f"cement content that can jointly satisfy compressive strength >= {strength_target:g} MPa and "
            f"permeability >= {permeability_target:g} mm/s with the pre-specified "
            f"{100.0 * reliability_target:g}% integrated joint-reliability requirement, after propagating "
            "the frozen manufacturing variability and predictive/model uncertainty?"
        ),
        "target_strength_mpa": strength_target,
        "target_permeability_mm_s": permeability_target,
        "required_joint_reliability": reliability_target,
        "acceptance_statistic": "one_sided_95pct_lower_confidence_bound",
        "cement_search_mode": "continuous (Phases A-C), then 1 kg/m3 integer re-optimization profile (Phase D)",
        "continuous_frontier_best_found_C_kg_m3": continuous_frontier_C,
        "lexicographic_selection_result": result,
        "user_experimental_variability_used": False,
        "gradation_shape_penalty_used": False,
        "gradation_shape_preference_used": False,
        "global_optimality_claim": False,
        "PHYSICAL_INPUTS_COMPLETE": False, "VOLUME_BASIS_CHECKED": False, "EXPERIMENTALLY_VERIFIED": False,
        "batch_ready": "NOT_YET_DECLARED",
        "proposal_kind": "manufacturing-robust Stage-One computational proposal",
    }

    if result["status"] == "FEASIBLE_CANDIDATE_FOUND":
        selected_level = result["minimum_passing_cement_level_kg_m3"]
        try:
            BC.verify_adjacent_bracket(selected_level, tested_levels_status)
            decision["adjacent_integer_bracket_established"] = True
        except BC.BracketNotEstablishedError as e:
            decision["adjacent_integer_bracket_established"] = False
            decision["bracket_error"] = str(e)
            decision["status_override"] = "STAGE1_NOT_READY_TO_FREEZE"

        decision["minimum_passing_cement_level_kg_m3"] = selected_level
        decision["selected_candidate_id"] = result["selected_candidate"].get("cid")
        decision["selected_candidate_R_LCB95"] = result["selected_candidate"]["R_LCB95_joint"]
        lower = selected_level - 1.0
        if lower in tested_levels_status:
            decision["next_lower_tested_level_kg_m3"] = lower
            decision["next_lower_level_status"] = tested_levels_status[lower]
    else:
        decision["status_override"] = "STAGE1_NOT_READY_TO_FREEZE"

    with open(os.path.join(RUN, "decision.json"), "w") as f:
        json.dump(decision, f, indent=2, default=str)
    print(json.dumps(decision, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--continuous-frontier-C", type=float, default=None)
    a = ap.parse_args()
    main(a.continuous_frontier_C)
