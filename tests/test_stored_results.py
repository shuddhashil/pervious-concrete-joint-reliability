"""Stored GPU-run outputs carry the values reported in the paper (seconds, CPU)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def test_design_space_map_counts():
    c = json.load(open(RES / "design_space_map/counts_v3.json"))
    assert c["n_targets"] == 920 and c["supported_final_cells"] == 389 and c["data_backed_cells"] == 154
    assert c["monotonic"]["final"]["n_violations"] == 0


def test_six_target_verdicts():
    t = {(x["S_target"], x["K_target"]): x["engine_verdict"] for x in json.load(open(RES / "design_space_map/six_targets_v3.json"))["targets"]}
    for st in [(10, 8), (14, 6), (16, 10), (20, 4)]:
        assert t[st] == "certified"
    for st in [(18, 13), (25, 13)]:
        assert t[st] != "certified"


def test_study_grouped_accuracy_of_both_forward_models():
    import numpy as np
    for kind, cs, lnk in [("MT", 2.753, 0.308), ("IND", 2.821, 0.302)]:
        e = {"S": [], "Q": []}
        for k in range(5):
            d = json.load(open(RES / f"forward_model_folds/{kind}_fold{k}.json"))
            p = d["protocols"]["near3"]; ev = np.asarray(p["eval"], int); mu = np.asarray(p["mu"])[ev]
            e["S"] += list(np.asarray(d["y_CS"])[ev] - mu[:, 0]); e["Q"] += list(np.asarray(d["y_lnk"])[ev] - mu[:, 1])
        rmse = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in e.items()}
        assert abs(rmse["S"] - cs) < 5e-3 and abs(rmse["Q"] - lnk) < 5e-3, (kind, rmse)
