"""End-to-end reproduction of the held-out reliability audit from the stored study-grouped fold predictions (CPU, about 3 min).

1. build_bank.py rebuilds the paired residual bank from the five fold prediction sets and checks it against the bank shipped with the engine.
2. audit_mtind.py re-runs the porosity-conditional audit over the 10,096 held-out comparisons.
The engine is copied to a temporary folder, so the repository is not modified."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code/03_validation_and_audit"


def run(args, cwd):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, *map(str, args)], cwd=cwd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


@pytest.mark.slow
def test_bank_and_audit_reproduce_paper(tmp_path):
    eng = tmp_path / "design_engine/engine"
    shutil.copytree(ROOT / "code/04_design_engine/engine", eng, ignore=shutil.ignore_patterns("__pycache__"))
    out = run([CODE / "build_bank.py", eng, "P_check", "legacy", "near3icm", ROOT / "results/forward_model_folds/lab_effect"], tmp_path)
    info = json.loads(out[out.rindex("BUILD DONE") + len("BUILD DONE"):].strip().splitlines()[0])
    assert info["P_check_same_rows"] and info["P_check_max_abs_diff_vs_paper_bank"] < 1e-9
    assert info["n_cal"] == 631 and info["n_studies"] == 35
    assert abs(info["oos_rmse_CS"] - 2.753) < 5e-4 and abs(info["oos_rmse_lnk"] - 0.3078) < 5e-4

    run([CODE / "audit_mtind.py", CODE / "audit_code", tmp_path / "design_engine/variants", tmp_path / "audit", "P_check"], tmp_path)
    m = json.load(open(tmp_path / "audit/audit_summary.json"))["P_check"]["paired"]["metrics"]
    stored = json.load(open(ROOT / "results/audit/audit_summary.json"))["P_check"]["paired"]["metrics"]
    assert m["n_pairs"] == 10096 and m["n_accept"] == 495
    assert round(m["cal_slope"], 3) == 1.012
    assert round(m["accept_observed_rate"], 3) == 0.935 and round(m["accept_mean_predicted"], 3) == 0.956
    for k in ("brier", "cal_slope", "accept_observed_rate", "accept_mean_predicted"):
        assert abs(m[k] - stored[k]) < 1e-9, k
