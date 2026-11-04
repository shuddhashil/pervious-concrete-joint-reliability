"""Checks on the compiled records and the study-grouped folds (seconds, CPU)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "code/04_design_engine/engine/base/core/data_inputs/phase1_reconciled_eligible.csv"
FOLDS = ROOT / "data/splits_grouped5.json"


def test_record_and_study_counts():
    d = pd.read_csv(RECORDS)
    assert len(d) == 631
    study = json.load(open(FOLDS))
    studies = set().union(*(set(f["studies"]) for f in study.values()))
    assert len(studies) == 35


def test_folds_partition_records_and_hold_out_whole_studies():
    folds = json.load(open(FOLDS))
    assert len(folds) == 5
    test_rows = []
    for f in folds.values():
        tr, te = set(f["train_idx"]), set(f["test_idx"])
        assert not tr & te
        assert len(tr) + len(te) == 631
        test_rows += f["test_idx"]
    # every record is held out exactly once
    assert sorted(test_rows) == list(range(631))
    # no study is held out in two folds
    held = [s for f in folds.values() for s in f["studies"]]
    assert len(held) == len(set(held))
