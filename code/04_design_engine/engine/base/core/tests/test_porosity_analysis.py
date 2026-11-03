import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))


def test_replicate_groups_are_same_study_and_have_tight_tolerances():
    import porosity_analysis as pa
    groups = pa.identify_replicate_groups(pa.load_training_data())
    assert len(groups) >= 10
    assert (groups["n_rows"] >= 2).all()
    assert groups["study_id"].nunique() >= 5


def test_replicate_evidence_is_reproducible():
    import porosity_analysis as pa
    a, sa = pa.replicate_scatter_evidence(n_bootstrap=100, seed=7)
    b, sb = pa.replicate_scatter_evidence(n_bootstrap=100, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert sa == sb
    assert np.isfinite(sa["pooled_within_group_sd_pp"])


def test_candidate_loader_and_contract():
    import porosity_analysis as pa
    c = pa.load_candidate(pa.TEMPLATE / "tables" / "FINAL_STAGE1_CANDIDATE.csv")
    assert set(c) == {"C", "r", "A", "P", "S", "phi", "g"}
    assert np.isclose(np.sum(c["g"]), 1.0)
    contract = pa.load_porosity_contract()
    assert contract["deployment_domain_pp"] == [20.0, 30.0]
    assert contract["guaranteed_tolerance_delta_pp"] == [0.0, 1.0, 2.0]


def test_crn_porosity_draws_share_order_and_differ_by_setting():
    import porosity_analysis as pa
    banks = pa.build_common_random_banks([(0.5, 0.0), (2.0, 1.0)], 128, 10)
    a = banks[(0.5, 0.0)].eps_phi_pp
    b = banks[(2.0, 1.0)].eps_phi_pp
    assert not np.array_equal(a, b)
    assert banks[(0.5, 0.0)].delta_C is banks[(2.0, 1.0)].delta_C


def test_window_is_contiguous_and_reports_failure_binding():
    import porosity_analysis as pa
    grid = [20.0, 20.25, 20.5, 20.75, 21.0]
    rows = [{"R_LCB95_joint": x, "permeability_only_failure": 0.2, "strength_only_failure": 0.01} for x in [0.91, 0.92, 0.88, 0.91, 0.92]]
    out = pa._largest_contiguous_window(grid, rows, target=0.9, design_phi=20.0)
    assert out["phi_low"] == 20.0 and out["phi_high"] == 20.25
    assert out["binding_low"] == "permeability"
