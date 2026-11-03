# Paper item → code → result file

| Paper item | Code | Stored result |
|---|---|---|
| Section 2.2, Figure 1: gradation harmonisation | `code/01_data_preparation/gradation_part1.py`, `gradation_part2_selected_bins.py` | `data/data_with_passing_sizes.xlsx` |
| Section 2.3: CS size conversion | `code/01_data_preparation/conversion of strength to 100x200 (batch).py` | `data/data_with_passing_sizes.xlsx` (`comp_converted`) |
| Section 3.2, Table 7: kernel comparison | `code/02_forward_model/kernel_comparison/` | — |
| Section 4.1: random-split accuracy | `code/02_forward_model/lab_fits.py`, `lab_fits2.py` | `results/forward_model_folds/lab_effect/` |
| Section 4.1, Table 8: study-grouped accuracy, coupled vs independent | `code/02_forward_model/mtind_fits.py`, `code/03_validation_and_audit/compare.py` | `results/forward_model_folds/MT_fold*.json`, `IND_fold*.json` |
| Table 6, Figure 5: four outside studies | `code/03_validation_and_audit/outside_studies/` | `results/outside_studies/VALIDATION_TABLE.*` |
| Figure 7: posterior sensitivity slopes | `code/02_forward_model/sensitivity_slopes/open_gp.py` | `results/sensitivity_slopes/` |
| Section 4.3: residual bank by porosity band | `code/03_validation_and_audit/build_bank.py` | — |
| Section 4.4, Figure 8, Table 8: held-out reliability audit | `code/03_validation_and_audit/audit_mtind.py`, `audit_code/`, `compare.py` | `results/audit/audit_summary.json` |
| Section 4.4: nested-bank audit | `code/03_validation_and_audit/nested_audit/nested_audit.py` | `results/audit/nested/nested_audit_run.json` |
| Section 4.4, Figure 9: no, random and nearest trial batches; trial-batch distance | `mtind_fits.py`, `compare.py`, `code/03_validation_and_audit/trial_batch_distance/` | `results/audit/audit_summary.json` |
| Section 4.4: porosity-region coverage (check iv) | `code/03_validation_and_audit/porosity_region_coverage/` | — |
| Section 4.5, Figure 10: design space | `code/05_design_space_map/run_map.py` (pipeline), `code/07_figures/make_figure_v3.py` | `results/design_space_map/` (`counts_v3.json`: 389 of 920) |
| Section 4.7, Tables 10–11, Figures 11–14: designs for the six targets | `code/04_design_engine/engine/` | `results/designs/S*_TABLES.md`, `S*_global_scan.csv` |
| Table 12: joint vs separate acceptance | `code/06_joint_vs_separate/` | `results/joint_vs_separate/` |
| Table 13: score components; designs of the independent GPs; cross-model scores | `code/04_design_engine/cross_model/` | `results/designs/`, `results/forward_model_folds/IND_*` |
