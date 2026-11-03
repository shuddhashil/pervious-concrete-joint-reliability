# Joint-reliability inverse design of low-cement pervious concrete

Code, data and result files for the manuscript *Joint-Reliability Inverse Design of Low-Cement Pervious Concrete Using a Laboratory-Adaptive Gaussian Process*.

The framework does four things:

1. It fits a laboratory-adaptive Gaussian process to compiled pervious-concrete records. The model predicts compressive strength (CS) and ln permeability jointly.
2. It estimates the joint probability that a mixture meets both a strength target and a permeability target.
3. It screens candidates against empirical-support checks.
4. It searches for the lowest-cement design that reaches a required joint reliability.

`PAPER_MAP.md` lists which script and which result file correspond to each table and figure of the paper.

## Layout

| Folder | Contents |
|---|---|
| `data/` | Compiled mix-level records with harmonised gradation and converted CS (`data_with_passing_sizes.xlsx`).<br>Fold definitions: `splits_grouped5.json` holds the five study-grouped folds; `splits_A3-1_remapped.json` holds the five random 80/20 splits.<br>The four outside studies are in `outside_studies/`. |
| `code/01_data_preparation/` | Gradation harmonisation to six bins, and CS size conversion to 100 × 200 mm cylinders. |
| `code/02_forward_model/` | Model code (`LCMGP_without_head.py`).<br>Fits with the laboratory term (`lab_fits.py`, `lab_fits2.py`).<br>Coupled versus independent GP fits under the three trial-batch protocols (`mtind_fits.py`, `indkern.py`).<br>Kernel comparison (`kernel_comparison/`) and posterior sensitivity slopes (`sensitivity_slopes/`). |
| `code/03_validation_and_audit/` | Residual bank (`build_bank.py`), the held-out reliability audit (`audit_mtind.py`, `audit_code/`) and comparison with whole-study bootstrap intervals (`compare.py`).<br>Also: variant engine bases (`variant_bases/`), the nested-bank audit, porosity-region coverage, the outside-study validation, and the trial-batch distance analysis. |
| `code/04_design_engine/` | The inverse-design engine (`engine/`), described next.<br>Cross-model design evaluation and the independent-GP search: `cross_model/`. |
| `code/05_design_space_map/` | Candidate pool, classification, edge refinement, strict final evaluation and assembly of the design-space map. |
| `code/06_joint_vs_separate/` | Joint versus separate acceptance rules (Table 12). |
| `code/07_figures/` | Figure scripts. |
| `results/` | Outputs of the runs reported in the paper. |

The engine in `code/04_design_engine/engine/` contains:

- `gengine/`: search, cement descent, final evaluation and tables;
- `base/core/`: the frozen model, support models, residual bank and contracts;
- `stages/`: gradation families;
- `run_targets.py`: the entry point.

## Environment

- Python 3.12.
- Package versions are listed in `requirements.txt`.
- Model fitting, the inverse search and the design-space map were run on Google Colab with an NVIDIA T4 GPU, one target per runtime. Each target took several hours.
- The audit, comparison and table scripts run on a CPU.

## Running

Each script states its usage in its first lines. The order is:

1. Data preparation: `01_data_preparation`.
2. Forward-model fits:
   - `02_forward_model/lab_fits.py` for the full data and the random splits;
   - `mtind_fits.py` for the study-grouped folds with no, three random or three nearest trial batches, coupled and independent.
3. Residual bank and audit:
   - `03_validation_and_audit/build_bank.py`;
   - then `audit_mtind.py`;
   - then `compare.py <out_mtind> <audit_dir> <legacy_dir> <out_json>`.
4. Designs: `04_design_engine/engine/run_targets.py`, one target per run, at R* = 0.90.
5. Design-space map: `05_design_space_map/run_map.py`.
6. Joint versus separate acceptance: `06_joint_vs_separate/driver.py` and `bonf_driver.py`.

Notes:

- Some scripts still contain the absolute input and output paths of the original workstation. Set these to your local folders before running.
- The stored files in `results/` let you check the reported numbers without refitting. For example, running `compare.py` on `results/forward_model_folds` reproduces the following study-grouped numbers:
  - CS RMSE 2.75 MPa and ln k RMSE 0.308 with three nearest trial batches;
  - the independent-GP values;
  - the matched-subset values.

## Data

The records were compiled from the published studies listed in the paper's reference list and in Table 1; each record carries its source identifier. The outside-study files reproduce values reported in those publications.

## Licence

- Code: MIT (see `LICENSE`).
- Compiled data and result files: CC BY 4.0. Please cite the paper and the original studies.
