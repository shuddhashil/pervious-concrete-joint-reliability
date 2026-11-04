# Joint-reliability inverse design of low-cement pervious concrete

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23948289.svg)](https://doi.org/10.5281/zenodo.23948289)

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
| `inputs/` | Further inputs read by individual scripts (saved model variants, audit inputs, contract modules), one folder per script. |
| `tests/` | Automated checks (see Tests). |

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

- All paths are relative to the repository. Inputs that individual scripts read beyond `data/` and `results/` (saved model variants, audit inputs, contract modules) are in `inputs/<script name>/`.
- On Google Colab (when `/content` exists), `run_targets.py` and `run_map.py` write logs and result zips to `/content`; elsewhere they write to `runs/` in the repository.
- `results/` holds the outputs of the reported runs, so the numbers can be checked without refitting. For example, run `audit_mtind.py` on the fold files in `results/forward_model_folds`, then `compare.py` with `results/forward_model_folds/lab_effect` as the legacy folder. This reproduces the study-grouped numbers below. These were checked on 2026-11-03.
  - CS RMSE 2.753 MPa and ln k RMSE 0.308 with the three nearest trial batches;
  - the independent-GP values (2.821 and 0.302);
  - the matched-subset values (2.79-2.84 MPa).

## Tests

```
pip install -r requirements.txt pytest
python -m pytest            # about 1.5 min on a laptop CPU
python -m pytest -m "not slow"   # fast checks only, a few seconds
```

- `tests/test_data_and_folds.py`: 631 records from 35 studies; the five study-grouped folds hold out every record and every study exactly once.
- `tests/test_stored_results.py`: the stored GPU-run outputs carry the reported values (design-space map 389 of 920 targets, 154 backed by recorded mixtures; verdicts of the six targets; study-grouped accuracy of the coupled and independent GPs).
- `tests/test_reproduce_audit.py` (slow): rebuilds the paired residual bank from the stored fold predictions, checks it against the bank used by the design engine, and repeats the held-out audit over 10,096 comparisons (calibration slope 1.012; observed pass rate 0.935 against 0.956 stated where the stated reliability is at least 0.90). It works on a temporary copy of the engine and does not modify the repository.

## Runtime (as recorded in the reported runs)

| Step | Hardware | Time |
|---|---|---|
| Laboratory-adaptive GP, one study-grouped fold | Colab, NVIDIA T4 | 4–12 min |
| Coupled or independent GP, one fold, all adaptation protocols | Colab, NVIDIA T4 | 7.5–19 min |
| One cement-descent run (one seed, one gradation family), 10/8 target | Colab, NVIDIA T4 | 1–2 min |
| Final evaluation of one candidate, 100,000 simulated batches | Colab, NVIDIA T4 | 12–29 s |
| Boundary refinement of the design-space map | Colab, NVIDIA T4 | 33 min |
| Residual bank and held-out audit (test suite) | laptop CPU | about 1.5 min |

Sources: the `minutes` field of `results/forward_model_folds/*.json`, the `seconds` field of `results/design_space_map/certificates_v2.jsonl`, and `results/design_space_map/refinement_log.json`.

## Citation

See `CITATION.cff`. Archived on Zenodo: version v1.0.0, https://doi.org/10.5281/zenodo.23948289 (all versions: https://doi.org/10.5281/zenodo.23948288).

## Data

The records were compiled from the published studies listed in the paper's reference list and in Table 1; each record carries its source identifier. The outside-study files reproduce values reported in those publications.

## Licence

- Code: MIT (see `LICENSE`).
- Compiled data and result files: CC BY 4.0. Please cite the paper and the original studies.
