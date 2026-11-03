# base (A3-1) - GuidedEngine phase 1
Copy of DefensibleEngine_20261001 `core` + `defengine` (A3 base, 650 rows), with the A3-1 artefacts rebuilt by tools/build_base.py and tools/build_s17.py (see PHASE1_REPORT.md).
Rebuilt: models/*, tables/paired_residual_bank.csv + residual_bank.npz + RESIDUAL_BANK_VALIDATION.csv, UNCERTAINTY_CONTRACT.json (paired-bank block + model hashes),
state_support_stage17/*, state_support_gradation/* + GRADATION_SUPPORT_CONTRACT.json (eps 0.1014), data_inputs/*, tables/training_data_reconciled.csv.
STALE (650-row, documentation only, not read by the evaluator; do not cite for A3-1): tables/POROSITY_REPLICATE_*.csv, tables/uncertainty_parameters.csv, tables/JOINT_VS_MARGINAL.csv,
tables/run02_candidate_pool_reference.csv, figures/*, PARAMETER_PROVENANCE.md.
tests_old/ and reference/ are copies of the old engine's tests and 2000 reference designs (inputs only; their stored verdict labels belong to the OLD base).
Hash manifest: CORE_MANIFEST_A3-1.json (core) ; logs/A3-1_core_hash.json (combined). Porosity SD is NOT in the contract (kept 1.0 pp = frozen baseline); the engine passes it as a parameter.
