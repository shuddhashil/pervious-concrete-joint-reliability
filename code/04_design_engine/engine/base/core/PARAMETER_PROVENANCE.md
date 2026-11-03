# Achieved-porosity parameter provenance

The inverse-design coordinate `phi_nominal` is a designer-selected target
porosity in percent. The surrogate was trained on measured hardened porosity,
which is an observation of the realised specimen, not a target recorded at
design time. During reliability evaluation the realised value is
`phi_actual = phi_nominal + eps_phi`, where `eps_phi` is a zero-mean truncated
normal residual with a baseline SD of 1.0 percentage point and truncation at
plus or minus three SD. The value is retained as an engineering prior.

`template/src/porosity_analysis.py` checks the available direct evidence by
grouping rows within the same study whose cement, water, coarse aggregate,
sand, superplasticiser, and six reported gradation fractions agree within
0.001 kg/m3 for masses and 0.001 percentage points for gradation fractions.
The resulting counts and pooled/robust scatter are written to
`tables/POROSITY_REPLICATE_SCATTER.csv` and
`tables/POROSITY_REPLICATE_GROUPS.csv`. Bootstrap resampling is by usable
near-identical recipe group, not by row.

These replicate groups are supporting evidence only. They combine any
measurement, curing, and within-study recipe differences and therefore do not
identify a universal batch-to-batch achieved-porosity SD. No aggregate
density or volume-balance constant is inferred. The credibility check is the
explicit SD/bias sensitivity grid and the porosity-robust tolerance analysis,
not a claim that 1.0 percentage point is measured project-specific scatter.
