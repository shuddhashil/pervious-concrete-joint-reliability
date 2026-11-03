# Porosity semantics and robustness modes

* **Measured hardened porosity** is the `porosity` column in the reconciled
  training CSV. It is the coordinate on which the frozen surrogate was
  trained.
* **Nominal design porosity** (`phi_nominal`) is the target selected by the
  inverse designer. It is not asserted to equal the hardened value.
* **Achieved porosity** (`phi_actual`) is the nominal target plus `eps_phi`, a
  truncated-normal residual scatter used in each manufacturing scenario.
* **Guaranteed tolerance** (`delta`) is a deterministic offset used by the
  robust Option-B mode. The design is evaluated at `phi_nominal-delta`,
  `phi_nominal`, and `phi_nominal+delta`; it is not a second random error.

Total hardened porosity is used as a proxy for connected porosity because no
separate connected-porosity measurement is available in the supplied table.
This proxy is a model limitation, not a physical identity.

The deployment domain is 20--30 percent hardened porosity. The empirical
Stage-17 recipe/HDR support gate and OCSVM+KDE threshold 0.25 are applied
before performance reliability. Values outside the deployment window are
rejected/penalised rather than extrapolated.

Sensitivity, window, and robust-tolerance analyses reuse the same
manufacturing perturbations and paired residual rows across settings. Every
output records scenario count, target reliability, and whether `R_LCB95` used
Wilson only (smoke mode) or the minimum of Wilson and study-cluster bootstrap
bounds (final mode). A reduced smoke run is not final verification.
