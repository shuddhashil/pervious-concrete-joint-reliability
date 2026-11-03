"""A3 forward-model contract.

Model A3 = 11 inputs: the six recipe variables + five passing sizes q10/q25/q50/q75/q90 (log2 mm).
Kernel: global RBF over the six recipe inputs only (ARD); global RQ over all 11 inputs with ONE shared lengthscale
for the five passing sizes; Binder / Filler additive RBF groups; no gradation-only group. Bounds, priors, rank-1 task
covariance as in the production model. The kernel construction below is copied from
research/LCMGP_Retrain_20260930/retrain.py (_tied_kernels with rbf_recipe_only=True, activate for Q5a3) so that the
saved fulldata_Q5a3 state loads into an identical model. Nothing here trains anything.

install(L) patches the frozen LCMGP module's feature contract and kernel builder in the CURRENT process.
"""
import torch

RECIPE = ['porosity', 'w/c', 'cem (kg/m3)', 'CA (kg/m3)', 'SP (kg/m3)', 'sand (kg/m3)']
Q = ['q10', 'q25', 'q50', 'q75', 'q90']
FEATURES = RECIPE + Q
TRANSFORM_RECIPE = {'log_cols': ['cem (kg/m3)', 'CA (kg/m3)'], 'log1p_cols': ['SP (kg/m3)', 'sand (kg/m3)'],
                    'raw_cols': ['porosity', 'w/c'] + Q, 'categorical_cols': []}
CFG = {'kernel': 'rbf+rq', 'rank': 1, 'use_additive': True, 'add_kernel': 'rbf'}
CONTRACT_NAME = "A3-11D [porosity, w/c, cement, CA(log), SP, sand, q10, q25, q50, q75, q90 (log2 mm)]"


def groups():
    ix = FEATURES.index
    return {'Binder': [ix('w/c'), ix('cem (kg/m3)'), ix('SP (kg/m3)')], 'Filler': [ix('porosity'), ix('sand (kg/m3)')]}


def _tied_kernels(L, rbf_recipe_only=True):
    import gpytorch
    from gpytorch.constraints import Interval
    from gpytorch.priors import LogNormalPrior, GammaPrior
    n_rec = len(RECIPE); n_all = len(FEATURES)
    idx = list(range(n_rec)) + [n_rec] * (n_all - n_rec)          # map 11 inputs -> 7 distinct lengthscales

    def tie(cls):
        class Tied(cls):
            def __init__(self, **kw):
                super().__init__(ard_num_dims=n_rec + 1, **kw)
                self.register_buffer("tie_idx", torch.tensor(idx, dtype=torch.long))
                self.ard_num_dims = n_all

            @property
            def lengthscale(self):
                return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)[..., self.tie_idx]

            @lengthscale.setter
            def lengthscale(self, value):
                self._set_lengthscale(value)
        Tied.__name__ = "Tied" + cls.__name__
        return Tied

    def reg(k):
        k.register_constraint("raw_lengthscale", Interval(0.2, 4.0))
        k.register_prior("lengthscale_prior", LogNormalPrior(0.0, 0.45), lambda m: m.raw_lengthscale_constraint.transform(m.raw_lengthscale))
        with torch.no_grad():
            k.initialize(raw_lengthscale=k.raw_lengthscale_constraint.inverse_transform(torch.ones_like(k.raw_lengthscale)))
        return k

    def build(kernel_name, use_additive, add_kernel_name):
        cont = list(L.CONT_IDXS)
        k1 = (reg(gpytorch.kernels.RBFKernel(ard_num_dims=n_rec, active_dims=list(range(n_rec))))
              if rbf_recipe_only else reg(tie(gpytorch.kernels.RBFKernel)(active_dims=cont)))
        k2 = reg(tie(gpytorch.kernels.RQKernel)(active_dims=cont))
        k2.register_constraint("raw_alpha", Interval(0.25, 10.0)); k2.register_prior("alpha_prior", GammaPrior(2.0, 1.0), "alpha")
        with torch.no_grad(): k2.alpha = 2.0
        ks = k1 + k2
        if use_additive:
            for g in L._additive_groups().values():
                kg = gpytorch.kernels.RBFKernel(ard_num_dims=len(g), active_dims=g); L._reg_ls(kg); ks = ks + kg
        return ks
    return build


def install(L):
    """Patch the frozen module in this process; idempotent."""
    L.FEATURES = list(FEATURES); L.CONT_IDXS = list(range(len(FEATURES)))
    L.TRANSFORM_RECIPE = {k: list(v) for k, v in TRANSFORM_RECIPE.items()}
    g = groups()
    L._additive_groups = lambda: g
    L._make_cont_kernel = _tied_kernels(L, rbf_recipe_only=True)
    return list(FEATURES)
