"""Kernel candidate definitions K0-K7.

K0 is the authoritative input covariance, always built by the AUTHORITATIVE
module (never reimplemented). K1-K4 are expressible through the authoritative
(kernel, use_additive, add_kernel) contract. K5-K7 remove exactly one grouped
additive component; they reuse every authoritative prior/constraint/builder
except the group-skipping kernel sum, which mirrors _make_cont_kernel line by
line (same _reg_ls, same RQ alpha prior, same ARD dims).
"""
from __future__ import annotations

import copy

import gpytorch
import numpy as np
import torch

# Predeclared candidate table. `custom_skip` names the additive group to drop
# (None = use authoritative builder directly).
CANDIDATES = {
    "K0": {"label": "K0_CURRENT_FULL", "kernel": "rbf+rq", "use_additive": True,
           "add_kernel": "rbf", "custom_skip": None,
           "desc": "Authoritative full kernel: global ARD RBF+RQ + 3 grouped RBF (Binder/Filler/Aggregate)"},
    "K1": {"label": "K1_RBF_ARD", "kernel": "rbf", "use_additive": False,
           "add_kernel": "rbf", "custom_skip": None,
           "desc": "Single global ARD RBF over all transformed inputs"},
    "K2": {"label": "K2_MATERN52_ARD", "kernel": "matern52", "use_additive": False,
           "add_kernel": "rbf", "custom_skip": None,
           "desc": "Single global Matern-5/2 ARD over all transformed inputs"},
    "K3": {"label": "K3_GLOBAL_RBF_RQ", "kernel": "rbf+rq", "use_additive": False,
           "add_kernel": "rbf", "custom_skip": None,
           "desc": "Global RBF+RQ only; all grouped physical kernels removed"},
    "K4": {"label": "K4_NO_RQ", "kernel": "rbf", "use_additive": True,
           "add_kernel": "rbf", "custom_skip": None,
           "desc": "K0 without the global RQ component"},
    "K5": {"label": "K5_NO_BINDER", "kernel": "rbf+rq", "use_additive": True,
           "add_kernel": "rbf", "custom_skip": "Binder",
           "desc": "K0 without the binder-group kernel (w/c, cem, SP)"},
    "K6": {"label": "K6_NO_FILLER", "kernel": "rbf+rq", "use_additive": True,
           "add_kernel": "rbf", "custom_skip": "Filler",
           "desc": "K0 without the filler/interstitial-group kernel (porosity, sand)"},
    "K7": {"label": "K7_NO_AGG", "kernel": "rbf+rq", "use_additive": True,
           "add_kernel": "rbf", "custom_skip": "Aggregate",
           "desc": "K0 without the aggregate-group kernel (WAS, I_gap, Cu, Aag)"},
}

ORDER = ["K0", "K1", "K2", "K3", "K4", "K5", "K6", "K7"]

SIMPLICITY_PREFERENCE = ["K1", "K2", "K3"]  # conceptual simplicity order


def is_authoritative_path(key):
    return CANDIDATES[key]["custom_skip"] is None


def make_custom_cont_kernel(lcm, skip_group):
    """Mirror of authoritative _make_cont_kernel for rbf+rq + additive minus one group."""
    nc = len(lcm.CONT_IDXS)
    cont = list(lcm.CONT_IDXS)
    k_rbf = gpytorch.kernels.RBFKernel(ard_num_dims=nc, active_dims=cont)
    lcm._reg_ls(k_rbf)
    k_rq = gpytorch.kernels.RQKernel(ard_num_dims=nc, active_dims=cont)
    lcm._reg_ls(k_rq)
    k_rq.register_constraint("raw_alpha", lcm.Interval(0.25, 10.0))
    k_rq.register_prior("alpha_prior", lcm.GammaPrior(2.0, 1.0), "alpha")
    with torch.no_grad():
        k_rq.alpha = 2.0
    k_sum = k_rbf + k_rq
    for name, idxs in lcm._additive_groups().items():
        if name == skip_group:
            continue
        kg = gpytorch.kernels.RBFKernel(ard_num_dims=len(idxs), active_dims=list(idxs))
        lcm._reg_ls(kg)
        k_sum = k_sum + kg
    return k_sum


class CustomLCMGPModel(gpytorch.models.ExactGP):
    """Rank-1 LCM with custom (group-skipped) input kernel; otherwise authoritative."""

    def __init__(self, train_x, train_y, likelihood, lcm, skip_group):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = lcm.TwoTaskConstantMean()
        shared_cont = make_custom_cont_kernel(lcm, skip_group)
        import math as _math
        shared_data = lcm._scaled_kernel(
            shared_cont, init=0.8, lo=0.05, hi=3.0,
            prior_log_mean=_math.log(0.8), prior_sd=0.4)
        self.shared_covar_module = gpytorch.kernels.MultitaskKernel(
            shared_data, num_tasks=int(lcm.NUM_TASKS), rank=1)

    def forward(self, x):
        mean = self.mean_module(x)
        shared_cov = self.shared_covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean, shared_cov)


def train_custom(lcm, Xtr_np, Ytr_std_np, skip_group, iters, lr, restarts, seed, fast_mode=True):
    """Training loop mirroring authoritative train_single/train_lcmgp exactly."""
    import math as _math
    best_loss, best_model, best_lik = float("inf"), None, None
    errors = []
    for r in range(max(1, int(restarts))):
        try:
            s = int(seed + 97 * r)
            lcm.set_seed(s)
            Xt = torch.as_tensor(np.asarray(Xtr_np, float), dtype=lcm.DTYPE,
                                 device=lcm.DEVICE).contiguous()
            Yt = torch.as_tensor(np.asarray(Ytr_std_np, float), dtype=lcm.DTYPE,
                                 device=lcm.DEVICE).contiguous()
            likelihood = lcm._build_likelihood()
            model = CustomLCMGPModel(Xt, Yt, likelihood, lcm, skip_group)
            model = model.to(device=lcm.DEVICE, dtype=lcm.DTYPE)
            model.train()
            likelihood.train()
            seen, params = set(), []
            for p in list(model.parameters()) + list(likelihood.parameters()):
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
            opt = torch.optim.Adam(params, lr=float(lr))
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            best, bad = float("inf"), 0
            bmsd = blsd = None
            mdev = next(model.parameters()).device
            ldev = next(likelihood.parameters()).device
            patience = max(80, int(iters) // 5)
            jseq = (0.001, 0.003, 0.01, 0.03, 0.1) if lcm.DTYPE == torch.float32 else \
                (0.0001, 0.0003, 0.001, 0.003, 0.01)
            for it in range(1, int(iters) + 1):
                opt.zero_grad(set_to_none=True)
                loss = None
                for jit in jseq:
                    try:
                        with lcm._train_settings(fast_mode), lcm._max_chol(fast_mode), \
                             gpytorch.settings.max_cg_iterations(int(lcm.MAX_CG_ITERATIONS)), \
                             gpytorch.settings.cholesky_jitter(float(jit)), \
                             gpytorch.settings.cg_tolerance(float(lcm.CG_TOLERANCE)):
                            loss = -mll(model(Xt), Yt)
                        break
                    except Exception:
                        loss = None
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss in custom kernel training")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=2.0)
                opt.step()
                cur = float(loss.detach())
                if cur + 1e-6 < best:
                    best, bad = cur, 0
                    with torch.no_grad():
                        bmsd = {k: (v.detach().to(device=mdev).contiguous()
                                    if torch.is_tensor(v) else copy.deepcopy(v))
                                for k, v in model.state_dict().items()}
                        blsd = {k: (v.detach().to(device=ldev).contiguous()
                                    if torch.is_tensor(v) else copy.deepcopy(v))
                                for k, v in likelihood.state_dict().items()}
                else:
                    bad += 1
                if bad >= int(patience):
                    break
            model.load_state_dict(bmsd)
            likelihood.load_state_dict(blsd)
            model.eval()
            likelihood.eval()
            if best < best_loss:
                if best_model is not None:
                    del best_model, best_lik
                best_loss, best_model, best_lik = float(best), model, likelihood
            else:
                del model, likelihood
            lcm._gpu_cleanup()
        except Exception as e:
            errors.append(repr(e))
            lcm._gpu_cleanup()
    if best_model is None:
        raise RuntimeError("All custom-kernel restarts failed: " + " | ".join(errors[-3:]))
    return best_model, best_lik, best_loss


def count_parameters(model, likelihood):
    n = 0
    for p in list(model.parameters()) + list(likelihood.parameters()):
        n += int(np.asarray(p.detach().cpu().numpy()).size)
    return n
