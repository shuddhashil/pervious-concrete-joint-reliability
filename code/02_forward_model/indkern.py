"""Plain-torch evaluation of a fitted ScaleKernel(sum of RBF / RQ leaves) so the IND model can run inside the engine process without importing the
fit-time model classes (the engine already holds its own copy of LCMGP_without_head). Leaves are read from the fitted gpytorch kernel; the value
is checked against gpytorch in mtind_fits.export_ind before anything is saved."""
import torch


def leaves_of(scale_kernel):
    out = []
    for m in scale_kernel.base_kernel.modules():
        name = type(m).__mro__
        if any(c.__name__ == "RQKernel" for c in name): kind = "rq"
        elif any(c.__name__ == "RBFKernel" for c in name): kind = "rbf"
        else: continue
        dims = m.active_dims if m.active_dims is not None else torch.arange(m.lengthscale.shape[-1])
        out.append(dict(kind=kind, dims=[int(d) for d in dims], ls=m.lengthscale.detach().double().reshape(-1).cpu().tolist(),
                        alpha=float(m.alpha.detach()) if kind == "rq" else None))
    return dict(outputscale=float(scale_kernel.outputscale.detach()), leaves=out)


def K(spec, x1, x2):
    """spec from leaves_of; x1 (n,d), x2 (m,d) float64 tensors on one device -> (n,m)."""
    tot = None
    for lf in spec["leaves"]:
        ls = torch.as_tensor(lf["ls"], dtype=x1.dtype, device=x1.device); dims = lf["dims"]
        a, b = x1[:, dims] / ls, x2[:, dims] / ls
        d2 = torch.clamp((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T, min=0.0)
        v = torch.exp(-0.5 * d2) if lf["kind"] == "rbf" else (1.0 + d2 / (2.0 * lf["alpha"])) ** (-lf["alpha"])
        tot = v if tot is None else tot + v
    return spec["outputscale"] * tot
