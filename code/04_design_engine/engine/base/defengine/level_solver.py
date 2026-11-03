"""Level solver: for ONE fixed cement level, find the design (other 11 variables) that maximises joint reliability.
Batched jDE (self-adaptive differential evolution). Common random numbers: ONE manufacturing bank and ONE residual-draw set per solve,
so candidates are compared on identical noise. Gate-violating designs are never sent through the forward model (they get a penalty tier).
Search-time only; acceptance happens in verify.py with fresh seeds (optimiser's-curse protection)."""
import time, math
import numpy as np
from . import corelib as C, gates as G

D_FREE = 11                      # r, A, P, S, phi, m1..m6 (cement fixed)
FREE_IDX = list(range(1, 12))    # positions inside the frozen 12-vector x (x[0] = cement)


def _full(Zfree, cement):
    Z = np.zeros((len(Zfree), 12)); Z[:, 1:] = Zfree
    return C.decode_unit(Z, cement=cement)


def objective(noms, gate, bank, eS, eQ):
    """Larger is better. Tiers (lexicographic, nothing clipped inside a tier so the search always has a gradient):
    gate-violating (-1e4 - 1e3*penalty)  <  model-unsupported scenarios (-1e3 + 100*support_fraction)  <  fully supported (CVaR margin)."""
    n = len(noms); pen = gate.penalties(noms)["penalty"]
    J = -1.0e4 - 1.0e3 * pen
    ok = pen <= 0.0
    info = dict(rhat=np.zeros(n), cvar=np.full(n, -np.inf), support_fraction=np.zeros(n))
    if ok.any():
        idx = np.flatnonzero(ok); st = C.batch_search_stats([noms[i] for i in idx], bank, eS, eQ)
        for k in ("rhat", "cvar", "support_fraction"): info[k][idx] = st[k]
        c = st["cvar"]; J[idx] = np.where(np.isfinite(c), c, -1.0e3 + 100.0 * st["support_fraction"])
    return J, pen, info


def _lhs(rng, n, d):
    u = (rng.permuted(np.tile(np.arange(n), (d, 1)), axis=1).T + rng.uniform(size=(n, d))) / n
    return u


def initial_population(rng, P, cement, training_anchor, warm, exact):
    """EXACT copies of the best designs known from other levels (so a level can never do worse than its neighbours), perturbed copies of warm
    starts, data-anchored proposals (real recipe/gradation), and space-filling points."""
    exact = list(exact); n_ex = len(exact); n_warm = min(len(warm), P // 5); n_anchor = int(0.3 * P); n_lhs = max(P - n_ex - n_warm - n_anchor, 4)
    pop = [_lhs(rng, n_lhs, D_FREE)]
    if n_anchor:
        a = training_anchor[rng.integers(0, len(training_anchor), n_anchor)]; pop.append(np.clip(a + rng.normal(0, 0.02, a.shape), 0, 1))
    if n_warm:
        w = np.array(warm[:n_warm]); pop.append(np.clip(w + rng.normal(0, 0.01, w.shape), 0, 1))
    if n_ex: pop.append(np.array(exact))
    return np.vstack(pop)[:max(P, n_ex)]


def solve_level(cement, seed, P=80, generations=30, n_scen=256, warm=None, pool=None, n_exact=8, bank_seed=None, verbose=False, gate=None, rerank_n=1024):
    """Returns dict(best=[top distinct designs, ranked on a FRESH bank], history, evals, seconds). pool: list of unit-cube vectors found at other levels."""
    t0 = time.time(); rng = np.random.default_rng(seed); gate = gate or G.SearchGate()
    bank = C.mfg_bank(n_scen, seed=(bank_seed if bank_seed is not None else seed + 1)); eS, eQ, _ = C.res_bank_iid(n_scen, seed=seed + 2)
    anchor = _training_anchor_vectors(); exact = []
    if pool:
        Zp = np.array(pool); Jp, _, _ = objective(_full(Zp, cement), gate, bank, eS, eQ); exact = [Zp[i] for i in np.argsort(-Jp)[:n_exact]]
    pop = initial_population(rng, P, cement, anchor, warm or [], exact); P = len(pop)
    J, pen, info = objective(_full(pop, cement), gate, bank, eS, eQ); evals = len(pop)
    F = np.full(P, 0.5); CR = np.full(P, 0.9); hist = [float(J.max())]
    for g in range(generations):
        Fi = np.where(rng.random(P) < 0.1, rng.uniform(0.1, 1.0, P), F); CRi = np.where(rng.random(P) < 0.1, rng.random(P), CR)
        trial = np.empty_like(pop)
        for i in range(P):
            a, b, c = rng.choice(np.delete(np.arange(P), i), 3, replace=False)
            mutant = pop[a] + Fi[i] * (pop[b] - pop[c]); cross = rng.random(D_FREE) < CRi[i]; cross[rng.integers(D_FREE)] = True
            v = np.where(cross, mutant, pop[i]); lo = v < 0; hi = v > 1
            v = np.where(lo, -v, v); v = np.where(hi, 2 - v, v); trial[i] = np.clip(v, 0, 1)
        Jt, pent, infot = objective(_full(trial, cement), gate, bank, eS, eQ); evals += P
        better = Jt >= J
        pop[better] = trial[better]; J[better] = Jt[better]; pen[better] = pent[better]
        for k in info: info[k][better] = infot[k][better]
        F = np.where(better, Fi, F); CR = np.where(better, CRi, CR); hist.append(float(J.max()))
        if verbose: print(f"  C={cement:5.0f} gen {g+1:3d} best J {J.max():7.3f} rhat {info['rhat'][J.argmax()]:.3f} gate-ok {int((pen <= 0).sum())}/{P}", flush=True)
    order = np.argsort(-J); out = []; seen = []
    for i in order:
        if any(np.abs(pop[i] - s).max() < 0.02 for s in seen): continue
        seen.append(pop[i])
        out.append(dict(z=pop[i].tolist(), J=float(J[i]), penalty=float(pen[i]), rhat_search=float(info["rhat"][i]), cvar_search=float(info["cvar"][i])))
        if len(out) >= 6: break
    # re-rank on a FRESH bank (unbiased by the optimiser's selection)
    Zb = np.array([o["z"] for o in out]); noms = _full(Zb, cement)
    fb = C.mfg_bank(rerank_n, seed=seed + 777); fS, fQ, _ = C.res_bank_iid(rerank_n, seed=seed + 778)
    Jf, penf, inff = objective(noms, gate, fb, fS, fQ)
    for k, o in enumerate(out):
        o["rhat"] = float(inff["rhat"][k]); o["cvar"] = float(inff["cvar"][k]); o["support_fraction"] = float(inff["support_fraction"][k]); o["penalty"] = float(penf[k])
        o["nominal"] = {kk: (v.tolist() if hasattr(v, "tolist") else float(v)) for kk, v in noms[k].items()}
    out.sort(key=lambda o: (o["penalty"] > 0, -o["rhat"], -o["cvar"]))
    return dict(cement=cement, seed=seed, n_scen=n_scen, P=P, generations=generations, best=out, history=hist, evals=evals, seconds=time.time() - t0)


_ANCHOR = {}


def _training_anchor_vectors():
    """Unit-cube vectors (11 free dims) of the real training mixes (recipe + gradation as recorded): the data-anchored proposals."""
    if "a" not in _ANCHOR:
        import pandas as pd
        d = pd.read_csv(C.TRAIN_CSV); d = d[(d["perm (mm/s)"] > 0) & np.isfinite(d["comp_converted"])].reset_index(drop=True)
        g = d[C.E.BINS].to_numpy(float); g = g / g.sum(1, keepdims=True)
        B = C.SC.BND; z = np.zeros((len(d), 12))
        for j, (col, key) in enumerate(((None, "C"), ("w/c", "r"), ("CA (kg/m3)", "A"), ("SP (kg/m3)", "P"), ("sand (kg/m3)", "S"), ("porosity", "phi"))):
            if col is None: continue
            lo, hi = B[key]; v = d[col].to_numpy(float)
            z[:, j] = (np.log(np.clip(v, *B[key])) - C.SC.LNA_LO) / (C.SC.LNA_HI - C.SC.LNA_LO) if key == "A" else (np.clip(v, lo, hi) - lo) / (hi - lo)
        z[:, 6:12] = g
        _ANCHOR["a"] = np.clip(z[:, 1:], 0, 1)
    return _ANCHOR["a"]
