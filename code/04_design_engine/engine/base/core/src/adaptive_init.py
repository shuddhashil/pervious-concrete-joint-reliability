"""Target-agnostic adaptive initialiser (TASK A).

Stage 0 (space-filling): seeded space-filling sample for the
6 scalar coordinates in the unit cube (default 4096, contract seed). For the
gradation, mixture of (i) Dirichlet alpha=1, (ii) Dirichlet alpha=0.3,
(iii) data-fitted gradation prior (gradation_prior.py). Convert g to sticks
with codec11.g_to_sticks so the rest of the pipeline is unchanged.

Stage 0b (data-anchored): perturb/interpolate REAL training recipes in the
model's scaled feature space with random weights and small jitter, project
to bounds and simplex. Never copies rows verbatim.

Screening: bounds, recipe kNN gate + porosity HDR (stage17_state_support),
then surrogate reliability on a small scenario bank.  Every Stage-0 and
Stage-0b candidate is screened once.  Later rounds spend more scenarios on a
smaller, *non-dominated* reliability/cement/support frontier.

Adaptive refinement: cross-entropy / successive-halving loop fitting a
diagonal Gaussian on elites in the unit cube, resampling, re-screening with
growing scenario count. Elite = feasible (Rhat above working threshold)
with lowest C plus diversity rule (L2 distance threshold in unit cube).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

try:
    from scipy.stats import qmc
except Exception:  # pragma: no cover - bundled runtime includes scipy
    qmc = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import codec11 as K  # noqa: E402
import gradation_prior as GP  # noqa: E402

try:  # Blind-repair helpers are optional at import time (tests stub them).
    import continuous_score as CS  # noqa: E402
    import basin_archive as BA  # noqa: E402
    import proposal_engine as PE  # noqa: E402
except Exception:  # pragma: no cover - production bundle includes them
    CS = None
    BA = None
    PE = None

ORDER = "coarse_first"
DEFAULT_N0 = 4096
MIX_W = (0.4, 0.3, 0.3)


def _dirichlet_mixture(rng, n, w=MIX_W):
    n1, n2 = int(n * w[0]), int(n * w[1])
    n3 = n - n1 - n2
    parts = []
    if n1:
        parts.append(rng.dirichlet([1.0] * 6, size=n1))
    if n2:
        parts.append(rng.dirichlet([0.3] * 6, size=n2))
    if n3:
        parts.append(GP.sample_prior(rng, n3))
    g = np.concatenate(parts, axis=0) if parts else np.empty((0, 6))
    rng.shuffle(g)
    return g


def _latin_hypercube(n, d, seed):
    """Deterministic maximin-free LHS fallback for runtimes without qmc."""
    r = np.random.default_rng(int(seed))
    u = (np.arange(n, dtype=float)[:, None] + r.random((n, d))) / max(n, 1)
    for j in range(d):
        r.shuffle(u[:, j])
    return u


def _space_filling_scalars(n, seed, method="sobol"):
    """Return a reproducible six-dimensional unit-cube design.

    Scrambled Sobol is used for the production initializer.  The fallback is
    a seeded Latin hypercube, never an unconstrained iid cloud, so every
    scalar coordinate is covered before adaptive refinement starts.
    """
    if n <= 0:
        return np.empty((0, 6), dtype=float)
    if method == "sobol" and qmc is not None:
        return np.asarray(qmc.Sobol(d=6, scramble=True, seed=int(seed)).random(n), dtype=float)
    return _latin_hypercube(n, 6, seed)


def stage0_sample(rng, n=DEFAULT_N0, seed_scalars=None, scalar_method="sobol"):
    # This is intentionally target-agnostic: only the declared seed and the
    # data-fitted gradation prior influence proposals; no result coordinates
    # are an input to this sampler.
    if seed_scalars is None:
        seed_scalars = int(rng.integers(0, 2**32 - 1))
    u_scalars = _space_filling_scalars(n, seed_scalars, scalar_method)
    g = _dirichlet_mixture(rng, n)
    sticks = np.array([K.g_to_sticks(gg, ORDER) for gg in g])
    return np.concatenate([u_scalars, sticks], axis=1), g


def stage0b_data_anchored(rng, n, training_csv=None, jitter=0.02, k=3):
    training_csv = training_csv or GP.TRAINING_CSV
    df = pd.read_csv(training_csv)
    # map training columns to design keys where possible
    cols = {"cem (kg/m3)": "C", "w/c": "r", "CA (kg/m3)": "A",
            "SP (kg/m3)": "P", "sand (kg/m3)": "S", "porosity": "phi"}
    avail = [c for c in cols if c in df.columns]
    base = df[avail + GP.BIN_COLS].dropna()
    idx = rng.integers(0, len(base), size=(n, k))
    w = rng.dirichlet([1.0] * k, size=n)
    out_X, out_g = [], []
    for i in range(n):
        rows = base.iloc[idx[i]]
        # blend scalars
        cand = {}
        for c in avail:
            vals = rows[c].to_numpy(dtype=float)
            cand[cols[c]] = float((vals * w[i]).sum() + rng.normal(0, jitter * (vals.std() + 1e-9)))
        # blend gradation bins then renormalise
        gb = rows[GP.BIN_COLS].to_numpy(dtype=float)
        gb = gb / gb.sum(axis=1, keepdims=True).clip(min=1e-12)
        g = (gb * w[i][:, None]).sum(axis=0)
        g = np.clip(g + rng.normal(0, jitter / 6.0, size=6), 0, None)
        g = g / g.sum() if g.sum() > 0 else np.full(6, 1.0 / 6)
        out_g.append(g)
        # encode scalars via codec bounds
        tmp = {"C": cand.get("C", 300.0), "r": cand.get("r", 0.3),
               "A": cand.get("A", 1400.0), "P": max(0.0, cand.get("P", 0.0)),
               "S": max(0.0, cand.get("S", 0.0)), "phi": cand.get("phi", 25.0), "g": g}
        try:
            out_X.append(K.encode11(tmp, ORDER))
        except Exception:
            continue
    return np.array(out_X), np.array(out_g)


def _frontier_order(rhats, cements, supported, target_reliability=0.90):
    """Return a deterministic, low-cement/reliability frontier order.

    ``supported`` is a hard scientific admissibility gate.  Within supported
    designs, neither cement nor reliability is hidden behind a 0.92 cut-off:
    candidates are ordered by Pareto layers in (reliability shortfall to 0.90,
    cement), then by the same two quantities for deterministic tie-breaking.

    Feasibility discovery is kept separate from cement minimization: callers
    that have a continuous search score should use
    :func:`feasibility_cement_frontier` instead, which keeps a feasibility
    archive (ordered by smooth search quality) alongside the cement frontier
    so a low-cement but hopeless candidate can never displace a substantially
    more feasible one.
    """
    rhats = np.asarray(rhats, dtype=float)
    cements = np.asarray(cements, dtype=float)
    supported = np.asarray(supported, dtype=bool)
    deficit = np.maximum(0.0, float(target_reliability) - rhats)
    remaining = list(np.flatnonzero(supported))
    ordered = []
    while remaining:
        layer = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                no_worse = deficit[j] <= deficit[i] and cements[j] <= cements[i]
                better = deficit[j] < deficit[i] or cements[j] < cements[i]
                if no_worse and better:
                    dominated = True
                    break
            if not dominated:
                layer.append(i)
        layer.sort(key=lambda i: (deficit[i], cements[i], -rhats[i], i))
        ordered.extend(layer)
        layer_set = set(layer)
        remaining = [i for i in remaining if i not in layer_set]
    # Unsupported designs are retained only as diagnostic fallbacks.  They
    # cannot displace a supported candidate from the discovery frontier.
    ordered.extend(sorted(np.flatnonzero(~supported),
                          key=lambda i: (deficit[i], cements[i], -rhats[i], i)))
    return np.asarray(ordered, dtype=int)


def feasibility_cement_frontier(cvars, cements, supported, rhats=None):
    """Two-archive ordering: feasibility archive first, cement frontier second.

    ``cvars`` is the continuous search-only score (higher is better;
    -inf for support failures).  Supported candidates are ordered by
    (feasibility quality, then cement); unsupported diagnostics trail and can
    never displace a supported candidate.  Returns (order, feasibility_rank)
    where ``feasibility_rank[i]`` is the position of candidate i in the
    feasibility archive.
    """
    cvars = np.asarray(cvars, dtype=float)
    cements = np.asarray(cements, dtype=float)
    supported = np.asarray(supported, dtype=bool)
    n = len(cements)
    feas_idx = np.flatnonzero(supported)
    # Feasibility archive: smooth quality first, cement only as tie-break.
    feas_order = sorted(feas_idx.tolist(), key=lambda i: (-cvars[i], cements[i], i))
    other = sorted(np.flatnonzero(~supported).tolist(),
                   key=lambda i: (-cvars[i], cements[i], i))
    order = np.asarray(feas_order + other, dtype=int)
    feasibility_rank = np.full(n, -1, dtype=int)
    for rank, i in enumerate(feas_order):
        feasibility_rank[int(i)] = int(rank)
    return order, feasibility_rank


def blind_proposals(n, seed, training_csv=None, mix=(0.35, 0.35, 0.15, 0.15)):
    """Support-aware, target-independent proposals with source labels.

    Thin wrapper over :mod:`proposal_engine` so the initializer owns one
    proposal entry point.  Fails closed when support cannot be established.
    """
    if PE is None:
        raise RuntimeError("proposal_engine module unavailable; refusing blind proposals")
    return PE.generate_proposals(int(n), int(seed), training_csv=training_csv, mix=mix)


def _diverse_take(X, order, n_take, diversity_L2):
    chosen = []
    for idx in order:
        x = X[idx]
        if not chosen or all(np.linalg.norm(x - X[j]) >= diversity_L2 for j in chosen):
            chosen.append(int(idx))
        if len(chosen) >= n_take:
            break
    # A very concentrated promising basin must not make the initializer
    # return too few points for DE; fill deterministically from the frontier.
    if len(chosen) < n_take:
        chosen_set = set(chosen)
        chosen.extend(int(i) for i in order if int(i) not in chosen_set)
    return np.asarray(chosen[:n_take], dtype=int)


def _as_diagnostic_rows(diagnostics, n):
    if diagnostics is None:
        return [{} for _ in range(n)]
    if isinstance(diagnostics, dict):
        return [{k: (v[i].item() if hasattr(v[i], "item") else v[i])
                 for k, v in diagnostics.items()} for i in range(n)]
    return [dict(x) for x in diagnostics]


def screen_and_refine(X0, evaluate_fn, seed=0, rounds=3, elite_frac=0.1,
                      diversity_L2=0.05, scenario_schedule=(64, 128, 256),
                      refinement_sizes=(768, 256), global_sampler=None,
                      candidate_log=None, cvar_fn=None, basin_radius=0.12,
                      initial_sources=None, target_reliability=0.90):
    """Screen all blind starts and concentrate later work on a frontier.

    ``evaluate_fn(X, n_scenarios)`` returns ``(rhats, cements, supported)``
    or ``(rhats, cements, supported, diagnostics)``.  It must use common
    random numbers within a round.  The first round always evaluates every
    supplied Stage-0/0b point; subsequent rounds evaluate a bounded
    population resampled around diverse Pareto-frontier elites.  This gives
    discovery useful global coverage without turning a failed search into
    hours of expensive repeated verification.

    ``cvar_fn`` optionally maps ``(rhats, cements, supported)`` to a
    continuous search-quality array (higher is better); when supplied, a
    separate feasibility archive ordered by that quality is maintained
    alongside the cement frontier, and per-round basin clustering is logged
    (basin IDs, membership, proposal sources, survival reasons).  Every
    candidate's proposal ``source`` is recorded in ``candidate_log``.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X0, dtype=float)
    if X.ndim != 2 or len(X) == 0:
        raise ValueError("X0 must be a non-empty 2-D candidate array")
    log = []
    if global_sampler is None:
        global_sampler = lambda n: rng.uniform(0.0, 1.0, size=(n, X.shape[1]))
    sources = (np.full(len(X), "stage0", dtype=object) if initial_sources is None
               else np.asarray(initial_sources, dtype=object))
    if len(sources) != len(X):
        raise ValueError("initial_sources length must match X0")
    archive_X, archive_r, archive_c, archive_s, archive_q, archive_sources = [], [], [], [], [], []
    cumulative_candidates = 0
    for rd in range(rounds):
        n_sc = scenario_schedule[min(rd, len(scenario_schedule) - 1)]
        t_eval = __import__("time").perf_counter()
        evaluated = evaluate_fn(X, n_sc)
        elapsed_eval = __import__("time").perf_counter() - t_eval
        if len(evaluated) == 4:
            rhats, cements, supported, diagnostics = evaluated
        else:
            rhats, cements, supported = evaluated
            diagnostics = None
        rhats = np.asarray(rhats, dtype=float)
        cements = np.asarray(cements, dtype=float)
        supported = np.asarray(supported, dtype=bool)
        diagnostics = _as_diagnostic_rows(diagnostics, len(X))
        cumulative_candidates += len(X)
        archive_X.append(X.copy())
        archive_r.append(rhats.copy())
        archive_c.append(cements.copy())
        archive_s.append(supported.copy())
        diag_scores = np.asarray(
            [d.get("continuous_score", np.nan) for d in diagnostics], dtype=float)
        if cvar_fn is not None:
            try:
                cvars = np.asarray(cvar_fn(rhats, cements, supported), dtype=float)
            except Exception:
                cvars = np.asarray(rhats, dtype=float)
        elif np.any(np.isfinite(diag_scores)):
            cvars = np.where(np.isfinite(diag_scores), diag_scores, float("-inf"))
        else:
            cvars = np.asarray(rhats, dtype=float)
        cvars = np.where(supported, cvars, float("-inf"))
        # Reliability is primary.  Continuous margins only break near-ties;
        # raw CVaR previously caused Rhat=0.03 candidates to outrank Rhat=0.65.
        finite_margin = np.where(np.isfinite(cvars), np.clip(cvars, -100.0, 100.0), -100.0)
        rank_quality = np.where(supported, rhats + 1e-4 * finite_margin, float("-inf"))
        archive_q.append(rank_quality.copy())
        archive_sources.extend(str(s) for s in sources)
        if candidate_log is not None:
            for i, (x, diag) in enumerate(zip(X, diagnostics)):
                row = {"round": int(rd), "candidate_index": int(i),
                       "source": str(sources[i]), "n_scenarios": int(n_sc),
                       "Rhat": float(rhats[i]), "C": float(cements[i]),
                       "supported": bool(supported[i])}
                row.update({f"x{j+1}": float(v) for j, v in enumerate(x)})
                row.update(diag)
                candidate_log.append(row)
        cement_order = _frontier_order(
            rhats, cements, supported, target_reliability=target_reliability)
        feasibility_order, _ = feasibility_cement_frontier(
            rank_quality, cements, supported, rhats=rhats)
        # Preserve both archives.  Alternating their ranks prevents either a
        # cheap hopeless point or one expensive high-quality basin from
        # monopolising the continuation set.
        combined = []
        for a, b in zip(feasibility_order, cement_order):
            for idx in (int(a), int(b)):
                if idx not in combined:
                    combined.append(idx)
        combined.extend(int(i) for i in feasibility_order if int(i) not in combined)
        combined.extend(int(i) for i in cement_order if int(i) not in combined)
        order = np.asarray(combined, dtype=int)
        n_elite = max(32, min(192, int(len(X) * elite_frac)))
        elite_idx = _diverse_take(X, order, min(n_elite, len(X)), diversity_L2)
        elites = X[elite_idx]
        round_entry = {"round": rd, "n": int(len(X)), "n_scenarios": int(n_sc),
                       "elite_size": int(len(elites)),
                       "supported": int(np.sum(supported)),
                       "best_C": float(np.min(cements[elite_idx])),
                       "best_Rhat": float(np.max(rhats[elite_idx])),
                       "mean_Rhat": float(np.mean(rhats[elite_idx]))}
        round_entry["feasibility_best_cvar"] = float(np.max(cvars[supported])) if np.any(supported) else float("-inf")
        round_entry["feasibility_archive_size"] = int(np.sum(supported))
        arch = None
        if BA is not None:
            try:
                arch = BA.BasinArchive(radius=float(basin_radius))
                all_x = np.concatenate(archive_X, axis=0)
                all_q = np.concatenate(archive_q, axis=0)
                all_s = np.concatenate(archive_s, axis=0)
                arch.cluster(all_x, all_q, all_s, sources=archive_sources)
                alloc, n_novel = arch.allocate(
                    refinement_sizes[min(rd, len(refinement_sizes) - 1)]
                    if rd < rounds - 1 else min(len(X), 64))
                round_entry["n_basins"] = int(len(arch.basins))
                round_entry["basin_sizes"] = [int(len(b["members"])) for b in arch.basins]
                round_entry["basin_ids"] = [int(b["id"]) for b in arch.basins]
                round_entry["novelty_quota"] = int(n_novel)
                round_entry["survival_rule"] = ("diverse Pareto frontier + per-basin minimum "
                                                "allocation + fresh-global quota")
            except Exception:
                pass
        log.append(round_entry)
        if rd < rounds - 1:
            n_next = refinement_sizes[min(rd, len(refinement_sizes) - 1)]
            if arch is not None and arch.basins:
                alloc, n_global = arch.allocate(n_next)
                local, local_sources = arch.propose(rng, alloc, X.shape[1])
            else:
                # Safe fallback for a round with no supported basin yet.
                mu, sd = elites.mean(axis=0), elites.std(axis=0) + 1e-6
                n_global = min(n_next, max(16, int(0.25 * n_next)))
                local = np.clip(
                    rng.normal(mu, sd, size=(n_next - n_global, X.shape[1])), 0, 1)
                local_sources = ["adaptive_fallback"] * len(local)
            injected_result = global_sampler(n_global)
            if isinstance(injected_result, tuple):
                injected, injected_sources = injected_result
            else:
                injected, injected_sources = injected_result, ["fresh_global"] * n_global
            injected = np.asarray(injected, dtype=float)
            if injected.shape != (n_global, X.shape[1]):
                raise ValueError("global_sampler returned an invalid candidate matrix")
            X = np.concatenate([local, np.clip(injected, 0, 1)], axis=0)
            sources = np.asarray(list(local_sources) + list(injected_sources), dtype=object)
        log[-1].update({"cumulative_candidates": int(cumulative_candidates),
                        "elapsed_s": float(elapsed_eval),
                        "cost_per_candidate_s": float(elapsed_eval / max(len(rhats), 1))})

    # Build a multi-round archive before selecting the DE population.  The
    # archive is intentionally not just the final local cloud: fresh global
    # injections and early low-cement basins remain eligible for continuation.
    AX = np.concatenate(archive_X, axis=0)
    AR = np.concatenate(archive_r, axis=0)
    AC = np.concatenate(archive_c, axis=0)
    AS = np.concatenate(archive_s, axis=0)
    AQ = np.concatenate(archive_q, axis=0)
    q_order, _ = feasibility_cement_frontier(AQ, AC, AS, rhats=AR)
    c_order = _frontier_order(AR, AC, AS, target_reliability=target_reliability)
    merged = []
    for a, b in zip(q_order, c_order):
        for idx in (int(a), int(b)):
            if idx not in merged:
                merged.append(idx)
    merged.extend(int(i) for i in q_order if int(i) not in merged)
    merged.extend(int(i) for i in c_order if int(i) not in merged)
    archive_order = np.asarray(merged, dtype=int)
    n_return = min(len(AX), max(32, min(192, len(X))))
    chosen = _diverse_take(AX, archive_order, n_return, diversity_L2)
    if candidate_log is not None:
        chosen_set = set(int(i) for i in chosen)
        offset = 0
        for rd, block in enumerate(archive_X):
            for i in range(len(block)):
                if offset + i in chosen_set:
                    candidate_log[offset + i]["archive_selected"] = True
            offset += len(block)
    log[-1]["archive_size"] = int(len(AX))
    log[-1]["archive_selected"] = int(len(chosen))
    return AX[chosen], log
