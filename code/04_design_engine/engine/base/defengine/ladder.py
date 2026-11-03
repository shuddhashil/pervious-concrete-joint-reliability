"""Frontier tracing (the new search). No monotonicity is assumed: every coarse level is solved. Stages:
 A1 cheap sweep of the whole cement range; A2 deeper solve where the sweep found promise; B refine (5 then 1 kg/m3) around every point where
 the reliability frontier crosses the target, warm-started from the neighbouring levels' best designs; C independent verification with
 fresh seeds (two replicates) of the best designs near each crossing. Results are written per level so a run can be resumed.
Run with the engine python. Parallel over levels with a process pool (each worker loads the frozen core once)."""
import os, sys, json, time, math
from concurrent.futures import ProcessPoolExecutor
import numpy as np

_W = {}


def _gpu():
    return os.environ.get("DEFENGINE_GPU") == "1"


def _init_worker():
    from . import corelib as C, gates as G
    C.engine(); _W["gate"] = G.SearchGate()
    if _gpu():
        from . import gpu_solver as GS
        GS._state()


def _solve_task(task):
    from . import level_solver as LS, gpu_solver as GS
    fn = GS.solve_level_gpu if _gpu() else LS.solve_level
    r = fn(task["cement"], seed=task["seed"], P=task["P"], generations=task["generations"], n_scen=task["n_scen"], warm=task.get("warm"), pool=task.get("pool"), gate=_W.get("gate"))
    r["stage"] = task["stage"]; return r


def _chain_task(task):
    """Walk a feasible design DOWN in cement: at each level start from the previous level's best designs (exact copies), short DE, fresh-bank re-rank.
    Every level is saved to disk the moment it finishes (and reloaded on a restart), so a crash or kill never loses finished work."""
    from . import level_solver as LS
    c = task["start"]; seed = task["seed"]; pool = list(task["pool"]); out = []; below = 0; tag = "CH" + str(task["chain"])
    while c >= task["floor"]:
        fn = f"level_{c}_{tag}.json"; got = None
        for d in (task.get("out_dir"), task.get("resume")):
            if d and os.path.exists(os.path.join(d, fn)): got = json.load(open(os.path.join(d, fn))); break
        if got is None:
            fn = __import__("defengine.gpu_solver", fromlist=["x"]).solve_level_gpu if _gpu() else LS.solve_level
            got = fn(c, seed=seed + c, P=task["P"], generations=task["generations"], n_scen=task["n_scen"], pool=pool, n_exact=10, gate=_W.get("gate"))
            got["stage"] = tag
            if task.get("out_dir"): _save(os.path.join(task["out_dir"], fn), got)
        r = got; out.append(r); pool = [b["z"] for b in r["best"][:4]] + pool[:6]
        if task.get("out_dir"):
            line = time.strftime("%H:%M:%S") + " level " + str(c) + " best fresh rhat " + format(best_rhat(r), ".3f") + " (" + format(r["seconds"], ".0f") + "s)"
            with open(os.path.join(task["out_dir"], "chain_" + str(task["chain"]) + ".log"), "a") as f: f.write(line + chr(10))
        below = below + 1 if best_rhat(r) < task["stop_below"] else 0
        if below >= 2: break
        c -= task["step"]
    return out


def run_chain_single(task):
    with ProcessPoolExecutor(max_workers=1, initializer=_init_worker) as p:
        return p.submit(_chain_task, task).result()


def _verify_task(task):
    from . import verify as V
    return V.verify_design(task["cid"], task["nominal"], replicate=task["replicate"]) | dict(cement=task["cement"], stage="verify")


def run_tasks(fn, tasks, workers, on_result=None):
    """Run tasks; with one worker they run in this process and on_result(r) fires the moment each finishes (progress + crash-safe saving)."""
    if workers <= 1:
        _init_worker(); out = []
        for t in tasks:
            r = fn(t); out.append(r)
            if on_result: on_result(r)
        return out
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        out = list(pool.map(fn, tasks))
    if on_result:
        for r in out: on_result(r)
    return out


def _save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def best_rhat(r):
    return max((b["rhat"] for b in r["best"]), default=0.0)


def _pool(best, per_level=3):
    """Unit-cube vectors of the best designs found at ANY level (shared across levels)."""
    z = []
    for c in sorted(best):
        z += [b["z"] for b in best[c]["best"][:per_level]]
    return z


def frontier_search(out_dir, profile, workers=4, seed0=1000, log=print, resume=None):
    """profile: dict of budgets. Returns dict with per-level results and the crossing summary."""
    t0 = time.time(); os.makedirs(out_dir, exist_ok=True); res = {}; best = {}
    lo, hi, step = profile["range"]; coarse = list(range(int(lo), int(hi) + 1, int(step)))
    tgt = profile["search_target"]

    def load_or_run(stage, tasks):
        got, todo = [], []
        for t in tasks:
            f = os.path.join(resume, f"level_{t['cement']}_{stage}.json") if resume else None
            if f and os.path.exists(f): got.append(json.load(open(f)))
            else: todo.append(t)
        def cb(r):
            r["stage"] = stage; _save(os.path.join(out_dir, f"level_{r['cement']}_{stage}.json"), r)
            log(f"    [{stage}] level {r['cement']}: best fresh reliability {best_rhat(r):.3f} ({r['seconds']:.0f}s, {r['evals']} evals)")
        return got + (run_tasks(_solve_task, todo, workers, on_result=cb) if todo else [])

    def absorb(results, stage):
        for r in results:
            r["stage"] = stage; c = r["cement"]; res[(c, stage)] = r; _save(os.path.join(out_dir, f"level_{c}_{stage}.json"), r)
            if c not in best or best_rhat(r) > best_rhat(best[c]): best[c] = r

    log(f"[A1] sweep {len(coarse)} levels {coarse[0]}..{coarse[-1]} step {step} ({profile['A1']})")
    absorb(load_or_run("A1", [dict(cement=c, seed=seed0 + c, stage="A1", **profile["A1"]) for c in coarse]), "A1")
    v = lambda: {c: best_rhat(best[c]) for c in sorted(best)}
    log("[A1] fresh-bank reliability per level: " + ", ".join(f"{c}:{x:.2f}" for c, x in v().items()))
    prom = [c for c in coarse if v()[c] >= profile["promising"]]
    if len(prom) < 4: prom = sorted(coarse, key=lambda c: -v()[c])[:6]
    log(f"[A2] deeper solve (shared pool) at {len(prom)} levels: {sorted(prom)}")
    pool = _pool(best)
    absorb(load_or_run("A2", [dict(cement=c, seed=seed0 + 7919 + c, stage="A2", pool=pool, **profile["A2"]) for c in sorted(prom)]), "A2")
    log("[A2] best per level: " + ", ".join(f"{c}:{x:.2f}" for c, x in v().items()))
    # B - continuation chains: walk the best designs DOWN in cement (no monotonicity assumed: chains start from several independent basins)
    starts = sorted([c for c in best if best_rhat(best[c]) >= profile["chain_start_min"]], key=lambda c: -best_rhat(best[c]))[:profile.get("n_chains", workers)]
    log(f"[B] descending chains from levels {starts} (step {profile['chain_step']}, stop when two consecutive levels < {profile['chain_stop']})")
    gpool = _pool(best, 2)
    tasks = [dict(start=c, chain=i, seed=seed0 + 99991 + 17 * i, pool=[x["z"] for x in best[c]["best"][:4]] + gpool[:6], floor=profile["range"][0], step=profile["chain_step"],
                  stop_below=profile["chain_stop"], out_dir=out_dir, resume=resume, **profile["B"]) for i, c in enumerate(starts)]
    if workers <= 1: _init_worker(); outs = [_chain_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool_:
            outs = list(pool_.map(_chain_task, tasks))
    for chain in outs: absorb(chain, chain[0]["stage"])
    log("[B] after chains: " + ", ".join(f"{c}:{x:.2f}" for c, x in v().items()))
    # C0 - fine walk (1 kg/m3) around the lowest level that reaches the target, from the best designs known
    ok = [c for c in best if best_rhat(best[c]) >= tgt]
    if ok:
        c0 = min(ok); fine = [dict(start=c0 - 1, chain=90, seed=seed0 + 55555, pool=[x["z"] for x in best[c0]["best"][:4]] + _pool(best, 2)[:6], floor=max(profile["range"][0], c0 - 25), step=1,
                                    stop_below=tgt - 0.03, out_dir=out_dir, resume=resume, **profile["B"])]
        log(f"[C0] fine 1-kg walk down from {c0}")
        absorb(_chain_task(fine[0]) if workers <= 1 else run_chain_single(fine[0]), "CH90")
    cross = [(a2, b2) for a2, b2 in zip(sorted(v()), sorted(v())[1:]) if b2 - a2 <= 10 and v()[a2] < tgt <= v()[b2]]; cross5 = cross
    summary = dict(levels={c: dict(best_rhat=best_rhat(best[c]), stage=best[c]["stage"], best=best[c]["best"][:3]) for c in sorted(best)}, crossings_coarse=cross, crossings_5=cross5,
                   seconds=time.time() - t0, profile=profile)
    _save(os.path.join(out_dir, "frontier_search.json"), summary); log(f"[search] done in {time.time() - t0:.0f}s; {len(best)} levels solved")
    return summary


def verify_stage(out_dir, summary, window, workers=4, replicates=2, log=print):
    """C - independent verification: for every solved level in the target window verify the top designs (frozen estimator, fresh seeds)."""
    tasks = []
    for c, info in summary["levels"].items():
        c = int(c)
        if not (window[0] <= c <= window[1]): continue
        for k, b in enumerate(info["best"][:2]):
            for rep in range(replicates):
                tasks.append(dict(cid=f"DE_C{c}_d{k}", nominal=b["nominal"], replicate=rep, cement=c))
    log(f"[C] verifying {len(tasks)} (design, replicate) pairs")
    out = run_tasks(_verify_task, tasks, workers); _save(os.path.join(out_dir, "verification.json"), out); return out
