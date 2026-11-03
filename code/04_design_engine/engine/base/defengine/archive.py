"""Design archive: collect EVERY design any search stage ever found, score all of them at every level of interest on ONE fresh, large noise sample (no selection bias),
and report the best unbiased reliability per cement level. Acceptance still happens only in verify.py (independent seeds, frozen estimator); the archive just makes sure the search
effort is never wasted and that the frontier is the best of everything found.
python -m defengine.archive <out_json> <run_dir> [<run_dir> ...] --levels a,b,step --n 4096"""
import os, sys, json, glob, argparse, time
import numpy as np


def collect(run_dirs):
    zs, src = [], []
    for d in run_dirs:
        for f in sorted(glob.glob(os.path.join(d, "level_*.json"))):
            r = json.load(open(f))
            for b in r["best"][:6]: zs.append(b["z"]); src.append(os.path.basename(f))
    Z = np.array(zs); keep = [0]
    for i in range(1, len(Z)):
        if np.abs(Z[keep] - Z[i]).max(axis=1).min() > 0.003: keep.append(i)
    return Z[keep], [src[i] for i in keep]


def rescore(Z, levels, n, seed=987654, log=print):
    from . import corelib as C, level_solver as LS, gates as G
    C.engine(); gate = G.SearchGate(); bank = C.mfg_bank(n, seed=seed); eS, eQ, _ = C.res_bank_iid(n, seed=seed + 1); out = {}
    for c in levels:
        noms = LS._full(Z, c); pen = gate.penalties(noms)["penalty"]; ok = np.flatnonzero(pen <= 0)
        rhat = np.full(len(Z), -1.0); sup = np.zeros(len(Z)); cv = np.full(len(Z), -np.inf)
        if len(ok):
            st = C.batch_search_stats([noms[i] for i in ok], bank, eS, eQ)
            rhat[ok] = st["rhat"]; sup[ok] = st["support_fraction"]; cv[ok] = st["cvar"]
        order = np.argsort(-rhat)[:5]
        out[int(c)] = [dict(idx=int(i), rhat=float(rhat[i]), support=float(sup[i]), cvar=float(cv[i]), z=Z[i].tolist(), nominal={k: (v.tolist() if hasattr(v, "tolist") else float(v)) for k, v in noms[i].items()}) for i in order]
        log(f"  level {c}: best unbiased reliability {rhat[order[0]]:.3f} (of {len(ok)} gate-passing designs)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("runs", nargs="+"); ap.add_argument("--levels", default="300,420,10"); ap.add_argument("--n", type=int, default=4096)
    a = ap.parse_args(); lo, hi, st = [int(x) for x in a.levels.split(",")]
    Z, src = collect(a.runs); print(f"{len(Z)} distinct designs in the archive", flush=True)
    t = time.time(); res = rescore(Z, list(range(lo, hi + 1, st)), a.n); json.dump(dict(n_designs=len(Z), n_scen=a.n, levels=res, seconds=time.time() - t), open(a.out, "w"), indent=1); print(f"done in {time.time() - t:.0f}s")
