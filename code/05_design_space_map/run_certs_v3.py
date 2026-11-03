"""Stage C (v2, after round 3 and the notch fix, main thread, 2026-10-04): strict certificates only for the K* columns whose edge information changed since the round-2 certificates
(top supported cell under the acceptance rule, or the candidate designs of the top cell / the cell above it). Unchanged columns keep their round-2 result (edge_certs_v2.json) and their runs in certificates_v2.jsonl.
Same code as run_certs_v2 (certify.edge_walk: frozen estimator, 2 replicates, nominal + worst shift, porosity SD 1.5, up to 3 designs per cell, escalating N). CPU, 3 workers.
Reads data_v2/cands_final.pkl (new) and data_v2/cands_final_round2.pkl (copy of the old one). Writes data_v2/edge_certs_v2.json (merged; the round-2 file is kept as edge_certs_v2_round2.json) and appends new runs to certificates_v2.jsonl."""
import sys, os, json, pickle, time, shutil; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp
import numpy as np
import certify as CE
SM = CE.SM; S_GRID = np.arange(5.0, 51.0, 1.0); K_GRID = np.geomspace(0.5, 30.0, 20); M_WALK = 0.002; R = 0.90; nS, nK = len(S_GRID), len(K_GRID)


def sig(B, b):
    L = np.nan_to_num(B["best"], nan=-1.0).reshape(nS, nK); ix = np.flatnonzero(L[:, b] >= R + M_WALK); a0 = int(ix.max()) if len(ix) else -1
    return (a0, tuple(tuple(d for lb, d, _, _ in B["cands"].get(a * nK + b, [])[:3]) for a in (a0, a0 + 1) if 0 <= a < nS))


def main():
    t0 = time.time(); D2 = os.path.join(SM, "data_v2"); B = pickle.load(open(os.path.join(D2, "cands_final.pkl"), "rb")); Bo = pickle.load(open(os.path.join(D2, "cands_final_round2.pkl"), "rb"))
    old = json.load(open(os.path.join(D2, "edge_certs_v2.json"))); shutil.copy(os.path.join(D2, "edge_certs_v2.json"), os.path.join(D2, "edge_certs_v2_round2.json")); keep = {r["b"]: r for r in old["results"]}
    L = np.nan_to_num(B["best"], nan=-1.0).reshape(nS, nK); changed = [b for b in range(nK) if sig(B, b) != sig(Bo, b)]; print("columns to re-certify:", changed, flush=True); tasks = []
    for b in changed:
        tasks.append(dict(b=b, Kval=float(K_GRID[b]), R=R, S=[float(x) for x in S_GRID], supB=[bool(L[a, b] >= R + M_WALK) for a in range(nS)], cands={a: B["cands"][a * nK + b] for a in range(nS) if (a * nK + b) in B["cands"] and B["cands"][a * nK + b]}))
    res = []; out = open(os.path.join(SM, "certificates_v2.jsonl"), "a")
    if tasks:
        with mp.get_context("spawn").Pool(3) as pool:
            for r in pool.imap_unordered(CE.edge_walk, tasks):
                res.append(r); [out.write(json.dumps(dict(run, task_K=r["K"], task_R=r["R"], phase="after_round3_and_notch_fix")) + "\n") for run in r["runs"]]; out.flush()
                print(f"K*={r['K']:.3f}: stageB top S={(S_GRID[r['stageB_highest']] if r['stageB_highest'] >= 0 else None)} confirmed top S={(S_GRID[r['confirmed_top']] if r['confirmed_top'] >= 0 else None)} promoted {r['promoted']} first unsupported {r['first_unsupported']} ({r['seconds']}s) [{len(res)}/{len(tasks)}]", flush=True)
    for r in res: r["runs"] = len(r["runs"]); keep[r["b"]] = r
    merged = [keep[b] for b in sorted(keep)]; json.dump(dict(results=merged, seconds=round(time.time() - t0, 1), recertified_columns=changed, note="columns not listed in recertified_columns keep their round-2 certificate result"), open(os.path.join(D2, "edge_certs_v2.json"), "w"), indent=1); print("CERTS DONE", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
