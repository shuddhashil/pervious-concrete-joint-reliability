"""Stage C (v2): strict certificates at the 0.90 edge only (frozen estimator via certify.py / gengine\cert.py, unchanged; 2 replicates, nominal + worst shift, porosity SD 1.5; up to 3 candidate designs per cell). One task per K* column, 3 CPU workers.
Per column: from the top supported cell downward until a cell passes strictly, then upward until the first cell that fails (certify.edge_walk, as v1). Writes data_v2/edge_certs_v2.json and certificates_v2.jsonl."""
import sys, os, json, pickle, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp
import numpy as np
import certify as CE
SM = CE.SM; S_GRID = np.arange(5.0, 51.0, 1.0); K_GRID = np.geomspace(0.5, 30.0, 20); M_WALK = 0.002; R = 0.90


def main():
    t0 = time.time(); B = pickle.load(open(os.path.join(SM, "data_v2", "cands_final.pkl"), "rb")); nS, nK = len(S_GRID), len(K_GRID); L = np.nan_to_num(B["best"], nan=-1.0).reshape(nS, nK); tasks = []
    for b in range(nK):
        tasks.append(dict(b=b, Kval=float(K_GRID[b]), R=R, S=[float(x) for x in S_GRID], supB=[bool(L[a, b] >= R + M_WALK) for a in range(nS)], cands={a: B["cands"][a * nK + b] for a in range(nS) if (a * nK + b) in B["cands"] and B["cands"][a * nK + b]}))
    tasks.sort(key=lambda t: -sum(t["supB"])); res = []; out = open(os.path.join(SM, "certificates_v2.jsonl"), "w")
    with mp.get_context("spawn").Pool(3) as pool:
        for r in pool.imap_unordered(CE.edge_walk, tasks):
            res.append(r); [out.write(json.dumps(dict(run, task_K=r["K"], task_R=r["R"])) + "\n") for run in r["runs"]]; out.flush()
            print(f"K*={r['K']:.3f}: stageB top S={(S_GRID[r['stageB_highest']] if r['stageB_highest'] >= 0 else None)} confirmed top S={(S_GRID[r['confirmed_top']] if r['confirmed_top'] >= 0 else None)} promoted {r['promoted']} first unsupported {r['first_unsupported']} ({r['seconds']}s) [{len(res)}/{len(tasks)}]", flush=True)
    for r in res: r["runs"] = len(r["runs"])
    json.dump(dict(results=res, seconds=round(time.time() - t0, 1)), open(os.path.join(SM, "data_v2", "edge_certs_v2.json"), "w"), indent=1); print("CERTS DONE", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
