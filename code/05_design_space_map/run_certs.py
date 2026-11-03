"""Stage C driver: strict certificates at the edge, one task per (K* column, R*), 3 CPU worker processes. Writes data/edge_certs.json and certificates.jsonl."""
import sys, os, json, pickle, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp
import numpy as np
import certify as CE
SM = CE.SM; S_GRID = np.arange(5.0, 51.0, 1.0); K_GRID = np.geomspace(0.5, 30.0, 20); RS = (0.80, 0.90, 0.95); M_WALK = 0.002


def main():
    t0 = time.time(); B = pickle.load(open(os.path.join(SM, "data", "stageB.pkl"), "rb")); nS, nK = len(S_GRID), len(K_GRID); L = np.nan_to_num(B["best_LB"], nan=-1.0).reshape(nS, nK); tasks = []
    for R in RS:
        for b in range(nK):
            tasks.append(dict(b=b, Kval=float(K_GRID[b]), R=R, S=[float(x) for x in S_GRID], supB=[bool(L[a, b] >= R + M_WALK) for a in range(nS)], cands={a: B["cands"][a * nK + b] for a in range(nS) if B["cands"][a * nK + b]}))
    tasks.sort(key=lambda t: -sum(t["supB"]))                                                  # longest first
    res = []; out = open(os.path.join(SM, "certificates.jsonl"), "w")
    with mp.get_context("spawn").Pool(3) as pool:
        for r in pool.imap_unordered(CE.edge_walk, tasks):
            res.append(r); [out.write(json.dumps(dict(run, task_K=r["K"], task_R=r["R"])) + "\n") for run in r["runs"]]; out.flush()
            print(f"R*={r['R']} K*={r['K']:.3f}: stageB top S={ (S_GRID[r['stageB_highest']] if r['stageB_highest']>=0 else None)} confirmed top S={(S_GRID[r['confirmed_top']] if r['confirmed_top']>=0 else None)} promoted {r['promoted']} first unsupported {r['first_unsupported']} ({r['seconds']}s) [{len(res)}/{len(tasks)}]", flush=True)
    for r in res: r["runs"] = len(r["runs"])
    json.dump(dict(results=res, seconds=round(time.time() - t0, 1)), open(os.path.join(SM, "data", "edge_certs.json"), "w"), indent=1); print("CERTS DONE", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
