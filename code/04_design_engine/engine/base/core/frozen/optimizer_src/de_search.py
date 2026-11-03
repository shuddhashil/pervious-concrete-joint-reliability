"""Simplex-aware constrained Differential Evolution (main search, batched).

DE/rand/1/bin, F=0.7, Cr=0.9, jitter; feasibility-first selection via the
authoritative merit. One evaluator batch per generation. Checkpointed.
Usage:
  python src/de_search.py --s 18 --k 13 --p 0.9 --seed 20260914 --out traces/de_18_13_p90_s20260914
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage1_evaluator as E
import solvers_common as SC

DIM = 12
F, CR = 0.7, 0.9


def build_init(rng, popsize, cohort_csv):
    df = pd.read_csv(cohort_csv)
    B = SC.BND
    enc = []
    for _, row in df.iterrows():
        c = {"C": float(row["cem (kg/m3)"]), "r": float(row["w/c"]), "A": float(row["CA (kg/m3)"]),
             "P": float(row["SP (kg/m3)"]), "S": float(row["sand (kg/m3)"]),
             "phi": float(row["porosity"])}
        if not (B["C"][0] <= c["C"] <= B["C"][1] and B["r"][0] <= c["r"] <= B["r"][1]
                and B["A"][0] <= c["A"] <= B["A"][1] and B["P"][0] <= c["P"] <= B["P"][1]
                and B["S"][0] <= c["S"] <= B["S"][1] and B["phi"][0] <= c["phi"] <= B["phi"][1]):
            continue
        g = np.array([row[b] for b in E.BINS], dtype=float)
        g = g / g.sum()
        c["g"] = g
        enc.append(SC.encode(c))
    enc = np.array(enc)
    rng.shuffle(enc)
    pop = []
    n_real = min(len(enc), popsize // 3)
    pop.extend(enc[:n_real])
    n_pert = min(len(enc), popsize // 3)
    for v in enc[:n_pert]:
        pop.append(np.clip(v + rng.normal(0, 0.05, DIM), 0, 1))
    specials = [np.eye(6)[k] for k in range(6)] + [np.full(6, 1 / 6)]
    for k in range(6):  # single-zero face centroids
        m = np.ones(6)
        m[k] = 0.0
        specials.append(m / m.sum())
    for m in specials:
        x = rng.uniform(0, 1, DIM)
        x[6:12] = m + rng.normal(0, 0.01, 6)
        pop.append(np.clip(x, 0, 1))
    while len(pop) < popsize:
        pop.append(rng.uniform(0, 1, DIM))
    return np.array(pop[:popsize]), n_real


def run(s_star, k_star, p, seed, popsize, generations, out_prefix, cohort_csv, resume=True):
    rng = np.random.default_rng(seed)
    bounds = {k: SC.BND[k] for k in ["C", "r", "A", "P", "S", "phi"]}
    ckpt = out_prefix + ".ckpt.npz"
    trace_path = out_prefix + ".trace.csv"
    start_gen, pop, merits, best_x, best_merit, trace = 0, None, None, None, np.inf, []
    if resume and os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=True)
        start_gen, pop = int(z["gen"]) + 1, z["pop"]
        merits, best_x, best_merit = z["merits"], z["best_x"], float(z["best_merit"])
        if os.path.exists(trace_path):
            trace = pd.read_csv(trace_path).to_dict("records")
    if pop is None:
        pop, n_real = build_init(rng, popsize, cohort_csv)
    else:
        n_real = -1
    n_eval = 0 if start_gen == 0 else (start_gen + 1) * popsize  # init + completed trial batches
    if start_gen == 0:
        recs0 = E.evaluate_batch(SC.decode_batch(pop), s_star, k_star, p=p, bounds=bounds, backend="fast")
        merits = SC.merit_batch(recs0)
        n_eval += popsize
        idx = int(np.argmin(merits))
        best_merit, best_x = float(merits[idx]), pop[idx].copy()
    else:
        pass  # resumed: merits/best restored from checkpoint
    for gen in range(start_gen, generations):
        trial = np.empty_like(pop)
        for i in range(popsize):
            ids = rng.choice(popsize, 3, replace=False)
            a, b_, c_ = pop[ids[0]], pop[ids[1]], pop[ids[2]]
            jitter = rng.uniform(0.5, 1.0)
            mut = np.clip(a + F * jitter * (b_ - c_), 0, 1)
            cross = rng.uniform(0, 1, DIM) < CR
            cross[rng.integers(0, DIM)] = True
            trial[i] = np.where(cross, mut, pop[i])
        tc = SC.decode_batch(trial)
        trecs = E.evaluate_batch(tc, s_star, k_star, p=p, bounds=bounds, backend="fast")
        tmer = SC.merit_batch(trecs)
        n_eval += popsize
        improve = tmer < merits
        pop[improve] = trial[improve]
        merits[improve] = tmer[improve]
        idx = int(np.argmin(merits))
        if merits[idx] < best_merit:
            best_merit, best_x = float(merits[idx]), pop[idx].copy()
        bi = int(np.argmin(tmer))
        br = trecs[bi]
        trace.append({"gen": gen, "evals": n_eval, "best_merit": best_merit,
                      "gen_best_C": float(br["C"]), "gen_feasible": bool(br["MODEL_SCREEN_FEASIBLE"]),
                      "support": float(br["support"]), "margin_S": float(br["margin_S"]),
                      "margin_Q": float(br["margin_Q"])})
        if (gen + 1) % 10 == 0 or gen == generations - 1:
            np.savez(ckpt, gen=gen, pop=pop, merits=merits,
                     best_x=best_x if best_x is not None else pop[0],
                     best_merit=best_merit)
            pd.DataFrame(trace).to_csv(trace_path, index=False)
    best_cand = SC.decode(best_x)
    best_rec = E.evaluate(best_cand, s_star, k_star, p=p, bounds=bounds, backend="fast")
    out = {"s_star": s_star, "k_star": k_star, "p": p, "seed": seed,
           "popsize": popsize, "generations": generations, "n_eval": n_eval,
           "n_real_init": n_real, "best_merit": best_merit,
           "best_cand": {k: (v if k == "g" else float(v)) for k, v in best_cand.items()},
           "best_rec": best_rec}
    out["best_cand"]["g"] = [float(x) for x in best_cand["g"]]
    with open(out_prefix + ".best.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=float, required=True)
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--popsize", type=int, default=64)
    ap.add_argument("--gens", type=int, default=125)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cohort", default=os.path.join(
        r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild",
        "materials", "data", "reconciled_phase2v", "phase1_reconciled_eligible.csv"))
    a = ap.parse_args()
    r = run(a.s, a.k, a.p, a.seed, a.popsize, a.gens, a.out, a.cohort)
    print(f"seed {a.seed}: best_merit={r['best_merit']:.3f} feasible={r['best_rec']['MODEL_SCREEN_FEASIBLE']} "
          f"C={r['best_rec']['C']:.2f} evals={r['n_eval']}")
