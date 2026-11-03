"""Driver: full DE battery (main @90 + sensitivity @80/95). Resumable per seed."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from de_search import run

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COHORT = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\data\reconciled_phase2v\phase1_reconciled_eligible.csv"

MAIN = [(18, 13), (15, 8), (12, 5)]
MAIN_SEEDS = list(range(20260914, 20260922))
SENS = [(18, 13), (12, 5)]
SENS_SEEDS = list(range(20260922, 20260926))

t0 = time.time()
jobs = []
for (s, k) in MAIN:
    for seed in MAIN_SEEDS:
        jobs.append((s, k, 0.9, seed, 64, 125))
for (s, k) in SENS:
    for p in (0.8, 0.95):
        for seed in SENS_SEEDS:
            jobs.append((s, k, p, seed, 64, 62))

for (s, k, p, seed, pop, gens) in jobs:
    tag = f"de_{s}_{k}_p{int(round(100*p))}_s{seed}"
    out = os.path.join(RUN, "traces", tag)
    if os.path.exists(out + ".best.json"):
        print(f"skip {tag} (done)", flush=True)
        continue
    r = run(s, k, p, seed, pop, gens, out, COHORT, resume=True)
    print(f"done {tag}: merit={r['best_merit']:.3f} feasible={r['best_rec']['MODEL_SCREEN_FEASIBLE']} "
          f"C={r['best_rec']['C']:.2f} evals={r['n_eval']} t={time.time()-t0:.0f}s", flush=True)
print(f"ALL DONE t={time.time()-t0:.0f}s", flush=True)
