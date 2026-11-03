"""Runs sep_eval.py for the six worked targets, one after the other, each in its own process with its own per-target core copy
(same set-up as run_target.py stage 0). Resumable: a target whose output json exists is skipped.
usage: python driver.py [smoke_n]   (smoke_n: score only n candidates of the first target, to check the set-up)"""
import os, sys, subprocess
ROOT = os.environ.get("GENGINE_ROOT", "/content/GuidedEngine_20261003"); J = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT); sys.dont_write_bytecode = True
from gengine import target as TG
TARGETS = [(10, 8), (14, 6), (16, 10), (20, 4), (18, 13), (25, 13)]
smoke = sys.argv[1] if len(sys.argv) > 1 else None
os.makedirs(f"{J}/out", exist_ok=True)
for S, K in (TARGETS[:1] if smoke else TARGETS):
    name = TG.name_of(S, K, 0.90); out = f"{J}/out/{name}{'_smoke' if smoke else ''}.json"
    if os.path.exists(out): print("done already:", name, flush=True); continue
    T = TG.make_target(S, K, 0.90)["dir"]
    env = dict(os.environ, GENGINE_BASE=T, GENGINE_CONFIG=os.path.join(T, "CONFIG.json"), GENGINE_ROOT=ROOT, PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", OPENBLAS_NUM_THREADS="3", NUMEXPR_NUM_THREADS="3")
    if not os.path.exists(os.path.join(T, "CONFIG.json")):
        subprocess.run([sys.executable, f"{ROOT}/gengine/target_config.py", str(float(S)), str(float(K)), "0.9", "0", "1", "2"], env=env, cwd=ROOT, check=True)
    print(f"=== {name}", flush=True)
    subprocess.run([sys.executable, "-u", f"{J}/sep_eval.py", f"{J}/logs/{name}", out] + ([smoke] if smoke else []), env=env, cwd=ROOT, check=True)
print("ALL DONE", flush=True)
