"""Runs bonf_eval.py for the four supported targets (same per-target set-up as driver.py). Resumable.
usage: python bonf_driver.py <results_dir_of_sep_eval>"""
import os, sys, subprocess
ROOT = os.environ.get("GENGINE_ROOT", "/content/GuidedEngine_20261003"); J = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT); sys.dont_write_bytecode = True
from gengine import target as TG
RES = sys.argv[1]; os.makedirs(f"{J}/out_bonf", exist_ok=True)
for S, K in [(10, 8), (14, 6), (16, 10), (20, 4)]:
    name = TG.name_of(S, K, 0.90); out = f"{J}/out_bonf/{name}.json"
    if os.path.exists(out): print("done already:", name, flush=True); continue
    T = TG.make_target(S, K, 0.90)["dir"]
    env = dict(os.environ, GENGINE_BASE=T, GENGINE_CONFIG=os.path.join(T, "CONFIG.json"), GENGINE_ROOT=ROOT, PYTHONDONTWRITEBYTECODE="1")
    if not os.path.exists(os.path.join(T, "CONFIG.json")):
        subprocess.run([sys.executable, f"{ROOT}/gengine/target_config.py", str(float(S)), str(float(K)), "0.9", "0", "1", "2"], env=env, cwd=ROOT, check=True)
    print(f"=== {name}", flush=True)
    subprocess.run([sys.executable, "-u", f"{J}/bonf_eval.py", f"{J}/logs/{name}", f"{RES}/{name}.json", out], env=env, cwd=ROOT, check=True)
print("ALL DONE", flush=True)
