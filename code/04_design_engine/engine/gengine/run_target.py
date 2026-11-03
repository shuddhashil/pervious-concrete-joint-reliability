"""Run the Guided engine for ANY target, one process per stage, each importing its OWN per-target copy of the frozen core.
usage: python gengine/run_target.py --S <MPa> --K <mm/s> --R <reliability> [--seeds 0 1 2] [--hours 2.0] [--cert-minutes 90]
Creates targets/S{S}_K{K}_R{R}/ : core/ + defengine/ (copy of base/, only core/TARGET_CONTRACT.json may differ, checked), CORE_MANIFEST.json + CORE_HASH.json, CONFIG.json (+ .sha256), stage/ (paste_floor.json, parents.json, walk logs,
certs.jsonl, SELECTION.json, TABLES.md, REDTEAM.json, run_target.log, STAGE_TIMES.json, GPU_TEMP.json). One GPU process at a time (stages run one after the other); CPU workers <= 3. Resumable: rerun the same command.
Old folders and base/ are never written."""
import sys, os, json, time, argparse, subprocess, threading
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
from gengine import target as TG


def gpu_temp():
    try: return int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip().split()[0])
    except Exception: return -1

ap = argparse.ArgumentParser(); ap.add_argument("--S", type=float, required=True); ap.add_argument("--K", type=float, required=True); ap.add_argument("--R", type=float, required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--hours", type=float, default=2.0); ap.add_argument("--cert-minutes", type=float, default=90.0)
ap.add_argument("--mode", choices=["walk", "global_walk"], default="walk", help="walk = the existing walk (default, unchanged); global_walk = global search + integrated walk + certificates + tables into <target>/stage_global_walk (baseline stage only read)")
ap.add_argument("--scan-hours", type=float, default=1.0); ap.add_argument("--gwalk-hours", type=float, default=0.6); a = ap.parse_args()
info = TG.make_target(a.S, a.K, a.R); T = info["dir"]; STAGE = os.path.join(T, "stage_global_walk" if a.mode == "global_walk" else "stage"); os.makedirs(STAGE, exist_ok=True); LOG = open(os.path.join(STAGE, "run_target.log"), "a")
def log(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
log("=" * 70); log("run_target", time.strftime("%Y-%m-%d %H:%M:%S"), "target", (a.S, a.K, a.R), "seeds", a.seeds, "dir", T)
log("per-target copy differs from base/ in:", info["differs_from_base"] or "nothing (byte-identical)", "| core manifest sha256", info["hashes"]["core_manifest_sha256"][:16], "combined", info["hashes"]["combined_core_and_defengine_sha256"][:16])
env = dict(os.environ, GENGINE_BASE=T, GENGINE_CONFIG=os.path.join(T, "CONFIG.json"), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", OPENBLAS_NUM_THREADS="3", NUMEXPR_NUM_THREADS="3")
peak = dict(max=gpu_temp(), samples=0, over80=0); stop = threading.Event()
def sampler():
    while not stop.wait(10):
        t = gpu_temp(); peak["samples"] += 1; peak["max"] = max(peak["max"], t); peak["over80"] += t > 80
threading.Thread(target=sampler, daemon=True).start()
times = json.load(open(os.path.join(STAGE, "STAGE_TIMES.json"))) if os.path.exists(os.path.join(STAGE, "STAGE_TIMES.json")) else {}
log("GPU temp at start:", gpu_temp())
stages = [("0_config", ["target_config.py", str(a.S), str(a.K), str(a.R)] + [str(x) for x in a.seeds]), ("1_walk", ["run_full.py", str(a.hours)]), ("2_certify", ["select_certify.py", str(a.cert_minutes)]),
          ("3_tables", ["make_tables.py"]), ("4_redteam", ["redteam.py"]), ("5_summary", ["run_summary.py"])]
if a.mode == "global_walk":      # needs the finished baseline stage of this target (CONFIG frozen there); never writes to it
    assert os.path.exists(os.path.join(T, "stage", "SELECTION.json")), "baseline walk stage missing: run the default mode first"
    stages = [("g1_global_search", ["global_search.py", str(a.scan_hours)]), ("g2_walk", ["run_global_walk.py", str(a.gwalk_hours)]), ("g3_certify", ["select_certify_global.py", str(a.cert_minutes)]), ("g4_tables", ["make_tables_global.py"])]
for nm, cmd in stages:
    t0 = time.time(); log(f"--- stage {nm}: {' '.join(cmd)}"); p = subprocess.Popen([sys.executable, os.path.join(HERE, cmd[0])] + cmd[1:], env=env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        LOG.write(line); LOG.flush()
        if nm in ("0_config", "3_tables", "5_summary", "g4_tables") or "family" in line[:12] or "paste floor" in line or "PASTE" in line or "WALK STAGE" in line or "[done]" in line or "DONE" in line or "WARNING" in line: print(line, end="", flush=True)
    rc = p.wait(); times[nm] = dict(seconds=round(times.get(nm, dict(seconds=0))["seconds"] + time.time() - t0, 1), rc=rc, gpu_temp_end=gpu_temp()); json.dump(times, open(os.path.join(STAGE, "STAGE_TIMES.json"), "w"), indent=1)
    log(f"--- stage {nm} rc={rc} {time.time() - t0:.0f}s GPU temp {gpu_temp()} (peak so far {peak['max']})")
    if rc != 0: log("STAGE FAILED, stopping"); stop.set(); sys.exit(rc)
stop.set(); json.dump(dict(peak_c=peak["max"], sampled_every_s=10, samples=peak["samples"], samples_over_80=peak["over80"]), open(os.path.join(STAGE, "GPU_TEMP.json"), "w")); log("ALL STAGES DONE | GPU peak", peak["max"], "C | tables:", os.path.join(STAGE, "TABLES.md"))
