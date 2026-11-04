"""Background runner for Colab (RobustEngine 2026-10-22): the two stages of GuidedEngine_Colab.ipynb STEP 3 per target, one target at a time, each finished
target zipped to /content/results. Resumable (finished targets are skipped). usage: python run_targets.py 10,8 14,6 ...   (R* = 0.90)"""
import subprocess, os, sys, json, zipfile, time
ROOT = os.path.dirname(os.path.abspath(__file__)); ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(ROOT)))
BASE = "/content" if os.path.isdir("/content") else os.path.join(REPO, "runs")
WORK = os.path.join(BASE, "work"); RES = os.path.join(BASE, "results")
os.makedirs(os.path.join(WORK, "logs"), exist_ok=True); os.makedirs(RES, exist_ok=True)


def name_of(S, K, R): return f"S{S:g}_K{K:g}_R{R:.2f}"


def stage(args, logf):
    p = subprocess.Popen(["python", "-u", "gengine/run_target.py"] + args, cwd=ROOT, env=ENV, stdout=logf, stderr=subprocess.STDOUT); return p.wait()


def is_done(name):
    try: return json.load(open(f"{ROOT}/targets/{name}/stage_global_walk/STAGE_TIMES.json")).get("g4_tables", {}).get("rc") == 0
    except Exception: return False


def make_zip(name):
    T = f"{ROOT}/targets/{name}"; out = os.path.join(RES, f"{name}_results.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("stage", "stage_global_walk"):
            for r, _, fs in os.walk(f"{T}/{sub}"):
                for f in fs: z.write(os.path.join(r, f), f"{name}/{sub}/" + os.path.relpath(os.path.join(r, f), f"{T}/{sub}"))
        z.write(os.path.join(WORK, "logs", f"{name}.log"), f"logs/{name}.log")
    return out


for tok in sys.argv[1:]:
    S, K = (float(x) for x in tok.split(",")); R = 0.90; name = name_of(S, K, R); T = f"{ROOT}/targets/{name}"; t0 = time.time()
    print(f"=== {name} started {time.strftime('%H:%M:%S')}", flush=True)
    if is_done(name): print("   already done", flush=True); continue
    base = ["--S", str(S), "--K", str(K), "--R", str(R)]
    with open(os.path.join(WORK, "logs", f"{name}.log"), "a") as lf:
        if not os.path.exists(f"{T}/stage/SELECTION.json"): stage(base + ["--hours", "24", "--cert-minutes", "600"], lf)
        if not os.path.exists(f"{T}/stage/SELECTION.json"): print(f"   {name}: no selection file after first half", flush=True); continue
        rc = stage(base + ["--mode", "global_walk", "--scan-hours", "24", "--gwalk-hours", "24", "--cert-minutes", "600"], lf)
    print(f"=== {name} {'finished' if is_done(name) else f'NOT finished (rc={rc})'} in {(time.time() - t0) / 60:.0f} min", flush=True)
    if is_done(name): print("   zip:", make_zip(name), flush=True)
print("ALL DONE", flush=True)
