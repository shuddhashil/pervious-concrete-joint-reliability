"""Item 3 driver: six worked targets x {MT, IND}, each in its own process on its own per-target copy (as JointVsSeparate driver.py), then a second
pass that final-evaluates each model's chosen design under the OTHER model. Resumable (finished json files are skipped).
usage: python run_design.py <GE_root (paper base)> <GE_IND_root (IND base)> <ind_oracle.pt> <candidates.json> <out_dir>"""
import os, sys, json, subprocess
GE, GEI, OPT, CANDS, OUT = [os.path.abspath(a) for a in sys.argv[1:6]]; KIT = os.path.dirname(os.path.abspath(__file__)); os.makedirs(OUT, exist_ok=True)
TARGETS = [(10, 8), (14, 6), (16, 10), (20, 4), (18, 13), (25, 13)]
if os.environ.get("REVERSE"): TARGETS = TARGETS[::-1]          # second machine works from the other end


def tdir(root, S, K):
    code = (f"import sys; sys.path.insert(0, {root!r}); sys.dont_write_bytecode=True\nfrom gengine import target as TG\n"
            f"print(TG.make_target({S}, {K}, 0.90)['dir'])")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=root).stdout.strip().splitlines()[-1]


def run(kind, S, K, out, extra=None):
    if os.path.exists(out): print("done already:", out, flush=True); return
    root = GE if kind == "MT" else GEI; T = tdir(root, S, K)
    env = dict(os.environ, GENGINE_BASE=T, GENGINE_CONFIG=os.path.join(T, "CONFIG.json"), GENGINE_ROOT=root, PYTHONDONTWRITEBYTECODE="1")
    if not os.path.exists(os.path.join(T, "CONFIG.json")):
        subprocess.run([sys.executable, f"{root}/gengine/target_config.py", str(float(S)), str(float(K)), "0.9", "0", "1", "2"], env=env, cwd=root, check=True)
    print(f"=== {kind} S{S} K{K}", flush=True)
    subprocess.run([sys.executable, "-u", f"{KIT}/design_eval.py", kind, CANDS, out, OPT if kind == "IND" else ""] + ([extra] if extra else []), env=env, cwd=root, check=True)


KINDS = os.environ.get("KINDS", "MT,IND").split(",")          # split runs: e.g. KINDS=IND on one machine, KINDS=MT on another
for S, K in TARGETS:
    for kind in KINDS: run(kind, S, K, f"{OUT}/{kind}_S{S}_K{K}.json")
for kind, other in ([] if os.environ.get("NOCROSS") else (("MT", "IND"), ("IND", "MT"))):           # cross final evaluation (needs both models' files)
    extra = {}
    for S, K in TARGETS:
        d = json.load(open(f"{OUT}/{other}_S{S}_K{K}.json")); lo = d["lowest_accepted"]
        if lo: extra[d["target"]] = [dict(family=lo["family"], C=lo["C"], Z=lo["Z"], tag=f"lowest_accepted_{other}_under_{kind}")]
    ep = f"{OUT}/extra_for_{kind}.json"; json.dump(extra, open(ep, "w"))
    for S, K in TARGETS:
        if f"S{S}_K{K}_R0.90" in extra: run(kind, S, K, f"{OUT}/{kind}_S{S}_K{K}_cross.json", ep)
print("ALL DONE", flush=True)
