"""Quick run summary from the stage folder of the active CONFIG (attempt counts, peak temperature, per-key timing, stops). usage: python gengine/run_summary.py (per-target: env GENGINE_CONFIG as in run_target.py)"""
import json, glob, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from gengine import config
OUT = config.out_dir(config.load()[0]); mx, n, nacc = 0, 0, 0
for p in glob.glob(os.path.join(OUT, "seed*", "F*", "*.jsonl")):
    for l in open(p):
        r = json.loads(l)
        if r["event"] == "attempt": n += 1; nacc += bool(r["accepted"]); mx = max(mx, r["gpu_temp"])
print("attempts", n, "accepted", nacc, "peak gpu temp (between rounds)", mx)
for k, v in json.load(open(os.path.join(OUT, "RUN_STATE.json"))).items(): print(k, round(v["seconds"]), v["gpu_temp_start"], v["gpu_temp_end"], v["stops"])
