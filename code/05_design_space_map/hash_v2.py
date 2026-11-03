"""N6 (v2): hash old folders, baseline stages and engine, tools, tests, targets. usage: hash_v2.py START or END (writes logs_v2 hash files; END also compares and writes logs_v2 N6_result.json)."""
import os, sys, json, subprocess, hashlib
SM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT = os.path.dirname(SM); L = os.path.join(SM, "logs_v2"); tag = sys.argv[1]
subprocess.run([sys.executable, os.path.join(ROOT, "tools", "hash_old.py"), os.path.join(L, f"old_hashes_{tag}.json")], check=True)
subprocess.run([sys.executable, os.path.join(ROOT, "tools", "hash_baseline_stages.py"), os.path.join(L, f"baseline_hashes_{tag}.json")], check=True)
out = {}
for sub in ["gengine", "tools", "tests"] + [os.path.join("targets", t) for t in os.listdir(os.path.join(ROOT, "targets"))]:
    for dp, dn, fn in os.walk(os.path.join(ROOT, sub)):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn: p = os.path.join(dp, f); out[os.path.relpath(p, ROOT).replace("\\", "/")] = hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump(out, open(os.path.join(L, f"engine_hashes_{tag}.json"), "w"), indent=0, sort_keys=True)
if tag == "END":
    res = {}
    for nm, a in (("old_folders", "old_hashes"), ("baseline_stages", "baseline_hashes"), ("gengine_tools_tests_targets", "engine_hashes")):
        A = json.load(open(os.path.join(L, a + "_START.json"))); B = json.load(open(os.path.join(L, a + "_END.json"))); diff = sorted(k for k in set(A) | set(B) if A.get(k) != B.get(k))
        res[nm] = dict(files_start=len(A), files_end=len(B), differences=len(diff), different_files=diff[:20])
    res["PASS"] = bool(all(v["differences"] == 0 for v in res.values())); json.dump(res, open(os.path.join(L, "N6_result.json"), "w"), indent=1); print(json.dumps(res, indent=1))
