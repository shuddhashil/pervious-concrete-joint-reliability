"""M6: re-hash everything at the end and compare with the START hashes (old folders 26,672 files, baseline stages 866 files, gengine / tools / tests / targets 897 files)."""
import os, sys, json, subprocess
SM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT = os.path.dirname(SM); L = os.path.join(SM, "logs"); res = {}
subprocess.run([sys.executable, os.path.join(ROOT, "tools", "hash_old.py"), os.path.join(L, "old_hashes_END.json")], check=True); subprocess.run([sys.executable, os.path.join(ROOT, "tools", "hash_baseline_stages.py"), os.path.join(L, "baseline_hashes_END.json")], check=True)
import hashlib
out = {}
for sub in ["gengine", "tools", "tests"]:
    for dp, dn, fn in os.walk(os.path.join(ROOT, sub)):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn: p = os.path.join(dp, f); out[os.path.relpath(p, ROOT).replace("\\", "/")] = hashlib.sha256(open(p, "rb").read()).hexdigest()
for t in ["S18_K13_R0.90", "S20_K10_R0.90", "S25_K13_R0.90"]:
    for dp, dn, fn in os.walk(os.path.join(ROOT, "targets", t)):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn: p = os.path.join(dp, f); out[os.path.relpath(p, ROOT).replace("\\", "/")] = hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump(out, open(os.path.join(L, "engine_hashes_END.json"), "w"), indent=0, sort_keys=True)
for nm, a, b in (("old_folders", "old_hashes_START.json", "old_hashes_END.json"), ("baseline_stages", "baseline_hashes_START.json", "baseline_hashes_END.json"), ("gengine_tools_tests_targets", "engine_hashes_START.json", "engine_hashes_END.json")):
    A = json.load(open(os.path.join(L, a))); B = json.load(open(os.path.join(L, b))); diff = sorted([k for k in set(A) | set(B) if A.get(k) != B.get(k)])
    res[nm] = dict(files_start=len(A), files_end=len(B), differences=len(diff), different_files=diff[:20])
res["PASS"] = bool(all(v["differences"] == 0 for v in res.values())); json.dump(res, open(os.path.join(L, "M6_result.json"), "w"), indent=1); print(json.dumps(res, indent=1))
