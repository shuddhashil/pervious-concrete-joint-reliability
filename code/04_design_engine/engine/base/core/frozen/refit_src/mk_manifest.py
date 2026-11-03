"""Build manifest.json + manifest.sha256 over the run directory."""
import hashlib, json, os
RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = {}
for root, dirs, fs in os.walk(RUN):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in sorted(fs):
        if f in ("manifest.json", "manifest.sha256"):
            continue
        p = os.path.join(root, f)
        rel = os.path.relpath(p, RUN).replace(os.sep, "/")
        files[rel] = hashlib.sha256(open(p, "rb").read()).hexdigest()
man = {"run_id": "run_01", "seed": 20260913, "files": files}
json.dump(man, open(os.path.join(RUN, "manifest.json"), "w"), indent=2, sort_keys=True)
h = hashlib.sha256(open(os.path.join(RUN, "manifest.json"), "rb").read()).hexdigest()
open(os.path.join(RUN, "manifest.sha256"), "w").write(h + "  manifest.json\n")
print(f"manifest: {len(files)} files, sha {h[:16]}...")
