"""Candidate pool for item 3 (2026-10-29): exactly the set sep_eval.py (JointVsSeparate_20261027) re-scored, written compactly so the Colab kit
does not need the 12 MB of run logs. Rule (copied from sep_eval.py): every walk/global-walk 'attempt' with Z whose reject reasons are a subset of
{reliability} (accepted or failed on reliability only), plus every global-scan parent; key = (family, C rounded 6, Z rounded 10)."""
import zipfile, json, glob, os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
src = {"S14_K6_R0.90": "S14_K6_R0.90_results.zip", "S16_K10_R0.90": "S16_K10_R0.90_results.zip", "S20_K4_R0.90": "S20_K4_R0.90_results.zip",
       "S18_K13_R0.90": "S18_K13_R0.90_FULLGRID_results.zip", "S25_K13_R0.90": "S25_K13_R0.90_results.zip"}
def files(name):
    if name == "S10_K8_R0.90":
        for p in glob.glob("RobustEngine_20261022/results/S10_K8_R0.90/**/*", recursive=True):
            if os.path.isfile(p): yield p.replace(os.sep, "/"), open(p, "rb").read()
    else:
        Z = zipfile.ZipFile("RawResultZips_20261023/" + src[name])
        for n in Z.namelist(): yield n, Z.read(n)
out = {}
for name in ["S10_K8_R0.90"] + list(src):
    c = {}
    for n, b in files(name):
        if n.endswith(".jsonl") and "/seed" in n and "/F" in n:
            for line in b.decode().splitlines():
                l = json.loads(line)
                if l.get("event") != "attempt" or "Z" not in l: continue
                if not set(l.get("reject_reasons") or []) <= {"reliability"}: continue
                k = (int(l["family"]), round(float(l["C"]), 6), tuple(round(float(x), 10) for x in l["Z"])); c.setdefault(k, "walk")
        elif n.endswith("global_parents.json"):
            for key, lst in json.loads(b).items():
                f = int(key.split("_F")[1])
                for g in lst: c.setdefault((f, round(float(g["C0"]), 6), tuple(round(float(x), 10) for x in g["Z"])), "parent")
    keys = sorted(c, key=lambda k: (k[1], k[0])); out[name] = [[k[0], k[1], list(k[2])] for k in keys]
    print(name, len(keys))
json.dump(out, open("SirReview_MTvsIND_20261029/kit/candidates.json", "w"))
print(os.path.getsize("SirReview_MTvsIND_20261029/kit/candidates.json") / 1e6, "MB")
