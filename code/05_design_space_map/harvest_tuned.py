"""Step 2 (v2): collect every certified / accepted design recorded by the earlier target-specific searches (read only).
(a) targets/S18_K13_R0.90/stage_global_walk: accepted walk steps (seed*/F*/*.jsonl), the best design of every (family, level, seed) global-scan run (global/runs.jsonl), the certified / chosen designs (SELECTION_GLOBAL.json);
    targets/S20_K10_R0.90/stage: accepted walk steps, certified / chosen designs (SELECTION.json);   (b) stages/full_G3: accepted walk steps, chosen designs (SELECTION.json).
Output data_v2/tuned_raw.npz (Z (n,11), C, fam, kind, origin text) + data_v2/tuned_harvest.json (counts). Duplicates (same family, cement, Z to 1e-9) are merged and the kinds are kept."""
import os, sys, json, glob
import numpy as np
SM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT = os.path.dirname(SM)
SRC = [("S18_gw", os.path.join(ROOT, "targets", "S18_K13_R0.90", "stage_global_walk")), ("S20", os.path.join(ROOT, "targets", "S20_K10_R0.90", "stage")), ("full_G3", os.path.join(ROOT, "stages", "full_G3"))]
SEL = {"S18_gw": "SELECTION_GLOBAL.json", "S20": "SELECTION.json", "full_G3": "SELECTION.json"}
found = {}; counts = {}; skipped_lines = 0


def add(src, kind, origin, fam, C, Z):
    key = (int(fam), round(float(C), 6), tuple(round(float(z), 9) for z in Z)); e = found.setdefault(key, dict(fam=int(fam), C=float(C), Z=[float(z) for z in Z], kinds=set(), origins=[]))
    e["kinds"].add(f"{src}:{kind}"); len(e["origins"]) < 3 and e["origins"].append(origin); counts[f"{src}:{kind}"] = counts.get(f"{src}:{kind}", 0) + 1


for src, d in SRC:
    for p in sorted(glob.glob(os.path.join(d, "seed*", "F*", "*.jsonl"))):
        for i, ln in enumerate(open(p)):
            if not ln.strip(): continue
            L = json.loads(ln)
            if L.get("event") == "attempt" and L.get("accepted"): add(src, "walk_accepted", os.path.relpath(p, ROOT) + f":{i}", L["family"], L["C"], L["Z"])
    rj = os.path.join(d, "global", "runs.jsonl")
    if os.path.exists(rj):
        for i, ln in enumerate(open(rj)):
            if ln.strip(): L = json.loads(ln); add(src, "global_best_passed" if L["passed"] else "global_best_failed", os.path.relpath(rj, ROOT) + f":{i}", L["family"], L["C"], L["Z"])
    sel = json.load(open(os.path.join(d, SEL[src])))
    for k, e in sel["entries"].items():
        for ck in ("chosen", "chosen_t2"):
            if ck in e and e[ck] and "l" in e[ck]:
                l = e[ck]["l"]; add(src, "certified_" + ck, os.path.relpath(os.path.join(d, SEL[src]), ROOT) + ":" + k, l["family"], l["C"], l["Z"])
keys = sorted(found); Z = np.array([found[k]["Z"] for k in keys]); Cv = np.array([found[k]["C"] for k in keys]); fam = np.array([found[k]["fam"] for k in keys])
kinds = np.array(["|".join(sorted(found[k]["kinds"])) for k in keys]); origin = np.array([found[k]["origins"][0] for k in keys])
np.savez(os.path.join(SM, "data_v2", "tuned_raw.npz"), Z=Z, C=Cv, fam=fam, kinds=kinds, origin=origin)
js = dict(records_by_kind=counts, n_unique=int(len(keys)), by_family={str(f): int((fam == f).sum()) for f in (0, 1, 2)}, cement_min=float(Cv.min()), cement_max=float(Cv.max()))
json.dump(js, open(os.path.join(SM, "data_v2", "tuned_harvest.json"), "w"), indent=1); print(json.dumps(js, indent=1))
