"""Step 1a (v3): harvest every design that the six earlier target-specific searches recorded (read only; adapted from harvest_tuned.py, 2026-10-22).
Sources = every ROOT/supportmap_v3/inputs/six_targets/<name>/stage_global_walk: (a) accepted walk attempts (seed*/F*/*.jsonl, event "attempt" and accepted), (b) the best design of every run in global/runs.jsonl
(passed or not), (c) the chosen designs of SELECTION_GLOBAL.json (entries -> chosen / chosen_t2 -> "l" -> family, C, Z; a missing key is skipped).
Output data_v2/tuned_raw.npz (Z (n,11), C, fam, kinds, origin) + data_v2/tuned_harvest.json. Duplicates (same family, cement to 1e-6, Z to 1e-9) are merged and all kinds are kept."""
import os, sys, json, glob
import numpy as np
SM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT = os.path.dirname(SM)
D2 = os.path.join(SM, "data_v2"); os.makedirs(D2, exist_ok=True)
SRC = sorted(glob.glob(os.path.join(SM, "inputs", "six_targets", "*", "stage_global_walk")))
found = {}; counts = {}; skipped = []


def add(src, kind, origin, fam, C, Z):
    if Z is None or len(Z) != 11: skipped.append(origin); return
    key = (int(fam), round(float(C), 6), tuple(round(float(z), 9) for z in Z)); e = found.setdefault(key, dict(fam=int(fam), C=float(C), Z=[float(z) for z in Z], kinds=set(), origins=[]))
    e["kinds"].add(f"{src}:{kind}"); len(e["origins"]) < 3 and e["origins"].append(origin); counts[f"{src}:{kind}"] = counts.get(f"{src}:{kind}", 0) + 1


for d in SRC:
    src = os.path.basename(os.path.dirname(d))
    for p in sorted(glob.glob(os.path.join(d, "seed*", "F*", "*.jsonl"))):
        for i, ln in enumerate(open(p, encoding="utf-8")):
            if not ln.strip(): continue
            try: L = json.loads(ln)
            except Exception: skipped.append(os.path.relpath(p, ROOT) + f":{i} (bad json)"); continue
            if L.get("event") == "attempt" and L.get("accepted") and all(k in L for k in ("family", "C", "Z")):
                add(src, "walk_accepted", os.path.relpath(p, ROOT) + f":{i}", L["family"], L["C"], L["Z"])
    rj = os.path.join(d, "global", "runs.jsonl")
    if os.path.exists(rj):
        for i, ln in enumerate(open(rj, encoding="utf-8")):
            if not ln.strip(): continue
            try: L = json.loads(ln)
            except Exception: skipped.append(os.path.relpath(rj, ROOT) + f":{i} (bad json)"); continue
            if all(k in L for k in ("family", "C", "Z")): add(src, "global_best_passed" if L.get("passed") else "global_best_failed", os.path.relpath(rj, ROOT) + f":{i}", L["family"], L["C"], L["Z"])
            else: skipped.append(os.path.relpath(rj, ROOT) + f":{i} (no design)")
    sp = os.path.join(d, "SELECTION_GLOBAL.json")
    if os.path.exists(sp):
        try: sel = json.load(open(sp, encoding="utf-8"))
        except Exception: skipped.append(os.path.relpath(sp, ROOT) + " (bad json)"); sel = {}
        for k, e in (sel.get("entries") or {}).items():
            for ck in ("chosen", "chosen_t2"):
                c = e.get(ck) if isinstance(e, dict) else None; l = c.get("l") if isinstance(c, dict) else None
                if isinstance(l, dict) and all(x in l for x in ("family", "C", "Z")): add(src, "certified_" + ck, os.path.relpath(sp, ROOT) + ":" + k, l["family"], l["C"], l["Z"])

keys = sorted(found); Z = np.array([found[k]["Z"] for k in keys], float).reshape(-1, 11); Cv = np.array([found[k]["C"] for k in keys], float); fam = np.array([found[k]["fam"] for k in keys], int)
kinds = np.array(["|".join(sorted(found[k]["kinds"])) for k in keys], dtype=str); origin = np.array([found[k]["origins"][0] for k in keys], dtype=str)
np.savez(os.path.join(D2, "tuned_raw.npz"), Z=Z, C=Cv, fam=fam, kinds=kinds, origin=origin)
js = dict(n_sources=len(SRC), sources=[os.path.basename(os.path.dirname(d)) for d in SRC], records_by_kind=counts, n_unique=int(len(keys)), by_family={str(f): int((fam == f).sum()) for f in (0, 1, 2)},
          cement_min=float(Cv.min()) if len(Cv) else None, cement_max=float(Cv.max()) if len(Cv) else None, skipped=skipped[:50], n_skipped=len(skipped))
json.dump(js, open(os.path.join(D2, "tuned_harvest.json"), "w"), indent=1); print(json.dumps(js, indent=1))
