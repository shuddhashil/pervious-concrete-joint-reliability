"""Short red team + paste-guard side tables from logs (CPU, light). usage: python gengine/redteam.py -> <stage>/REDTEAM.json (stage of the active CONFIG)
R1 old 253 design: paste h of the design and its 10 shifts against THIS target's floor (none when the target has no floor); the Phase-2 E1 reliability numbers (logs/PHASE2_tests.json) are included ONLY when this config's target equals the target of the root CONFIG.json under which E1 ran; R2 seed spread of the per-family minima (SELECTION.json); R3 nesting check across n_min per parent log;
R4 what-if floors (report only, along the accepted steps of the walks that ran with the real floor): lowest accepted cement whose h_min_over_11 >= p5 / p10 floor."""
import os, sys, json, glob
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
import numpy as np
from gengine import config, checks
cfg, H = config.load(); OUT = config.out_dir(cfg); res = {}; pg = cfg["paste_guard"]; floor = pg["h_min_mm"]; old_floor = pg["h_min_cs_only_mm"]
# ---- R1
nomd = [t for t in json.load(open(os.path.join(os.path.dirname(ROOT), "DefensibleEngine_20261001", "runs", "full2", "verify_w7_tasks.json"))) if t["cid"] == "FF_C253_d0"][0]["nominal"]
D = np.array([21.79449471770337, 15.411035007422441, 10.897247358851684, 7.736278174936576, 5.470374758643141, 3.348133808556641])
def h(C, r, A, g): return (C / 3150 + r * C / 1000) / (A / 2700 * 6 * (np.asarray(g) / D).sum())
hs = [h(nomd["C"], nomd["r"], nomd["A"], nomd["g"])]
for j in range(5):
    for s, d in ((j, j + 1), (j + 1, j)):
        g = np.array(nomd["g"], float)
        if g[s] - 0.02 >= 0: g[s] -= 0.02; g[d] += 0.02; hs.append(h(nomd["C"], nomd["r"], nomd["A"], g))
res["R1_old253"] = dict(C=nomd["C"], h_nominal=float(hs[0]), h_min_over_shifts=float(min(hs)), floor=floor, cs_only_floor=old_floor, paste_ok_cs_only_floor=None if old_floor is None else bool(min(hs) >= old_floor), paste_ok=None if floor is None else bool(min(hs) >= floor),
                        note="paste check skipped for this target (no floor)" if floor is None else "")
root_cfg = json.load(open(os.path.join(ROOT, "CONFIG.json")))["target"]
if cfg["target"] == root_cfg:                       # E1 (Phase 2) ran under the root CONFIG's target only
    e1 = json.load(open(os.path.join(ROOT, "logs", "PHASE2_tests.json")))["E1"]
    res["R1_old253"].update(accepted_by_walk_rule=e1["accepted_by_walk_rule"], reject_reason=e1["reject_reason"], LB_nominal=e1["LB_nominal"], LB_worst_2pct=e1["LB_worst_2pct"], n_near_pass_units=e1["n_near_pass_units"], near_pass_ids=e1["near_pass_ids"],
                            by_lambda_n_near_pass=e1["by_lambda_n_near_pass"], conflict=e1["conflict"]["flag"], cpu_cert_nominal=e1["cpu_certificate_nominal"], cpu_cert_worst=e1["cpu_certificate_worst"])
else: res["R1_old253"]["E1_reliability_numbers"] = "not available: the Phase-2 E1 test was run for a different target; not reproduced here"
# ---- R2 / R3 / R4
sel = json.load(open(os.path.join(OUT, "SELECTION.json")))["entries"]; res["R2_seed_spread"] = {k: dict(model_min=e["model_min"], data_min=e["data_min"], certified=(e["chosen"]["l"]["C"] if "chosen" in e else None), status=e.get("status")) for k, e in sel.items()}
nest = {}; whatif = {}
for sid in cfg["walk"]["seeds"]:
    for f in range(3):
        acc = []
        for p in glob.glob(os.path.join(OUT, f"seed{sid}", f"F{f}", "*.jsonl")):
            L = checks.load_log(p); v = checks.nesting_violations(checks.parent_summary(L))
            if v: nest[f"seed{sid}_F{f}_{os.path.basename(p)[:-6]}"] = v
            acc += [dict(l, pid=os.path.basename(p)[:-6]) for l in checks.accepted(L)]
        w = {}
        for nm, fl in (("p5", pg["h_p5_mm"]), ("p10", pg["h_p10_mm"]), ("real_floor", pg["h_min_mm"])):
            if fl is None: continue                                                # floor not defined for this target
            c = [l for l in acc if l["accept"]["paste"]["h_min_over_11"] >= fl]; c1 = [l for l in c if checks.data_ok(l, 1)]
            w[nm] = dict(floor=fl, model_min=min((l["C"] for l in c), default=None), data_min_n1=min((l["C"] for l in c1), default=None))
        whatif[f"seed{sid}_F{f}"] = w
res["R3_nesting_violations_all_parent_logs"] = nest; res["R4_whatif_floor_report_only"] = whatif
json.dump(res, open(os.path.join(OUT, "REDTEAM.json"), "w"), indent=1, default=str); print(json.dumps(res, indent=1, default=str)[:6000])
