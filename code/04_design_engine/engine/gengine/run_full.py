"""Full walk run for the target of the active CONFIG (default CONFIG.json; a per-target run sets GENGINE_BASE / GENGINE_CONFIG, see run_target.py). 3 families x seeds, parents by select_parents (G-3 rule; model-chosen starts only for a zero-near-unit family), paste-film guard in the acceptance (skipped when the target has no paste floor).
Seeds = cfg walk.seeds (bank seeds only differ; parents, config, estimator identical). Output <stage>/seed{s}/F{f}/<pid>.jsonl (+ stage log). Resumable (JSONL replay). One GPU process.
usage: python gengine/run_full.py [deadline_hours]   (default 2.0 h for the walk stage)"""
import sys, os, json, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
import numpy as np, torch
from gengine import config, walk
from gengine.common import gpu_temp
cfg, H = config.load(); OUT = config.out_dir(cfg); os.makedirs(OUT, exist_ok=True); LOGF = open(os.path.join(OUT, "full_run.log"), "a"); STATE = os.path.join(OUT, "RUN_STATE.json")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
hours = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0; t_all = time.time(); deadline = t_all + hours * 3600
log("CONFIG", H, "start", time.strftime("%Y-%m-%d %H:%M:%S"), "GPU temp:", gpu_temp(), "deadline h:", hours)
log("target S* K* R*:", cfg["target"]); ctx = walk.Ctx(cfg, H); lab = np.array(cfg["families"]["labels"]); lou = lab[ctx.data.blend_of]
log("paste floor h_min (mm, full target):", ctx.h_min, "unit", ctx.pf["argmin_unit"], "| units meeting the full target:", ctx.pf["n_units_full_target"], "| CS-only floor:", ctx.pf["h_min_cs_only"], "unit", ctx.pf["argmin_unit_cs_only"])
if ctx.pf["reason"]: log("PASTE FLOOR NULL:", ctx.pf["reason"], "| paste check SKIPPED; every result is 'model-suggested, no paste floor'")
if ctx.pf["warning"]: log(ctx.pf["warning"])
par = {}
for f in range(3):
    p, info = walk.select_parents(ctx, f, lou, cfg["walk"]["K_parents"]); par[f] = (p, info); log(f"family {f}: {len(p)} parents, start rule: {info['rule']}, {info}")
json.dump({str(f): dict(info=v[1], parents=v[0]) for f, v in par.items()}, open(os.path.join(OUT, "parents.json"), "w"), indent=1)
timing = json.load(open(STATE)) if os.path.exists(STATE) else {}; done_all = True
for sid in cfg["walk"]["seeds"]:
    ctx.sid = sid
    for f in range(3):
        key = f"seed{sid}_F{f}"; t0 = time.time(); d = os.path.join(OUT, f"seed{sid}", f"F{f}"); temp0 = gpu_temp()
        sts = walk.run_family(ctx, f, par[f][0], d, cfg, H, log=log, deadline=deadline); fin = all(s["stop"] is not None for s in sts.values())
        rec = timing.get(key, dict(seconds=0.0)); rec.update(seconds=rec["seconds"] + time.time() - t0, finished=fin, gpu_temp_start=temp0, gpu_temp_end=gpu_temp(), stops={k: s["stop"] for k, s in sts.items()}); timing[key] = rec
        json.dump(timing, open(STATE, "w"), indent=1); log(f"[done] {key} finished={fin} {time.time() - t0:.0f}s total {time.time() - t_all:.0f}s temp {gpu_temp()}")
        if not fin: done_all = False
        if time.time() > deadline: break
    if time.time() > deadline: break
log("WALK STAGE", "COMPLETE" if done_all else "INCOMPLETE (deadline; rerun the same command to resume)", f"{time.time() - t_all:.0f}s", "GPU temp end", gpu_temp())
