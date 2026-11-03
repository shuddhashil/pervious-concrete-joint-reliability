"""Integrated walk: recorded parents (UNCHANGED, source "recorded") + global-seeded parents (source "global"). One GPU process; runs INSIDE a per-target process (env GENGINE_BASE / GENGINE_CONFIG).
usage: python gengine/run_global_walk.py [deadline_hours]
Inputs : <target>/stage (baseline walk, READ only: parents.json, paste_floor.json, seed*/F*/u*.jsonl, certs.jsonl) and <target>/stage_global_walk/global/runs.jsonl (global scan).
Output : <target>/stage_global_walk/ parents.json (recorded + global), global_parents.json, seed{s}/F{f}/*.jsonl, RUN_STATE_GW.json, global_walk.log.
Recorded parents: select_parents is called exactly as in the baseline run (asserted: same parent ids as the baseline parents.json); their walk logs are BYTE COPIES of the baseline logs (same code, same config, same bank seeds: the walk is deterministic per
parent and the baseline logs are the baseline result), so they are not recomputed. Global parents: per family, the best global designs of walk seed s at (a) the lowest passing level and (b) the next scanned level up (fallback: that seed's own lowest passing level, flagged);
each starts as an accepted 'repair' step (feasible start, no repair phase), its parent record has source="global", recorded_passer=False; evidence is counted at the design itself (walk accept metrics, data-only)."""
import sys, os, json, time, shutil, hashlib
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
import numpy as np
from gengine import config, walk
from gengine import common as CM
from gengine.common import gpu_temp
from gengine.global_search import stage_dirs, tier_of


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def global_pid(level, seed): return f"g{level:g}_s{seed}"


def load_runs(gl):
    runs = {}
    for l in open(os.path.join(gl, "runs.jsonl")):
        if l.strip(): r = json.loads(l); runs[(r["family"], r["C"], r["seed"])] = r
    return runs


def global_levels(runs, fam):
    """(a, b): a = lowest level where any seed passed; b = next scanned level above a (any seed ran there). None if nothing passed."""
    passed = sorted({c for (f, c, s), r in runs.items() if f == fam and r["passed"]})
    if not passed: return None, None
    a = passed[0]; above = sorted({c for (f, c, s) in runs if f == fam and c > a}); return a, (above[0] if above else None)


def global_parents(runs, fam, seed):
    """Global parent records for (family, walk seed): from that seed's own search runs. Returns list of dict(parent record, 'start_line' accept metrics)."""
    a, b = global_levels(runs, fam); out = []; seen = set()
    if a is None: return out
    for tag, L in (("a_lowest_passing_level", a), ("b_next_level_up", b)):
        if L is None: continue
        r = runs.get((fam, float(L), seed)); fb = None
        if r is None or not r["passed"]:                       # fallback: this seed's own lowest passing level at or above L
            ok = sorted(c for (f, c, s), x in runs.items() if f == fam and s == seed and x["passed"] and c >= L)
            if not ok: continue
            fb = ok[0]; r = runs[(fam, float(fb), seed)]
        if r["C"] in seen: continue
        seen.add(r["C"]); nd = r["nominal_design"]; rec = [r["C"], nd["r"], nd["A"], nd["P"], nd["S"], nd["phi"]]
        out.append(dict(pid=global_pid(r["C"], seed), unit=None, C0=float(r["C"]), rec=rec, Z=r["Z"], recorded_passer=False, recorded_passer_raw=False, phi_clamped=False, phi_clamped_to_window=False, clamped_coords=[], source="global",
                        start_rule="global_search_seed", model_suggested=None, global_tag=tag, global_level_requested=float(L), global_fallback_level=fb, global_search_seed=int(seed), global_LB_min=r["LB_min_all"], global_tier_at_start=r["tier"], global_n_near_pass=r["n_near_pass"],
                        point_rhat_at_recorded=None, support_fraction=r["accept"]["nominal"]["support_fraction"], recorded_phi=nd["phi"], blend=None, CS=None, K=None, accept=r["accept"], run=r))
    return out


def start_line(p, fam):
    """The global design as the first (accepted) 'repair' attempt of a walk log: exactly what a walk logs when its repair step passes, so replay puts the parent in the descend phase at this cement."""
    r = p["run"]
    return dict(event="attempt", family=fam, pid=p["pid"], round=0, phase="repair", delta=10.0, C=p["C0"], Z=p["Z"], accepted=True, reject_reason=None, reject_reasons=[], source="global",
                polish=dict(source="global_search", search_seed=r["seed"], exact_bank_tag=r["exact_bank_tag"], note="start accepted by the global search's exact objective (= walk acceptance), bank differs from walk banks"),
                accept=r["accept"], n_active=1, polish_batch_s=0.0, accept_batch_s=0.0, round_wall_s=0.0, global_runtime_s=r["runtime_s"], gpu_temp=r["gpu_temp"])


def main():
    cfg, H = config.load(); BASE, OUT = stage_dirs(cfg); GL = os.path.join(OUT, "global"); os.makedirs(OUT, exist_ok=True); LOGF = open(os.path.join(OUT, "global_walk.log"), "a")
    def log(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + chr(10)); LOGF.flush()
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0; t_all = time.time(); deadline = t_all + hours * 3600
    log("global walk start", time.strftime("%Y-%m-%d %H:%M:%S"), "target", cfg["target"], "config", H[:12], "GPU temp", gpu_temp(), "deadline h", hours)
    ctx = walk.Ctx(cfg, H); lab = np.array(cfg["families"]["labels"]); lou = lab[ctx.data.blend_of]; runs = load_runs(GL); base_par = json.load(open(os.path.join(BASE, "parents.json")))
    par, info = {}, {}
    for f in range(3):
        p, inf = walk.select_parents(ctx, f, lou, cfg["walk"]["K_parents"]); assert [x["pid"] for x in p] == [x["pid"] for x in base_par[str(f)]["parents"]], f"F{f}: recorded parents differ from the baseline"
        par[f] = [dict(x, source="recorded") for x in p]; info[f] = inf
    allpar = {}                                                                         # (seed, fam) -> parents of that walk seed
    for sid in cfg["walk"]["seeds"]:
        for f in range(3): allpar[(sid, f)] = par[f] + global_parents(runs, f, sid)
    json.dump({str(f): dict(info=info[f], levels_a_b=list(global_levels(runs, f)), parents=par[f]) for f in range(3)}, open(os.path.join(OUT, "parents.json"), "w"), indent=1)
    json.dump({f"seed{sid}_F{f}": [{k: v for k, v in p.items() if k not in ("accept", "run")} for p in allpar[(sid, f)] if p["source"] == "global"] for (sid, f) in allpar}, open(os.path.join(OUT, "global_parents.json"), "w"), indent=1)
    for (sid, f), ps in allpar.items():
        log(f"seed {sid} F{f}: {sum(p['source'] == 'recorded' for p in ps)} recorded + {sum(p['source'] == 'global' for p in ps)} global parents: " + ", ".join(f"{p['pid']}({p['global_tag'][0]}{'*' if p['global_fallback_level'] else ''})" for p in ps if p["source"] == "global"))
    # copy the baseline recorded-parent logs (bytes), never overwrite an existing new log
    for sid in cfg["walk"]["seeds"]:
        for f in range(3):
            d = os.path.join(OUT, f"seed{sid}", f"F{f}"); os.makedirs(d, exist_ok=True)
            for p in par[f]:
                src = os.path.join(BASE, f"seed{sid}", f"F{f}", p["pid"] + ".jsonl"); dst = os.path.join(d, p["pid"] + ".jsonl")
                if not os.path.exists(dst): shutil.copyfile(src, dst); assert sha(src) == sha(dst)
            for p in allpar[(sid, f)]:
                if p["source"] == "global":
                    dst = os.path.join(d, p["pid"] + ".jsonl")
                    if not os.path.exists(dst): open(dst, "w").write(json.dumps(start_line(p, f)) + chr(10))
    walk.cool_down = lambda log=print: CM.cool_down(78, 70, log=log)                        # heat rule 78 -> 70 C for this process
    STATE = os.path.join(OUT, "RUN_STATE_GW.json"); timing = json.load(open(STATE)) if os.path.exists(STATE) else {}; done_all = True; peak = gpu_temp()
    for sid in cfg["walk"]["seeds"]:
        ctx.sid = sid
        for f in range(3):
            key = f"seed{sid}_F{f}"; t0 = time.time(); d = os.path.join(OUT, f"seed{sid}", f"F{f}")
            if not any(p["source"] == "global" for p in allpar[(sid, f)]): log(f"[{key}] no global parent (no pass in the scan); recorded logs only"); continue
            sts = walk.run_family(ctx, f, allpar[(sid, f)], d, cfg, H, log=log, deadline=deadline); fin = all(s["stop"] is not None for s in sts.values()); peak = max(peak, gpu_temp())
            rec = timing.get(key, dict(seconds=0.0)); rec.update(seconds=rec["seconds"] + time.time() - t0, finished=fin, gpu_temp_end=gpu_temp(), stops={k: s["stop"] for k, s in sts.items() if k.startswith("g")}); timing[key] = rec
            json.dump(timing, open(STATE, "w"), indent=1); log(f"[done] {key} finished={fin} {time.time() - t0:.0f}s temp {gpu_temp()}")
            done_all &= fin
            if time.time() > deadline: break
        if time.time() > deadline: break
    json.dump(dict(peak_c_between_families=peak), open(os.path.join(OUT, "gpu_temp_walk.json"), "w"))
    log("GLOBAL WALK STAGE", "COMPLETE" if done_all else "INCOMPLETE (deadline; rerun resumes)", f"{time.time() - t_all:.0f}s", "GPU temp end", gpu_temp())


if __name__ == "__main__":
    main()
