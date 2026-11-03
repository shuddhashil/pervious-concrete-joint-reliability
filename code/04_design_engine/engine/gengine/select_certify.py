"""Selection from the walk logs, strict CPU certificates, tables (; generic target 2026-10-04). CPU only (<= 3 threads). Resumable: certificates cached in <stage>/certs.jsonl (cid, rep, sd).
usage: python gengine/select_certify.py [budget_minutes]  -> <stage>/SELECTION.json  (stage of the active CONFIG; per-target: env as set by run_target.py)
Selection per (seed, family): headline n_min = 1 (cfg tiers.n_min): Tier 1 family (has a recorded deployable passer AND the target has a paste floor) -> lowest accepted step with >= 1 near recorded passer and no conflict flag; every other family is Tier 2 -> lowest accepted step (model-suggested, never recommended; labelled "no paste floor" when the target has none).
Certificate = frozen estimator, 2 replicates, nominal + worst-shift design, porosity SD 1.5 pp (PASS = all 4 pass); roll back to the next-higher-cement accepted step of the same family on failure (plan section 8, up to 3 fallbacks).
SD 1.0 / 2.0 pp and the no-lab bootstrap variants (recipe_group, row) are report-only and run on the passing design."""
import os, sys, json, time, hashlib, glob
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[k] = "3"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
import numpy as np, torch
torch.set_num_threads(3)
from gengine.common import C, W_LOG, SHIFT_LABELS, gpu_temp
from gengine import config, checks, cert
cfg, H = config.load(); OUT = config.out_dir(cfg); CACHE = os.path.join(OUT, "certs.jsonl"); budget = (float(sys.argv[1]) if len(sys.argv) > 1 else 90.0) * 60; t_start = time.time()
cert.RG_CSV = os.path.join(ROOT, "stages", "00_base", "recipe_groups_631.csv"); UNITS = ("study", "recipe_group", "row"); SDS = [cfg["porosity_sd_pp"]["main"]] + cfg["porosity_sd_pp"]["checks"]
par = json.load(open(os.path.join(OUT, "parents.json"))); cache = {}; NO_FLOOR = cfg["paste_guard"]["h_min_mm"] is None
if os.path.exists(CACHE):
    for l in open(CACHE): r = json.loads(l); cache[(r["cid"], r["rep"], r["sd"])] = r


def cid_of(d): return "G3_" + hashlib.sha1(json.dumps({k: (round(float(v), 10) if k != "g" else [round(float(x), 10) for x in v]) for k, v in sorted(d.items())}).encode()).hexdigest()[:14]


def worst_design(nd, label, tau):
    g = np.array(nd["g"], float)
    if label != "nominal": a, b = map(int, label.split("->")); g[a] -= tau; g[b] += tau
    return dict(nd, g=g.tolist())


def certify(des, rep, sd):
    cid = cid_of(des); k = (cid, rep, sd)
    if k not in cache:
        r = cert.verify_design_cfg(cid, des, rep, sd_pp=sd, n_boot=100000, units=UNITS)
        cache[k] = dict(cid=cid, rep=rep, sd=sd, R=r["R_LCB95_joint"], wilson=r["mc_wilson_lower_95"], boot_study=r["calib_bootstrap_R_LCB95"], R_nolab_recipe_group=r.get("R_LCB95_recipe_group"), R_nolab_row=r.get("R_LCB95_row"),
                        support=r["support_fraction"], frozen_gate=bool(r["frozen_gate"]["all_pass"]), PASS=bool(r["PASS"]), N=r["n_scenarios"], note=r["escalation_note"], seconds=r["elapsed_s"])
        open(CACHE, "a").write(json.dumps(cache[k]) + "\n")
    return cache[k]


def load_family(sid, f):
    L = {}
    for p in glob.glob(os.path.join(OUT, f"seed{sid}", f"F{f}", "*.jsonl")): L[os.path.basename(p)[:-6]] = checks.load_log(p)
    return L


def stop_of(lines):
    s = [l for l in lines if l.get("event") in ("stop", "merged")]; return (s[-1].get("reason") or "merged into " + s[-1]["into"]) if s else "running/unfinished"


sel = dict(config_sha256=H, seeds=cfg["walk"]["seeds"], entries={}); tau = cfg["tau"]
for sid in cfg["walk"]["seeds"]:
    for f in range(3):
        L = load_family(sid, f)
        if not L: continue
        key = f"seed{sid}_F{f}"; has_passer = par[str(f)]["info"]["n_passers"] > 0 and not NO_FLOOR; allacc = [dict(l, pid=pid) for pid, ls in L.items() for l in checks.accepted(ls)]; E = dict(seed=sid, family=f, has_recorded_passer=has_passer, start_rule=par[str(f)]["info"].get("rule", "G3_near_target_units"), paste_floor_status="none" if NO_FLOOR else "computed", n_parents=len(L), n_attempts=sum(len(checks.attempts(ls)) for ls in L.values()))
        E["finished"] = all(stop_of(ls) != "running/unfinished" for ls in L.values()); E["model_min"] = min((l["C"] for l in allacc), default=None)
        E["data_min"] = {n: min((l["C"] for l in allacc if checks.data_ok(l, n)), default=None) for n in (1, 2, 3)}
        # paste guard statistics (report only)
        at = [dict(l, pid=pid) for pid, ls in L.items() for l in checks.attempts(ls)]
        E["fail_counts"] = {n: sum(1 for l in at if n in l.get("reject_reasons", [])) for n in ("gate", "family", "support", "paste_floor", "reliability")}; E["n_rejected"] = sum(1 for l in at if not l["accepted"])        # G-3: ALL failing checks per attempt
        E["paste_stats"] = dict(n_attempts=len(at), n_rejected_paste_floor=sum(1 for l in at if "paste_floor" in l.get("reject_reasons", [])),
                                n_paste_only_blockers=sum(1 for l in at if (not l["accept"]["paste"]["ok"]) and l["accept"]["gates_all_pass"] and l["accept"]["family_all_ok"] and l["accept"]["support_all"] and l["accept"]["paste"]["reliability_ok"]),
                                lowest_attempted_C_passing_everything_except_paste=min((l["C"] for l in at if l["accept"]["gates_all_pass"] and l["accept"]["family_all_ok"] and l["accept"]["support_all"] and l["accept"]["paste"]["reliability_ok"]), default=None),
                                lowest_accepted_h_min_over_11=min((l["accept"]["paste"]["h_min_over_11"] for l in allacc), default=None))
        if not allacc: E["status"] = "no accepted step"; sel["entries"][key] = E; continue
        cands = sorted([l for l in allacc if (checks.data_ok(l, 1) if has_passer else True)], key=lambda l: (l["C"], l["pid"]))
        E["tier"] = "Tier 1" if has_passer else ("Tier 2: model-suggested, no paste floor (never recommended, never data-backed)" if NO_FLOOR else "Tier 2 (model-suggested, never recommended)")
        if has_passer and not cands: E["status"] = "model-only: recorded passers exist but no accepted step has a near recorded passer"; E["certificate"] = None; sel["entries"][key] = E; continue
        tries = []
        for ci, l in enumerate(cands):
            if ci > 3 or time.time() - t_start > budget: break
            a = l["accept"]; nd = a["nominal_design"]; wl = a["worst"]["label"]; dn, dw = nd, worst_design(nd, wl, tau); res = {}
            for tag, des in (("nominal", dn), ("worst", dw)):
                for rep in (0, 1): res[f"{tag}_rep{rep}"] = certify(des, rep, SDS[0])
            ok = all(r["PASS"] for r in res.values()); tries.append(dict(pid=l["pid"], C=l["C"], round=l["round"], worst_label=wl, pass_all4=ok, results={k: {kk: vv for kk, vv in v.items() if kk != "cid"} for k, v in res.items()}))
            print(key, "candidate", l["pid"], l["C"], "PASS" if ok else "FAIL", {k: round(v["R"], 4) for k, v in res.items()}, f"t={time.time() - t_start:.0f}s", flush=True)
            if ok: E["chosen"] = dict(l=l, certificate=tries[-1], fallback_index=ci); break
        E["tries"] = tries
        if "chosen" in E:
            l = E["chosen"]["l"]; a = l["accept"]; nd = a["nominal_design"]; dn, dw = nd, worst_design(nd, a["worst"]["label"], tau); sdres = {}
            for sd in SDS[1:]:
                for tag, des in (("nominal", dn), ("worst", dw)):
                    for rep in (0, 1):
                        if time.time() - t_start > budget: sdres[f"sd{sd}_{tag}_rep{rep}"] = "skipped (budget)"; continue
                        r = certify(des, rep, sd); sdres[f"sd{sd}_{tag}_rep{rep}"] = {kk: vv for kk, vv in r.items() if kk != "cid"}
            E["chosen"]["sd_checks"] = sdres; E["status"] = "certified"
        else: E["status"] = "no candidate certified (or budget used)" if tries else "not run (budget)"
        sel["entries"][key] = E; json.dump(sel, open(os.path.join(OUT, "SELECTION.json"), "w"), indent=1, default=str)
sel["seconds"] = time.time() - t_start; json.dump(sel, open(os.path.join(OUT, "SELECTION.json"), "w"), indent=1, default=str); print("DONE", round(sel["seconds"]), "s")
