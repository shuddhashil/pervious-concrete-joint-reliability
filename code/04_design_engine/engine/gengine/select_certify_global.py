"""Selection + strict CPU certificates for the integrated (recorded + global) walk. CPU only (<= 3 threads), resumable (<stage_global_walk>/certs.jsonl, seeded with a COPY of the baseline certs.jsonl so recorded designs
that are identical to the baseline are not re-certified: the cache key is the design content). usage: python gengine/select_certify_global.py [budget_minutes]
Per (seed, family, source): TIER 1 candidates = accepted steps with >= 1 near recorded passing mix (data_ok n_min=1, no conflict flag) and a paste floor; the lowest is certified (frozen estimator, 2 replicates, nominal + worst 2 % shift, SD 1.5 pp, PASS = all 4),
roll back to the next-higher accepted Tier-1 step on failure (up to 3 fallbacks, as in the baseline). TIER 2 (model-suggested): the lowest accepted step overall is certified too ONLY when there is no certified Tier-1 design for that (seed, family, source) or the Tier-2
design is more than 10 kg/m3 cheaper than it (these are never recommended). SD 1.0 / 2.0 pp and no-lab variants (report only) are run on the lowest certified design of each (family, source, tier) only.
The certificate code is gengine/cert.py (unchanged); cid_of / worst_design / certify below are the same wrappers as in select_certify.py."""
import os, sys, json, time, hashlib, glob, shutil
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[k] = "3"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
import numpy as np, torch
torch.set_num_threads(3)
from gengine.common import C
from gengine import config, checks, cert
from gengine.global_search import stage_dirs, tier1_candidates
cfg, H = config.load(); BASE, OUT = stage_dirs(cfg); CACHE = os.path.join(OUT, "certs.jsonl"); budget = (float(sys.argv[1]) if len(sys.argv) > 1 else 60.0) * 60; t_start = time.time()
cert.RG_CSV = os.path.join(ROOT, "stages", "00_base", "recipe_groups_631.csv"); UNITS = ("study", "recipe_group", "row"); SDS = [cfg["porosity_sd_pp"]["main"]] + cfg["porosity_sd_pp"]["checks"]; tau = cfg["tau"]
NO_FLOOR = cfg["paste_guard"]["h_min_mm"] is None; FLAG_KG = 10.0
if not os.path.exists(CACHE) and os.path.exists(os.path.join(BASE, "certs.jsonl")): shutil.copyfile(os.path.join(BASE, "certs.jsonl"), CACHE)
cache = {}
if os.path.exists(CACHE):
    for l in open(CACHE): r = json.loads(l); cache[(r["cid"], r["rep"], r["sd"])] = r
par = json.load(open(os.path.join(OUT, "parents.json"))); gpar = json.load(open(os.path.join(OUT, "global_parents.json")))


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


def try_certify(cands, key, label):
    tries = []
    for ci, l in enumerate(cands):
        if ci > 3 or time.time() - t_start > budget: break
        a = l["accept"]; nd = a["nominal_design"]; wl = a["worst"]["label"]; dn, dw = nd, worst_design(nd, wl, tau); res = {}
        for tag, des in (("nominal", dn), ("worst", dw)):
            for rep in (0, 1): res[f"{tag}_rep{rep}"] = certify(des, rep, SDS[0])
        ok = all(r["PASS"] for r in res.values()); tries.append(dict(pid=l["pid"], C=l["C"], round=l["round"], worst_label=wl, pass_all4=ok, results={k: {kk: vv for kk, vv in v.items() if kk != "cid"} for k, v in res.items()}))
        print(key, label, "candidate", l["pid"], l["C"], "PASS" if ok else "FAIL", {k: round(v["R"], 4) for k, v in res.items()}, f"t={time.time() - t_start:.0f}s", flush=True)
        if ok: return dict(l=l, certificate=tries[-1], fallback_index=ci), tries
    return None, tries


sel = dict(config_sha256=H, seeds=cfg["walk"]["seeds"], entries={}, flag_kg=FLAG_KG)
for sid in cfg["walk"]["seeds"]:
    for f in range(3):
        L = load_family(sid, f)
        if not L: continue
        src_of = {p["pid"]: "recorded" for p in par[str(f)]["parents"]}; src_of.update({p["pid"]: "global" for p in gpar.get(f"seed{sid}_F{f}", [])})
        for source in ("recorded", "global"):
            Ls = {pid: ls for pid, ls in L.items() if src_of.get(pid) == source}
            if not Ls: continue
            key = f"seed{sid}_F{f}_{source}"; allacc = [dict(l, pid=pid, source=source) for pid, ls in Ls.items() for l in checks.accepted(ls)]; at = [l for ls in Ls.values() for l in checks.attempts(ls)]
            E = dict(seed=sid, family=f, source=source, n_parents=len(Ls), n_attempts=len(at), finished=all(stop_of(ls) != "running/unfinished" for ls in Ls.values()), model_min=min((l["C"] for l in allacc), default=None),
                     data_min={n: min((l["C"] for l in allacc if checks.data_ok(l, n)), default=None) for n in (1, 2, 3)}, stops={pid: stop_of(ls) for pid, ls in Ls.items()},
                     fail_counts={n: sum(1 for l in at if n in l.get("reject_reasons", [])) for n in ("gate", "family", "support", "paste_floor", "reliability")})
            if not allacc: E["status"] = "no accepted step"; sel["entries"][key] = E; continue
            t1 = tier1_candidates(allacc, NO_FLOOR)
            allsorted = sorted(allacc, key=lambda l: (l["C"], l["pid"])); E["tier1_available"] = bool(t1)
            ch1, tr1 = try_certify(t1, key, "T1") if t1 else (None, []); E["tries_t1"] = tr1
            if ch1: E["chosen"] = ch1
            low = allsorted[0]; E["lowest_accepted_overall"] = dict(C=low["C"], pid=low["pid"], tier1=bool(checks.data_ok(low, 1)) and not NO_FLOOR)
            need_t2 = (ch1 is None or low["C"] < ch1["l"]["C"] - FLAG_KG) and not (checks.data_ok(low, 1) and not NO_FLOOR)
            if need_t2:
                t2c = [l for l in allsorted if not (checks.data_ok(l, 1) and not NO_FLOOR)]; ch2, tr2 = try_certify(t2c, key, "T2"); E["tries_t2"] = tr2
                if ch2: E["chosen_t2"] = ch2
            E["status"] = "certified" if ch1 else ("model-suggested only (Tier 2) certified" if E.get("chosen_t2") else ("no Tier-1 candidate" if not t1 else "no candidate certified (or budget used)"))
            sel["entries"][key] = E
            json.dump(sel, open(os.path.join(OUT, "SELECTION_GLOBAL.json"), "w"), indent=1, default=str)
# SD 1.0 / 2.0 checks on the lowest certified design of each (family, source, tier)
for f in range(3):
    for source in ("recorded", "global"):
        for ck in ("chosen", "chosen_t2"):
            es = [e for e in sel["entries"].values() if e["family"] == f and e["source"] == source and ck in e]
            if not es: continue
            e = min(es, key=lambda e: e[ck]["l"]["C"]); l = e[ck]["l"]; nd = l["accept"]["nominal_design"]; dn, dw = nd, worst_design(nd, l["accept"]["worst"]["label"], tau); sdres = {}
            for sd in SDS[1:]:
                for tag, des in (("nominal", dn), ("worst", dw)):
                    for rep in (0, 1):
                        if time.time() - t_start > budget: sdres[f"sd{sd}_{tag}_rep{rep}"] = "skipped (budget)"; continue
                        r = certify(des, rep, sd); sdres[f"sd{sd}_{tag}_rep{rep}"] = {kk: vv for kk, vv in r.items() if kk != "cid"}
            e[ck]["sd_checks"] = sdres; e[ck]["sd_checks_basis"] = "lowest certified design of this (family, source, tier) over the seeds"
sel["seconds"] = time.time() - t_start; json.dump(sel, open(os.path.join(OUT, "SELECTION_GLOBAL.json"), "w"), indent=1, default=str); print("DONE", round(sel["seconds"]), "s")
