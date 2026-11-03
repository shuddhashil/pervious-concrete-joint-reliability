"""Joint vs separate acceptance on the SAME candidate designs (2026-10-27). Forward-only: nothing is fitted, the engine is unchanged.
Candidates = every design the six-target runs (RobustEngine, 2026-10-22) evaluated at acceptance that failed on reliability only or was accepted
(walk + global-walk logs, plus the global-scan parents). Each is re-scored with the engine's own acceptance machinery (11 gradation
neighbours, both tie rules, N_accept scenarios, one fixed bank per target = common random numbers) under three pass definitions:
  joint : CS >= S* and ln k >= ln K*   (the paper's rule)
  S     : CS >= S* only                (ln K* set to -inf)
  Q     : ln k >= ln K* only           (S* set to -inf)
separate rule = min(LB_S, LB_Q) >= R_ACC, i.e. each property accepted on its own lower bound. Then the lowest-cement design accepted by the
separate rule gets the strict final evaluation (cert.verify_design_cfg, nominal + worst shift, 2 replicates) under all three definitions.
usage (env GENGINE_BASE / GENGINE_CONFIG set to the target copy): python sep_eval.py <logs_root_for_this_target> <out_json>"""
import os, sys, json, glob, time, math
sys.dont_write_bytecode = True
ROOT = os.environ.get("GENGINE_ROOT", "/content/GuidedEngine_20261003"); sys.path.insert(0, ROOT)
import numpy as np, torch
from gengine.common import C, T, decode, nom_dicts, SHIFT_LABELS, gpu_lb
from gengine import config, cert
from gengine.walk import Ctx

LOGS, OUT = sys.argv[1], sys.argv[2]; LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else None     # LIMIT = smoke test on the first n candidates
cfg, H = config.load(); ctx = Ctx(cfg, H); R_ACC = ctx.R_ACC; tau = ctx.tau
EV = C.EV; S0, Q0 = C.S_STAR, C.LOGK_STAR
cert.RG_CSV = os.path.join(ROOT, "stages", "00_base", "recipe_groups_631.csv")
print("target", (C.S_STAR, C.K_STAR, C.REL_TARGET), "R_ACC", R_ACC, flush=True)


def set_mode(m):
    s, q = {"joint": (S0, Q0), "S": (S0, -1e9), "Q": (-1e9, Q0)}[m]
    C.S_STAR, C.LOGK_STAR = s, q; EV.S_STAR, EV.LOGK_STAR = s, q


# ---- candidates ----
cands = {}
for p in glob.glob(os.path.join(LOGS, "**", "seed*", "F*", "*.jsonl"), recursive=True):
    for line in open(p):
        l = json.loads(line)
        if l.get("event") != "attempt" or "Z" not in l: continue
        if not set(l.get("reject_reasons") or []) <= {"reliability"}: continue
        k = (int(l["family"]), round(float(l["C"]), 6), tuple(round(float(x), 10) for x in l["Z"]))
        cands.setdefault(k, dict(src=os.path.relpath(p, LOGS), logged_LB=l.get("accept", {}).get("LB_min_all") if isinstance(l.get("accept"), dict) else None, accepted=bool(l.get("accepted"))))
for p in glob.glob(os.path.join(LOGS, "**", "global_parents.json"), recursive=True):
    for key, lst in json.load(open(p)).items():
        f = int(key.split("_F")[1])
        for g in lst:
            k = (f, round(float(g["C0"]), 6), tuple(round(float(x), 10) for x in g["Z"])); cands.setdefault(k, dict(src="global_parents:" + key, logged_LB=None, accepted=None))
keys = sorted(cands, key=lambda k: (k[1], k[0]))
if LIMIT: keys = keys[::max(1, len(keys) // LIMIT)][:LIMIT]
print("candidates", len(keys), flush=True)

# ---- re-score: one bank for the target, all modes ----
bank, u = ctx.bank(cfg["walk"]["N_accept"], seed=20261027); ctx.eng.set_lowprec(False)
res = {m: np.full(len(keys), np.nan) for m in ("joint", "S", "Q")}; worst = {m: [None] * len(keys) for m in res}; feas = np.zeros(len(keys), bool)
t0 = time.time(); B = 24
for f in range(3):
    idx = [i for i, k in enumerate(keys) if k[0] == f]
    for a in range(0, len(idx), B):
        ii = idx[a:a + B]; Z = T([keys[i][2] for i in ii]); Cv = T([keys[i][1] for i in ii]); nom = decode(Z, Cv)
        nc, owner, kk, pen, fok, fd = ctx.exact_feasible(nom, f); own = owner.cpu().numpy(); kkn = kk.cpu().numpy()
        for m in res:
            set_mode(m); r1, r2 = gpu_lb.lcb95_tie_bracket(ctx.LB, nc, bank, u=u, units=("study",)); LB = np.minimum(r1["LB"], r2["LB"]); SF = r1["support_fraction"]
            for j, i in enumerate(ii):
                ix = np.flatnonzero(own == j); res[m][i] = float(LB[ix].min()); worst[m][i] = SHIFT_LABELS[int(kkn[ix[int(np.argmin(LB[ix]))]])]
                if m == "joint": feas[i] = bool((pen.cpu().numpy()[ix] <= 0).all() and fok.cpu().numpy()[ix].all() and (SF[ix] >= 1 - 1e-12).all())
        set_mode("joint")
    print(f"family {f} scored, {time.time() - t0:.0f}s", flush=True)
sep = np.minimum(res["S"], res["Q"])
acc_j = feas & (res["joint"] >= R_ACC); acc_s = feas & (sep >= R_ACC)
low = lambda acc: int(np.flatnonzero(acc)[0]) if acc.any() else None          # keys are sorted by cement
ij, isp = low(acc_j), low(acc_s)
summary = dict(target=dict(S=S0, K=C.K_STAR, R=C.REL_TARGET), R_ACC=R_ACC, n_candidates=len(keys), n_feasible=int(feas.sum()), n_accept_joint=int(acc_j.sum()), n_accept_separate=int(acc_s.sum()),
               n_separate_not_joint=int((acc_s & ~acc_j).sum()),
               lowest_joint=None if ij is None else dict(C=keys[ij][1], family=keys[ij][0], LB_joint=res["joint"][ij], LB_S=res["S"][ij], LB_Q=res["Q"][ij]),
               lowest_separate=None if isp is None else dict(C=keys[isp][1], family=keys[isp][0], LB_joint=res["joint"][isp], LB_S=res["S"][isp], LB_Q=res["Q"][isp], worst_label=worst["joint"][isp]))
print(json.dumps(summary, indent=1), flush=True)


# ---- strict final evaluation of the lowest separate-accepted design (and the lowest joint one, as the check that rescoring reproduces the paper) ----
def nominal_dict(i):
    return nom_dicts(decode(T([keys[i][2]]), T([keys[i][1]])))[0]


def shifted(nd, label):
    g = np.array(nd["g"], float)
    if label != "nominal": a, b = map(int, label.split("->")); g[a] -= tau; g[b] += tau
    return dict(nd, g=g.tolist())


final = {}
for tag, i, modes in (("lowest_separate", isp, ("joint", "S", "Q")), ("lowest_joint", ij, ("joint",))):
    if i is None or (tag == "lowest_joint" and i == isp): continue
    nd = nominal_dict(i); final[tag] = dict(C=keys[i][1], family=keys[i][0], design=nd, runs={})
    labels = sorted({"nominal"} | {worst[m][i] for m in modes})
    for lab in labels:
        for m in modes:
            set_mode(m)
            for rep in (0, 1):
                r = cert.verify_design_cfg(f"JVS_{tag}_{lab}", shifted(nd, lab), rep, sd_pp=cfg["porosity_sd_pp"]["main"], n_boot=100000, units=("study",))
                final[tag]["runs"][f"{lab}|{m}|rep{rep}"] = dict(R_LCB95=r["R_LCB95_joint"], Rhat=r["Rhat_joint"], wilson=r["mc_wilson_lower_95"], boot=r["calib_bootstrap_R_LCB95"], support=r["support_fraction"], N=r["n_scenarios"], frozen_gate=bool(r["frozen_gate"]["all_pass"]))
                print(tag, lab, m, rep, round(r["R_LCB95_joint"], 4), round(r["Rhat_joint"], 4), flush=True)
            set_mode("joint")
    rr = final[tag]["runs"]
    for m in modes: final[tag]["min_R_" + m] = min(v["R_LCB95"] for k, v in rr.items() if k.split("|")[1] == m)
    if "S" in modes: final[tag]["min_R_separate"] = min(final[tag]["min_R_S"], final[tag]["min_R_Q"])
summary["final_evaluation"] = final
summary["table"] = [dict(C=k[1], family=k[0], feasible=bool(feas[i]), LB_joint=res["joint"][i], LB_S=res["S"][i], LB_Q=res["Q"][i], src=cands[k]["src"], logged_LB=cands[k]["logged_LB"]) for i, k in enumerate(keys)]
json.dump(summary, open(OUT, "w"), indent=1, default=float); print("saved", OUT, f"{time.time() - t0:.0f}s", flush=True)
