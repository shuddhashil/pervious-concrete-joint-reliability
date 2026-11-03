"""0.95/0.95 baseline (2026-10-28): the lowest-cement candidate whose CS-alone and k-alone lower bounds are both >= 0.95 + 0.002
(read from the sep_eval results table) gets the same strict final evaluation as Table 11 (cert.verify_design_cfg, nominal + worst
shift, 2 replicates, joint / S / Q). Candidate keys are rebuilt exactly as in sep_eval.py and checked against the table.
usage (env as for sep_eval.py): python bonf_eval.py <logs_root_for_this_target> <sep_eval_results_json> <out_json>"""
import os, sys, json, glob
sys.dont_write_bytecode = True
ROOT = os.environ.get("GENGINE_ROOT", "/content/GuidedEngine_20261003"); sys.path.insert(0, ROOT)
import numpy as np
from gengine.common import C, T, decode, nom_dicts, SHIFT_LABELS, gpu_lb
from gengine import config, cert
from gengine.walk import Ctx

LOGS, RES, OUT = sys.argv[1:4]; THR = 0.952
cfg, H = config.load(); ctx = Ctx(cfg, H); tau = ctx.tau; EV = C.EV; S0, Q0 = C.S_STAR, C.LOGK_STAR
cert.RG_CSV = os.path.join(ROOT, "stages", "00_base", "recipe_groups_631.csv")


def set_mode(m):
    s, q = {"joint": (S0, Q0), "S": (S0, -1e9), "Q": (-1e9, Q0)}[m]
    C.S_STAR, C.LOGK_STAR = s, q; EV.S_STAR, EV.LOGK_STAR = s, q


# same candidate keys and order as sep_eval.py
cands = set()  # keys as in sep_eval.py
for p in glob.glob(os.path.join(LOGS, "**", "seed*", "F*", "*.jsonl"), recursive=True):
    for line in open(p):
        l = json.loads(line)
        if l.get("event") != "attempt" or "Z" not in l or not set(l.get("reject_reasons") or []) <= {"reliability"}: continue
        cands.add((int(l["family"]), round(float(l["C"]), 6), tuple(round(float(x), 10) for x in l["Z"])))
for p in glob.glob(os.path.join(LOGS, "**", "global_parents.json"), recursive=True):
    for key, lst in json.load(open(p)).items():
        for g in lst: cands.add((int(key.split("_F")[1]), round(float(g["C0"]), 6), tuple(round(float(x), 10) for x in g["Z"])))
tab = json.load(open(RES))["table"]
assert len(cands) == len(tab), "candidate set differs from sep_eval"
ok = [i for i, t in enumerate(tab) if t["feasible"] and min(t["LB_S"], t["LB_Q"]) >= THR]
if not ok:
    json.dump(dict(design=None), open(OUT, "w")); print("no 0.95/0.95 design"); sys.exit()
row = tab[ok[0]]
# the table row identifies a design only by cement and family, and file order differs between machines, so every candidate with
# that cement and family is re-scored on the same bank as sep_eval and the one whose three bounds match the row is taken
same = [k for k in cands if k[0] == row["family"] and k[1] == row["C"]]
bank, u = ctx.bank(cfg["walk"]["N_accept"], seed=20261027); ctx.eng.set_lowprec(False)


def score(k):
    nom = decode(T([k[2]]), T([k[1]])); nc, owner, kk, pen, fok, fd = ctx.exact_feasible(nom, k[0]); kkn = kk.cpu().numpy(); lb, worst = {}, {}
    for m in ("joint", "S", "Q"):
        set_mode(m); r1, r2 = gpu_lb.lcb95_tie_bracket(ctx.LB, nc, bank, u=u, units=("study",)); LB = np.minimum(r1["LB"], r2["LB"])
        lb[m] = float(LB.min()); worst[m] = SHIFT_LABELS[int(kkn[int(np.argmin(LB))])]
    set_mode("joint")
    return nom, lb, worst


best = None
for k in same:
    nom, lb, worst = score(k); dist = max(abs(lb["joint"] - row["LB_joint"]), abs(lb["S"] - row["LB_S"]), abs(lb["Q"] - row["LB_Q"]))
    print("  candidate", k[1], "LB joint/S/Q", round(lb["joint"], 4), round(lb["S"], 4), round(lb["Q"], 4), "dist", round(dist, 5), flush=True)
    if best is None or dist < best[0]: best = (dist, k, nom, lb, worst)
dist, key, nom, lb, worst = best
assert dist < 0.005, f"no candidate reproduces the table row (closest {dist})"
f = key[0]
print("0.95/0.95 design: C", key[1], "family", f, "row LB S/Q/joint", row["LB_S"], row["LB_Q"], row["LB_joint"], "| match", round(dist, 5), flush=True)
nd = nom_dicts(nom)[0]


def shifted(label):
    g = np.array(nd["g"], float)
    if label != "nominal": a, b = map(int, label.split("->")); g[a] -= tau; g[b] += tau
    return dict(nd, g=g.tolist())


runs = {}
for lab in sorted({"nominal"} | set(worst.values())):
    for m in ("joint", "S", "Q"):
        set_mode(m)
        for rep in (0, 1):
            r = cert.verify_design_cfg(f"BONF_{lab}", shifted(lab), rep, sd_pp=cfg["porosity_sd_pp"]["main"], n_boot=100000, units=("study",))
            runs[f"{lab}|{m}|rep{rep}"] = dict(R_LCB95=r["R_LCB95_joint"], Rhat=r["Rhat_joint"], N=r["n_scenarios"], frozen_gate=bool(r["frozen_gate"]["all_pass"]))
            print(lab, m, rep, round(r["R_LCB95_joint"], 4), round(r["Rhat_joint"], 4), flush=True)
        set_mode("joint")
res = dict(C=key[1], family=f, design=nd, search=row, rescored=lb, match=dist, runs=runs,
           **{"min_R_" + m: min(v["R_LCB95"] for k, v in runs.items() if k.split("|")[1] == m) for m in ("joint", "S", "Q")},
           pass_nominal_joint=float(np.mean([v["Rhat"] for k, v in runs.items() if k.startswith("nominal|joint")])))
json.dump(res, open(OUT, "w"), indent=1, default=float); print("saved", OUT, flush=True)
