"""Stage C: strict certificates at the edge of the map. CPU only (frozen estimator through gengine/cert.py, unchanged), 2 replicates, nominal + worst 2 % shift, porosity SD 1.5 pp.
The frozen core reads its target from TARGET_CONTRACT.json at import; instead of one per-target core copy for every cell, this worker assigns the module constants that the core derives from that file
(runtime_targets.STRENGTH / PERMEABILITY / RELIABILITY, corelib.S_STAR / K_STAR / LOGK_STAR / REL_TARGET, efficient_verification.S_STAR / K_STAR / LOGK_STAR) to the cell's own (S*, K*, R*) before each certificate.
No frozen file is written. tests M7 checks that this gives the SAME numbers as the per-target-copy certificates stored by the earlier phases (18/13 and 20/10). Escalation (N = 20000 -> 50000 -> 100000) is judged against the cell's own R*."""
import os, sys, json, math, time, hashlib
SM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT = os.path.dirname(SM); TGT = os.path.join(ROOT, "targets", "S18_K13_R0.90")
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[k] = "1"
os.environ["GENGINE_BASE"] = TGT; os.environ["GENGINE_CONFIG"] = os.path.join(TGT, "CONFIG.json"); os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True; sys.path.insert(0, ROOT); sys.path.insert(0, TGT)
import numpy as np
from defengine import corelib as C
from gengine import cert
cert.RG_CSV = None
TAU = 0.02; SD = 1.5; _cache = {}


def set_target(S, K, R):
    import runtime_targets as RT
    RT.STRENGTH, RT.PERMEABILITY, RT.RELIABILITY = float(S), float(K), float(R)
    C.S_STAR, C.K_STAR, C.LOGK_STAR, C.REL_TARGET = float(S), float(K), math.log(float(K)), float(R)
    C.EV.S_STAR, C.EV.K_STAR, C.EV.LOGK_STAR = float(S), float(K), math.log(float(K))


def cid_of(d):
    """same content hash as select_certify.cid_of (so the seeds equal the earlier phases')."""
    return "G3_" + hashlib.sha1(json.dumps({k: (round(float(v), 10) if k != "g" else [round(float(x), 10) for x in v]) for k, v in sorted(d.items())}).encode()).hexdigest()[:14]


def worst_design(nd, label, tau=TAU):
    g = np.array(nd["g"], float)
    if label != "nominal": a, b = map(int, label.split("->")); g[a] -= tau; g[b] += tau
    return dict(nd, g=g.tolist())


def one(des, rep, S, K, R):
    k = (cid_of(des), rep, S, K, R)
    if k not in _cache:
        set_target(S, K, R); r = cert.verify_design_cfg(cid_of(des), des, rep, sd_pp=SD, n_boot=100000, units=("study",))
        _cache[k] = dict(cid=k[0], rep=rep, S=S, K=K, Rstar=R, R=r["R_LCB95_joint"], wilson=r["mc_wilson_lower_95"], boot_study=r["calib_bootstrap_R_LCB95"], support=r["support_fraction"], frozen_gate=bool(r["frozen_gate"]["all_pass"]), N=r["n_scenarios"], seconds=r["elapsed_s"])
    return _cache[k]


def strict(nd, shift_labels, S, K, R, out):
    """certificate of one design at cell (S, K) and level R: nominal and the given worst shift(s), 2 replicates each; stops at the first failing run. Returns (pass, runs). `out` collects every run."""
    runs = {}
    for tag, des in [("nominal", nd)] + [(f"shift_{l}", worst_design(nd, l)) for l in shift_labels]:
        for rep in (0, 1):
            r = one(des, rep, S, K, R); runs[f"{tag}_rep{rep}"] = r; out.append(r)
            if not (r["R"] >= R and r["support"] >= 1 - 1e-12 and r["frozen_gate"]): return False, runs
    return True, runs


def cell_cert(cands, S, K, R, log):
    """cands: [(LB, design_id, LB11 list, Z, C, nominal dict)] best first. Tries up to 3 designs; the first design that passes all four runs makes the cell strictly supported."""
    out = []; tries = []
    for lb, d, lb11, nd in cands[:3]:
        lb11 = np.array(lb11, float); sh = lb11[1:]; allsh = [f"{s}->{t}" for jj in range(5) for s, t in ((jj, jj + 1), (jj + 1, jj))]; label = None
        if np.isfinite(sh).any(): label = allsh[int(np.nanargmin(sh))]; labels = [label]                                       # shift rows were evaluated: the worst one
        else: labels = [l for l in allsh if nd["g"][int(l.split("->")[0])] - TAU >= 0]                                          # cut at the nominal blend: shifts unknown -> all valid shifts
        ok, runs = strict(nd, labels, S, K, R, out); tries.append(dict(design=int(d), LB_stageB=float(np.nanmin(lb11)), worst_shift=label, shifts_certified=labels, passed=bool(ok), runs={k: dict(R=v["R"], N=v["N"], support=v["support"]) for k, v in runs.items()}))
        if ok: return True, tries, out
    return False, tries, out


def edge_walk(task):
    """task = dict(b, Kval, R, nS, S (list), supB (list bool), cands {a: [...]}) -> dict. Downward from the highest stage-B-supported cell until a cell passes strictly, then upward until the first cell that fails strictly."""
    t0 = time.time(); b, K, R, S = task["b"], task["Kval"], task["R"], task["S"]; supB = task["supB"]; cands = {int(a): v for a, v in task["cands"].items()}; log = []; allruns = []; nS = len(S)
    def cert_cell(a):
        c = cands.get(a, [])
        if not c: return None, []
        ok, tries, out = cell_cert(c, S[a], K, R, log); allruns.extend(out); return ok, tries
    ix = [a for a in range(nS) if supB[a]]; a0 = max(ix) if ix else -1; tested = {}; confirmed = -1; a = a0
    while a >= 0:                                                                  # downward
        if not supB[a]: a -= 1; continue
        ok, tr = cert_cell(a); tested[a] = dict(stageB_supported=True, strict_pass=ok, tries=tr)
        if ok: confirmed = a; break
        a -= 1
    confirmed_down = confirmed; promoted = []; a = confirmed + 1; first_unsup = None
    while a < nS:                                                                  # upward from the confirmed cell
        ok, tr = cert_cell(a)
        tested[a] = dict(stageB_supported=bool(supB[a]), strict_pass=ok, tries=tr)
        if ok: promoted.append(a); confirmed = a; a += 1
        else: first_unsup = a; break
    return dict(b=b, K=K, R=R, stageB_highest=a0, confirmed_down=confirmed_down, confirmed_top=confirmed, promoted=promoted, first_unsupported=first_unsup, tested={str(k): v for k, v in tested.items()}, runs=allruns, seconds=round(time.time() - t0, 1))
