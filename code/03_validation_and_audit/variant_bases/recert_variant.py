"""Re-certify a reported design under a variant engine base (sandbox, 2026-10-21). The frozen certificate (gengine/cert.py) is used unchanged;
only the folder it reads (core/ + defengine/) differs between variants. One target per process (the core reads its target at import).
Worst shift = the 2 % adjacent-bin shift with the lowest certificate (rep 0, SD 1.5), as in analysis/volume_closure_2026-10-10/recert_closed.py.
usage: python recert_variant.py <variant_base_dir> <S> <K> <design_json> <out_json> [sd ...]"""
import os, sys, json, shutil, hashlib
vb, S, K, dj, out_json = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
SDS = tuple(float(x) for x in sys.argv[6:]) or (1.5,)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
GE = os.path.join(REPO, "code", "04_design_engine", "engine")
work = os.path.join(os.path.dirname(os.path.abspath(out_json)), "_work", os.path.basename(out_json)[:-5])
if not os.path.exists(os.path.join(work, "core")):
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(os.path.join(vb, "core"), os.path.join(work, "core"), ignore=ign); shutil.copytree(os.path.join(vb, "defengine"), os.path.join(work, "defengine"), ignore=ign)
    tc = os.path.join(work, "core", "TARGET_CONTRACT.json"); c = json.load(open(tc, encoding="utf-8"))
    c.update(strength_target_mpa=S, permeability_target_mm_s=K, required_joint_reliability=0.9); open(tc, "w", encoding="utf-8").write(json.dumps(c, indent=2) + "\n")
os.environ.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="3", MKL_NUM_THREADS="3"); sys.dont_write_bytecode = True
sys.path.insert(0, work); sys.path.insert(1, GE)
import numpy as np
from defengine import corelib as C
from gengine import cert
assert (C.S_STAR, C.K_STAR) == (S, K), (C.S_STAR, C.K_STAR)
cert.RG_CSV = os.path.join(GE, "stages", "00_base", "recipe_groups_631.csv")
TAU = 0.02


def cid_of(d): return "G3_" + hashlib.sha1(json.dumps({k: (round(float(v), 10) if k != "g" else [round(float(x), 10) for x in v]) for k, v in sorted(d.items())}).encode()).hexdigest()[:14]


nom = json.load(open(dj)); nom = {k: (list(map(float, v)) if k == "g" else float(v)) for k, v in nom.items()}


def shifts(d):
    g = np.array(d["g"], float); out = {}
    for i in range(5):
        for a, b in ((i, i + 1), (i + 1, i)):
            if g[a] >= TAU - 1e-12:
                gg = g.copy(); gg[a] -= TAU; gg[b] += TAU; out[f"{a}->{b}"] = dict(d, g=gg.tolist())
    return out


def certify(d, rep, sd):
    r = cert.verify_design_cfg(cid_of(d), d, rep, sd_pp=sd, n_boot=100000, units=("study",))
    return dict(R=r["R_LCB95_joint"], Rhat=r["Rhat_joint"], wilson=r["mc_wilson_lower_95"], boot=r["calib_bootstrap_R_LCB95"], support=r["support_fraction"],
                gate=bool(r["frozen_gate"]["all_pass"]), PASS=bool(r["PASS"]), N=r["n_scenarios"], n_studies=r["n_studies_cluster_bootstrap"])


sh = shifts(nom); scan = {lab: certify(d, 0, 1.5) for lab, d in sh.items()}; worst = min(scan, key=lambda k: scan[k]["R"])
certs = {}
for sd in SDS:
    for tag, d in (("nominal", nom), ("worst", sh[worst])):
        for rep in (0, 1):
            certs[f"{sd}_{tag}_{rep}"] = scan[worst] if (sd == 1.5 and tag == "worst" and rep == 0) else certify(d, rep, sd)
res = dict(variant_base=vb, S=S, K=K, nominal=nom, oracle_sha256=C.model_hash(), bank_sha256=C.sha256_file(C.BANK_CSV), worst_label=worst,
           shift_scan_sd1_5_rep0={k: round(v["R"], 4) for k, v in scan.items()}, certs=certs,
           table={str(sd): [min(certs[f"{sd}_nominal_{r}"]["R"] for r in (0, 1)), min(certs[f"{sd}_worst_{r}"]["R"] for r in (0, 1))] for sd in SDS},
           PASS_sd1_5=all(certs[f"1.5_{t}_{r}"]["PASS"] for t in ("nominal", "worst") for r in (0, 1)) if 1.5 in SDS else None)
json.dump(res, open(out_json, "w"), indent=1)
print(f"S{S:g}/K{K:g}: R_LCB95 nominal/worst =", {k: [round(x, 3) for x in v] for k, v in res["table"].items()}, "PASS", res["PASS_sd1_5"])
