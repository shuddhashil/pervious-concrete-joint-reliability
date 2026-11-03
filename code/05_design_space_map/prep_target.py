"""Step 0 (v3): make the 18 MPa / 13 mm/s / R 0.90 context that every map script imports (v3, 2026-10-22).
Exactly gengine/run_target.py stage "0_config": target.make_target(18, 13, 0.90), then gengine/target_config.py S K R seeds in a subprocess with the same env that run_target.py builds (GENGINE_BASE, GENGINE_CONFIG,
PYTHONDONTWRITEBYTECODE, 3-thread limits), cwd ROOT. Writes targets/S18_K13_R0.90/ (CONFIG.json, paste_floor.json); skipped when CONFIG.json already exists."""
import os, sys, subprocess
SM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT = os.path.dirname(SM); sys.path.insert(0, ROOT)
from gengine import target as TG

S, K, R = 18.0, 13.0, 0.90; T = os.path.join(ROOT, "targets", TG.name_of(S, K, R)); CJ = os.path.join(T, "CONFIG.json")
L = os.path.join(SM, "logs_v3"); os.makedirs(L, exist_ok=True); LOGF = open(os.path.join(L, "prep_target.log"), "a")


def log(*a):
    s = " ".join(str(v) for v in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()


if os.path.exists(CJ): log("skip: CONFIG already exists", CJ)
else:
    info = TG.make_target(S, K, R); T = info["dir"]; log("per-target copy differs from base/ in:", info["differs_from_base"] or "nothing (byte-identical)")
    env = dict(os.environ, GENGINE_BASE=T, GENGINE_CONFIG=os.path.join(T, "CONFIG.json"), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", OPENBLAS_NUM_THREADS="3", NUMEXPR_NUM_THREADS="3")
    cmd = [sys.executable, os.path.join(ROOT, "gengine", "target_config.py"), "18.0", "13.0", "0.9", "0", "1", "2"]
    log("---", " ".join(cmd)); p = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in p.stdout: sys.stdout.write(line); sys.stdout.flush(); LOGF.write(line); LOGF.flush()
    rc = p.wait(); log("--- target_config rc", rc)
    if rc: sys.exit(rc)
log("context ready:", T)
