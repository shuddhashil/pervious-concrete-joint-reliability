"""Command line: python -m defengine.run_engine <profile: dev|full> [--workers 4] [--out DIR]   (run from DefensibleEngine_20261001 with the engine python)"""
import os, sys, time, json, argparse
PROFILES = {
    "dev":  dict(range=(250, 600, 50), A1=dict(P=48, generations=20, n_scen=128), promising=0.30, A2=dict(P=60, generations=40, n_scen=256), B=dict(P=60, generations=30, n_scen=256), search_target=0.80, chain_start_min=0.60, chain_step=10, chain_stop=0.50),
    "explore": dict(range=(200, 700, 50), A1=dict(P=60, generations=40, n_scen=256), promising=0.35, A2=dict(P=80, generations=80, n_scen=256), B=dict(P=48, generations=25, n_scen=256), search_target=0.935, chain_start_min=0.80, chain_step=10, chain_stop=0.70),
    "gpu": dict(range=(250, 600, 25), A1=dict(P=100, generations=60, n_scen=256), promising=0.35, A2=dict(P=150, generations=150, n_scen=512), B=dict(P=100, generations=50, n_scen=512), search_target=0.935, chain_start_min=0.80, chain_step=10, chain_stop=0.70, n_chains=4),
    "gpu2": dict(range=(200, 500, 25), A1=dict(P=100, generations=60, n_scen=256), promising=0.35, A2=dict(P=150, generations=150, n_scen=512), B=dict(P=100, generations=50, n_scen=512), search_target=0.935, chain_start_min=0.88, chain_step=10, chain_stop=0.70, n_chains=3),
    "full": dict(range=(200, 700, 25), A1=dict(P=64, generations=30, n_scen=128), promising=0.40, A2=dict(P=100, generations=100, n_scen=256), B=dict(P=64, generations=30, n_scen=512), search_target=0.935, chain_start_min=0.80, chain_step=5, chain_stop=0.70),
}
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("profile", choices=list(PROFILES)); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--out", default=None)
    ap.add_argument("--no-verify", action="store_true"); ap.add_argument("--resume", default=None); ap.add_argument("--gpu", action="store_true"); a = ap.parse_args()
    if a.gpu: os.environ["DEFENGINE_GPU"] = "1"
    from defengine import ladder, corelib as C
    bad = C.verify_core(); assert not bad, f"frozen core changed: {bad[:3]}"
    out = a.out or os.path.join("runs", time.strftime("%Y%m%d_%H%M%S") + "_" + a.profile); os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "run.log"), "a", encoding="utf-8")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    log(f"profile {a.profile} | workers {a.workers} | model hash {C.model_hash()[:16]} | targets {C.S_STAR} MPa / {C.K_STAR} mm/s / {C.REL_TARGET}")
    summ = ladder.frontier_search(out, PROFILES[a.profile], workers=a.workers, log=log, resume=a.resume)
    if not a.no_verify:
        lv = [int(c) for c, i in summ["levels"].items() if i["best_rhat"] >= PROFILES[a.profile]["search_target"] - 0.15]
        if lv: ladder.verify_stage(out, summ, (min(lv), max(lv)) if len(lv) < 8 else (min(lv), min(lv) + 20), workers=a.workers, log=log)
    log("DONE")
