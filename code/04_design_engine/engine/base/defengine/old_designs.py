"""The 15 designs logged by the OLD production search, re-scored in A3.
Old-model numbers come from the old run's saved verification traces (read-only); A3 numbers are computed here (point predictions, gates, and
independent verification with the frozen estimator). python -m defengine.old_designs"""
import os, sys, json, glob, math
import numpy as np, pandas as pd
OLD = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\Final Inverse Mix Design\01_optimiser_engine_copy_2\runs\Stage1_Automated_20260922_212440\run_01\traces"
if __name__ == "__main__":
    from defengine import corelib as C, gates as G, ladder
    rows = []
    for f in sorted(glob.glob(os.path.join(OLD, "decisive_verification_2*.json"))):
        d = json.load(open(f)); cand = d["cand"]; lvl = int(round(cand["C"]))
        if not (252 <= lvl <= 284): continue
        rows.append(dict(level=lvl, cand=cand, old_rlcb=d["R_LCB95_joint"], old_rhat=d["Rhat_joint"], old_cs_med=C.S_STAR + d["strength_margin_p50"], old_k_med=C.K_STAR * math.exp(d["logk_margin_p50"]),
                         old_pass=d.get("PASS_FAIL")))
    rows.sort(key=lambda r: r["level"]); print(len(rows), "old designs found", flush=True)
    noms = [dict(C=r["cand"]["C"], r=r["cand"]["r"], A=r["cand"]["A"], P=r["cand"]["P"], S=r["cand"]["S"], phi=r["cand"]["phi"], g=np.array(r["cand"]["g"], float)) for r in rows]
    recs = C.E.evaluate_batch(noms, C.S_STAR, C.K_STAR, backend="fast")
    for r, nom, rec in zip(rows, noms, recs):
        fg = G.frozen_gate(nom)
        r.update(a3_cs=rec["mu_S"], a3_k=rec["k_pred"], a3_cs_sd=rec["s_S_eff"], a3_lnk_sd=rec["s_Q_eff"], a3_support=rec["support"], gate=fg["all_pass"], recipe=fg["recipe_pass"], hdr=fg["porosity_hdr"], w1=fg["w1_gate"])
    tasks = [dict(cid=f"OLD_C{r['level']}", nominal={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in n.items()}, replicate=0, cement=r["level"]) for r, n in zip(rows, noms)]
    ver = {v["cement"]: v for v in ladder.run_tasks(ladder._verify_task, tasks, 4)}
    out = []
    for r in rows:
        v = ver[r["level"]]
        out.append(dict(cement=r["level"], old_CS_median=round(r["old_cs_med"], 1), old_k_median=round(r["old_k_med"], 1), old_Rhat=round(r["old_rhat"], 3), old_R_LCB95=round(r["old_rlcb"], 4), old_result=r["old_pass"],
                        A3_CS=round(r["a3_cs"], 1), A3_CS_sd=round(r["a3_cs_sd"], 1), A3_k=round(r["a3_k"], 1), A3_Rhat=round(v["Rhat_joint"], 3), A3_R_LCB95=round(v["R_LCB95_joint"], 4),
                        A3_strength_fail=round(v["probability_strength_fail_only"], 3), A3_perm_fail=round(v["probability_perm_fail_only"], 3), A3_both_fail=round(v["probability_both_fail"], 3), A3_pass=v["PASS"], frozen_gates=r["gate"]))
    df = pd.DataFrame(out); df.to_csv("runs/old_designs_in_A3.csv", index=False); json.dump([{k: v for k, v in r.items()} for r in rows], open("runs/old_designs_detail.json", "w"), indent=1, default=str)
    pd.set_option("display.width", 250); print(df.to_string(index=False))
