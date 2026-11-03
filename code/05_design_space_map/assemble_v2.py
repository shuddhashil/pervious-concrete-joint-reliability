"""Assemble the v2 tables (CPU). Reads data_v2/state_step2.pkl, state_final.pkl, stageA_v2.npz, edge_certs_v2.json, data/stageB.pkl (v1), edge_table.csv (v1), logs_v2/step2.json, refinement_log.json.
Writes grid_supported_v2.csv, edge_table_v2.csv, counts_v2.json, monotonic_v2.json, before_after_v2.csv. Never touches a v1 file."""
import sys, os, json, pickle
import numpy as np, pandas as pd
SM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D1 = os.path.join(SM, "data"); D2 = os.path.join(SM, "data_v2")
S_GRID = np.arange(5.0, 51.0, 1.0); K_GRID = np.geomspace(0.5, 30.0, 20); nS, nK = len(S_GRID), len(K_GRID); THR = 0.902


def ur_any(F):
    G = F.copy().astype(bool)
    for a in range(nS - 2, -1, -1): G[a] |= G[a + 1]
    for b in range(nK - 2, -1, -1): G[:, b] |= G[:, b + 1]
    return G


def top_of(F): return np.array([int(np.flatnonzero(F[:, b]).max()) if F[:, b].any() else -1 for b in range(nK)])


def main():
    B1 = pickle.load(open(os.path.join(D1, "stageB.pkl"), "rb")); L1 = np.nan_to_num(B1["best_LB"], nan=-1.0).reshape(nS, nK); f1 = L1 >= THR
    s2 = pickle.load(open(os.path.join(D2, "state_step2.pkl"), "rb")); L2 = np.nan_to_num(s2["best"], nan=-1.0).reshape(nS, nK); f2 = L2 >= THR
    sf = pickle.load(open(os.path.join(D2, "state_final.pkl"), "rb")); L3 = np.nan_to_num(sf["best"], nan=-1.0).reshape(nS, nK); f3 = L3 >= THR; bd = sf["bd"].reshape(nS, nK); floors = sf["floors"].reshape(nS, nK)
    P = np.load(os.path.join(D2, "stageA_v2.npz")); src = P["source"]; E = json.load(open(os.path.join(D2, "edge_certs_v2.json"))); res = {r["b"]: r for r in E["results"]}
    fin = np.zeros((nS, nK), bool); dem = prom = 0
    for b in range(nK):
        r = res[b]; down = r["confirmed_down"]
        for a in range(nS):
            if a <= down: fin[a, b] = f3[a, b]
            elif a in r["promoted"]: fin[a, b] = True; prom += 1
            elif f3[a, b]: dem += 1
    rows = []
    for a in range(nS):
        for b in range(nK):
            d = int(bd[a, b]); rows.append(dict(S_star=S_GRID[a], K_star=round(float(K_GRID[b]), 6), best_R_LCB95_stageB=round(float(L3[a, b]), 5) if L3[a, b] >= 0 else "none", best_pool_design=d, best_design_source=str(src[d]) if d >= 0 else "none", paste_floor_mm=round(float(floors[a, b]), 4) if np.isfinite(floors[a, b]) else "none",
                             **{"supported_0.90": bool(fin[a, b]), "stageB_rule_0.902": bool(f3[a, b]), "v1_supported_0.90": bool(f1[a, b]), "after_step2_rule_0.902": bool(f2[a, b])}))
    pd.DataFrame(rows).to_csv(os.path.join(SM, "grid_supported_v2.csv"), index=False)
    t1, t2, t3, tf = top_of(f1), top_of(f2), top_of(f3), top_of(fin); v1e = pd.read_csv(os.path.join(SM, "edge_table.csv")); er = []
    S = lambda i: float(S_GRID[i]) if i >= 0 else None
    for b in range(nK):
        r = res[b]; fu = r["first_unsupported"]; v1top = v1e["highest_supported_S_0.90"].iloc[b]; v1top = None if pd.isna(v1top) else float(v1top)
        er.append(dict(K_star=round(float(K_GRID[b]), 4), v1_highest_S=v1top, after_step2_highest_S=S(t2[b]), after_step3_rule_highest_S=S(t3[b]), highest_certified_S=S(tf[b]), top_cell_strictly_certified=bool(tf[b] >= 0 and tf[b] == r["confirmed_top"]), first_unsupported_S=S(fu) if fu is not None else None,
                       first_unsupported_best_design_strict_pass=(r["tested"].get(str(fu), {}).get("strict_pass") if fu is not None else None), moved_by_step2_MPa=(S(t2[b]) or 0) - (v1top or 0) if (t2[b] >= 0 or v1top) else 0, moved_by_step3_MPa=(S(tf[b]) or 0) - (S(t2[b]) or 0), moved_total_MPa=(S(tf[b]) or 0) - (v1top or 0)))
    ed = pd.DataFrame(er); ed.to_csv(os.path.join(SM, "edge_table_v2.csv"), index=False)
    ba = ed[["K_star", "v1_highest_S", "after_step2_highest_S", "highest_certified_S", "moved_by_step3_MPa", "moved_total_MPa"]].rename(columns={"highest_certified_S": "v2_final_highest_S"}); ba.to_csv(os.path.join(SM, "before_after_v2.csv"), index=False)
    mv = ed["moved_by_step3_MPa"].to_numpy(float); mv_all = ed["moved_total_MPa"].to_numpy(float); rl = json.load(open(os.path.join(SM, "refinement_log.json"))); s2j = json.load(open(os.path.join(SM, "logs_v2", "step2.json")))
    q = lambda x: dict(min=float(np.min(x)), p25=float(np.percentile(x, 25)), median=float(np.median(x)), p75=float(np.percentile(x, 75)), max=float(np.max(x)), mean=round(float(np.mean(x)), 2), n_columns_moved=int((np.asarray(x) > 0).sum()), n_columns_unchanged=int((np.asarray(x) == 0).sum()), n_columns_lower=int((np.asarray(x) < 0).sum()))
    cj = dict(note="targets supported at R_LCB95 >= 0.90 (920 targets: 46 strengths x 20 permeabilities). Interior cells rest on the acceptance rule (min over 11 blends of R_LCB95 >= 0.902, GPU, N=4096); the edge cells (top supported cell and first unsupported cell of each K* column) were additionally certified strictly (frozen estimator, 2 replicates, nominal and worst shift, porosity SD 1.5).",
              v1_pool_only=int(f1.sum()), after_step2_tuned_designs_added=int(f2.sum()), after_step3_acceptance_rule=int(f3.sum()), after_step3_final_with_strict_edge_check=int(fin.sum()), strict_edge_check=dict(demoted=int(dem), promoted=int(prom)),
              gained_by_step2=int((f2 & ~f1).sum()), gained_by_step3_rule=int((f3 & ~f2).sum()), gained_total_final_vs_v1=int((fin & ~f1).sum()), lost_vs_v1_final=int((f1 & ~fin).sum()), per_round_newly_supported=[r["newly_supported"] for r in rl["rounds"]], per_round_supported_after=[r["supported_after"] for r in rl["rounds"]],
              edge_move_by_step3_MPa=q(mv), edge_move_total_vs_v1_MPa=q(mv_all), pool_size=dict(v1=int(s2["n_v1"]), tuned_added=int((src == "tuned").sum()), refined_added=int((src == "refined").sum()), total=int(len(src))),
              best_design_source_among_final_supported={k: int(((src[bd.ravel()] == k) & fin.ravel()).sum()) for k in ("sobol", "anchor", "tuned", "refined")})
    json.dump(cj, open(os.path.join(SM, "counts_v2.json"), "w"), indent=1)
    mono = dict(note="a supported target must imply every lower S* and K* is supported; violation = unsupported cell with a supported cell at an equal-or-harder target (listed, not closed)"); viol = []
    for tag, F in (("final", fin), ("acceptance_rule", f3)):
        cl = ur_any(F); v = cl & ~F; vl = []
        for a, b in zip(*np.nonzero(v)):
            a2, b2 = [(x, y) for x in range(a, nS) for y in range(b, nK) if F[x, y]][0]; vl.append(dict(S=float(S_GRID[a]), K=round(float(K_GRID[b]), 4), best_LB_here=round(float(L3[a, b]), 4), supported_at=dict(S=float(S_GRID[a2]), K=round(float(K_GRID[b2]), 4), best_LB_there=round(float(L3[a2, b2]), 4))))
        mono[tag] = dict(n_supported=int(F.sum()), n_violations=int(v.sum()), violations=vl[:200])
    json.dump(mono, open(os.path.join(SM, "monotonic_v2.json"), "w"), indent=1); print(json.dumps({k: v for k, v in cj.items() if k != "note"}, indent=1)); print({k: v["n_violations"] for k, v in mono.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
