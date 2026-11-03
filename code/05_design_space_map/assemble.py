"""Assemble the final classification, tables and diagnostics. CPU only. Reads data/stageA.npz, data/stageB.pkl, data/edge_certs.json."""
import sys, os, json, pickle, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
SM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.join(SM, "data")
S_GRID = np.arange(5.0, 51.0, 1.0); K_GRID = np.geomspace(0.5, 30.0, 20); RS = (0.80, 0.90, 0.95); M = 0.002; nS, nK = len(S_GRID), len(K_GRID)
OLD = {"0.80": 543, "0.90": 493, "0.95": 437}
BND_PHI = (6.2, 38.02)


def ur_any(F):
    G = F.copy().astype(bool)
    for a in range(nS - 2, -1, -1): G[a] |= G[a + 1]
    for b in range(nK - 2, -1, -1): G[:, b] |= G[:, b + 1]
    return G


def final_flags(Bk, E):
    """stage-B flags per level, then the edge check of stage C: cells <= the strictly confirmed edge keep their stage-B flag, strictly promoted cells become supported, everything above the confirmed edge is unsupported."""
    L = np.nan_to_num(Bk["best_LB"], nan=-1.0).reshape(nS, nK); out = {}; info = {}
    res = {(round(r["R"], 2), r["b"]): r for r in E["results"]}
    for R in RS:
        sb = L >= R + M; fin = np.zeros((nS, nK), bool); dem = 0; prom = 0
        for b in range(nK):
            r = res[(round(R, 2), b)]; top = r["confirmed_top"]; down = r["confirmed_down"]
            for a in range(nS):
                if a <= down: fin[a, b] = sb[a, b]
                elif a in r["promoted"]: fin[a, b] = True; prom += 1
                elif sb[a, b]: dem += 1
        out[R] = fin; info[R] = dict(stageB=int(sb.sum()), final=int(fin.sum()), demoted_by_edge_check=int(dem), promoted_by_edge_check=int(prom))
    return L, out, info


def main():
    B = pickle.load(open(os.path.join(D, "stageB.pkl"), "rb")); E = json.load(open(os.path.join(D, "edge_certs.json"))); L, fin, info = final_flags(B, E); floors = B["floors"].reshape(nS, nK)
    sbf = {R: L >= R + M for R in RS}
    # ---- grid table
    rows = []
    for a in range(nS):
        for b in range(nK):
            t = a * nK + b; c = B["cands"][t]
            rows.append(dict(S_star=S_GRID[a], K_star=round(float(K_GRID[b]), 6), best_R_LCB95_stageB=round(float(L[a, b]), 5) if L[a, b] >= 0.80 + M else "<0.802 (cut early)", n_designs_evaluated=int(B["n_eval"][t]), best_pool_design=int(B["best_design"][t]),
                             paste_floor_mm=round(float(floors[a, b]), 4) if np.isfinite(floors[a, b]) else "none", **{f"supported_{R:.2f}": bool(fin[R][a, b]) for R in RS}, **{f"stageB_only_{R:.2f}": bool(sbf[R][a, b]) for R in RS}))
    pd.DataFrame(rows).to_csv(os.path.join(SM, "grid_supported.csv"), index=False)
    # ---- counts
    counts = dict(note="targets supported at each reliability level R* (920 targets: 46 strengths x 20 permeabilities). The old numbers come from the OLD engine (old 650-record model, porosity scatter 1.0, old gates, no batching-shift test, no paste floor); they are shown beside the new ones only for orientation.",
                  new_engine_final={f"{R:.2f}": int(fin[R].sum()) for R in RS}, new_engine_stageB_only={f"{R:.2f}": int(sbf[R].sum()) for R in RS}, old_engine_for_orientation=OLD, edge_check=info,
                  nested_levels=bool((fin[0.95] <= fin[0.90]).all() and (fin[0.90] <= fin[0.80]).all()))
    json.dump(counts, open(os.path.join(SM, "counts.json"), "w"), indent=1)
    # ---- edge table
    er = {(round(r["R"], 2), r["b"]): r for r in E["results"]}; rows = []
    for b in range(nK):
        row = dict(K_star=round(float(K_GRID[b]), 4))
        for R in RS:
            r = er[(R, b)]; ix = np.flatnonzero(fin[R][:, b]); top = int(ix.max()) if len(ix) else -1; strict_top = r["confirmed_top"]; row[f"highest_supported_S_{R:.2f}"] = float(S_GRID[top]) if top >= 0 else ""
            row[f"top_cell_strictly_certified_{R:.2f}"] = bool(top >= 0 and top == strict_top); row[f"stageB_highest_S_{R:.2f}"] = float(S_GRID[r["stageB_highest"]]) if r["stageB_highest"] >= 0 else ""
            fu = r["first_unsupported"]; row[f"first_unsupported_S_{R:.2f}"] = float(S_GRID[fu]) if fu is not None else ""
            row[f"first_unsupported_best_design_strict_pass_{R:.2f}"] = (r["tested"].get(str(fu), {}).get("strict_pass") if fu is not None else "")
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(SM, "edge_table.csv"), index=False)
    # ---- monotonicity (M3): a supported cell must imply every cell below-left of it is supported
    mono = dict(note="a design that supports (S*, K*) supports every lower S* and K*; violation = unsupported cell that has a supported cell at an equal-or-harder target", levels={})
    for tag, flags in (("final", fin), ("stageB_only", sbf)):
        for R in RS:
            F = flags[R]; closed = ur_any(F); viol = closed & ~F; vl = []
            for a, b in zip(*np.nonzero(viol)):
                up = [(a2, b2) for a2 in range(a, nS) for b2 in range(b, nK) if F[a2, b2]][:1]; a2, b2 = up[0]
                vl.append(dict(S=float(S_GRID[a]), K=round(float(K_GRID[b]), 4), floor_here=None if not np.isfinite(floors[a, b]) else round(float(floors[a, b]), 4), best_LB_here=round(float(L[a, b]), 4), supported_at=dict(S=float(S_GRID[a2]), K=round(float(K_GRID[b2]), 4), floor_there=None if not np.isfinite(floors[a2, b2]) else round(float(floors[a2, b2]), 4), best_LB_there=round(float(L[a2, b2]), 4))))
            mono["levels"][f"{tag}_{R:.2f}"] = dict(n_supported=int(F.sum()), n_violations=int(viol.sum()), n_closed_region=int(closed.sum()), violations=vl[:200])
    json.dump(mono, open(os.path.join(SM, "monotonic.json"), "w"), indent=1)
    # ---- pool coverage
    A = np.load(os.path.join(D, "stageA.npz")); ok = A["supf"] >= 1 - 1e-12; ok11 = ok & A["gate11"] & A["fam11"]; Z, Cv = A["Z"], A["C"]; phi = BND_PHI[0] + Z[:, 4] * (BND_PHI[1] - BND_PHI[0]); cs, km = A["csm"], A["km"]
    q = lambda x: {k: round(float(v), 3) for k, v in zip(("min", "p5", "p25", "median", "p75", "p95", "max"), np.percentile(x, [0, 5, 25, 50, 75, 95, 100]))}
    sub = lambda m: dict(n=int(m.sum()), cement_kg_m3=q(Cv[m]), porosity_pct=q(phi[m]), predicted_CS_median_MPa=q(cs[m]), predicted_K_median_mm_s=q(km[m]))
    cov = dict(note="pool = Sobol designs that pass the engine's gate, family rule and support check at the nominal blend; 'strict pool' also passes gates + family rule at all 11 blends", n_pool_nominal_pass=int(ok.sum()), n_pool_all_11_blend_gate_family=int(ok11.sum()), all=sub(ok), strict=sub(ok11),
               by_family={str(f): sub(ok & (A["fam"] == f)) for f in (0, 1, 2)}, by_stratum={n: sub(ok & (A["stratum"] == i)) for i, n in enumerate(("whole box", "high-strength corner", "high-permeability corner"))},
               counts_by_family_stratum={f"F{f}/{n}": int((ok & (A["fam"] == f) & (A["stratum"] == i)).sum()) for f in (0, 1, 2) for i, n in enumerate(("box", "strength", "perm"))},
               n_pred_CS_ge={"30": int((ok & (cs >= 30)).sum()), "40": int((ok & (cs >= 40)).sum()), "45": int((ok & (cs >= 45)).sum())}, n_pred_K_ge={"15": int((ok & (km >= 15)).sum()), "20": int((ok & (km >= 20)).sum()), "25": int((ok & (km >= 25)).sum())},
               n_cement_ge_500=int((ok & (Cv >= 500)).sum()), n_porosity_lt_22=int((ok & (phi < 22)).sum()), n_porosity_gt_28=int((ok & (phi > 28)).sum()))
    wil = A["counts"].astype(float) / 4096; cov["highest_S_with_a_pool_design_at_pass_fraction_0.90_per_K"] = {f"{K_GRID[b]:.3f}": (float(S_GRID[np.flatnonzero((wil[ok][:, np.arange(nS) * nK + b] >= 0.90).any(0))].max()) if (wil[ok][:, np.arange(nS) * nK + b] >= 0.90).any() else None) for b in range(nK)}
    json.dump(cov, open(os.path.join(SM, "pool_coverage.json"), "w"), indent=1)
    print(json.dumps(counts["new_engine_final"]), json.dumps(counts["new_engine_stageB_only"]), json.dumps(info)); print({k: v["n_violations"] for k, v in mono["levels"].items()}); print("pool", cov["n_pool_nominal_pass"], cov["n_pool_all_11_blend_gate_family"])


if __name__ == "__main__":
    main()
