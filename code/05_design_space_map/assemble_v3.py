"""Assemble the v3 tables (CPU). Reads data_v2/state_final.pkl (else state_step2.pkl), data_v2/stageA_v2.npz, data_v2/islands.pkl, data_v2/edge_certs_v2.json (if present),
logs_v2/monotone_fix.json (if present, referenced only), refinement_log.json (if present). Writes grid_supported_v3.csv, edge_table_v3.csv, counts_v3.json, six_targets_v3.json.
Supported = acceptance rule best >= 0.902; strict edge results come from run_certs_v2."""
import sys, os, json, pickle
import numpy as np, pandas as pd
SM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D2 = os.path.join(SM, "data_v2"); L2 = os.path.join(SM, "logs_v2")
S_GRID = np.arange(5.0, 51.0, 1.0); K_GRID = np.geomspace(0.5, 30.0, 20); nS, nK = len(S_GRID), len(K_GRID); THR = 0.902
TARGETS = [(10.0, 8.0), (14.0, 6.0), (16.0, 10.0), (20.0, 4.0), (18.0, 13.0), (25.0, 13.0)]
ENGINE = {(10.0, 8.0): "certified", (14.0, 6.0): "certified", (16.0, 10.0): "certified", (20.0, 4.0): "certified", (18.0, 13.0): "not certified", (25.0, 13.0): "not certified"}


def ur_any(F):
    """cells at an easier target than some supported cell (same rule as assemble_v2)."""
    G = F.copy().astype(bool)
    for a in range(nS - 2, -1, -1): G[a] |= G[a + 1]
    for b in range(nK - 2, -1, -1): G[:, b] |= G[:, b + 1]
    return G


def top_of(F): return np.array([int(np.flatnonzero(F[:, b]).max()) if F[:, b].any() else -1 for b in range(nK)])


def loadj(p):
    return json.load(open(p)) if os.path.exists(p) else None


def mono(F, best):
    """count and list the monotonicity violations of a support grid (assemble_v2 logic)."""
    cl = ur_any(F); v = cl & ~F; vl = []
    for a, b in zip(*np.nonzero(v)):
        a2, b2 = [(x, y) for x in range(a, nS) for y in range(b, nK) if F[x, y]][0]
        vl.append(dict(S=float(S_GRID[a]), K=round(float(K_GRID[b]), 4), best_LB_here=round(float(best[a, b]), 4),
                       supported_at=dict(S=float(S_GRID[a2]), K=round(float(K_GRID[b2]), 4), best_LB_there=round(float(best[a2, b2]), 4))))
    return dict(n_supported=int(F.sum()), n_violations=int(v.sum()), violations=vl[:200])


def main():
    sf = os.path.join(D2, "state_final.pkl")
    if not os.path.exists(sf): sf = os.path.join(D2, "state_step2.pkl")
    if not os.path.exists(sf): raise FileNotFoundError(f"{sf}: run classify0.py / run_refine.py first")
    st = pickle.load(open(sf, "rb")); L = np.nan_to_num(st["best"], nan=-1.0).reshape(nS, nK); bd = np.asarray(st["bd"], int).reshape(nS, nK); floors = np.asarray(st["floors"], float).reshape(nS, nK); F = L >= THR
    P = np.load(os.path.join(D2, "stageA_v2.npz")); src = P["source"]
    isl = pickle.load(open(os.path.join(D2, "islands.pkl"), "rb")) if os.path.exists(os.path.join(D2, "islands.pkl")) else None
    DB = np.asarray(isl["data_backed"], bool).reshape(nS, nK) if (isl is not None and "data_backed" in isl) else np.zeros((nS, nK), bool)
    E = loadj(os.path.join(D2, "edge_certs_v2.json")); res = {int(r["b"]): r for r in E["results"]} if E else {}

    sc = np.full((nS, nK), "", dtype=object)                                                      # strict certificate of the edge cells run_certs_v2 tested
    for b, r in res.items():
        for a_s, v in r.get("tested", {}).items():
            a = int(a_s); sp = v.get("strict_pass")
            if a < nS: sc[a, b] = "pass" if sp is True else ("fail" if sp is False else "")

    fin = F.copy(); dem = prom = 0
    if E:
        for b in range(nK):
            r = res.get(b)
            if not r: continue
            down = r["confirmed_down"]; pro = set(r.get("promoted", []))
            for a in range(nS):
                if a <= down: fin[a, b] = F[a, b]
                elif a in pro: fin[a, b] = True; prom += 1
                elif F[a, b]: fin[a, b] = False; dem += 1          # above the strictly confirmed edge: demoted (v2 started from zeros)

    rows = []
    for a in range(nS):
        for b in range(nK):
            d = int(bd[a, b]); ok = d >= 0
            rows.append({"S_star": float(S_GRID[a]), "K_star": round(float(K_GRID[b]), 6), "best_LB": round(float(L[a, b]), 5) if L[a, b] >= 0 else "none",
                             "supported_0.90": bool(F[a, b]), "supported_final": bool(fin[a, b]), "data_backed": bool(DB[a, b] and fin[a, b]), "best_design_id": d, "best_design_source": str(src[d]) if ok and d < len(src) else "none",
                             "paste_floor": round(float(floors[a, b]), 4) if np.isfinite(floors[a, b]) else "none", "strict_cert": str(sc[a, b])})
    pd.DataFrame(rows).to_csv(os.path.join(SM, "grid_supported_v3.csv"), index=False)

    tops = top_of(fin); er = []
    for b in range(nK):
        t = int(tops[b]); above = t + 1 if t >= 0 else -1
        er.append(dict(K_star=round(float(K_GRID[b]), 4), highest_supported_S=float(S_GRID[t]) if t >= 0 else None,
                       top_cell_strict_cert=str(sc[t, b]) if t >= 0 else "", cell_above_S=float(S_GRID[above]) if t >= 0 and above < nS else None,
                       cell_above_strict_cert=str(sc[above, b]) if t >= 0 and above < nS else ""))
    pd.DataFrame(er).to_csv(os.path.join(SM, "edge_table_v3.csv"), index=False)

    rl = loadj(os.path.join(SM, "refinement_log.json")); mf = loadj(os.path.join(L2, "monotone_fix.json"))
    sv, cn = np.unique(src, return_counts=True); mf_final = dict(supported=mf.get("supported_after"), violations=len(mf.get("violations_after", []))) if mf else None
    cj = dict(note="targets supported at R_LCB95 >= 0.90 (920 targets: 46 strengths x 20 permeabilities). Interior cells rest on the acceptance rule (min over 11 blends of R_LCB95 >= 0.902, GPU, N=4096); the edge cells (top supported cell and the first unsupported cell of each K* column) were additionally certified strictly (frozen estimator, 2 replicates, nominal and worst shift, porosity SD 1.5).",
              supported_cells=int(F.sum()), supported_final_cells=int(fin.sum()), data_backed_cells=int((DB & fin).sum()), n_targets=int(nS * nK),
              strict_edge_check=dict(demoted=int(dem), promoted=int(prom), columns_certified=len(res), final_with_strict_edge=int(fin.sum())),
              per_round_newly_supported=[r.get("newly_supported") for r in rl.get("rounds", [])] if rl else [], refinement_stopped=(rl or {}).get("stopped"),
              monotone_fix=mf_final, monotonic=dict(acceptance_rule=mono(F, L), final=mono(fin, L)),
              pool_size=dict(total=int(len(src)), by_source={str(k): int(v) for k, v in zip(sv, cn)}),
              best_design_source_among_supported={str(k): int(((src[bd] == k) & F).sum()) for k in sv})
    json.dump(cj, open(os.path.join(SM, "counts_v3.json"), "w"), indent=1)

    six = []
    for St, Kt in TARGETS:
        hard = [(a, b) for a in range(nS) for b in range(nK) if fin[a, b] and S_GRID[a] >= St - 1e-9 and K_GRID[b] >= Kt - 1e-9]
        inside = len(hard) > 0; ev = ENGINE[(St, Kt)]
        alo = int(np.clip(np.searchsorted(S_GRID, St, side="right") - 1, 0, nS - 1)); ahi = int(np.clip(np.searchsorted(S_GRID, St, side="left"), 0, nS - 1))
        blo = int(np.clip(np.searchsorted(K_GRID, Kt, side="right") - 1, 0, nK - 1)); bhi = int(np.clip(np.searchsorted(K_GRID, Kt, side="left"), 0, nK - 1))
        if ahi == alo: ahi = min(alo + 1, nS - 1)                                                       # target on a grid strength: take that cell and the next one up (still four distinct nearest cells)
        if bhi == blo: bhi = min(blo + 1, nK - 1)
        near = []
        for a, sb in ((alo, "S_lo"), (ahi, "S_hi")):
            for b, kb in ((blo, "K_lo"), (bhi, "K_hi")):
                near.append({"role": f"{sb}/{kb}", "S_star": float(S_GRID[a]), "K_star": round(float(K_GRID[b]), 4), "best_LB": round(float(L[a, b]), 5) if L[a, b] >= 0 else None, "supported_final": bool(fin[a, b])})
        six.append(dict(S_target=St, K_target=Kt, engine_verdict=ev, map_verdict="inside" if inside else "outside", flag="AGREES" if (inside == (ev == "certified")) else "DISAGREES",
                        n_supported_cells_at_or_above=len(hard), nearest_cells=near))
    json.dump(dict(engine_verdict_source="the six target-specific engine runs: certified for 10/8, 14/6, 16/10, 20/4; not certified for 18/13 and 25/13",
                   rule="map 'inside' = some supported grid cell has S >= S_target and K >= K_target (monotone region); 'outside' = none. Nearest cells = the four grid cells around the target (floor/ceil S x floor/ceil K); when the target lies on a grid strength the two S cells are that cell and the next one up.",
                   targets=six), open(os.path.join(SM, "six_targets_v3.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in cj.items() if k not in ("note", "monotonic")}, indent=1)); print({t["S_target"]: (t["map_verdict"], t["flag"]) for t in six})


if __name__ == "__main__":
    main()
