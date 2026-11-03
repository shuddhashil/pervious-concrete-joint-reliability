"""Step 3 (v2): refine the 0.90 edge. Per round: for every K* column the highest supported cell and the first two cells above it get a short search (refine_lib.GS2) that starts from the best pool design of the cell and the best design of its best supported neighbour;
all exactly scored candidates are added to the pool (screen, stage A, 11-blend rules), the classification at all 920 targets is redone (v2_core.classify_pool), repeat until newly supported <= 3 or four rounds or the time cap. Round 1 = L-SHADE + polish; from round 2 on,
a sample of cells is run in both variants and the polish-only variant is used for the rest only if it gives the same decisions on the sample. Writes refinement_log.json, logs_v2/refine_records.jsonl, data_v2/stageA_v2.npz, stageB_v2_cache.pkl, state_final.pkl. GPU, one process.
usage: python run_refine.py [minutes budget, default 75]"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_core import *
from refine_lib import GS2

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 75.0; T0 = time.time(); DEADLINE = T0 + BUDGET * 60
log = lambda *a: (print(*a, flush=True), open(os.path.join(L2, "refine.log"), "a").write(" ".join(str(x) for x in a) + "\n"))
cfg, H, ctx = get_ctx(); tg = Gpu(log); grid = Grid(S_GRID, K_GRID); gs = GS2(ctx); gs.hook = tg.cheap; P = load_pool(); floors = paste_floors(ctx, grid); sb = StageB(ctx, grid, tg=tg); load_cache(sb)
sf = os.path.join(D2, "state_final.pkl"); st = pickle.load(open(sf if os.path.exists(sf) else os.path.join(D2, "state_step2.pkl"), "rb")); best, bd = st["best"], st["bd"]; n_v1 = st["n_v1"]
REC = os.path.join(L2, "refine_records.jsonl"); LOGJ = os.path.join(SM, "refinement_log.json")
rlog = json.load(open(LOGJ)) if os.path.exists(LOGJ) else dict(rule="per round: cells = highest supported cell + the first two cells above it in every K* column (0.90, acceptance rule 0.902); short search per cell; designs added to the pool; reclassify all 920 targets", rounds=[], equivalence_check=None)
done_rounds = len(rlog["rounds"]); mode_later = (rlog["equivalence_check"] or {}).get("mode_used_for_rest"); last_fp = {int(k): tuple(v) for k, v in st.get("last_fp", {}).items()}; log("start", time.strftime("%H:%M:%S"), "rounds already done", done_rounds)


def pick_cells(best, bd, rank, cands):
    L = np.nan_to_num(best, nan=-1.0).reshape(grid.nS, grid.nK) >= THR90; top = top_cells(best, grid); cells = []
    for b in range(grid.nK):
        a0 = int(top[b])
        for a in ([a0, a0 + 1, a0 + 2] if a0 >= 0 else [0, 1]):
            if 0 <= a < grid.nS: cells.append((a, b, a0))
    out = []
    for a, b, a0 in cells:
        t = grid.tid(a, b); d1 = int(bd[t]) if bd[t] >= 0 else (int(rank[0, t]) if rank[0, t] >= 0 else -1)
        nbs = [(a + da, b + db) for da in (-1, 0, 1) for db in (-1, 0, 1) if (da or db) and 0 <= a + da < grid.nS and 0 <= b + db < grid.nK and L[a + da, b + db]]
        d2 = -1
        if nbs:
            a2, b2 = max(nbs, key=lambda x: (np.nan_to_num(best[grid.tid(*x)], nan=-1.0), -x[0])); d2 = int(bd[grid.tid(a2, b2)])
            if d2 == d1:
                alt = [d for lb, d in cands[grid.tid(a2, b2)] if d != d1]; d2 = int(alt[0]) if alt else -1
        out.append(dict(a=a, b=b, t=t, a0=a0, d1=d1, d2=d2))
    return out


def run_cell(cell, rnd, mode):
    a, b, t, d1, d2 = cell["a"], cell["b"], cell["t"], cell["d1"], cell["d2"]
    if d1 < 0 and d2 < 0: return None
    dl = d1 if d1 >= 0 else d2; fam = int(P["fam"][dl]); c = float(P["C"][dl]); sts = [P["Z"][dl]]; note = ""
    if d1 >= 0 and d2 >= 0:
        if int(P["fam"][d2]) == fam: sts.append(P["Z"][d2])
        else: note = "neighbour design in another family: not used as a start"
    tg(); r = gs.search(float(S_GRID[a]), float(K_GRID[b]), fam, c, 1000 * rnd + t, np.array(sts), mode); r.update(round=rnd, tid=int(t), a=int(a), b=int(b), d1=d1, d2=d2, start_note=note, starts=[list(map(float, z)) for z in sts])
    open(REC, "a").write(json.dumps(r) + "\n"); time.sleep(3); return r


while done_rounds < 4:
    rnd = done_rounds + 1; t_round = time.time()
    if time.time() > DEADLINE: log("TIME CAP reached before round", rnd); rlog["stopped"] = "time cap"; break
    cands0 = best_from_cache(sb, grid.nT)[2]
    elig = eligible(P); wilA = wilson(torch.as_tensor(P["counts"], dtype=torch.float64, device=DEV), 4096.0); rank = rankings(wilA, elig, P["hmin11"], floors); del wilA; torch.cuda.empty_cache()
    cells = pick_cells(best, bd, rank, cands0); f_before = np.nan_to_num(best, nan=-1.0) >= THR90; n_before = int(f_before.sum()); edge_before = edge_summary(best, grid)
    todo = []; skipped = 0
    for c_ in cells:
        fp = (c_["d1"], c_["d2"])
        if rnd >= 2 and last_fp.get(c_["t"]) == fp: skipped += 1; continue
        todo.append(c_)
    log(f"round {rnd}: {len(cells)} cells, {skipped} skipped (same start designs as the previous search of that cell), {len(todo)} to search, supported before {n_before}")
    results = []; truncated = False; eq = None
    mode = "lshade" if rnd == 1 else (mode_later or "lshade")
    if rnd >= 2 and mode_later is None and len(todo) >= 4:                      # equivalence check on a sample: L-SHADE+polish vs polish-only, same starts, same seeds
        samp = todo[::max(1, len(todo) // 8)][:8]; rows = []
        for c_ in samp:
            r1 = run_cell(c_, rnd, "lshade"); r2 = run_cell(c_, rnd, "polish"); last_fp[c_["t"]] = (c_["d1"], c_["d2"])
            if r1 and r2: rows.append(dict(S=r1["S"], K=round(r1["K"], 3), lshade=dict(passed=r1["passed"], LB=round(r1["LB_min_all"], 4), s=r1["runtime_s"]), polish=dict(passed=r2["passed"], LB=round(r2["LB_min_all"], 4), s=r2["runtime_s"]))); results += [r1, r2]
        same = all(x["lshade"]["passed"] == x["polish"]["passed"] for x in rows); eq = dict(round=rnd, sample=rows, same_decisions=bool(same), n_sample=len(rows), mode_used_for_rest="polish" if same else "lshade"); rlog["equivalence_check"] = eq; mode_later = eq["mode_used_for_rest"]; mode = mode_later; log("equivalence", json.dumps(eq))
        todo = [c_ for c_ in todo if c_ not in samp]
    for i, c_ in enumerate(todo):
        if time.time() > DEADLINE: truncated = True; log("TIME CAP inside round", rnd, "after", i, "cells"); break
        r = run_cell(c_, rnd, mode); last_fp[c_["t"]] = (c_["d1"], c_["d2"])
        if r: results.append(r); log(f"  [{rnd}:{i + 1}/{len(todo)}] S*={r['S']:g} K*={r['K']:.3f} F{r['fam']} C={r['C']:g} {r['mode']} pass={r['passed']} LBmin={r['LB_min_all']:.4f} acc={r['n_accepted_candidates']}/{len(r['cand_ok'])} {r['runtime_s']:.0f}s T={r['gpu_temp']}")
    for c_ in cells: last_fp.setdefault(c_["t"], (c_["d1"], c_["d2"]))
    t_search = time.time() - t_round
    Zn = np.array([z for r in results for z in r["cand_Z"]]).reshape(-1, 11); Cn = np.array([r["C"] for r in results for z in r["cand_Z"]]); Fn = np.array([r["fam"] for r in results for z in r["cand_Z"]], int)
    t1 = time.time(); n_add, cnt, ids, keep = append_designs(ctx, grid, P, Zn, Cn, Fn, "refined", rnd, tg=tg, log=log); t_app = time.time() - t1; save_pool(P)
    t1 = time.time(); R = classify_pool(ctx, grid, P, sb, floors, tg, log=log, tag=f" r{rnd}"); t_cl = time.time() - t1; best, bd = R["best"], R["bd"]; f_after = np.nan_to_num(best, nan=-1.0) >= THR90; n_after = int(f_after.sum())
    newly = int((f_after & ~f_before).sum()); lost = int((f_before & ~f_after).sum()); save_cache(sb)
    ent = dict(round=rnd, mode=mode, equivalence_sample_run=bool(eq), cells_in_rule=len(cells), cells_skipped_same_starts=skipped, cells_searched=len(results), truncated_by_time=truncated, searches_passed_own_acceptance=int(sum(r["passed"] for r in results)), designs_returned=int(len(Zn)), designs_added_after_screen=int(n_add), screen=cnt,
               supported_before=n_before, supported_after=n_after, newly_supported=newly, lost=lost, edge_before=edge_before, edge_after=edge_summary(best, grid), seconds=dict(search=round(t_search, 1), append_stageA=round(t_app, 1), classify=round(t_cl, 1), total=round(time.time() - t_round, 1)), gpu_peak_so_far=tg.peak)
    rlog["rounds"].append(ent); done_rounds += 1; json.dump(rlog, open(LOGJ, "w"), indent=1); pickle.dump(dict(best=best, bd=bd, floors=floors, n_v1=n_v1, last_fp=last_fp, mode_later=mode_later), open(os.path.join(D2, "state_final.pkl"), "wb"))
    log(f"ROUND {rnd} DONE: newly supported {newly} (lost {lost}), supported {n_before} -> {n_after}, {ent['seconds']}, GPU peak {tg.peak}")
    if truncated: rlog["stopped"] = "time cap inside round"; break
    if newly <= 3: rlog["stopped"] = f"converged: round {rnd} added {newly} (<= 3)"; break
else: rlog["stopped"] = "four rounds done; last round added %d" % newly
rlog.update(gpu_peak=tg.peak, pauses=tg.pauses, total_seconds=round(time.time() - T0, 1), budget_minutes=BUDGET); json.dump(rlog, open(LOGJ, "w"), indent=1); save_pool(P); save_cache(sb); log("REFINE DONE", rlog["stopped"], round(time.time() - T0), "s"); tg.stop()
