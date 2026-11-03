"""Step 1d (v3): which 0.90-supported cells are DATA-BACKED (2026-10-22, rewritten by the main thread: the first version checked only the reliability shortlist of each cell, whose designs sit away from the data).
A cell is data-backed iff some pool design is BOTH certified there (stage-B min-over-11-blends R_LCB95 >= THR90, full support) AND meets the Tier-1 evidence rule of gengine/data.py Data.evidence for that cell:
paste floor of the cell exists; near recorded units (d_grad <= Rg and d_rec <= Rr, nominal blend) include units passing THIS cell (CS >= S*, k >= K*, porosity in the window) from >= MIN_STUDIES studies;
absolute volume within VOL_TOL of 1 m3; no conflict flag. The evidence part is data-only, so it is computed for every pool design and every cell at once; stage B then runs (v1 `classify`, shared cache) only on
the designs that pass it, ranked per cell by the stage-A Wilson bound (same TOPN/TOPK/SCREEN as the main classification). Writes data_v2/islands.pkl, logs_v3/islands.json; updates stageB_v2_cache.pkl. GPU."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_core import *
from gengine import data as GD

T0 = time.time(); L3 = os.path.join(SM, "logs_v3"); os.makedirs(L3, exist_ok=True); LOGF = open(os.path.join(L3, "islands.log"), "a")
log = lambda *a: (print(*a, flush=True), LOGF.write(" ".join(str(x) for x in a) + "\n"), LOGF.flush())
cfg, H, ctx = get_ctx(); tg = Gpu(log); grid = Grid(S_GRID, K_GRID); P = load_pool(); D, nT = len(P["C"]), grid.nT; dat = ctx.data
sf = os.path.join(D2, "state_final.pkl"); st = pickle.load(open(sf if os.path.exists(sf) else os.path.join(D2, "state_step2.pkl"), "rb")); floors = st["floors"]; F = np.nan_to_num(st["best"], nan=-1.0) >= THR90
sb = StageB(ctx, grid, tg=tg); load_cache(sb)

# ---- data-only evidence, all designs x all cells
studies = sorted(set().union(*dat.unit_studies)); M = torch.zeros(dat.nU, len(studies), device=DEV)
for u, ss in enumerate(dat.unit_studies):
    for s in ss: M[u, studies.index(s)] = 1.0
UCS, UK = T(dat.U.CS.to_numpy()), T(dat.U.K.to_numpy()); win = torch.as_tensor(dat.phi_in_window, device=DEV)
Pc = ((UCS[:, None] >= grid.Sg.repeat_interleave(grid.nK)[None, :]) & (UK[:, None] >= torch.exp(grid.Qg).repeat(grid.nS)[None, :] - 1e-9) & win[:, None]).float()   # (nU, nT) passers per cell
ev = torch.zeros(D, nT, dtype=torch.bool, device=DEV); vol_ok = np.zeros(D, bool); conf = np.zeros(D, bool); n_near = np.zeros(D, int)
with torch.no_grad():
    for a in range(0, D, 2000):
        sl = slice(a, a + 2000); nom = decode(T(P["Z"][sl]), T(P["C"][sl])); rec = torch.stack([nom[k] for k in ("C", "r", "A", "P", "S", "phi")], 1); dg, dr = dat.dists(rec, nom["g"])
        near = ((dg <= dat.Rg) & (dr <= dat.Rr)).float(); n_near[sl] = near.sum(1).long().cpu().numpy()
        V = GD.volume(*(nom[k] for k in ("C", "r", "A", "S", "P", "phi"))); vol_ok[sl] = ((V - 1.0).abs() <= GD.VOL_TOL).cpu().numpy()
        for j, row in enumerate(near.bool().cpu().numpy()): conf[a + j] = bool(dat.conflict(np.flatnonzero(row).tolist())["flag"]) if row.any() else False
        for c0 in range(0, nT, 64):                                                        # studies covered by near passers of each cell: (near * pass_cell) @ incidence > 0
            cc = slice(c0, c0 + 64); Wc = (Pc[:, cc, None] * M[:, None, :]).reshape(dat.nU, -1); cov = (near @ Wc).view(near.shape[0], -1, M.shape[1]) > 0; ev[sl, cc] = cov.sum(2) >= GD.MIN_STUDIES
ok_d = torch.as_tensor(vol_ok & ~conf, device=DEV)[:, None]; fl_ok = torch.as_tensor(~np.isnan(floors), device=DEV)[None, :]; ev &= ok_d & fl_ok
log(f"evidence: {int(ev.any(1).sum())} of {D} designs data-backed at some cell; cells with any such design {int(ev.any(0).sum())}; volume ok {int(vol_ok.sum())}, conflict {int(conf.sum())}")

# ---- certification of the evidence-passing designs only (same ranking rule as classify.rankings, restricted to ev)
elig = torch.as_tensor(eligible(P), device=DEV)[:, None]; wilA = wilson(torch.as_tensor(P["counts"], dtype=torch.float64, device=DEV), 4096.0)
hm = torch.as_tensor(P["hmin11"], device=DEV)[:, None]; flt = torch.as_tensor(floors, device=DEV)[None, :]
el = elig & ev & (torch.isnan(flt) | (hm >= flt - 1e-12)) & (wilA >= SCREEN) & torch.as_tensor(F, device=DEV)[None, :]
idx = torch.arange(D, device=DEV, dtype=torch.float64)[:, None]; v, i = torch.where(el, wilA - 1e-12 * idx / D, torch.full_like(wilA, -1.0)).topk(min(TOPN, D), dim=0)
rank = torch.where(v > 0, i, torch.full_like(i, -1)).cpu().numpy(); del wilA, el; torch.cuda.empty_cache()
R = classify(sb, rank, design_nom_fn(P["Z"], P["C"]), log=log, tag=" islands"); save_cache(sb)
bdb = np.nan_to_num(R["best_LB"], nan=-1.0); data_backed = F & (bdb >= THR90); des = np.where(data_backed, R["best_design"], -1)
evn = ev.cpu().numpy(); stu = {}
for t in np.flatnonzero(data_backed).tolist():
    d = int(des[t]); nom = decode(T(P["Z"][d][None, :]), T([P["C"][d]])); rec = torch.stack([nom[k] for k in ("C", "r", "A", "P", "S", "phi")], 1); dg, dr = dat.dists(rec, nom["g"])
    near = np.flatnonzero(((dg <= dat.Rg) & (dr <= dat.Rr))[0].cpu().numpy()); npd = near[Pc[near, t].cpu().numpy() > 0]; stu[t] = dict(studies=sorted(set().union(*[dat.unit_studies[u] for u in npd])), n_passers=int(len(npd)), LB=float(bdb[t]), source=str(P["source"][d]))
pickle.dump(dict(supported=F, data_backed=data_backed, design=des, detail=stu, best_LB_data_backed=bdb), open(os.path.join(D2, "islands.pkl"), "wb"))
js = dict(n_supported=int(F.sum()), n_data_backed=int(data_backed.sum()), n_cells_with_evidence_design=int((evn.any(0) & F).sum()),
          cells=[[float(grid.S[t // grid.nK]), round(float(grid.K[t % grid.nK]), 4)] for t in np.flatnonzero(data_backed).tolist()],
          note="data-backed = a pool design passes both the Tier-1 evidence rule of that cell and the 0.90 acceptance there; stage B evaluated only on evidence-passing designs (top designs by stage-A bound per cell): a lower bound over the pool",
          gpu_peak=tg.peak, seconds=round(time.time() - T0, 1))
json.dump(js, open(os.path.join(L3, "islands.json"), "w"), indent=1); log(json.dumps({k: v for k, v in js.items() if k != "cells"}, indent=1)); tg.stop()
