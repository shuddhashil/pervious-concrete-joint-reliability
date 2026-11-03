"""Guided walk: parent selection, per-parent state machine, batched polish + acceptance, resumable JSONL logs.
Layout: pure state machine (new_state / propose_cement / update) + two evaluator calls (polish, accept) that run_family() drives; tests use a mock evaluator for the same loop.
Seeds: sha256(config hash | family | round | tag) mod 2^31. The banks of one round are shared by all parents of the family (common random numbers, lets the GPU batch them)."""
import os, json, math, time, hashlib
import numpy as np, torch
from gengine.common import (C, GP, GG, T, DEV, ROOT, nbhd, decode, encode, nom_dicts, Fam, SHIFT_LABELS, gpu_temp, cool_down, sha256_file, gpu_lb)
from gengine import diff as DF, paste as PS
from gengine.data import Data


def seed_for(cfgh, fam, rnd, tag, sid=0):
    """sid = seed index (cfg walk.seeds); sid 0 keeps the original string, so sid 0 reproduces the Phase-2 seeding rule."""
    return int(hashlib.sha256((f"{cfgh}|{fam}|{rnd}|{tag}" + (f"|s{sid}" if sid else "")).encode()).hexdigest(), 16) % (2 ** 31)


# ------------------------------------------------------------------ state machine (no GPU)
def new_state(pid, Z, C0, rec, cfg):
    return dict(pid=pid, phase="repair", Zc=list(map(float, Z)), Zcenter=list(map(float, Z)), C0=float(C0), rec=list(map(float, rec)), repair_j=0, Cacc=None, delta=float(cfg["walk"]["delta0"]),
                n_acc=0, n_rej1=0, steps=0, attempts=0, stop=None, merged_into=None, wall=0.0)


def propose_cement(st, cfg):
    w = cfg["walk"]
    return st["C0"] + w["repair_step"] * st["repair_j"] if st["phase"] == "repair" else max(st["Cacc"] - st["delta"], w["floor"])


def update(st, line, cfg):
    """Apply one logged attempt line to the state (identical code for live runs and for replay on resume)."""
    w = cfg["walk"]; st["attempts"] += 1
    if line["accepted"]:
        st["Zc"] = st["Zcenter"] = list(line["Z"]); st["n_rej1"] = 0
        if st["phase"] == "repair": st.update(phase="descend", Cacc=line["C"], delta=float(w["delta0"]), n_acc=0)
        else:
            st["Cacc"] = line["C"]; st["steps"] += 1; st["n_acc"] += 1
            if st["n_acc"] >= 3 and st["delta"] < w["delta0"]: st["delta"] = min(2 * st["delta"], float(w["delta0"])); st["n_acc"] = 0
        if st["Cacc"] <= w["floor"] + 1e-9 and st["stop"] is None: st["stop"] = "floor"
    elif st["phase"] == "repair":
        st["repair_j"] += 1; st["Zc"] = st["Zcenter"] = list(line["Z"])
        if st["repair_j"] > w["repair_max_steps"]: st["stop"] = "no_feasible_start"
    else:
        st["n_acc"] = 0
        if st["delta"] <= w["delta_min"] + 1e-9:
            st["n_rej1"] += 1
            if st["n_rej1"] >= 2: st["stop"] = line["reject_reason"]
        else: st["delta"] = max(st["delta"] / 2, float(w["delta_min"]))
    if st["stop"] is None and st["attempts"] >= w["max_attempts"]: st["stop"] = "max_attempts"
    return st


def run_family(ev, fam, parents, outdir, cfg, cfgh, stop_after=None, log=print, deadline=None):
    """parents: list of dict(pid, Z, C0, rec). ev: evaluator with polish(todo, fam, rnd) and accept(todo, props, fam, rnd). Resumes from existing logs. Returns the final states."""
    os.makedirs(outdir, exist_ok=True); w = cfg["walk"]; states = {}; paths = {}
    for p in parents:
        paths[p["pid"]] = os.path.join(outdir, f"{p['pid']}.jsonl"); st = new_state(p["pid"], p["Z"], p["C0"], p["rec"], cfg); states[p["pid"]] = st
        if os.path.exists(paths[p["pid"]]):
            for ln in open(paths[p["pid"]]):
                if not ln.strip(): continue
                L = json.loads(ln)
                if L["event"] == "attempt": update(st, L, cfg); st["wall"] += L.get("round_wall_s", 0.0)
                elif L["event"] == "stop": st["stop"] = L["reason"]
                elif L["event"] == "merged": st["stop"] = "merged"; st["merged_into"] = L["into"]

    def wr(pid, L):
        with open(paths[pid], "a") as f: f.write(json.dumps(L) + "\n")
    while True:
        active = [s for s in states.values() if s["stop"] is None]
        if not active: break
        rnd = min(s["attempts"] for s in active); todo = [s for s in active if s["attempts"] == rnd]
        if stop_after is not None and rnd >= stop_after: break
        if deadline is not None and time.time() > deadline: log(f"[F{fam}] DEADLINE reached before round {rnd}: stopping cleanly (resumable)"); break
        cool_down(log=log); t0 = time.time(); props = ev.polish(todo, fam, rnd); t1 = time.time(); res = ev.accept(todo, props, fam, rnd); t2 = time.time(); tmp = gpu_temp()
        for s, p, r in zip(todo, props, res):
            line = dict(event="attempt", family=fam, pid=s["pid"], round=rnd, phase=s["phase"], delta=s["delta"], C=p["C"], Z=p["Z"], accepted=r["accepted"], reject_reason=r["reject_reason"], reject_reasons=r.get("reject_reasons", [r["reject_reason"]] if r["reject_reason"] else []),
                        polish=p["info"], accept=r["metrics"], n_active=len(todo), polish_batch_s=t1 - t0, accept_batch_s=t2 - t1, round_wall_s=t2 - t0, gpu_temp=tmp)
            wr(s["pid"], line); update(s, line, cfg); s["wall"] += t2 - t0
        log(f"[F{fam}] round {rnd}: " + ", ".join(f"{s['pid']} C={s['Cacc']} d={s['delta']} {'STOP:' + s['stop'] if s['stop'] else ''}" for s in todo) + f"  polish {t1 - t0:.1f}s accept {t2 - t1:.1f}s temp {tmp}C")
        for s in todo:
            if s["stop"] is None and s["wall"] > w["parent_time_cap_s"]: s["stop"] = "time"
            if s["stop"] is not None: wr(s["pid"], dict(event="stop", reason=s["stop"], C=s["Cacc"], steps=s["steps"], attempts=s["attempts"]))
        live = [s for s in todo if s["stop"] is None and s["phase"] == "descend"]            # merge: two parents within the Z tolerance (plan 7.2.6), lower id kept
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if a["stop"] is None and b["stop"] is None and max(abs(x - y) for x, y in zip(a["Zc"], b["Zc"])) < w["merge_tol"]:
                    lo, hi = sorted((a, b), key=lambda s: s["pid"]); hi["stop"] = "merged"; hi["merged_into"] = lo["pid"]; wr(hi["pid"], dict(event="merged", into=lo["pid"], C=hi["Cacc"]))
    return states


# ------------------------------------------------------------------ the real evaluator
class Ctx:
    def __init__(self, cfg, cfgh, sid=0):
        self.cfg, self.cfgh, self.sid = cfg, cfgh, sid; C.engine(); self.eng = GP.GPUEngine(); self.gate = GG.GPUGate(); DF.prepare(self.eng)
        self.LB = gpu_lb.LB(self.eng, recipe_groups_csv=os.path.join(ROOT, "stages", "00_base", "recipe_groups_631.csv"), n_boot=cfg["bootstrap"]["n_boot"], boot_seed=cfg["bootstrap"]["seed"])
        bl = np.load(os.path.join(ROOT, "stages", "01_families", "blends83.npy")); self.fam = Fam(bl, cfg["families"]["labels"], cfg["families"]["r_fam"]); self.data = Data(cfg)
        assert np.allclose(bl, self.data.Bl, atol=1e-12)
        self.SD = cfg["porosity_sd_pp"]["main"]; self.tau = cfg["tau"]; self.R_ACC = cfg["target"]["R"] + cfg["walk"]["m_walk"]
        assert (cfg["target"]["S"], cfg["target"]["K"], cfg["target"]["R"]) == (C.S_STAR, C.K_STAR, C.REL_TARGET)
        self.pf = PS.paste_floor(self.data, cfg["target"]["S"], cfg["target"]["K"]); self.h_min = self.pf["h_min"]           # paste-film floor for THIS target (S*, K*); None = no recorded unit meets the target -> paste check skipped (every result "model-suggested, no paste floor")
        pg = cfg["paste_guard"]
        if (cfg["target"]["S"], cfg["target"]["K"]) == (pg["S_frozen"], pg["K_frozen"]):
            assert (self.h_min is None and pg["h_min_mm"] is None) or abs(self.h_min - pg["h_min_mm"]) < 1e-12, "paste floor differs from the frozen value"

    def bank(self, n, seed, sd=None):
        return C.mfg_bank(n, seed=seed, porosity_sd_pp=self.SD if sd is None else sd), T(np.random.default_rng(seed + 3).random(n))

    def exact_feasible(self, nom, fam):
        """gates (GPU search gate) + family rule at all valid neighbours."""
        nc, owner, kk = nbhd(nom, self.tau); pen = torch.nan_to_num(self.gate.penalties(nc)["penalty"], posinf=1e6); fok, fd, _ = self.fam.check(nc["g"], fam); return nc, owner, kk, pen, fok, fd

    @staticmethod
    def worst_over(v, owner, n, op, init):
        return torch.full((n,), init, dtype=v.dtype, device=DEV).scatter_reduce(0, owner, v, op, include_self=True)

    def score_polish(self, Z, Cv, fam, bk):
        nom = decode(Z, Cv); n = Z.shape[0]; nc, owner, kk, pen, fok, fd = self.exact_feasible(nom, fam); st = self.eng.stats_t(nc, bk[0], None, None, u=bk[1])
        ok = (pen <= 0) & fok & (st["support_fraction"] >= 1 - 1e-12); good = self.worst_over(ok.double(), owner, n, "amin", 1.0) > 0.5; rh = self.worst_over(st["rhat"], owner, n, "amin", 2.0)
        return good, rh

    def polish(self, todo, fam, rnd):
        cfg = self.cfg; w = cfg["walk"]; pc = w["polish"]; seed = seed_for(self.cfgh, fam, rnd, "polish", self.sid); bk = self.bank(w["N_polish"], seed)
        Z0 = T([s["Zc"] for s in todo]); cen = T([s["Zcenter"] for s in todo]); Cv = T([propose_cement(s, cfg) for s in todo]); n = Z0.shape[0]
        lo, hi = torch.clamp(cen - pc["box"], 0, 1), torch.clamp(cen + pc["box"], 0, 1); self.eng.set_lowprec(True)
        Zc = Z0.clone(); m = torch.zeros_like(Zc); v = torch.zeros_like(Zc); cd = 4
        for t in range(1, pc["iters"] + 1):
            Zr = Zc.clone().requires_grad_(True)
            for a in range(0, n, cd):
                R = DF.smooth_reliability(self.eng, Zr[a:a + cd], Cv[a:a + cd], self.tau, bk, temp=pc["temp"], chunk_designs=cd); (-R.sum()).backward()
            gr = torch.nan_to_num(Zr.grad); m = 0.9 * m + 0.1 * gr; v = 0.999 * v + 0.001 * gr * gr
            step = (m / (1 - 0.9 ** t)) / (torch.sqrt(v / (1 - 0.999 ** t)) + 1e-8); Zc = torch.minimum(torch.maximum(Zc - pc["lr"] * step, lo), hi)
        res = []
        with torch.no_grad():
            for tt in pc["cand_t"]:
                Zt = Z0 + tt * (Zc - Z0); g2 = self.fam.project(decode(Zt, Cv)["g"], fam); Zt = torch.cat([Zt[:, :5], g2], 1); good, rh = self.score_polish(Zt, Cv, fam, bk); res.append((tt, Zt, good, rh))
            props = []
            for i in range(n):
                pick = None
                for tt, Zt, good, rh in res:
                    sc = float(rh[i]) - (0.0 if bool(good[i]) else 10.0)
                    if pick is None or sc > pick[0]: pick = (sc, tt, Zt[i], bool(good[i]), float(rh[i]))
                props.append(dict(Z=pick[2].cpu().numpy().tolist(), C=float(Cv[i]), info=dict(t_chosen=pick[1], polish_bank_worst_rhat=pick[4], polish_feasible=pick[3], seed=seed, N=w["N_polish"],
                                                                                              all_t={str(tt): (float(rh[i]), bool(good[i])) for tt, _, good, rh in res}, box_displacement_max=float((Zc[i] - Z0[i]).abs().max()))))
        return props

    def accept(self, todo, props, fam, rnd):
        cfg = self.cfg; w = cfg["walk"]; seed = seed_for(self.cfgh, fam, rnd, "accept", self.sid); N = w["N_accept"]; bank, u = self.bank(N, seed); self.eng.set_lowprec(False)
        Z = T([p["Z"] for p in props]); Cv = T([p["C"] for p in props]); nom = decode(Z, Cv); nc, owner, kk, pen, fok, fd = self.exact_feasible(nom, fam)
        r1, r2 = gpu_lb.lcb95_tie_bracket(self.LB, nc, bank, u=u, units=("study", "recipe_group", "row"))
        mn = lambda k: np.minimum(r1[k], r2[k]); LBs, LBg, LBr, Wil, Boo, Rh = mn("LB"), mn("LB_recipe_group"), mn("LB_row"), mn("wilson"), mn("boot_study"), mn("rhat"); SF = r1["support_fraction"]
        own = owner.cpu().numpy(); kkn = kk.cpu().numpy(); penn = pen.cpu().numpy(); fokn = fok.cpu().numpy(); fdn = fd.cpu().numpy(); t0 = time.time(); ev = self.data.evidence(nom); t_ev = time.time() - t0
        nomd = nom_dicts(nom); out = []; hh = PS.h_mm(nc["C"], nc["r"], nc["A"], nc["g"]).cpu().numpy()
        for i, s in enumerate(todo):
            ix = np.flatnonzero(own == i); i0 = ix[kkn[ix] == 0][0]; sh = ix[kkn[ix] > 0]; iw = sh[np.argmin(LBs[sh])] if len(sh) else i0
            pick = lambda j: dict(LB=float(LBs[j]), LB_lowest_index_rule=float(r1["LB"][j]), LB_highest_index_rule=float(r2["LB"][j]), rhat=float(Rh[j]), wilson=float(Wil[j]), boot_study=float(Boo[j]),
                                  LB_recipe_group=float(LBg[j]), LB_row=float(LBr[j]), support_fraction=float(SF[j]), gate_penalty=float(penn[j]), family_ok=bool(fokn[j]), family_dist=float(fdn[j]), band=int(r1["band"][j]), phi_med=float(r1["phi_med"][j]))
            gate_all = bool((penn[ix] <= 0).all()); fam_all = bool(fokn[ix].all()); sup_all = bool((SF[ix] >= 1 - 1e-12).all()); lb_min = float(LBs[ix].min()); rel = lb_min >= self.R_ACC
            h_min_all = float(hh[ix].min()); paste_ok = True if self.h_min is None else h_min_all >= self.h_min - 1e-12; h_worst = SHIFT_LABELS[int(kkn[ix[int(np.argmin(hh[ix]))]])]
            reasons = [n for n, ok in (("gate", gate_all), ("family", fam_all), ("support", sup_all), ("paste_floor", paste_ok), ("reliability", rel)) if not ok]      # G-3: ALL failing checks, in the old test order
            reason = reasons[0] if reasons else None                                                                                                             # old single field = first failing check (unchanged)
            rec = np.array([nomd[i]["C"], nomd[i]["r"], nomd[i]["A"], nomd[i]["P"], nomd[i]["S"], nomd[i]["phi"]]); drift = float(np.sqrt((((rec - np.array(s["rec"])) / self.data.scale) ** 2).sum()))
            met = dict(nominal=pick(i0), worst=dict(label=SHIFT_LABELS[int(kkn[iw])], **pick(iw)), LB_min_all=lb_min, R_ACC=self.R_ACC, gates_all_pass=gate_all, paste=dict(h_nominal=float(hh[i0]), h_min_over_11=h_min_all, h_worst_label=h_worst, h_floor=self.h_min, ok=bool(paste_ok), floor_status="none (check skipped)" if self.h_min is None else "computed", reliability_ok=bool(rel)), family_all_ok=fam_all, support_all=sup_all, n_designs=int(len(ix)),
                       N=N, bank_seed=seed, nominal_design=nomd[i], evidence=ev[i], d_rec_drift_from_parent=drift, evidence_batch_s=t_ev, sd_pp=self.SD)
            out.append(dict(accepted=reason is None, reject_reason=reason, reject_reasons=reasons, metrics=met))
        return out


# ------------------------------------------------------------------ parents (plan 7.1)
def recorded_design(data, u):
    R = data.Ru[u]; rec = dict(C=float(R[0]), r=float(R[1]), A=float(R[2]), P=float(R[3]), S=float(R[4]), phi=float(R[5]))
    lo, hi = C.S17.DEPLOY_LO, C.S17.DEPLOY_HI; phi_clamped = not (lo <= rec["phi"] <= hi); rec["phi"] = min(max(rec["phi"], lo), hi)
    Z, clamped = encode(rec, data.Gu[u]); return rec, Z, clamped, phi_clamped


def select_parents(ctx, fam, labels_of_unit, K, margin=None):
    """G-3 parent rule (author-approved correction 2). Candidates = de-duplicated units whose recorded blend is in the family AND CS >= S* - margin_S AND k >= K* - margin_K (margins from cfg parents; default 3 MPa / 3 mm/s).
    Parents = all deployable recorded passers (porosity inside 20-30 %) + farthest-point sampling (A3 standardised inputs) ONLY inside the candidate set, up to the cap K. A family with fewer candidates than the cap uses just those
    (never filled with far-from-target mixes). A window-clamped recorded passer is a candidate (it meets the thresholds) but is not flagged recorded_passer (data.passU).
    Generic engine, ONLY for a family with ZERO candidates (author-approved): model-chosen starts, the G-2 style = farthest-point sampling inside the model's top-`zero_parent_pool_top` (24) pool of the family's units ranked by point reliability, first = best-scoring, up to the cap K.
    Every parent from that rule is flagged model_suggested / start_rule 'G2_model_chosen_zero_near_units' (Tier 2 only: never data-backed, never recommended); the family info carries `rule`."""
    d = ctx.data; cfg = ctx.cfg; mS, mK = (cfg["parents"]["margin_S"], cfg["parents"]["margin_K"]) if margin is None else margin; S0, K0 = cfg["target"]["S"], cfg["target"]["K"]
    fam_ids = np.flatnonzero(labels_of_unit == fam); ok = (np.asarray(d.U.CS)[fam_ids] >= S0 - mS) & (np.asarray(d.U.K)[fam_ids] >= K0 - mK); ids = fam_ids[ok]; eng = ctx.eng; eng.set_lowprec(False)
    zero = len(ids) == 0; n_cand = int(len(ids)); pool_top = int(cfg["parents"].get("zero_parent_pool_top", 24))
    if zero: ids = fam_ids                              # zero near-target units: score every unit of the family, then keep the model's top pool (below)
    Zs, recs, flags = [], [], []
    for u in ids:
        rec, Z, cl, pc = recorded_design(d, u); Zs.append(Z); recs.append(rec); flags.append((cl, pc))
    Z = T(np.array(Zs)); Cv = T([r["C"] for r in recs]); nom = decode(Z, Cv); bank, uu = ctx.bank(1024, 20261003)
    with torch.no_grad():
        st = eng.stats_t(nom, bank, None, None, u=uu); score = st["rhat"].cpu().numpy(); sf = st["support_fraction"].cpu().numpy()
        X6 = torch.stack([nom["phi"], nom["r"], nom["C"], nom["A"], nom["P"], nom["S"]], 1); F = eng.standardise(X6, eng.quantiles(nom["g"]))[0].cpu().numpy()
    key = [hashlib.sha256(f"{u}".encode()).hexdigest() for u in ids]; passer = d.passU[ids]
    order = sorted(range(len(ids)), key=lambda i: (-score[i], key[i])); chosen = [i for i in order if passer[i]][:K]
    if zero: order = order[:pool_top]; chosen = []      # zero case: no passers by construction (a passer meets CS >= S*, k >= K*), pool = top-24 by point reliability
    pool = [i for i in order if i not in chosen]
    if not chosen and pool: chosen = [pool.pop(0)]
    while len(chosen) < K and pool:
        dm = [min(np.linalg.norm(F[i] - F[j]) for j in chosen) for i in pool]; j = int(np.argmax(dm)); chosen.append(pool.pop(j))
    out = []
    for i in chosen:
        u = int(ids[i]); out.append(dict(pid=f"u{u}", unit=u, C0=recs[i]["C"], rec=[recs[i][k] for k in ("C", "r", "A", "P", "S", "phi")], Z=Zs[i].tolist(), recorded_passer=bool(passer[i]), point_rhat_at_recorded=float(score[i]),
                                         support_fraction=float(sf[i]), recorded_phi=float(d.Ru[u, 5]), clamped_coords=flags[i][0], phi_clamped_to_window=flags[i][1], phi_clamped=flags[i][1], recorded_passer_raw=bool(d.passU_raw[u]), blend=int(d.blend_of[u]), CS=float(d.U.CS[u]), K=float(d.U.K[u]),
                                         start_rule="G2_model_chosen_zero_near_units" if zero else "G3_near_target_units", model_suggested=bool(zero)))
    return out, dict(n_family_units=int(len(fam_ids)), n_candidates=n_cand, rule="G2_model_chosen_zero_near_units (top-%d pool by point reliability, farthest-point; every result model-suggested)" % pool_top if zero else "G3_near_target_units", zero_near_units=bool(zero), n_passers=int(passer.sum()), n_passers_raw_incl_window_clamped=int(d.passU_raw[fam_ids].sum()), n_parents=len(out), cap=K, margin_S=mS, margin_K=mK)
