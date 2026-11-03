"""Family-restricted GLOBAL search (L-SHADE + gradient polish) on the Guided stack. Free variables: w/c, CA, SP, sand, porosity (inside the 20-30 % window) and the six-bin blend (inside the family);
cement is fixed per run. Objective = the walk's own acceptance, evaluated by the walk's own functions:
  * exact objective  = walk.Ctx.accept (N = 4096, stricter tie rule, nominal + all valid 2 % single adjacent-bin shifts, gates, family rule, support, paste floor, LB_min >= R + m_walk). Used to rank and decide pass/fail.
  * search proxy     = the walk's own polish scoring (Ctx.exact_feasible + eng.stats_t worst-over-11 point reliability on a small bank, feasibility from the same gate / family / support / paste checks) with graded penalties so the
                       evolution can climb out of infeasible regions; the gradient polish uses the walk's DF.smooth_reliability. Blends are projected into the family (Fam.project) after every move.
usage (inside a per-target process, env GENGINE_BASE / GENGINE_CONFIG as set by run_target.py --mode global_walk): python gengine/global_search.py [hours]
Output: <target>/stage_global_walk/global/ (runs.jsonl, scan_summary.json, global_scan.csv, global_scan_by_seed.csv, GLOBAL_SCAN.md, gpu_temp_global.json)."""
import sys, os, json, time, math
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
import numpy as np, torch
from gengine import config, walk, optim as O, diff as DF, paste as PS, checks
from gengine.common import C, T, DEV, BND, decode, nbhd, gpu_temp, cool_down, SHIFT_LABELS

PHI_WIN = (float(C.S17.DEPLOY_LO), float(C.S17.DEPLOY_HI))
DEFAULT = dict(P=60, max_gens=80, patience=40, t_lshade=1.0e9, n_polish=8, t_polish=1.0e9, steps_max=30, n_exact=6, n_lo=64, n_hi=256, cap_s=30.0, cool_s=0.0, chunk=100, lr=0.02, n_pool=12, frac_pool=0.15, frac_noisy=0.15)
GRID_COARSE = list(range(420, 259, -20)); GRID_EXT = [440, 460, 480, 500]


def tier_of(accept, paste_floor_ok=True):
    """Tier rule (author-locked): Tier 1 / data-backed iff at least one recorded passing mix is near the design (n_min = 1; walk evidence = data-only near passers at the frozen lambda, window-clamped never counted),
    no conflict flag, and the paste floor exists. Everything else is Tier 2 model-suggested (never recommended). Works on an accept-metrics dict (walk log line['accept'])."""
    ev = accept["evidence"]; n = int(ev["n_near_pass"]); t1 = n >= 1 and not ev["conflict"]["flag"] and paste_floor_ok
    return dict(tier="Tier 1" if t1 else "Tier 2", n_near_pass=n, n_near_pass_n2=n >= 2, n_near_pass_n3=n >= 3, conflict=bool(ev["conflict"]["flag"]))


def tier1_candidates(accepted_lines, no_floor):
    """Accepted walk steps that may be certified as Tier 1 / recommended: >= 1 near recorded passing mix, no conflict flag, and the target has a paste floor. Sorted by cement. Anything else is Tier 2 model-suggested."""
    return [] if no_floor else sorted([l for l in accepted_lines if checks.data_ok(l, 1)], key=lambda l: (l["C"], l["pid"]))


def recommend(entries):
    """Recommended design = cheapest CERTIFIED Tier-1 design (entry['chosen']; Tier-2 designs live under 'chosen_t2' and are never considered). None if there is none."""
    c = [(e["chosen"]["l"]["C"], k) for k, e in entries.items() if "chosen" in e]; return min(c) if c else None


class GlobalSearch:
    def __init__(self, ctx, **kw):
        self.ctx = ctx; self.p = dict(DEFAULT, **kw); self.eng = ctx.eng; self.tau = ctx.tau
        z = lambda v: (v - BND["phi"][0]) / (BND["phi"][1] - BND["phi"][0]); self.zphi = (float(z(PHI_WIN[0])), float(z(PHI_WIN[1])))
        lab = np.array(ctx.cfg["families"]["labels"]); self.lou = lab[ctx.data.blend_of]; self.blab = lab; self.r_fam = ctx.fam.r

    # ------------------------------------------------------------------ box / family repair
    def repair(self, Z, fam):
        Z = Z.clone(); Z[:, 4] = torch.clamp(Z[:, 4], *self.zphi); g = decode(Z, torch.zeros(Z.shape[0], dtype=torch.float64, device=DEV))["g"]
        Z[:, 5:] = self.ctx.fam.project(g, fam); return Z

    # ------------------------------------------------------------------ objectives
    def fast_J(self, Z, c, fam, bk):
        """Proxy of the acceptance objective on a small bank: worst-over-neighbourhood point reliability, minus 1 + graded violation when a gate / family / support / paste check fails anywhere in the neighbourhood."""
        ctx = self.ctx; out = []; self.eng.set_lowprec(True)
        with torch.no_grad():
            for a in range(0, Z.shape[0], self.p["chunk"]):
                Zc = Z[a:a + self.p["chunk"]]; n = Zc.shape[0]; nom = decode(Zc, torch.full((n,), float(c), dtype=torch.float64, device=DEV)); nc, owner, kk, pen, fok, fd = ctx.exact_feasible(nom, fam)
                st = self.eng.stats_t(nc, bk[0], None, None, u=bk[1]); sup = st["support_fraction"]; h = PS.h_mm(nc["C"], nc["r"], nc["A"], nc["g"]); hm = ctx.h_min
                hv = torch.zeros_like(h) if hm is None else torch.clamp((hm - h) / hm, min=0.0)
                ok = (pen <= 0) & fok & (sup >= 1 - 1e-12) & (hv <= 1e-12)
                good = ctx.worst_over(ok.double(), owner, n, "amin", 1.0) > 0.5; rh = ctx.worst_over(st["rhat"], owner, n, "amin", 2.0)
                viol = ctx.worst_over(torch.clamp(pen, max=5.0) + torch.clamp(fd - self.r_fam, min=0.0) * 10 + (1 - sup) + hv * 5, owner, n, "amax", 0.0)
                out.append(torch.where(good, rh, rh - 1.0 - viol))
        return torch.cat(out)

    def smooth(self, Z, c, bk, temp):
        Cv = torch.full((Z.shape[0],), float(c), dtype=torch.float64, device=DEV); return DF.smooth_reliability(self.eng, Z, Cv, self.tau, bk, temp=temp, chunk_designs=Z.shape[0])

    def exact(self, Z, c, fam, rnd):
        """THE global objective = walk acceptance. Z (n,11) tensor or array, c cement. Returns list of dict(accepted, reasons, LB_min_all, LB_nominal, LB_worst, worst_label, metrics) from walk.Ctx.accept."""
        Zn = np.asarray(Z.cpu() if torch.is_tensor(Z) else Z, float); nom = decode(T(Zn), T(np.full(len(Zn), float(c)))); recs = []
        for i in range(len(Zn)): recs.append([float(c), float(nom["r"][i]), float(nom["A"][i]), float(nom["P"][i]), float(nom["S"][i]), float(nom["phi"][i])])
        todo = [dict(rec=r) for r in recs]; props = [dict(Z=Zn[i].tolist(), C=float(c)) for i in range(len(Zn))]; res = self.ctx.accept(todo, props, fam, rnd); out = []
        for r in res:
            m = r["metrics"]; out.append(dict(accepted=bool(r["accepted"]), reasons=list(r["reject_reasons"]), LB_min_all=m["LB_min_all"], LB_nominal=m["nominal"]["LB"], LB_worst=m["worst"]["LB"], worst_label=m["worst"]["label"], metrics=m))
        return out

    # ------------------------------------------------------------------ population
    def init_pop(self, run, fam, c, P):
        """V2-style start: (a) the n_pool best recorded recipes of the family re-cemented at this level (all family units scored once on the large bank), (b) noisy copies of the best 4, (c) the rest random:
        Latin hypercube over w/c, CA, SP, sand, porosity (window) + blends that are random mixtures of two recorded blends of the family; everything repaired into the family / porosity window."""
        d = self.ctx.data; p = self.p; ids = np.flatnonzero(self.lou == fam); gen = np.random.default_rng(run.seed); Zu = T(np.array([walk.recorded_design(d, int(u))[1] for u in ids]))
        Zu = self.repair(Zu, fam); Ju = run.J(Zu, run.hi); top = Zu[Ju.argsort(descending=True)[:p["n_pool"]]]; parts = [top]
        n_n = int(p["frac_noisy"] * P); parts.append(torch.clamp(top[torch.randint(0, min(len(top), 4), (n_n,), generator=run.gen, device=DEV)] + 0.02 * run.nrm(n_n, 11), 0, 1))
        n_l = P - sum(len(x) for x in parts); lhs = (torch.stack([torch.as_tensor(gen.permutation(n_l), device=DEV) for _ in range(5)], 1).to(torch.float64) + run.rnd(n_l, 5)) / n_l
        bl = np.flatnonzero(self.blab == fam); bi = gen.integers(0, len(bl), (n_l, 2)); w = run.rnd(n_l, 1); Bl = T(d.Bl); g = w * Bl[bl[bi[:, 0]]] + (1 - w) * Bl[bl[bi[:, 1]]]
        parts.append(torch.cat([lhs, g], 1)); return self.repair(torch.cat(parts, 0), fam)

    # ------------------------------------------------------------------ one (family, cement, seed) run
    def run_level(self, fam, c, seed, t_lshade=None, t_polish=None, log=print):
        p = self.p; t0 = time.time(); ctx = self.ctx; run_seed = int(seed) * 100003 + int(round(c * 4)) * 7 + fam * 1000003
        bank_lo = ctx.bank(p["n_lo"], run_seed + 1); bank_hi = ctx.bank(p["n_hi"], run_seed + 2)
        run = O.Run(lambda Z, bk: self.fast_J(Z, c, fam, bk), lambda Z, bk, temp: self.smooth(Z, c, bk, temp), lambda Z: self.repair(Z, fam), run_seed, DEV, bank_lo, bank_hi)
        pop = self.init_pop(run, fam, c, p["P"]); pop, J = O.lshade(run, pop, run.elapsed() + (t_lshade or p["t_lshade"]), patience=p["patience"], max_gens=p["max_gens"], time_driven=False); t_ls = run.elapsed()
        starts = O.distinct_top(pop, J, p["n_polish"]); self.eng.set_lowprec(True); Zb, Jb = O.grad_polish(run, starts, run.elapsed() + (t_polish or p["t_polish"]), lr=p["lr"], chunk=4, steps_max=p["steps_max"]); t_gp = run.elapsed()
        cand = torch.cat([O.distinct_top(pop, J, p["n_exact"]), Zb], 0); Jc = run.J(cand, run.hi); keep = [int(i) for i in Jc.argsort(descending=True).cpu().numpy()]; Zn = cand.cpu().numpy(); sel = []
        for i in keep:
            if all(np.abs(Zn[i] - Zn[j]).max() > 0.003 for j in sel): sel.append(i)
            if len(sel) >= p["n_exact"]: break
        rnd = f"glob{c:g}s{seed}"; ex = self.exact(cand[sel], c, fam, rnd); t_ex = run.elapsed()
        order = sorted(range(len(sel)), key=lambda k: (not ex[k]["accepted"], len(ex[k]["reasons"]), -ex[k]["LB_min_all"])); b = order[0]; e = ex[b]; m = e["metrics"]
        reason_counts = {}
        for x in ex:
            for r in x["reasons"]: reason_counts[r] = reason_counts.get(r, 0) + 1
        rec = dict(family=fam, C=float(c), seed=int(seed), runtime_s=time.time() - t0, t_lshade_s=t_ls, t_polish_s=t_gp - t_ls, t_exact_s=t_ex - t_gp, evals=int(run.evals), n_exact=len(sel), exact_bank_tag=rnd,
                   passed=bool(e["accepted"]), reject_reasons=e["reasons"], reject_reason_counts_over_candidates=reason_counts, LB_nominal=e["LB_nominal"], LB_worst=e["LB_worst"], LB_min_all=e["LB_min_all"], worst_label=e["worst_label"],
                   Z=cand[sel[b]].cpu().numpy().tolist(), nominal_design=m["nominal_design"], h_min_over_11=m["paste"]["h_min_over_11"], accept=m, proxy_best_J_hi=float(Jc[sel[b]]),
                   candidates=[dict(accepted=x["accepted"], reasons=x["reasons"], LB_min_all=x["LB_min_all"]) for x in ex], gpu_temp=gpu_temp())
        rec.update(tier_of(m, self.ctx.h_min is not None)); return rec


# ---------------------------------------------------------------------------------------------- scan driver
def scan(ctx, out_dir, seeds, log=print, deadline=None, **kw):
    """Cement scan per family: coarse grid 420 -> 260 step 20 (descending; a family stops after two consecutive levels where all seeds fail: nesting rule, lower cement is never easier), extension 440..500 only for a family where
    nothing passed up to 420, then refinement in 5 kg/m3 steps between the lowest passing level and the one below it. Resumable (runs.jsonl). One GPU process, cool-down >= cool_s between runs, heat rule 78 -> 70 C."""
    gs = GlobalSearch(ctx, **kw); os.makedirs(out_dir, exist_ok=True); RF = os.path.join(out_dir, "runs.jsonl"); done = {}; temps = dict(peak=gpu_temp(), pauses=[])
    if os.path.exists(RF):
        for l in open(RF):
            if l.strip(): r = json.loads(l); done[(r["family"], r["C"], r["seed"])] = r
    def run_one(fam, c, s):
        if (fam, float(c), s) in done: return done[(fam, float(c), s)]
        if deadline and time.time() > deadline: raise TimeoutError("global search deadline")
        paused = cool_down(78, 70, log=log)
        if paused: temps["pauses"].append(dict(family=fam, C=c, seconds=paused))
        r = gs.run_level(fam, float(c), s, log=log); done[(fam, float(c), s)] = r; open(RF, "a").write(json.dumps(r) + "\n"); temps["peak"] = max(temps["peak"], r["gpu_temp"])
        log(f"[global F{fam} C={c:g} s{s}] pass={r['passed']} LBmin={r['LB_min_all']:.4f} nom={r['LB_nominal']:.4f} worst={r['LB_worst']:.4f} reasons={r['reject_reasons']} {r['tier']} nnear={r['n_near_pass']} {r['runtime_s']:.1f}s temp {r['gpu_temp']}")
        time.sleep(gs.p["cool_s"]); return r
    def level(fam, c):
        rs = [run_one(fam, c, s) for s in seeds]; return any(r["passed"] for r in rs)
    summary = {}
    for fam in (2, 1, 0):
        status = {}; fails = 0
        for c in GRID_COARSE:
            if fails >= 2: status[c] = "skipped (two lower-or-equal-cost levels above failed for all seeds; nesting rule)"; continue
            ok = level(fam, c); status[c] = "pass" if ok else "fail"; fails = 0 if ok else fails + 1
        passed = sorted(c for c, v in status.items() if v == "pass")
        if not passed:
            for c in GRID_EXT:
                ok = level(fam, c); status[c] = "pass" if ok else "fail"
                if ok: passed = [c]; break
        refine = []
        if passed:
            lo = min(passed)
            for c in (lo - 5, lo - 10, lo - 15):
                if c < 230: continue
                ok = level(fam, c); status[c] = "pass" if ok else "fail"; refine.append(c)
                if not ok: break                         # lower than a fully failing refinement level: stop (nesting)
        summary[fam] = dict(status={str(k): v for k, v in sorted(status.items(), reverse=True)}, lowest_pass=min([c for c, v in status.items() if v == "pass"], default=None))
        json.dump(summary, open(os.path.join(out_dir, "scan_summary.json"), "w"), indent=1)
    json.dump(temps, open(os.path.join(out_dir, "gpu_temp_global.json"), "w")); return done, summary, temps


def stage_dirs(cfg):
    """(baseline stage folder, new stage_global_walk folder) of the active per-target config; the baseline folder is only read."""
    base = config.out_dir(cfg); return base, os.path.join(os.path.dirname(base), "stage_global_walk")


def write_tables(gl, done, summary):
    """Scan tables: per (family, level, seed) and per (family, level) best-over-seeds (the reliability-vs-cement curve), CSV + a short markdown."""
    import csv
    rows = sorted(done.values(), key=lambda r: (r["family"], -r["C"], r["seed"])); hdr = ["family", "cement", "seed", "passed", "LB_min_all", "LB_nominal", "LB_worst", "worst_shift", "reject_reasons", "tier", "n_near_pass", "h_min_over_11", "runtime_s"]
    with open(os.path.join(gl, "global_scan_by_seed.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr)
        for r in rows: w.writerow([r["family"], r["C"], r["seed"], r["passed"], round(r["LB_min_all"], 5), round(r["LB_nominal"], 5), round(r["LB_worst"], 5), r["worst_label"], "+".join(r["reject_reasons"]) or "-", r["tier"], r["n_near_pass"], round(r["h_min_over_11"], 4), round(r["runtime_s"], 1)])
    md = ["| family | cement | seeds run | seeds passing | best min-LB (nominal / worst shift) | LB_min per seed | best design tier | reasons of the best |", "|---|---|---|---|---|---|---|---|"]
    with open(os.path.join(gl, "global_scan.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["family", "cement", "n_seeds", "n_pass", "best_LB_min_all", "best_LB_nominal", "best_LB_worst", "LB_min_per_seed", "best_tier", "best_reasons", "scan_status"])
        for fam in (0, 1, 2):
            for c in sorted({r["C"] for r in rows if r["family"] == fam}, reverse=True):
                rs = [r for r in rows if r["family"] == fam and r["C"] == c]; b = max(rs, key=lambda r: (r["passed"], r["LB_min_all"])); st = summary.get(fam, {}).get("status", {}).get(f"{c:g}", summary.get(fam, {}).get("status", {}).get(str(int(c)) if c == int(c) else str(c), ""))
                per = "/".join(f"{r['LB_min_all']:.4f}" for r in rs); w.writerow([fam, c, len(rs), sum(r["passed"] for r in rs), round(b["LB_min_all"], 5), round(b["LB_nominal"], 5), round(b["LB_worst"], 5), per, b["tier"], "+".join(b["reject_reasons"]) or "-", st])
                md.append(f"| F{fam} | {c:g} | {len(rs)} | {sum(r['passed'] for r in rs)} | {b['LB_min_all']:.4f} ({b['LB_nominal']:.4f} / {b['LB_worst']:.4f}) | {per} | {b['tier']} | {'+'.join(b['reject_reasons']) or '-'} |")
    skipped = [f"F{f}: " + ", ".join(k for k, v in s["status"].items() if v.startswith("skipped")) for f, s in summary.items() if any(v.startswith("skipped") for v in s["status"].values())]
    open(os.path.join(gl, "GLOBAL_SCAN.md"), "w", encoding="utf-8").write("# Global scan (best design per family and cement level; LB = R_LCB95, N=4096, stricter tie rule, min over nominal + 10 shifts)" + chr(10) * 2 + chr(10).join(md) + (chr(10) * 2 + "Levels not run (nesting rule after two failing levels): " + "; ".join(skipped) if skipped else "") + chr(10))


def main():
    cfg, H = config.load(); BASE, OUT = stage_dirs(cfg); GL = os.path.join(OUT, "global"); os.makedirs(GL, exist_ok=True); LOGF = open(os.path.join(GL, "global_scan.log"), "a")
    def log(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + chr(10)); LOGF.flush()
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0; log("global scan start", time.strftime("%Y-%m-%d %H:%M:%S"), "target", cfg["target"], "config", H[:12], "GPU temp", gpu_temp(), "deadline h", hours)
    ctx = walk.Ctx(cfg, H); log("paste floor", ctx.h_min); done, summary, temps = {}, {}, {}
    try: done, summary, temps = scan(ctx, GL, [int(s) for s in cfg["walk"]["seeds"]], log=log, deadline=time.time() + hours * 3600)
    except TimeoutError as e:
        log("DEADLINE:", e, "(partial scan; rerun resumes)"); done = {}
        for l in open(os.path.join(GL, "runs.jsonl")):
            if l.strip(): r = json.loads(l); done[(r["family"], r["C"], r["seed"])] = r
        summary = json.load(open(os.path.join(GL, "scan_summary.json"))) if os.path.exists(os.path.join(GL, "scan_summary.json")) else {}
        summary = {int(k): v for k, v in summary.items()}
    write_tables(GL, done, summary); log("GLOBAL SCAN DONE", len(done), "runs; GPU peak", temps.get("peak"))


if __name__ == "__main__":
    main()
