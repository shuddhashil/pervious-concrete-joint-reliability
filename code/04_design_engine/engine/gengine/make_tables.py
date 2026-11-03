"""Markdown tables from SELECTION.json + walk logs + parents.json of the active CONFIG's stage folder (no GPU, no CPU-heavy work), with a one-line verdict at the top. usage: python gengine/make_tables.py -> <stage>/TABLES.md"""
import os, sys, json, glob, math
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
import numpy as np
from gengine import config, checks
from gengine.common import W_LOG
cfg, H = config.load(); OUT = config.out_dir(cfg); sel = json.load(open(os.path.join(OUT, "SELECTION.json"))); par = json.load(open(os.path.join(OUT, "parents.json"))); E = sel["entries"]
PG = cfg["paste_guard"]; NO_FLOOR = PG["h_min_mm"] is None; TG = cfg["target"]; TNAME = f"S* {TG['S']:g} MPa, K* {TG['K']:g} mm/s, R* {TG['R']:.2f}"
FAMN = {f: f"F{f} ({cfg['families']['sizes'][f]} blends; {par[str(f)]['info']['n_passers']} recorded deployable passer(s), {par[str(f)]['info']['n_candidates']} near-target unit(s); parents: {'model-chosen starts' if par[str(f)]['info'].get('zero_near_units') else 'near-target recorded units'})" for f in range(3)}
TIER2 = "Tier 2: model-suggested, no paste floor (never recommended, never data-backed)" if NO_FLOOR else "Tier 2 (model-suggested, never recommended)"
w1 = lambda a, b: float((np.abs(np.cumsum(a)[:5] - np.cumsum(b)[:5]) * W_LOG).sum())
f1 = lambda x, n=1: "-" if x is None else f"{x:.{n}f}"; pct = lambda g: "/".join(f"{100 * x:.1f}" for x in g)
SDS = [cfg["porosity_sd_pp"]["main"]] + cfg["porosity_sd_pp"]["checks"]; out = []


def rmin(ch, sd, tag, k="R"):
    v = [ch["sd_checks"][f"sd{sd}_{tag}_rep{r}"] if sd != SDS[0] else ch["certificate"]["results"][f"{tag}_rep{r}"] for r in (0, 1)]
    v = [x[k] for x in v if isinstance(x, dict) and x.get(k) is not None]; return min(v) if v else None


def design_row(e):
    ch = e["chosen"]; l = ch["l"]; a = l["accept"]; nd = a["nominal_design"]; ev = a["evidence"]; p = a["paste"]
    return dict(C=l["C"], pid=l["pid"], blend=pct(nd["g"]), other=f"w/c {nd['r']:.4f}, CA {nd['A']:.1f}, SP {nd['P']:.2f}, sand {nd['S']:.1f}, phi {nd['phi']:.2f}", h=f"{p['h_nominal']:.3f} / min11 {p['h_min_over_11']:.3f} ({p['h_worst_label']}) vs floor " + ("none (no recorded unit meets the target; paste check skipped)" if p["h_floor"] is None else f"{p['h_floor']:.4f} (CS-only floor {cfg['paste_guard']['h_min_cs_only_mm']:.4f})"),
                near=f"{ev['n_near_pass']} (lam {cfg['radii']['lambda']}); by lam 1.5/2.0/2.5/3.0: " + "/".join(str(ev['by_lambda_n_near_pass'][k]) for k in ("1.5", "2.0", "2.5", "3.0")), conflict=str(ev["conflict"]["flag"]),
                R="; ".join(f"SD{sd}: {f1(rmin(ch, sd, 'nominal'), 4)} / {f1(rmin(ch, sd, 'worst'), 4)}" for sd in SDS),
                nolab=f"recipe-group {f1(rmin(ch, SDS[0], 'nominal', 'R_nolab_recipe_group'), 4)} / {f1(rmin(ch, SDS[0], 'worst', 'R_nolab_recipe_group'), 4)}; row {f1(rmin(ch, SDS[0], 'nominal', 'R_nolab_row'), 4)} / {f1(rmin(ch, SDS[0], 'worst', 'R_nolab_row'), 4)}",
                n2=ev["n_near_pass"] >= 2, n3=ev["n_near_pass"] >= 3, fb=ch["fallback_index"])


# ---------------- 0. per (family, seed) selection table
out.append("### T0 per (family, seed): log-derived minima (cement kg/m3), certificate status\n")
out.append("| family | seed | tier | parents | attempts | finished | model_min | data_min n1 | data_min n2 | data_min n3 | status | certified C (n_min=1 headline) | fallback used | attempts failing each check (one attempt counts under EVERY check it fails): gate / family / support / paste_floor / reliability (of n rejected) | paste-only blockers | lowest attempted C passing all but paste |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, e in E.items():
    c = e.get("chosen"); ps = e["paste_stats"]
    out.append(f"| F{e['family']} | {e['seed']} | {e.get('tier', '-')} | {e['n_parents']} | {e['n_attempts']} | {e['finished']} | {f1(e['model_min'], 2)} | {f1(e['data_min']['1'] if '1' in e['data_min'] else e['data_min'][1], 2)} | {f1(e['data_min'].get('2'), 2)} | {f1(e['data_min'].get('3'), 2)} | {e.get('status')} | {f1(c['l']['C'], 2) if c else '-'} | {c['fallback_index'] if c else '-'} | {' / '.join(str(e['fail_counts'][n]) for n in ('gate', 'family', 'support', 'paste_floor', 'reliability'))} (of {e['n_rejected']}) | {ps['n_paste_only_blockers']} | {f1(ps['lowest_attempted_C_passing_everything_except_paste'], 2)} |")

# ---------------- 1. family table (lowest-cement seed design) + per-seed cement
out.append("\n### T1 FAMILY TABLE\n")
out.append("| family | tier (n_min 1 / 2 / 3) | lowest certified C per seed in the order of the seeds (min .. max) | blend six bins % (lowest seed design) | other mix values | h vs floor | near passers | conflict | R_LCB95 nominal / worst at SD 1.0 / 1.5 / 2.0 (min of 2 reps) | no-lab variants SD 1.5 nominal / worst |\n|---|---|---|---|---|---|---|---|---|---|")
best_tier1 = None
for f in range(3):
    rows = [E.get(f"seed{s}_F{f}") for s in cfg["walk"]["seeds"]]; cs = [(e["chosen"]["l"]["C"] if e and "chosen" in e else None) for e in rows]; ok = [c for c in cs if c is not None]
    e0 = rows[0] if rows and rows[0] else None; has = (e0["has_recorded_passer"] and not NO_FLOOR) if e0 else None
    tier1 = "Tier 1" if has else TIER2
    dm = {n: [(e["data_min"][str(n)] if str(n) in e["data_min"] else e["data_min"][n]) if e else None for e in rows] for n in (1, 2, 3)}
    t = " / ".join(f"n{n}: {'reachable' if has and any(x is not None for x in dm[n]) else ('no recorded passer' if not has else 'not reachable')}" for n in (1, 2, 3))
    if not ok: out.append(f"| {FAMN[f]} | {tier1}; {t} | none certified ({', '.join(str(e.get('status')) if e else 'missing' for e in rows)}) | - | - | - | - | - | - | - |"); continue
    jb = int(np.argmin([c if c is not None else 1e9 for c in cs])); r = design_row(rows[jb])
    if has and (best_tier1 is None or r["C"] < best_tier1[0]): best_tier1 = (r["C"], f, rows[jb]["seed"], rows[jb])
    out.append(f"| {FAMN[f]} | {tier1}; {t} | {' / '.join(f1(c, 2) for c in cs)} ({min(ok):.2f} .. {max(ok):.2f}) | {r['blend']} (s{rows[jb]['seed']}, {r['pid']}) | {r['other']} | {r['h']} | {r['near']} | {r['conflict']} | {r['R']} | {r['nolab']} |")

# ---------------- 2. all seeds detail
out.append("\n### T2 certified design of every (family, seed)\n")
out.append("| family | seed | parent | C | blend six bins % | other | h | near passers | conflict | R_LCB95 nominal / worst SD 1.0 / 1.5 / 2.0 | no-lab | meets n>=2 / n>=3 (same design) |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, e in E.items():
    if "chosen" not in e: continue
    r = design_row(e); out.append(f"| F{e['family']} | {e['seed']} | {r['pid']} | {r['C']:.2f} | {r['blend']} | {r['other']} | {r['h']} | {r['near']} | {r['conflict']} | {r['R']} | {r['nolab']} | {r['n2']} / {r['n3']} |")

# ---------------- 3. recommended
out.append("\n### T3 recommended design\n")
if best_tier1: C0, f, s, e = best_tier1; r = design_row(e); out.append(f"Cheapest certified Tier-1 design: family {f}, seed {s}, parent {r['pid']}, cement {C0:.2f} kg/m3. Blend % {r['blend']}; {r['other']}; h {r['h']}; near passers {r['near']}; conflict {r['conflict']}; R_LCB95 {r['R']}; no-lab {r['nolab']}.")
else: out.append("none (no Tier-1 family has a certified data-backed design)")
# ---------------- verdict (first line of the file) + header
n_t1f = sum(1 for f in range(3) if par[str(f)]["info"]["n_passers"] > 0 and not NO_FLOOR)
if best_tier1 and not NO_FLOOR: C0, f, s_, e = best_tier1; r = design_row(e); VERDICT = f"recommended design = family {f}, seed {s_}, parent {r['pid']}, cement {C0:.2f} kg/m3 (Tier 1, data-backed; certified); blend % {r['blend']}; {r['other']}"
elif NO_FLOOR: VERDICT = f"none, because no recorded mix reaches {TG['S']:g} MPa AND {TG['K']:g} mm/s ({PG['reason']}); there is no paste floor and no data-backed design is possible - every design below is model-suggested, no paste floor, not a recommendation"
elif n_t1f == 0: VERDICT = "none, because no family has a recorded deployable passer, so every design below is model-suggested (Tier 2) and none may be recommended"
else: VERDICT = "none, because no Tier-1 family produced a design that is certified AND has a near recorded passer (see T0 status column); the Tier-2 rows below are model-suggested only"
fl = "none (null; no recorded unit meets the target)" if NO_FLOOR else f"{PG['h_min_mm']:.6f} mm (smallest paste thickness among {PG['n_units_full_target']} recorded unit(s) meeting the target, unit u{PG['argmin_unit']})"
hdr = [f"**VERDICT ({TNAME}): {VERDICT}.**", "", f"Target {TNAME}. Config {H}. Paste floor: {fl}." + (f" {PG['warning']}" if PG.get("warning") else ""),
       "Parent rule per family: " + "; ".join(f"F{f}: {par[str(f)]['info'].get('rule', 'G3_near_target_units')}" for f in range(3)) + ".", ""]
out[:0] = hdr

# ---------------- 4. per-parent
out.append("\n### T4 per-parent table (recorded cement / CS / k, lowest accepted cement per seed, W1 of the blend of the lowest accepted design from the parent blend, stop reason and counts of ALL failing checks per seed)\n")
out.append("Counts: attempts failing each check among the parent's attempts (g gate, f family, s support, p paste_floor, r reliability); one attempt can count under several checks.\n")
out.append("| family | parent | start rule | recorded C / CS / k | recorded passer (deployable) | porosity clamped | lowest accepted per seed | W1 moved per seed | stop per seed | failing-check counts per seed |\n|---|---|---|---|---|---|---|---|---|---|")
for f in range(3):
    for p in par[str(f)]["parents"]:
        lo, mv, st, fc = [], [], [], []
        gp = np.array(p["Z"][5:])        # Z[5:11] of a parent = its recorded blend (encode keeps g / sum(g))
        for s in cfg["walk"]["seeds"]:
            fp = os.path.join(OUT, f"seed{s}", f"F{f}", p["pid"] + ".jsonl")
            if not os.path.exists(fp): lo.append("-"); mv.append("-"); st.append("missing"); fc.append("-"); continue
            L = checks.load_log(fp); low = checks.lowest(L); ev = [l for l in L if l.get("event") in ("stop", "merged")]; at = checks.attempts(L)
            lo.append(f"{low['C']:.2f}" if low else "none"); mv.append(f"{w1(gp, np.array(low['accept']['nominal_design']['g'])):.4f}" if low else "-")
            st.append((ev[-1].get("reason") or "merged into " + ev[-1]["into"]) if ev else "unfinished")
            cnt = {n: sum(1 for l in at if n in l.get("reject_reasons", [])) for n in ("gate", "family", "support", "paste_floor", "reliability")}
            fc.append(f"g{cnt['gate']} f{cnt['family']} s{cnt['support']} p{cnt['paste_floor']} r{cnt['reliability']} (of {sum(1 for l in at if not l['accepted'])} rej, {len(at)} att)")
        out.append(f"| F{f} | {p['pid']} | {p.get('start_rule', 'G3_near_target_units')} | {p['C0']:.2f} / {p['CS']:.2f} / {p['K']:.2f} | {p['recorded_passer']} | {p['phi_clamped']} (phi {p['recorded_phi']:.2f}) | {' / '.join(lo)} | {' / '.join(mv)} | {' / '.join(st)} | {' / '.join(fc)} |")
open(os.path.join(OUT, "TABLES.md"), "w", encoding="utf-8").write("\n".join(out)); print("\n".join(out))
