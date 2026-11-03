"""Tables for the integrated (recorded + global) walk of one target: <stage_global_walk>/TABLES.md with a one-line VERDICT at the top. No GPU. usage: python gengine/make_tables_global.py"""
import os, sys, json, glob, math
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
import numpy as np
from gengine import config, checks
from gengine.common import W_LOG
from gengine.global_search import stage_dirs, recommend
cfg, H = config.load(); BASE, OUT = stage_dirs(cfg); sel = json.load(open(os.path.join(OUT, "SELECTION_GLOBAL.json"))); E = sel["entries"]; bsel = json.load(open(os.path.join(BASE, "SELECTION.json")))["entries"]
par = json.load(open(os.path.join(OUT, "parents.json"))); gpar = json.load(open(os.path.join(OUT, "global_parents.json")))
runs = [json.loads(l) for l in open(os.path.join(OUT, "global", "runs.jsonl")) if l.strip()]
PG = cfg["paste_guard"]; NO_FLOOR = PG["h_min_mm"] is None; TG = cfg["target"]; TNAME = f"S* {TG['S']:g} MPa, K* {TG['K']:g} mm/s, R* {TG['R']:.2f}"; SEEDS = cfg["walk"]["seeds"]; FLAG = sel.get("flag_kg", 10.0); SDS = [cfg["porosity_sd_pp"]["main"]] + cfg["porosity_sd_pp"]["checks"]
W1_SMALL, DC_SMALL = 0.02, 5.0
w1 = lambda a, b: float((np.abs(np.cumsum(a)[:5] - np.cumsum(b)[:5]) * W_LOG).sum()); f1 = lambda x, n=1: "-" if x is None else f"{x:.{n}f}"; pct = lambda g: "/".join(f"{100 * x:.1f}" for x in g); out = []
FN = {0: "F0 coarse/medium mixed", 1: "F1", 2: "F2 fine single-size (Li/Feric)"}


def rmin(ch, sd, tag, k="R"):
    v = [ch["sd_checks"].get(f"sd{sd}_{tag}_rep{r}") if sd != SDS[0] else ch["certificate"]["results"][f"{tag}_rep{r}"] for r in (0, 1)] if (sd == SDS[0] or "sd_checks" in ch) else []
    v = [x[k] for x in v if isinstance(x, dict) and x.get(k) is not None]; return min(v) if v else None


def row(e, ck):
    ch = e[ck]; l = ch["l"]; a = l["accept"]; nd = a["nominal_design"]; ev = a["evidence"]; p = a["paste"]; sdok = "sd_checks" in ch
    return dict(C=l["C"], pid=l["pid"], seed=e["seed"], fam=e["family"], source=e["source"], g=nd["g"], blend=pct(nd["g"]), other=f"w/c {nd['r']:.4f}, CA {nd['A']:.1f}, SP {nd['P']:.2f}, sand {nd['S']:.1f}, phi {nd['phi']:.2f}",
                h=f"{p['h_nominal']:.3f} / min11 {p['h_min_over_11']:.3f} vs floor " + ("none" if p["h_floor"] is None else f"{p['h_floor']:.4f}"), n=ev["n_near_pass"], near=f"{ev['n_near_pass']} (n>=2: {ev['n_near_pass'] >= 2}, n>=3: {ev['n_near_pass'] >= 3})",
                tier="Tier 1" if (checks.data_ok(l, 1) and not NO_FLOOR) else "Tier 2 (model-suggested, never recommended)", cert15=f"{f1(rmin(ch, SDS[0], 'nominal'), 4)} / {f1(rmin(ch, SDS[0], 'worst'), 4)}",
                sdtxt="; ".join(f"SD{sd}: {f1(rmin(ch, sd, 'nominal'), 4)} / {f1(rmin(ch, sd, 'worst'), 4)}" for sd in SDS) if sdok else f"SD{SDS[0]}: {f1(rmin(ch, SDS[0], 'nominal'), 4)} / {f1(rmin(ch, SDS[0], 'worst'), 4)} (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier))",
                fb=ch["fallback_index"], tag=ck)


def best_cert(f, source, ck, sid=None):
    es = [e for e in E.values() if e["family"] == f and e["source"] == source and ck in e and (sid is None or e["seed"] == sid)]; return min(es, key=lambda e: e[ck]["l"]["C"]) if es else None


def base_C(f, s):
    e = bsel.get(f"seed{s}_F{f}"); return e["chosen"]["l"]["C"] if e and "chosen" in e else None


def model_min(f, s, source): e = E.get(f"seed{s}_F{f}_{source}"); return e["model_min"] if e else None


# ---------------- recommended (cheapest certified Tier-1 over family, source, seed)
_r = recommend(E); rec = (_r[0], E[_r[1]]) if _r else None
best_rec_src = {f: min([e["chosen"]["l"]["C"] for e in E.values() if e["family"] == f and e["source"] == "recorded" and "chosen" in e], default=None) for f in range(3)}
flags = []
for e in E.values():
    if e["source"] != "global": continue
    for ck in ("chosen", "chosen_t2"):
        if ck in e:
            br = best_rec_src[e["family"]]; C0 = e[ck]["l"]["C"]; ref = br if br is not None else None
            if ref is None or C0 < ref - FLAG: flags.append((e, ck, C0, ref))

# ---------------- header
if rec:
    r = row(rec[1], "chosen"); VERDICT = f"recommended design = family {r['fam']}, seed {r['seed']}, source {r['source']}, parent {r['pid']}, cement {r['C']:.2f} kg/m3 (Tier 1: {r['n']} near recorded passing mix(es); certified); blend % {r['blend']}; {r['other']}"
else: VERDICT = "none, because no certified Tier-1 (data-backed) design exists; every design below is model-suggested (Tier 2) and none may be recommended"
fl = "none (null)" if NO_FLOOR else f"{PG['h_min_mm']:.6f} mm (unit u{PG['argmin_unit']}, {PG['n_units_full_target']} recorded unit(s) meet the target)"
out += [f"**VERDICT ({TNAME}, integrated global + recorded walk): {VERDICT}.**", "", f"Target {TNAME}. Config {H}. Paste floor {fl}. Tier rule: Tier 1 only if at least one recorded passing mix is near the design (n_min = 1, frozen near radii, window-clamped mixes never count); everything else Tier 2 model-suggested, never recommended. Baseline = the existing walk of this target (stage\\), unchanged.", ""]
if flags:
    out.append("**FLAG: a global-seeded design is more than %g kg/m3 cheaper than the best recorded-parent walk design of its family - treat as MODEL-SUGGESTED until checked in the lab:**" % FLAG)
    for e, ck, C0, ref in flags: r = row(e, ck); out.append(f"- F{e['family']} seed {e['seed']} global: {C0:.2f} kg/m3 vs best recorded-parent walk design {('NONE (the recorded-parent walks found no accepted design in this family)' if ref is None else f1(ref, 2))} ({'Tier 1' if r['tier'] == 'Tier 1' else 'Tier 2'}); {r['near']} near passers; certificate SD 1.5 nominal / worst {r['cert15']}")
    out.append("")

# ---------------- global scan summary (global only)
out.append("### G0 global search only (best design per family over all seeds and levels; full curve in global/global_scan.csv)\n")
out.append("| family | lowest passing level (any seed) | seeds passing there | best min-LB at that level (nominal / worst) | tier of that design | levels scanned | passes at (all levels) |\n|---|---|---|---|---|---|---|")
for f in range(3):
    rf = [r for r in runs if r["family"] == f]; ps = sorted({r["C"] for r in rf if r["passed"]})
    if not rf: out.append(f"| F{f} | not run | - | - | - | - | - |"); continue
    lv = sorted({r["C"] for r in rf}, reverse=True)
    if ps: L = ps[0]; at = [r for r in rf if r["C"] == L]; b = max(at, key=lambda r: (r["passed"], r["LB_min_all"])); out.append(f"| F{f} | {L:g} | {sum(r['passed'] for r in at)} of {len(at)} | {b['LB_min_all']:.4f} ({b['LB_nominal']:.4f} / {b['LB_worst']:.4f}) | {b['tier']} (near passers {b['n_near_pass']}) | {', '.join(f'{c:g}' for c in lv)} | {', '.join(f'{c:g}' for c in ps)} |")
    else: b = max(rf, key=lambda r: r["LB_min_all"]); out.append(f"| F{f} | none (best min-LB {b['LB_min_all']:.4f} at {b['C']:g}) | 0 | {b['LB_min_all']:.4f} | - | {', '.join(f'{c:g}' for c in lv)} | none |")

# ---------------- T0 per (family, seed, source)
out.append("\n### T0 per (family, seed, source): log-derived minima (cement kg/m3) and certificate status\n")
out.append("| family | seed | source | parents | attempts | finished | lowest accepted (any tier) | lowest accepted with >= 1 near passer (Tier-1 candidate) | n2 | n3 | status | certified Tier 1 | certified Tier 2 (model-suggested) | baseline walk certified | failing-check counts g/f/s/p/r |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for f in range(3):
    for s in SEEDS:
        for src in ("recorded", "global"):
            e = E.get(f"seed{s}_F{f}_{src}")
            if not e: continue
            c1 = e["chosen"]["l"]["C"] if "chosen" in e else None; c2 = e["chosen_t2"]["l"]["C"] if "chosen_t2" in e else None; d = e["data_min"]
            out.append(f"| F{f} | {s} | {src} | {e['n_parents']} | {e['n_attempts']} | {e['finished']} | {f1(e['model_min'], 2)} | {f1(d.get('1', d.get(1)), 2)} | {f1(d.get('2', d.get(2)), 2)} | {f1(d.get('3', d.get(3)), 2)} | {e['status']} | {f1(c1, 2)} | {f1(c2, 2)} | {f1(base_C(f, s), 2) if src == 'recorded' else '-'} | {' / '.join(str(e['fail_counts'][n]) for n in ('gate', 'family', 'support', 'paste_floor', 'reliability'))} |")

# ---------------- T1 family x source table
out.append("\n### T1 FAMILY x SOURCE TABLE (lowest certified design per seed; Tier-1 designs unless marked)\n")
out.append("| family | source | tier | lowest certified C per seed s0 / s1 / s2 | blend six bins % (lowest seed design) | other mix values | h vs floor | near passers (n>=1; n>=2, n>=3 beside) | certificates nominal / worst (min of 2 reps) at SD 1.5; SD 1.0 and 2.0 | baseline walk certified C per seed (recorded source) |\n|---|---|---|---|---|---|---|---|---|---|")
for f in range(3):
    for src in ("recorded", "global"):
        for ck, lab in (("chosen", "Tier 1"), ("chosen_t2", "Tier 2 model-suggested")):
            es = [E.get(f"seed{s}_F{f}_{src}") for s in SEEDS]; cs = [(e[ck]["l"]["C"] if e and ck in e else None) for e in es]; ok = [c for c in cs if c is not None]
            if not ok:
                if ck == "chosen": out.append(f"| F{f} | {src} | Tier 1 | none certified ({', '.join(str(e['status']) if e else 'no run' for e in es)}) | - | - | - | - | - | " + (" / ".join(f1(base_C(f, s), 2) for s in SEEDS) if src == "recorded" else "-") + " |")
                continue
            jb = int(np.argmin([c if c is not None else 1e9 for c in cs])); r = row(es[jb], ck)
            out.append(f"| F{f} | {src} | {lab} | {' / '.join(f1(c, 2) for c in cs)} | {r['blend']} (s{r['seed']}, {r['pid']}) | {r['other']} | {r['h']} | {r['near']} | {r['sdtxt']} | " + (" / ".join(f1(base_C(f, s), 2) for s in SEEDS) if src == "recorded" else "-") + " |")

# ---------------- T2 every certified design
out.append("\n### T2 certified design of every (family, seed, source, tier)\n")
out.append("| family | seed | source | tier | parent | C | blend six bins % | other | h | near passers | certificate SD 1.5 nominal / worst | SD checks | fallback index |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for e in E.values():
    for ck in ("chosen", "chosen_t2"):
        if ck in e: r = row(e, ck); out.append(f"| F{r['fam']} | {r['seed']} | {r['source']} | {r['tier']} | {r['pid']} | {r['C']:.2f} | {r['blend']} | {r['other']} | {r['h']} | {r['near']} | {r['cert15']} | {r['sdtxt']} | {r['fb']} |")

# ---------------- convergence
out.append(f"\n### T3 CONVERGENCE: do global-seeded and recorded-parent walks end in the same place? (same = cement within {DC_SMALL:g} kg/m3 AND blend W1 <= {W1_SMALL})\n")
out.append("Compared per (family, seed): (A) lowest CERTIFIED Tier-1 design of each source; (B) lowest ACCEPTED step of each source (any tier, before certificates).\n")
out.append("| family | seed | (A) recorded C | (A) global C | dC | W1 of blends | same? | (B) recorded lowest accepted | (B) global lowest accepted | dC | W1 | same? |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
differ = []; nsame = ntot = 0
def lowest_acc(f, s, src):
    e = E.get(f"seed{s}_F{f}_{src}")
    if not e or e["model_min"] is None: return None, None
    best = None
    for p in glob.glob(os.path.join(OUT, f"seed{s}", f"F{f}", "*.jsonl")):
        pid = os.path.basename(p)[:-6]; isg = pid.startswith("g")
        if (src == "global") != isg: continue
        l = checks.lowest(checks.load_log(p))
        if l and (best is None or l["C"] < best["C"]): best = l
    return best["C"], best["accept"]["nominal_design"]["g"]
for f in range(3):
    for s in SEEDS:
        er, eg = E.get(f"seed{s}_F{f}_recorded"), E.get(f"seed{s}_F{f}_global")
        a_r = (er["chosen"]["l"]["C"], er["chosen"]["l"]["accept"]["nominal_design"]["g"]) if er and "chosen" in er else (None, None); a_g = (eg["chosen"]["l"]["C"], eg["chosen"]["l"]["accept"]["nominal_design"]["g"]) if eg and "chosen" in eg else (None, None)
        b_r, b_g = lowest_acc(f, s, "recorded"), lowest_acc(f, s, "global")
        def cmp(x, y):
            if x[0] is None and y[0] is None: return None, None, "n/a (neither source)"
            if x[0] is None or y[0] is None: return None, None, "DIFFERENT (only one source has a design)"
            dC = y[0] - x[0]; d = w1(x[1], y[1]); same = abs(dC) <= DC_SMALL and d <= W1_SMALL; return dC, d, "yes" if same else "DIFFERENT"
        dA, wA, sA = cmp(a_r, a_g); dB, wB, sB = cmp(b_r, b_g)
        if sA != "n/a (neither source)": ntot += 1; nsame += sA == "yes"
        if sA.startswith("DIFF") or sB.startswith("DIFF"): differ.append(f"F{f} seed {s}: (A) {sA}" + (f" dC {dA:+.2f} W1 {wA:.4f}" if dA is not None else "") + f"; (B) {sB}" + (f" dC {dB:+.2f} W1 {wB:.4f}" if dB is not None else ""))
        out.append(f"| F{f} | {s} | {f1(a_r[0], 2)} | {f1(a_g[0], 2)} | {'-' if dA is None else f'{dA:+.2f}'} | {'-' if wA is None else f'{wA:.4f}'} | {sA} | {f1(b_r[0], 2)} | {f1(b_g[0], 2)} | {'-' if dB is None else f'{dB:+.2f}'} | {'-' if wB is None else f'{wB:.4f}'} | {sB} |")
out.append(f"\nConvergence verdict (A, certified Tier-1 designs): {nsame} of {ntot} comparable (family, seed) cases end in the same place." + ("" if not differ else " Cases that differ:"))
out += [f"- {d}" for d in differ] if differ else ["- none"]

# ---------------- recommended
out.append("\n### T4 recommended design\n")
if rec: r = row(rec[1], "chosen"); out.append(f"Cheapest certified Tier-1 design: family {r['fam']}, seed {r['seed']}, source {r['source']}, parent {r['pid']}, cement {r['C']:.2f} kg/m3; blend % {r['blend']}; {r['other']}; h {r['h']}; near recorded passers {r['near']}; certificate SD 1.5 nominal / worst {r['cert15']}; {r['sdtxt']}.")
else: out.append("none")
open(os.path.join(OUT, "TABLES.md"), "w", encoding="utf-8").write("\n".join(out)); print("\n".join(out))
