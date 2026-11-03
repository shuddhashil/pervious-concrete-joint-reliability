"""Item 3 follow-up (2026-10-30, author: "run the IND search then"): the inverse search itself run with the independent-GP engine, so IND gets a
candidate pool of its own instead of the pool built by MT searches. Per target, the same stages and settings as the 2026-10-22 runs
(run_target.py defaults: walk 2.0 h, global scan 1.0 h, global walk 0.6 h deadlines; seeds 0 1 2), each stage run through ind_patch (both the exact
scorer and the gradient polish use IND). The CPU certification stages are not run (they would use the MT model); instead the IND pool is harvested
with the extract_candidates rule, joined with the MT pool, and the joined pool is scored by BOTH models with design_eval.py (re-score + GPU final).
usage: python ind_search.py <GE_root> <GE_IND_root> <ind_oracle.pt> <mt_candidates.json> <out_dir> S,K [S,K ...]
       python ind_search.py ... --score-only S,K   (skip the search stages; harvest + score)
       python ind_search.py ... --search-only S,K  (search stages only; no harvest or scoring)"""
import os, sys, json, glob, subprocess
a = [x for x in sys.argv[1:] if not x.startswith("--")]; SCORE_ONLY = "--score-only" in sys.argv; SEARCH_ONLY = "--search-only" in sys.argv
GE, GEI, OPT, MTC, OUT = [os.path.abspath(x) for x in a[:5]]; TGTS = [tuple(float(v) for v in t.split(",")) for t in a[5:]]
KIT = os.path.dirname(os.path.abspath(__file__)); os.makedirs(OUT, exist_ok=True)
LAUNCH = ("import sys, runpy; sys.dont_write_bytecode=True; root, kit, script = sys.argv[1:4]; sys.path[:0] = [root, kit]\n"
          "import gengine.common, ind_patch\nsys.argv = [script] + sys.argv[4:]; runpy.run_path(script, run_name='__main__')")


def tdir(root, S, K):
    code = f"import sys; sys.path.insert(0, {root!r}); sys.dont_write_bytecode=True\nfrom gengine import target as TG\nprint(TG.make_target({S}, {K}, 0.90)['dir'])"
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=root).stdout.strip().splitlines()[-1]


def env_for(root, T):
    return dict(os.environ, GENGINE_BASE=T, GENGINE_CONFIG=os.path.join(T, "CONFIG.json"), GENGINE_ROOT=root, IND_ORACLE=OPT, PYTHONDONTWRITEBYTECODE="1")


def harvest(T):
    """extract_candidates.py rule on this target's own logs: walk + global-walk attempts that are accepted or failed on reliability only, plus global parents."""
    c = {}
    for p in glob.glob(os.path.join(T, "stage*", "seed*", "F*", "*.jsonl")):
        for line in open(p):
            l = json.loads(line)
            if l.get("event") != "attempt" or "Z" not in l or not set(l.get("reject_reasons") or []) <= {"reliability"}: continue
            c.setdefault((int(l["family"]), round(float(l["C"]), 6), tuple(round(float(x), 10) for x in l["Z"])), 1)
    for p in glob.glob(os.path.join(T, "stage*", "global_parents.json")):
        for key, lst in json.load(open(p)).items():
            for g in lst: c.setdefault((int(key.split("_F")[1]), round(float(g["C0"]), 6), tuple(round(float(x), 10) for x in g["Z"])), 1)
    return c


mt_pool = json.load(open(MTC))
for S, K in TGTS:
    name = f"S{S:g}_K{K:g}_R0.90"; T = tdir(GEI, S, K); env = env_for(GEI, T); LOG = open(os.path.join(OUT, f"search_{name}.log"), "a")
    if not os.path.exists(os.path.join(T, "CONFIG.json")):
        subprocess.run([sys.executable, f"{GEI}/gengine/target_config.py", str(S), str(K), "0.9", "0", "1", "2"], env=env, cwd=GEI, check=True)
    stages = [] if SCORE_ONLY else [("walk", "run_full.py", os.environ.get("H_WALK", "2.0")), ("global_scan", "global_search.py", os.environ.get("H_SCAN", "1.0")), ("global_walk", "run_global_walk.py", os.environ.get("H_GWALK", "0.6"))]          # wall-clock caps; a slower GPU gets longer caps so the search stops by its own rules, as on the T4
    for nm, script, hrs in stages:
        done = os.path.join(T, f"IND_{nm}.done")
        if os.path.exists(done): print("done already:", name, nm, flush=True); continue
        print(f"=== {name} {nm}", flush=True)
        r = subprocess.run([sys.executable, "-u", "-c", LAUNCH, GEI, KIT, os.path.join(GEI, "gengine", script), hrs], env=env, cwd=GEI, stdout=LOG, stderr=subprocess.STDOUT)
        if r.returncode != 0: print(f"STAGE FAILED {name} {nm} rc={r.returncode} (see {LOG.name})", flush=True); sys.exit(r.returncode)
        open(done, "w").write("ok")
    if SEARCH_ONLY: print(f'{name}: search stages done', flush=True); continue
    ind = harvest(T); mt = {(int(f), float(cc), tuple(z)): 1 for f, cc, z in mt_pool[name]}
    union = sorted(set(ind) | set(mt), key=lambda k: (k[1], k[0]))
    pools = dict(union=[[f, cc, list(z)] for f, cc, z in union], ind_only=[[f, cc, list(z)] for f, cc, z in sorted(set(ind) - set(mt), key=lambda k: (k[1], k[0]))])
    json.dump({name: pools["union"]}, open(os.path.join(OUT, f"pool_union_{name}.json"), "w")); json.dump(pools, open(os.path.join(OUT, f"pools_{name}.json"), "w"))
    print(f"{name}: IND pool {len(ind)} (new {len(pools['ind_only'])}), MT pool {len(mt)}, union {len(union)}", flush=True)
    for kind, root in (("MT", GE), ("IND", GEI)):
        out = os.path.join(OUT, f"{kind}_{name}_union.json")
        if os.path.exists(out): continue
        Tk = tdir(root, S, K); ek = env_for(root, Tk)
        if not os.path.exists(os.path.join(Tk, "CONFIG.json")):
            subprocess.run([sys.executable, f"{root}/gengine/target_config.py", str(S), str(K), "0.9", "0", "1", "2"], env=ek, cwd=root, check=True)
        print(f"=== score {kind} {name} on the union pool", flush=True)
        subprocess.run([sys.executable, "-u", f"{KIT}/design_eval.py", kind, os.path.join(OUT, f"pool_union_{name}.json"), out, OPT if kind == "IND" else ""], env=ek, cwd=root, check=True, stdout=LOG, stderr=subprocess.STDOUT)
        d = json.load(open(out)); lo = d["lowest_accepted"]; fe = [f for f in d["final"] if f["tag"].startswith("lowest")]
        print(f"   {kind} {name}: lowest accepted {lo and lo['C']} | final {fe and round(fe[0]['final_R_LCB95'], 4)} | accepted {d['n_accepted']}/{d['n_candidates']}", flush=True)
print("ALL DONE", flush=True)
