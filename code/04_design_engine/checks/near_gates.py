import json, glob, sys, collections
# per target: over every logged design with evidence, which gate kills near-passer evidence
for t in sys.argv[1:]:
    rows = []
    for f in (g.replace(chr(92), "/") for g in glob.glob(f"{t}/*/stage*/seed*/F*/*.jsonl")):
        for x in open(f):
            if '"evidence"' not in x: continue
            r = json.loads(x); a = r.get("accept") or {}; ev = a.get("evidence")
            if ev: rows.append((r.get("C"), a.get("accepted"), ev, f.split("/")[-4] + "/" + f.split("/")[-2] + "/" + f.split("/")[-1][:-6]))
    acc = [r for r in rows if r[1]] or rows
    c = collections.Counter()
    for C, ok, ev, _ in acc:
        c["designs"] += 1; c["any near unit"] += ev["n_near"] > 0; c["near passer units>=1"] += ev["n_near_pass_units"] > 0
        c["passers from>=2 studies"] += ev["n_near_pass_studies"] >= 2; c["volume ok"] += ev["volume_ok"]; c["counted (final)"] += ev["n_near_pass"] > 0
        for l, n in ev["by_lambda_n_near_pass"].items(): c[f"passers at radius x{l}"] += n > 0
    print(f"== {t}: {len(rows)} logged designs, {sum(1 for r in rows if r[1])} accepted"); [print(f"   {k}: {v}") for k, v in c.items()]
    best = sorted([r for r in acc if r[2]["n_near_pass_units"] > 0], key=lambda r: r[0] or 1e9)[:3]
    for C, ok, ev, w in best: print(f"   cheapest with a near passer unit: C {C:.1f} {w} studies {ev['near_pass_studies']} vol {ev['volume_m3']:.3f}")
