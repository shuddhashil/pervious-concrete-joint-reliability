"""Extract the mixtures of the author's two papers into one table (2026-10-26).
TiDE (mix design).docx: Box-Behnken, 17 mixtures, aggregate/cement/water in g per batch (Table 2), porosity, k (cm/s) and 28-day CS (Table 5).
RIENGG (gradation).docx: simplex-centroid, 24 runs, A/C = 4, W/C = 0.27, sand 5 % of aggregate weight, gradation A-D (Table 3), porosity, k (mm/s), CS (Table 4).
Reads the pandoc markdown of both papers (path arguments); writes own_papers_raw.csv."""
import re, sys, csv
tide, rieng = sys.argv[1], sys.argv[2]
rows = []
# ---- TiDE
t = open(tide, encoding="utf8").read()
_i = t.index("| 1       | -0+"); t2 = t[_i:t.index("**Table 2**", _i)]
mix = {}
for ln in t2.splitlines():
    c = [x.strip() for x in ln.strip("|").split("|")]
    if len(c) == 8 and c[0].isdigit():
        mix[int(c[0])] = dict(pattern=c[1], A=int(c[2]), CA=int(c[3]), WC=int(c[4]), agg=float(c[5]), cem=float(c[6]), wat=float(c[7]))
_i = t.index("**Mix No.**"); t5 = t[_i:t.index("**Table 5**", _i)]
res = {}
for ln in t5.splitlines():
    m = re.match(r"\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", ln)
    if m: res[int(m.group(1))] = (float(m.group(2)), float(m.group(3)) * 10.0, float(m.group(4)))   # porosity %, k cm/s -> mm/s, CS MPa
assert len(mix) == 17 and len(res) == 17, (len(mix), len(res))
fracA = {-1: 0.0, 0: 0.5, 1: 1.0}                # share of the large (9.5-19 mm) aggregate X: level -1 = 100 % Y (small), 0 = 50/50, +1 = 100 % X (large)
for i in range(1, 18):
    m = mix[i]; p, k, cs = res[i]
    rows.append(dict(lab="TiDE", run=i, cement_g=m["cem"], water_g=m["wat"], aggregate_g=m["agg"], sand_g=0.0, sp_g=0.0, wc=m["wat"] / m["cem"], large_share=fracA[m["A"]],
                     groups="", porosity=p, perm_mm_s=k, cs_mpa=cs))
# ---- RIENGG
r = open(rieng, encoding="utf8").read()
_i = r.index("| 1       | A            |"); r3 = r[_i:r.index("**Table 3**", _i)]
gr = {}
for ln in r3.splitlines():
    c = [x.strip() for x in ln.strip("|").split("|")]
    if len(c) == 6 and c[0].isdigit(): gr[int(c[0])] = [float(x) for x in c[2:6]]
_i = r.index("**[Run]{.mark}**"); r4 = r[_i:r.index("**Table 4**", _i)]
rs = {}
for ln in r4.splitlines():
    nums = re.findall(r"\[([\d.]+)\]\{\.mark\}", ln)
    if len(nums) == 6: rs[int(float(nums[0]))] = [float(x) for x in nums[1:]]   # porosity, density, k mm/s, CS, tensile
assert len(gr) == 24 and len(rs) == 24, (len(gr), len(rs))
for i in range(1, 25):
    p, dens, k, cs, ts = rs[i]
    rows.append(dict(lab="RIENGG", run=i, cement_g=1.0, water_g=0.27, aggregate_g=4.0, sand_g=0.2, sp_g=0.0, wc=0.27, large_share="",
                     groups=" ".join(f"{x:.2f}" for x in gr[i]), porosity=p, perm_mm_s=k, cs_mpa=cs, density_g_cm3=dens))
keys = ["lab", "run", "cement_g", "water_g", "aggregate_g", "sand_g", "sp_g", "wc", "large_share", "groups", "porosity", "perm_mm_s", "cs_mpa", "density_g_cm3"]
with open("own_papers_raw.csv", "w", newline="", encoding="utf8") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(len(rows), "records written")
