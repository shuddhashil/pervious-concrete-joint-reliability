**VERDICT (S* 18 MPa, K* 13 mm/s, R* 0.90, integrated global + recorded walk): none, because no certified Tier-1 (data-backed) design exists; every design below is model-suggested (Tier 2) and none may be recommended.**

Target S* 18 MPa, K* 13 mm/s, R* 0.90. Config d590a8225023eb9bdd4d000fb8d5b4fb89aa486a7a91d48a778e669735fd693b. Paste floor 0.289498 mm (unit u63, 8 recorded unit(s) meet the target). Tier rule: Tier 1 only if at least one recorded passing mix is near the design (n_min = 1, frozen near radii, window-clamped mixes never count); everything else Tier 2 model-suggested, never recommended. Baseline = the existing walk of this target (stage\), unchanged.

### G0 global search only (best design per family over all seeds and levels; full curve in global/global_scan.csv)

| family | lowest passing level (any seed) | seeds passing there | best min-LB at that level (nominal / worst) | tier of that design | levels scanned | passes at (all levels) |
|---|---|---|---|---|---|---|
| F0 | none (best min-LB 0.8143 at 440) | 0 | 0.8143 | - | 500, 480, 460, 440, 420, 400 | none |
| F1 | none (best min-LB 0.8401 at 400) | 0 | 0.8401 | - | 500, 480, 460, 440, 420, 400 | none |
| F2 | none (best min-LB 0.8865 at 400) | 0 | 0.8865 | - | 500, 480, 460, 440, 420, 400 | none |

### T0 per (family, seed, source): log-derived minima (cement kg/m3) and certificate status

| family | seed | source | parents | attempts | finished | lowest accepted (any tier) | lowest accepted with >= 1 near passer (Tier-1 candidate) | n2 | n3 | status | certified Tier 1 | certified Tier 2 (model-suggested) | baseline walk certified | failing-check counts g/f/s/p/r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 0 | recorded | 5 | 65 | True | - | - | - | - | no accepted step | - | - | - | 25 / 14 / 3 / 0 / 65 |
| F0 | 1 | recorded | 5 | 65 | True | - | - | - | - | no accepted step | - | - | - | 22 / 9 / 3 / 0 / 65 |
| F0 | 2 | recorded | 5 | 65 | True | - | - | - | - | no accepted step | - | - | - | 13 / 4 / 0 / 0 / 65 |
| F1 | 0 | recorded | 6 | 78 | True | - | - | - | - | no accepted step | - | - | - | 40 / 20 / 14 / 0 / 78 |
| F1 | 1 | recorded | 6 | 78 | True | - | - | - | - | no accepted step | - | - | - | 38 / 24 / 6 / 0 / 78 |
| F1 | 2 | recorded | 6 | 78 | True | - | - | - | - | no accepted step | - | - | - | 37 / 18 / 11 / 0 / 78 |
| F2 | 0 | recorded | 12 | 156 | True | - | - | - | - | no accepted step | - | - | - | 6 / 0 / 1 / 9 / 156 |
| F2 | 1 | recorded | 12 | 156 | True | - | - | - | - | no accepted step | - | - | - | 9 / 1 / 0 / 8 / 156 |
| F2 | 2 | recorded | 12 | 156 | True | - | - | - | - | no accepted step | - | - | - | 0 / 0 / 0 / 11 / 156 |

### T1 FAMILY x SOURCE TABLE (lowest certified design per seed; Tier-1 designs unless marked)

| family | source | tier | lowest certified C per seed s0 / s1 / s2 | blend six bins % (lowest seed design) | other mix values | h vs floor | near passers (n>=1; n>=2, n>=3 beside) | certificates nominal / worst (min of 2 reps) at SD 1.5; SD 1.0 and 2.0 | baseline walk certified C per seed (recorded source) |
|---|---|---|---|---|---|---|---|---|---|
| F0 | recorded | Tier 1 | none certified (no accepted step, no accepted step, no accepted step) | - | - | - | - | - | - / - / - |
| F0 | global | Tier 1 | none certified (no run, no run, no run) | - | - | - | - | - | - |
| F1 | recorded | Tier 1 | none certified (no accepted step, no accepted step, no accepted step) | - | - | - | - | - | - / - / - |
| F1 | global | Tier 1 | none certified (no run, no run, no run) | - | - | - | - | - | - |
| F2 | recorded | Tier 1 | none certified (no accepted step, no accepted step, no accepted step) | - | - | - | - | - | - / - / - |
| F2 | global | Tier 1 | none certified (no run, no run, no run) | - | - | - | - | - | - |

### T2 certified design of every (family, seed, source, tier)

| family | seed | source | tier | parent | C | blend six bins % | other | h | near passers | certificate SD 1.5 nominal / worst | SD checks | fallback index |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

### T3 CONVERGENCE: do global-seeded and recorded-parent walks end in the same place? (same = cement within 5 kg/m3 AND blend W1 <= 0.02)

Compared per (family, seed): (A) lowest CERTIFIED Tier-1 design of each source; (B) lowest ACCEPTED step of each source (any tier, before certificates).

| family | seed | (A) recorded C | (A) global C | dC | W1 of blends | same? | (B) recorded lowest accepted | (B) global lowest accepted | dC | W1 | same? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 0 | - | - | - | - | n/a (neither source) | - | - | - | - | n/a (neither source) |
| F0 | 1 | - | - | - | - | n/a (neither source) | - | - | - | - | n/a (neither source) |
| F0 | 2 | - | - | - | - | n/a (neither source) | - | - | - | - | n/a (neither source) |
| F1 | 0 | - | - | - | - | n/a (neither source) | - | - | - | - | n/a (neither source) |
| F1 | 1 | - | - | - | - | n/a (neither source) | - | - | - | - | n/a (neither source) |
| F1 | 2 | - | - | - | - | n/a (neither source) | - | - | - | - | n/a (neither source) |
| F2 | 0 | - | - | - | - | n/a (neither source) | - | - | - | - | n/a (neither source) |
| F2 | 1 | - | - | - | - | n/a (neither source) | - | - | - | - | n/a (neither source) |
| F2 | 2 | - | - | - | - | n/a (neither source) | - | - | - | - | n/a (neither source) |

Convergence verdict (A, certified Tier-1 designs): 0 of 0 comparable (family, seed) cases end in the same place.
- none

### T4 recommended design

none