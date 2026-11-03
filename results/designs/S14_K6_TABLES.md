**VERDICT (S* 14 MPa, K* 6 mm/s, R* 0.90, integrated global + recorded walk): none, because no certified Tier-1 (data-backed) design exists; every design below is model-suggested (Tier 2) and none may be recommended.**

Target S* 14 MPa, K* 6 mm/s, R* 0.90. Config 4ffc13101dcc12dd7a510ca26a72fadbed2520de85225c40a1a2dd36008661c8. Paste floor 0.160289 mm (unit u66, 97 recorded unit(s) meet the target). Tier rule: Tier 1 only if at least one recorded passing mix is near the design (n_min = 1, frozen near radii, window-clamped mixes never count); everything else Tier 2 model-suggested, never recommended. Baseline = the existing walk of this target (stage\), unchanged.

**FLAG: a global-seeded design is more than 10 kg/m3 cheaper than the best recorded-parent walk design of its family - treat as MODEL-SUGGESTED until checked in the lab:**
- F0 seed 0 global: 332.50 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9047 / 0.9039
- F1 seed 0 global: 295.00 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9215 / 0.9142
- F2 seed 0 global: 265.00 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9051 / 0.9031
- F0 seed 1 global: 327.75 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9029 / 0.9021
- F1 seed 1 global: 280.00 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9029 / 0.9023
- F2 seed 1 global: 267.75 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9031 / 0.9020
- F0 seed 2 global: 339.00 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9036 / 0.9030
- F1 seed 2 global: 280.50 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9042 / 0.9038
- F2 seed 2 global: 259.00 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9036 / 0.9015

### G0 global search only (best design per family over all seeds and levels; full curve in global/global_scan.csv)

| family | lowest passing level (any seed) | seeds passing there | best min-LB at that level (nominal / worst) | tier of that design | levels scanned | passes at (all levels) |
|---|---|---|---|---|---|---|
| F0 | 330 | 1 of 3 | 0.9025 (0.9031 / 0.9025) | Tier 2 (near passers 0) | 420, 400, 380, 360, 340, 335, 330, 325, 320, 300 | 330, 335, 340, 360, 380, 400, 420 |
| F1 | 280 | 1 of 3 | 0.9026 (0.9031 / 0.9026) | Tier 2 (near passers 0) | 420, 400, 380, 360, 340, 320, 300, 280, 275, 260 | 280, 300, 320, 340, 360, 380, 400, 420 |
| F2 | 270 | 2 of 3 | 0.9223 (0.9231 / 0.9223) | Tier 2 (near passers 0) | 420, 400, 380, 360, 340, 320, 300, 280, 275, 270, 265, 260 | 270, 275, 280, 300, 320, 340, 360, 380, 400, 420 |

### T0 per (family, seed, source): log-derived minima (cement kg/m3) and certificate status

| family | seed | source | parents | attempts | finished | lowest accepted (any tier) | lowest accepted with >= 1 near passer (Tier-1 candidate) | n2 | n3 | status | certified Tier 1 | certified Tier 2 (model-suggested) | baseline walk certified | failing-check counts g/f/s/p/r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 0 | recorded | 12 | 174 | True | 290.00 | - | - | - | model-suggested only (Tier 2) certified | - | 290.00 | - | 42 / 19 / 0 / 0 / 133 |
| F0 | 0 | global | 1 | 8 | True | 332.50 | - | - | - | model-suggested only (Tier 2) certified | - | 332.50 | - | 0 / 0 / 0 / 0 / 6 |
| F0 | 1 | recorded | 12 | 177 | True | 308.75 | - | - | - | model-suggested only (Tier 2) certified | - | 308.75 | - | 42 / 16 / 0 / 0 / 132 |
| F0 | 1 | global | 2 | 20 | True | 327.75 | - | - | - | model-suggested only (Tier 2) certified | - | 327.75 | - | 0 / 0 / 0 / 0 / 13 |
| F0 | 2 | recorded | 12 | 173 | True | 292.50 | - | - | - | model-suggested only (Tier 2) certified | - | 292.50 | - | 31 / 14 / 0 / 0 / 137 |
| F0 | 2 | global | 1 | 9 | True | 339.00 | - | - | - | model-suggested only (Tier 2) certified | - | 339.00 | - | 0 / 0 / 0 / 0 / 7 |
| F1 | 0 | recorded | 12 | 167 | True | 318.10 | - | - | - | model-suggested only (Tier 2) certified | - | 318.10 | - | 83 / 74 / 1 / 0 / 107 |
| F1 | 0 | global | 1 | 8 | True | 295.00 | - | - | - | model-suggested only (Tier 2) certified | - | 295.00 | - | 6 / 0 / 0 / 0 / 4 |
| F1 | 1 | recorded | 12 | 164 | True | 309.10 | - | - | - | model-suggested only (Tier 2) certified | - | 309.10 | - | 86 / 71 / 1 / 0 / 106 |
| F1 | 1 | global | 2 | 17 | True | 280.00 | - | - | - | model-suggested only (Tier 2) certified | - | 280.00 | - | 0 / 0 / 0 / 0 / 12 |
| F1 | 2 | recorded | 12 | 177 | True | 293.10 | - | - | - | model-suggested only (Tier 2) certified | - | 293.10 | - | 83 / 73 / 1 / 0 / 111 |
| F1 | 2 | global | 1 | 12 | True | 280.50 | - | - | - | model-suggested only (Tier 2) certified | - | 280.50 | - | 0 / 0 / 0 / 0 / 6 |
| F2 | 0 | recorded | 12 | 163 | True | 280.25 | - | - | - | model-suggested only (Tier 2) certified | - | 280.25 | - | 22 / 0 / 2 / 0 / 125 |
| F2 | 0 | global | 2 | 17 | True | 265.00 | - | - | - | model-suggested only (Tier 2) certified | - | 265.00 | - | 0 / 0 / 0 / 0 / 12 |
| F2 | 1 | recorded | 12 | 158 | True | 280.50 | - | - | - | model-suggested only (Tier 2) certified | - | 280.50 | - | 27 / 0 / 3 / 0 / 120 |
| F2 | 1 | global | 1 | 10 | True | 267.75 | - | - | - | model-suggested only (Tier 2) certified | - | 267.75 | - | 0 / 0 / 0 / 0 / 6 |
| F2 | 2 | recorded | 12 | 161 | True | 281.50 | - | - | - | model-suggested only (Tier 2) certified | - | 281.50 | - | 20 / 0 / 2 / 0 / 126 |
| F2 | 2 | global | 2 | 20 | True | 259.00 | - | - | - | model-suggested only (Tier 2) certified | - | 259.00 | - | 0 / 0 / 0 / 0 / 12 |

### T1 FAMILY x SOURCE TABLE (lowest certified design per seed; Tier-1 designs unless marked)

| family | source | tier | lowest certified C per seed s0 / s1 / s2 | blend six bins % (lowest seed design) | other mix values | h vs floor | near passers (n>=1; n>=2, n>=3 beside) | certificates nominal / worst (min of 2 reps) at SD 1.5; SD 1.0 and 2.0 | baseline walk certified C per seed (recorded source) |
|---|---|---|---|---|---|---|---|---|---|
| F0 | recorded | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - / - / - |
| F0 | recorded | Tier 2 model-suggested | 290.00 / 308.75 / 292.50 | 1.9/13.7/55.8/23.6/2.5/2.5 (s0, u474) | w/c 0.2735, CA 1603.4, SP 1.95, sand 0.0, phi 21.13 | 0.464 / min11 0.454 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9046 / 0.9029; SD1.0: 0.9140 / 0.9127; SD2.0: 0.8887 / 0.8870 | - / - / - |
| F0 | global | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - |
| F0 | global | Tier 2 model-suggested | 332.50 / 327.75 / 339.00 | 0.9/65.9/33.2/0.0/0.0/0.0 (s1, g335_s1) | w/c 0.3270, CA 1533.2, SP 0.00, sand 241.3, phi 22.98 | 0.842 / min11 0.833 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9029 / 0.9021; SD1.0: 0.9120 / 0.9106; SD2.0: 0.8908 / 0.8893 | - |
| F1 | recorded | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - / - / - |
| F1 | recorded | Tier 2 model-suggested | 318.10 / 309.10 / 293.10 | 0.0/0.0/30.7/38.0/25.5/5.8 (s2, u358) | w/c 0.2752, CA 1611.0, SP 1.77, sand 0.0, phi 23.33 | 0.344 / min11 0.338 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9053 / 0.9028; SD1.0: 0.9103 / 0.9076; SD2.0: 0.8960 / 0.8929 | - / - / - |
| F1 | global | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - |
| F1 | global | Tier 2 model-suggested | 295.00 / 280.00 / 280.50 | 1.0/5.7/38.0/48.7/5.6/1.1 (s1, g280_s1) | w/c 0.3273, CA 1492.0, SP 0.02, sand 268.7, phi 20.05 | 0.472 / min11 0.463 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9029 / 0.9023; SD1.0: 0.9146 / 0.9138; SD2.0: 0.8833 / 0.8823 | - |
| F2 | recorded | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - / - / - |
| F2 | recorded | Tier 2 model-suggested | 280.25 / 280.50 / 281.50 | 1.6/1.5/1.6/33.5/39.6/22.2 (s0, u290) | w/c 0.2723, CA 1595.6, SP 1.78, sand 0.0, phi 24.71 | 0.252 / min11 0.249 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9118 / 0.9033; SD1.0: 0.9136 / 0.9047; SD2.0: 0.9036 / 0.8947 | - / - / - |
| F2 | global | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - |
| F2 | global | Tier 2 model-suggested | 265.00 / 267.75 / 259.00 | 1.5/0.6/1.6/34.5/48.0/13.8 (s2, g275_s2) | w/c 0.2736, CA 1591.5, SP 1.88, sand 0.0, phi 22.07 | 0.246 / min11 0.243 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9036 / 0.9015; SD1.0: 0.9117 / 0.9082; SD2.0: 0.8894 / 0.8866 | - |

### T2 certified design of every (family, seed, source, tier)

| family | seed | source | tier | parent | C | blend six bins % | other | h | near passers | certificate SD 1.5 nominal / worst | SD checks | fallback index |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 0 | recorded | Tier 2 (model-suggested, never recommended) | u474 | 290.00 | 1.9/13.7/55.8/23.6/2.5/2.5 | w/c 0.2735, CA 1603.4, SP 1.95, sand 0.0, phi 21.13 | 0.464 / min11 0.454 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9046 / 0.9029 | SD1.5: 0.9046 / 0.9029; SD1.0: 0.9140 / 0.9127; SD2.0: 0.8887 / 0.8870 | 0 |
| F0 | 0 | global | Tier 2 (model-suggested, never recommended) | g335_s0 | 332.50 | 1.3/57.1/40.9/0.0/0.2/0.5 | w/c 0.3203, CA 1527.2, SP 0.02, sand 241.4, phi 22.48 | 0.811 / min11 0.803 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9047 / 0.9039 | SD1.5: 0.9047 / 0.9039 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F1 | 0 | recorded | Tier 2 (model-suggested, never recommended) | u358 | 318.10 | 0.0/0.0/30.8/38.3/23.8/7.1 | w/c 0.2722, CA 1622.1, SP 1.92, sand 0.0, phi 23.07 | 0.365 / min11 0.359 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9555 / 0.9548 | SD1.5: 0.9555 / 0.9548 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F1 | 0 | global | Tier 2 (model-suggested, never recommended) | g300_s0 | 295.00 | 0.0/19.6/21.9/34.4/13.9/10.2 | w/c 0.2844, CA 1582.0, SP 1.90, sand 0.1, phi 22.83 | 0.380 / min11 0.373 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9215 / 0.9142 | SD1.5: 0.9215 / 0.9142 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F2 | 0 | recorded | Tier 2 (model-suggested, never recommended) | u290 | 280.25 | 1.6/1.5/1.6/33.5/39.6/22.2 | w/c 0.2723, CA 1595.6, SP 1.78, sand 0.0, phi 24.71 | 0.252 / min11 0.249 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9118 / 0.9033 | SD1.5: 0.9118 / 0.9033; SD1.0: 0.9136 / 0.9047; SD2.0: 0.9036 / 0.8947 | 0 |
| F2 | 0 | global | Tier 2 (model-suggested, never recommended) | g270_s0 | 265.00 | 2.0/1.0/3.2/66.0/22.2/5.6 | w/c 0.2743, CA 1603.4, SP 1.89, sand 0.0, phi 20.39 | 0.299 / min11 0.294 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9051 / 0.9031 | SD1.5: 0.9051 / 0.9031 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F0 | 1 | recorded | Tier 2 (model-suggested, never recommended) | u474 | 308.75 | 1.4/5.7/78.4/11.6/1.5/1.5 | w/c 0.2697, CA 1497.6, SP 1.88, sand 0.0, phi 23.28 | 0.554 / min11 0.548 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9065 / 0.9044 | SD1.5: 0.9065 / 0.9044 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F0 | 1 | global | Tier 2 (model-suggested, never recommended) | g335_s1 | 327.75 | 0.9/65.9/33.2/0.0/0.0/0.0 | w/c 0.3270, CA 1533.2, SP 0.00, sand 241.3, phi 22.98 | 0.842 / min11 0.833 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9029 / 0.9021 | SD1.5: 0.9029 / 0.9021; SD1.0: 0.9120 / 0.9106; SD2.0: 0.8908 / 0.8893 | 0 |
| F1 | 1 | recorded | Tier 2 (model-suggested, never recommended) | u358 | 309.10 | 0.0/0.0/31.9/40.6/25.1/2.4 | w/c 0.2724, CA 1607.5, SP 1.92, sand 0.0, phi 23.13 | 0.378 / min11 0.372 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9501 / 0.9490 | SD1.5: 0.9501 / 0.9490 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F1 | 1 | global | Tier 2 (model-suggested, never recommended) | g280_s1 | 280.00 | 1.0/5.7/38.0/48.7/5.6/1.1 | w/c 0.3273, CA 1492.0, SP 0.02, sand 268.7, phi 20.05 | 0.472 / min11 0.463 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9029 / 0.9023 | SD1.5: 0.9029 / 0.9023; SD1.0: 0.9146 / 0.9138; SD2.0: 0.8833 / 0.8823 | 0 |
| F2 | 1 | recorded | Tier 2 (model-suggested, never recommended) | u241 | 280.50 | 1.4/1.4/1.1/56.0/29.7/10.3 | w/c 0.2703, CA 1626.3, SP 2.03, sand 0.0, phi 22.29 | 0.285 / min11 0.281 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9080 / 0.9043 | SD1.5: 0.9080 / 0.9043 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F2 | 1 | global | Tier 2 (model-suggested, never recommended) | g275_s1 | 267.75 | 1.2/0.7/0.3/57.3/37.5/2.8 | w/c 0.2841, CA 1566.7, SP 1.86, sand 0.0, phi 22.04 | 0.303 / min11 0.299 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9031 / 0.9020 | SD1.5: 0.9031 / 0.9020 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F0 | 2 | recorded | Tier 2 (model-suggested, never recommended) | u474 | 292.50 | 5.5/8.8/66.6/18.1/0.5/0.5 | w/c 0.2717, CA 1607.3, SP 1.98, sand 0.0, phi 20.97 | 0.507 / min11 0.501 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9075 / 0.9029 | SD1.5: 0.9075 / 0.9029 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F0 | 2 | global | Tier 2 (model-suggested, never recommended) | g340_s2 | 339.00 | 0.0/60.5/39.5/0.0/0.0/0.0 | w/c 0.3366, CA 1609.1, SP 0.02, sand 262.7, phi 27.12 | 0.821 / min11 0.813 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9036 / 0.9030 | SD1.5: 0.9036 / 0.9030 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F1 | 2 | recorded | Tier 2 (model-suggested, never recommended) | u358 | 293.10 | 0.0/0.0/30.7/38.0/25.5/5.8 | w/c 0.2752, CA 1611.0, SP 1.77, sand 0.0, phi 23.33 | 0.344 / min11 0.338 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9053 / 0.9028 | SD1.5: 0.9053 / 0.9028; SD1.0: 0.9103 / 0.9076; SD2.0: 0.8960 / 0.8929 | 0 |
| F1 | 2 | global | Tier 2 (model-suggested, never recommended) | g300_s2 | 280.50 | 2.6/2.9/38.0/46.8/9.7/0.0 | w/c 0.3302, CA 1491.9, SP 0.01, sand 252.0, phi 20.03 | 0.472 / min11 0.462 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9042 / 0.9038 | SD1.5: 0.9042 / 0.9038 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F2 | 2 | recorded | Tier 2 (model-suggested, never recommended) | u290 | 281.50 | 1.6/1.6/1.8/33.7/39.9/21.4 | w/c 0.2701, CA 1608.1, SP 1.78, sand 0.0, phi 24.14 | 0.252 / min11 0.249 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9052 / 0.9036 | SD1.5: 0.9052 / 0.9036 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F2 | 2 | global | Tier 2 (model-suggested, never recommended) | g275_s2 | 259.00 | 1.5/0.6/1.6/34.5/48.0/13.8 | w/c 0.2736, CA 1591.5, SP 1.88, sand 0.0, phi 22.07 | 0.246 / min11 0.243 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9036 / 0.9015 | SD1.5: 0.9036 / 0.9015; SD1.0: 0.9117 / 0.9082; SD2.0: 0.8894 / 0.8866 | 0 |

### T3 CONVERGENCE: do global-seeded and recorded-parent walks end in the same place? (same = cement within 5 kg/m3 AND blend W1 <= 0.02)

Compared per (family, seed): (A) lowest CERTIFIED Tier-1 design of each source; (B) lowest ACCEPTED step of each source (any tier, before certificates).

| family | seed | (A) recorded C | (A) global C | dC | W1 of blends | same? | (B) recorded lowest accepted | (B) global lowest accepted | dC | W1 | same? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 0 | - | - | - | - | n/a (neither source) | 290.00 | 332.50 | +42.50 | 0.2710 | DIFFERENT |
| F0 | 1 | - | - | - | - | n/a (neither source) | 308.75 | 327.75 | +19.00 | 0.2757 | DIFFERENT |
| F0 | 2 | - | - | - | - | n/a (neither source) | 292.50 | 339.00 | +46.50 | 0.2507 | DIFFERENT |
| F1 | 0 | - | - | - | - | n/a (neither source) | 318.10 | 295.00 | -23.10 | 0.1433 | DIFFERENT |
| F1 | 1 | - | - | - | - | n/a (neither source) | 309.10 | 280.00 | -29.10 | 0.1497 | DIFFERENT |
| F1 | 2 | - | - | - | - | n/a (neither source) | 293.10 | 280.50 | -12.60 | 0.1746 | DIFFERENT |
| F2 | 0 | - | - | - | - | n/a (neither source) | 280.25 | 265.00 | -15.25 | 0.2059 | DIFFERENT |
| F2 | 1 | - | - | - | - | n/a (neither source) | 280.50 | 267.75 | -12.75 | 0.0480 | DIFFERENT |
| F2 | 2 | - | - | - | - | n/a (neither source) | 281.50 | 259.00 | -22.50 | 0.0475 | DIFFERENT |

Convergence verdict (A, certified Tier-1 designs): 0 of 0 comparable (family, seed) cases end in the same place. Cases that differ:
- F0 seed 0: (A) n/a (neither source); (B) DIFFERENT dC +42.50 W1 0.2710
- F0 seed 1: (A) n/a (neither source); (B) DIFFERENT dC +19.00 W1 0.2757
- F0 seed 2: (A) n/a (neither source); (B) DIFFERENT dC +46.50 W1 0.2507
- F1 seed 0: (A) n/a (neither source); (B) DIFFERENT dC -23.10 W1 0.1433
- F1 seed 1: (A) n/a (neither source); (B) DIFFERENT dC -29.10 W1 0.1497
- F1 seed 2: (A) n/a (neither source); (B) DIFFERENT dC -12.60 W1 0.1746
- F2 seed 0: (A) n/a (neither source); (B) DIFFERENT dC -15.25 W1 0.2059
- F2 seed 1: (A) n/a (neither source); (B) DIFFERENT dC -12.75 W1 0.0480
- F2 seed 2: (A) n/a (neither source); (B) DIFFERENT dC -22.50 W1 0.0475

### T4 recommended design

none