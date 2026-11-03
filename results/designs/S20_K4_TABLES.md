**VERDICT (S* 20 MPa, K* 4 mm/s, R* 0.90, integrated global + recorded walk): none, because no certified Tier-1 (data-backed) design exists; every design below is model-suggested (Tier 2) and none may be recommended.**

Target S* 20 MPa, K* 4 mm/s, R* 0.90. Config b51f91a4ea1b725e388dc5071d6d60b1e56ab0db3bab448d4a134b67adccc1cd. Paste floor 0.207267 mm (unit u235, 45 recorded unit(s) meet the target). Tier rule: Tier 1 only if at least one recorded passing mix is near the design (n_min = 1, frozen near radii, window-clamped mixes never count); everything else Tier 2 model-suggested, never recommended. Baseline = the existing walk of this target (stage\), unchanged.

**FLAG: a global-seeded design is more than 10 kg/m3 cheaper than the best recorded-parent walk design of its family - treat as MODEL-SUGGESTED until checked in the lab:**
- F1 seed 0 global: 355.00 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9144 / 0.9031
- F2 seed 0 global: 319.00 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9030 / 0.9013
- F0 seed 1 global: 376.25 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9207 / 0.9173
- F1 seed 1 global: 351.25 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9176 / 0.9032
- F2 seed 1 global: 318.75 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9058 / 0.9044
- F0 seed 2 global: 382.50 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9037 / 0.9014
- F1 seed 2 global: 348.75 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9131 / 0.9017
- F2 seed 2 global: 334.00 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9044 / 0.9022

### G0 global search only (best design per family over all seeds and levels; full curve in global/global_scan.csv)

| family | lowest passing level (any seed) | seeds passing there | best min-LB at that level (nominal / worst) | tier of that design | levels scanned | passes at (all levels) |
|---|---|---|---|---|---|---|
| F0 | 435 | 2 of 6 | 0.9712 (0.9717 / 0.9712) | Tier 2 (near passers 0) | 440, 435, 430, 420, 400 | 435, 440 |
| F1 | 365 | 4 of 6 | 0.9578 (0.9607 / 0.9578) | Tier 2 (near passers 0) | 420, 400, 380, 375, 370, 365, 360, 340 | 365, 370, 375, 380, 400, 420 |
| F2 | 345 | 2 of 6 | 0.9595 (0.9606 / 0.9595) | Tier 2 (near passers 0) | 420, 400, 380, 360, 355, 350, 345, 340, 320 | 345, 350, 355, 360, 380, 400, 420 |

### T0 per (family, seed, source): log-derived minima (cement kg/m3) and certificate status

| family | seed | source | parents | attempts | finished | lowest accepted (any tier) | lowest accepted with >= 1 near passer (Tier-1 candidate) | n2 | n3 | status | certified Tier 1 | certified Tier 2 (model-suggested) | baseline walk certified | failing-check counts g/f/s/p/r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 0 | recorded | 12 | 175 | True | 421.75 | - | - | - | model-suggested only (Tier 2) certified | - | 421.75 | - | 29 / 15 / 2 / 0 / 146 |
| F0 | 1 | recorded | 12 | 196 | True | 382.50 | - | - | - | model-suggested only (Tier 2) certified | - | 382.50 | - | 37 / 21 / 5 / 0 / 135 |
| F0 | 1 | global | 1 | 28 | True | 376.25 | - | - | - | model-suggested only (Tier 2) certified | - | 376.25 | - | 12 / 10 / 0 / 0 / 0 |
| F0 | 2 | recorded | 12 | 240 | True | 362.75 | - | - | - | model-suggested only (Tier 2) certified | - | 362.75 | - | 26 / 7 / 12 / 0 / 158 |
| F0 | 2 | global | 1 | 27 | True | 382.50 | - | - | - | model-suggested only (Tier 2) certified | - | 382.50 | - | 0 / 0 / 0 / 0 / 12 |
| F1 | 0 | recorded | 12 | 199 | True | 378.25 | - | - | - | model-suggested only (Tier 2) certified | - | 378.25 | 378.25 | 90 / 78 / 4 / 0 / 102 |
| F1 | 0 | global | 2 | 28 | True | 355.00 | - | - | - | model-suggested only (Tier 2) certified | - | 355.00 | - | 0 / 0 / 0 / 0 / 20 |
| F1 | 1 | recorded | 12 | 203 | True | 364.75 | - | - | - | model-suggested only (Tier 2) certified | - | 364.75 | 364.75 | 92 / 70 / 5 / 0 / 103 |
| F1 | 1 | global | 1 | 21 | True | 351.25 | - | - | - | model-suggested only (Tier 2) certified | - | 351.25 | - | 0 / 0 / 0 / 0 / 12 |
| F1 | 2 | recorded | 12 | 414 | True | 368.25 | - | - | - | model-suggested only (Tier 2) certified | - | 368.25 | 368.25 | 150 / 130 / 8 / 0 / 230 |
| F1 | 2 | global | 2 | 38 | True | 348.75 | - | - | - | model-suggested only (Tier 2) certified | - | 348.75 | - | 0 / 0 / 0 / 0 / 24 |
| F2 | 0 | recorded | 12 | 203 | True | 320.25 | - | - | - | model-suggested only (Tier 2) certified | - | 320.25 | - | 38 / 0 / 2 / 1 / 110 |
| F2 | 0 | global | 2 | 40 | True | 319.00 | - | - | - | model-suggested only (Tier 2) certified | - | 319.00 | - | 4 / 0 / 0 / 0 / 26 |
| F2 | 1 | recorded | 12 | 175 | True | 328.00 | - | - | - | model-suggested only (Tier 2) certified | - | 328.00 | - | 25 / 0 / 1 / 1 / 118 |
| F2 | 1 | global | 1 | 23 | True | 318.75 | - | - | - | model-suggested only (Tier 2) certified | - | 318.75 | - | 0 / 0 / 0 / 0 / 12 |
| F2 | 2 | recorded | 12 | 334 | True | 320.25 | - | - | - | model-suggested only (Tier 2) certified | - | 320.25 | - | 29 / 0 / 3 / 2 / 232 |
| F2 | 2 | global | 1 | 25 | True | 334.00 | - | - | - | model-suggested only (Tier 2) certified | - | 334.00 | - | 0 / 0 / 0 / 0 / 12 |

### T1 FAMILY x SOURCE TABLE (lowest certified design per seed; Tier-1 designs unless marked)

| family | source | tier | lowest certified C per seed s0 / s1 / s2 | blend six bins % (lowest seed design) | other mix values | h vs floor | near passers (n>=1; n>=2, n>=3 beside) | certificates nominal / worst (min of 2 reps) at SD 1.5; SD 1.0 and 2.0 | baseline walk certified C per seed (recorded source) |
|---|---|---|---|---|---|---|---|---|---|
| F0 | recorded | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - / - / - |
| F0 | recorded | Tier 2 model-suggested | 421.75 / 382.50 / 362.75 | 0.8/13.4/54.5/22.4/4.5/4.4 (s2, u529) | w/c 0.2673, CA 1584.6, SP 1.86, sand 0.0, phi 20.24 | 0.551 / min11 0.540 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9117 / 0.9029; SD1.0: 0.9208 / 0.9136; SD2.0: 0.8944 / 0.8844 | - / - / - |
| F0 | global | Tier 1 | none certified (no run, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - |
| F0 | global | Tier 2 model-suggested | - / 376.25 / 382.50 | 0.0/18.1/62.8/13.5/2.8/2.8 (s1, g435_s1) | w/c 0.2666, CA 1585.6, SP 1.90, sand 0.0, phi 20.41 | 0.622 / min11 0.608 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9207 / 0.9173; SD1.0: 0.9288 / 0.9251; SD2.0: 0.9044 / 0.9004 | - |
| F1 | recorded | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | 378.25 / 364.75 / 368.25 |
| F1 | recorded | Tier 2 model-suggested | 378.25 / 364.75 / 368.25 | 0.0/6.7/23.8/37.4/23.0/9.0 (s1, u472) | w/c 0.2682, CA 1578.7, SP 1.99, sand 0.0, phi 21.43 | 0.424 / min11 0.417 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9054 / 0.9017; SD1.0: 0.9098 / 0.9065; SD2.0: 0.8973 / 0.8934 | 378.25 / 364.75 / 368.25 |
| F1 | global | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - |
| F1 | global | Tier 2 model-suggested | 355.00 / 351.25 / 348.75 | 1.7/14.8/35.7/27.8/12.7/7.4 (s2, g365_s2) | w/c 0.2673, CA 1586.2, SP 1.97, sand 0.0, phi 20.00 | 0.466 / min11 0.457 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9131 / 0.9017; SD1.0: 0.9207 / 0.9034; SD2.0: 0.8973 / 0.8890 | - |
| F2 | recorded | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - / - / - |
| F2 | recorded | Tier 2 model-suggested | 320.25 / 328.00 / 320.25 | 0.0/0.0/0.0/59.1/40.9/0.0 (s0, u124) | w/c 0.2676, CA 1590.1, SP 1.92, sand 0.0, phi 20.34 | 0.351 / min11 0.345 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9039 / 0.9027; SD1.0: 0.9101 / 0.9088; SD2.0: 0.8929 / 0.8911 | - / - / - |
| F2 | global | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - |
| F2 | global | Tier 2 model-suggested | 319.00 / 318.75 / 334.00 | 1.4/1.0/1.0/51.2/27.7/17.7 (s1, g355_s1) | w/c 0.2715, CA 1586.3, SP 1.83, sand 0.0, phi 20.11 | 0.310 / min11 0.306 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9058 / 0.9044; SD1.0: 0.9107 / 0.9087; SD2.0: 0.8978 / 0.8969 | - |

### T2 certified design of every (family, seed, source, tier)

| family | seed | source | tier | parent | C | blend six bins % | other | h | near passers | certificate SD 1.5 nominal / worst | SD checks | fallback index |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 0 | recorded | Tier 2 (model-suggested, never recommended) | u529 | 421.75 | 0.0/19.4/56.7/19.9/2.0/2.0 | w/c 0.2676, CA 1579.8, SP 1.88, sand 0.0, phi 20.78 | 0.703 / min11 0.687 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | 0.9700 / 0.9695 | SD1.5: 0.9700 / 0.9695 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F1 | 0 | recorded | Tier 2 (model-suggested, never recommended) | u472 | 378.25 | 0.0/2.6/29.7/37.5/22.1/8.1 | w/c 0.2686, CA 1600.4, SP 2.11, sand 0.0, phi 21.85 | 0.439 / min11 0.432 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | 0.9056 / 0.9022 | SD1.5: 0.9056 / 0.9022 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F1 | 0 | global | Tier 2 (model-suggested, never recommended) | g365_s0 | 355.00 | 0.1/25.4/25.1/21.5/21.3/6.5 | w/c 0.2664, CA 1591.9, SP 1.94, sand 0.0, phi 20.00 | 0.466 / min11 0.458 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | 0.9144 / 0.9031 | SD1.5: 0.9144 / 0.9031 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F2 | 0 | recorded | Tier 2 (model-suggested, never recommended) | u124 | 320.25 | 0.0/0.0/0.0/59.1/40.9/0.0 | w/c 0.2676, CA 1590.1, SP 1.92, sand 0.0, phi 20.34 | 0.351 / min11 0.345 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | 0.9039 / 0.9027 | SD1.5: 0.9039 / 0.9027; SD1.0: 0.9101 / 0.9088; SD2.0: 0.8929 / 0.8911 | 0 |
| F2 | 0 | global | Tier 2 (model-suggested, never recommended) | g345_s0 | 319.00 | 1.7/1.7/0.2/52.2/39.1/5.0 | w/c 0.2642, CA 1582.9, SP 1.69, sand 0.0, phi 20.00 | 0.338 / min11 0.333 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | 0.9030 / 0.9013 | SD1.5: 0.9030 / 0.9013 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F0 | 1 | recorded | Tier 2 (model-suggested, never recommended) | u534 | 382.50 | 0.0/17.5/60.2/15.8/4.5/2.0 | w/c 0.2682, CA 1584.3, SP 1.90, sand 0.0, phi 20.38 | 0.629 / min11 0.615 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | 0.9434 / 0.9408 | SD1.5: 0.9434 / 0.9408 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F0 | 1 | global | Tier 2 (model-suggested, never recommended) | g435_s1 | 376.25 | 0.0/18.1/62.8/13.5/2.8/2.8 | w/c 0.2666, CA 1585.6, SP 1.90, sand 0.0, phi 20.41 | 0.622 / min11 0.608 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | 0.9207 / 0.9173 | SD1.5: 0.9207 / 0.9173; SD1.0: 0.9288 / 0.9251; SD2.0: 0.9044 / 0.9004 | 0 |
| F1 | 1 | recorded | Tier 2 (model-suggested, never recommended) | u472 | 364.75 | 0.0/6.7/23.8/37.4/23.0/9.0 | w/c 0.2682, CA 1578.7, SP 1.99, sand 0.0, phi 21.43 | 0.424 / min11 0.417 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | 0.9054 / 0.9017 | SD1.5: 0.9054 / 0.9017; SD1.0: 0.9098 / 0.9065; SD2.0: 0.8973 / 0.8934 | 0 |
| F1 | 1 | global | Tier 2 (model-suggested, never recommended) | g370_s1 | 351.25 | 0.6/16.3/35.3/29.6/8.5/9.6 | w/c 0.2686, CA 1565.3, SP 1.83, sand 0.0, phi 20.00 | 0.470 / min11 0.462 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | 0.9176 / 0.9032 | SD1.5: 0.9176 / 0.9032 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F2 | 1 | recorded | Tier 2 (model-suggested, never recommended) | u241 | 328.00 | 1.6/1.6/1.8/52.9/34.3/7.8 | w/c 0.2674, CA 1603.2, SP 1.94, sand 0.0, phi 20.69 | 0.341 / min11 0.336 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | 0.9056 / 0.9037 | SD1.5: 0.9056 / 0.9037 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F2 | 1 | global | Tier 2 (model-suggested, never recommended) | g355_s1 | 318.75 | 1.4/1.0/1.0/51.2/27.7/17.7 | w/c 0.2715, CA 1586.3, SP 1.83, sand 0.0, phi 20.11 | 0.310 / min11 0.306 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | 0.9058 / 0.9044 | SD1.5: 0.9058 / 0.9044; SD1.0: 0.9107 / 0.9087; SD2.0: 0.8978 / 0.8969 | 0 |
| F0 | 2 | recorded | Tier 2 (model-suggested, never recommended) | u529 | 362.75 | 0.8/13.4/54.5/22.4/4.5/4.4 | w/c 0.2673, CA 1584.6, SP 1.86, sand 0.0, phi 20.24 | 0.551 / min11 0.540 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | 0.9117 / 0.9029 | SD1.5: 0.9117 / 0.9029; SD1.0: 0.9208 / 0.9136; SD2.0: 0.8944 / 0.8844 | 0 |
| F0 | 2 | global | Tier 2 (model-suggested, never recommended) | g440_s2 | 382.50 | 0.0/33.1/61.4/1.8/1.8/1.9 | w/c 0.2656, CA 1605.1, SP 1.96, sand 0.0, phi 20.46 | 0.702 / min11 0.696 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | 0.9037 / 0.9014 | SD1.5: 0.9037 / 0.9014 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F1 | 2 | recorded | Tier 2 (model-suggested, never recommended) | u472 | 368.25 | 0.6/4.1/26.1/39.5/21.6/8.2 | w/c 0.2667, CA 1598.7, SP 2.01, sand 0.0, phi 21.32 | 0.427 / min11 0.420 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | 0.9051 / 0.9020 | SD1.5: 0.9051 / 0.9020 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F1 | 2 | global | Tier 2 (model-suggested, never recommended) | g365_s2 | 348.75 | 1.7/14.8/35.7/27.8/12.7/7.4 | w/c 0.2673, CA 1586.2, SP 1.97, sand 0.0, phi 20.00 | 0.466 / min11 0.457 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | 0.9131 / 0.9017 | SD1.5: 0.9131 / 0.9017; SD1.0: 0.9207 / 0.9034; SD2.0: 0.8973 / 0.8890 | 0 |
| F2 | 2 | recorded | Tier 2 (model-suggested, never recommended) | u124 | 320.25 | 0.0/0.0/0.0/59.1/40.9/0.0 | w/c 0.2663, CA 1587.3, SP 1.93, sand 0.0, phi 20.34 | 0.351 / min11 0.345 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | 0.9069 / 0.9046 | SD1.5: 0.9069 / 0.9046 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F2 | 2 | global | Tier 2 (model-suggested, never recommended) | g380_s2 | 334.00 | 0.7/0.4/0.1/61.9/36.5/0.4 | w/c 0.2719, CA 1598.8, SP 1.72, sand 0.0, phi 21.26 | 0.373 / min11 0.367 vs floor 0.2073 | 0 (n>=2: False, n>=3: False) | 0.9044 / 0.9022 | SD1.5: 0.9044 / 0.9022 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |

### T3 CONVERGENCE: do global-seeded and recorded-parent walks end in the same place? (same = cement within 5 kg/m3 AND blend W1 <= 0.02)

Compared per (family, seed): (A) lowest CERTIFIED Tier-1 design of each source; (B) lowest ACCEPTED step of each source (any tier, before certificates).

| family | seed | (A) recorded C | (A) global C | dC | W1 of blends | same? | (B) recorded lowest accepted | (B) global lowest accepted | dC | W1 | same? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 0 | - | - | - | - | n/a (neither source) | 421.75 | - | - | - | DIFFERENT (only one source has a design) |
| F0 | 1 | - | - | - | - | n/a (neither source) | 382.50 | 376.25 | -6.25 | 0.0198 | DIFFERENT |
| F0 | 2 | - | - | - | - | n/a (neither source) | 362.75 | 382.50 | +19.75 | 0.1863 | DIFFERENT |
| F1 | 0 | - | - | - | - | n/a (neither source) | 378.25 | 355.00 | -23.25 | 0.1591 | DIFFERENT |
| F1 | 1 | - | - | - | - | n/a (neither source) | 364.75 | 351.25 | -13.50 | 0.1632 | DIFFERENT |
| F1 | 2 | - | - | - | - | n/a (neither source) | 368.25 | 348.75 | -19.50 | 0.1558 | DIFFERENT |
| F2 | 0 | - | - | - | - | n/a (neither source) | 320.25 | 319.00 | -1.25 | 0.0659 | DIFFERENT |
| F2 | 1 | - | - | - | - | n/a (neither source) | 328.00 | 318.75 | -9.25 | 0.0679 | DIFFERENT |
| F2 | 2 | - | - | - | - | n/a (neither source) | 320.25 | 334.00 | +13.75 | 0.0266 | DIFFERENT |

Convergence verdict (A, certified Tier-1 designs): 0 of 0 comparable (family, seed) cases end in the same place. Cases that differ:
- F0 seed 0: (A) n/a (neither source); (B) DIFFERENT (only one source has a design)
- F0 seed 1: (A) n/a (neither source); (B) DIFFERENT dC -6.25 W1 0.0198
- F0 seed 2: (A) n/a (neither source); (B) DIFFERENT dC +19.75 W1 0.1863
- F1 seed 0: (A) n/a (neither source); (B) DIFFERENT dC -23.25 W1 0.1591
- F1 seed 1: (A) n/a (neither source); (B) DIFFERENT dC -13.50 W1 0.1632
- F1 seed 2: (A) n/a (neither source); (B) DIFFERENT dC -19.50 W1 0.1558
- F2 seed 0: (A) n/a (neither source); (B) DIFFERENT dC -1.25 W1 0.0659
- F2 seed 1: (A) n/a (neither source); (B) DIFFERENT dC -9.25 W1 0.0679
- F2 seed 2: (A) n/a (neither source); (B) DIFFERENT dC +13.75 W1 0.0266

### T4 recommended design

none