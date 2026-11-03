**VERDICT (S* 16 MPa, K* 10 mm/s, R* 0.90, integrated global + recorded walk): none, because no certified Tier-1 (data-backed) design exists; every design below is model-suggested (Tier 2) and none may be recommended.**

Target S* 16 MPa, K* 10 mm/s, R* 0.90. Config 976bcf8395bbd345cc9d5467aa2070ea4e5d6c278b810e367f137c8f744dc0ef. Paste floor 0.289498 mm (unit u63, 28 recorded unit(s) meet the target). Tier rule: Tier 1 only if at least one recorded passing mix is near the design (n_min = 1, frozen near radii, window-clamped mixes never count); everything else Tier 2 model-suggested, never recommended. Baseline = the existing walk of this target (stage\), unchanged.

**FLAG: a global-seeded design is more than 10 kg/m3 cheaper than the best recorded-parent walk design of its family - treat as MODEL-SUGGESTED until checked in the lab:**
- F0 seed 0 global: 338.00 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9017 / 0.9013
- F1 seed 0 global: 337.50 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9067 / 0.9020
- F2 seed 0 global: 316.25 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9084 / 0.9024
- F0 seed 1 global: 352.75 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9017 / 0.9011
- F1 seed 1 global: 333.75 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9129 / 0.9044
- F2 seed 1 global: 318.75 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9067 / 0.9029
- F0 seed 2 global: 340.00 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9051 / 0.9032
- F1 seed 2 global: 335.00 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9085 / 0.9023
- F2 seed 2 global: 322.50 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9131 / 0.9039

### G0 global search only (best design per family over all seeds and levels; full curve in global/global_scan.csv)

| family | lowest passing level (any seed) | seeds passing there | best min-LB at that level (nominal / worst) | tier of that design | levels scanned | passes at (all levels) |
|---|---|---|---|---|---|---|
| F0 | 340 | 1 of 3 | 0.9022 (0.9043 / 0.9022) | Tier 2 (near passers 0) | 420, 400, 380, 360, 340, 335, 320, 300 | 340, 360, 380, 400, 420 |
| F1 | 340 | 2 of 3 | 0.9207 (0.9247 / 0.9207) | Tier 2 (near passers 0) | 420, 400, 380, 360, 340, 335, 320, 300 | 340, 360, 380, 400, 420 |
| F2 | 320 | 2 of 3 | 0.9127 (0.9179 / 0.9127) | Tier 2 (near passers 0) | 420, 400, 380, 360, 340, 320, 315, 300, 280 | 320, 340, 360, 380, 400, 420 |

### T0 per (family, seed, source): log-derived minima (cement kg/m3) and certificate status

| family | seed | source | parents | attempts | finished | lowest accepted (any tier) | lowest accepted with >= 1 near passer (Tier-1 candidate) | n2 | n3 | status | certified Tier 1 | certified Tier 2 (model-suggested) | baseline walk certified | failing-check counts g/f/s/p/r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 0 | recorded | 12 | 154 | True | 372.75 | - | - | - | model-suggested only (Tier 2) certified | - | 372.75 | - | 36 / 9 / 6 / 0 / 151 |
| F0 | 0 | global | 1 | 15 | True | 338.00 | - | - | - | model-suggested only (Tier 2) certified | - | 338.00 | - | 0 / 0 / 0 / 0 / 7 |
| F0 | 1 | recorded | 12 | 154 | True | 372.75 | - | - | - | model-suggested only (Tier 2) certified | - | 372.75 | - | 41 / 10 / 5 / 0 / 151 |
| F0 | 1 | global | 1 | 10 | True | 352.75 | - | - | - | model-suggested only (Tier 2) certified | - | 352.75 | - | 0 / 0 / 0 / 0 / 6 |
| F0 | 2 | recorded | 12 | 158 | True | 375.25 | - | - | - | model-suggested only (Tier 2) certified | - | 375.25 | - | 37 / 10 / 6 / 0 / 153 |
| F0 | 2 | global | 2 | 18 | True | 340.00 | - | - | - | model-suggested only (Tier 2) certified | - | 340.00 | - | 0 / 0 / 0 / 0 / 12 |
| F1 | 0 | recorded | 10 | 132 | True | 371.70 | - | - | - | model-suggested only (Tier 2) certified | - | 371.70 | - | 60 / 43 / 6 / 0 / 108 |
| F1 | 0 | global | 1 | 10 | True | 337.50 | - | - | - | model-suggested only (Tier 2) certified | - | 337.50 | - | 0 / 0 / 0 / 0 / 6 |
| F1 | 1 | recorded | 10 | 160 | True | 367.85 | - | - | - | model-suggested only (Tier 2) certified | - | 367.85 | - | 56 / 42 / 2 / 0 / 118 |
| F1 | 1 | global | 2 | 20 | True | 333.75 | - | - | - | model-suggested only (Tier 2) certified | - | 333.75 | - | 0 / 0 / 0 / 0 / 12 |
| F1 | 2 | recorded | 10 | 132 | True | 368.20 | - | - | - | model-suggested only (Tier 2) certified | - | 368.20 | - | 53 / 38 / 1 / 0 / 108 |
| F1 | 2 | global | 2 | 19 | True | 335.00 | - | - | - | model-suggested only (Tier 2) certified | - | 335.00 | - | 0 / 0 / 0 / 0 / 12 |
| F2 | 0 | recorded | 12 | 164 | True | 331.00 | - | - | - | model-suggested only (Tier 2) certified | - | 331.00 | - | 24 / 5 / 24 / 37 / 142 |
| F2 | 0 | global | 2 | 22 | True | 316.25 | - | - | - | model-suggested only (Tier 2) certified | - | 316.25 | - | 0 / 0 / 0 / 0 / 13 |
| F2 | 1 | recorded | 12 | 170 | True | 327.75 | - | - | - | model-suggested only (Tier 2) certified | - | 327.75 | - | 28 / 14 / 25 / 40 / 146 |
| F2 | 1 | global | 2 | 17 | True | 318.75 | - | - | - | model-suggested only (Tier 2) certified | - | 318.75 | - | 0 / 0 / 0 / 0 / 12 |
| F2 | 2 | recorded | 12 | 166 | True | 320.00 | - | - | - | model-suggested only (Tier 2) certified | - | 320.00 | - | 18 / 6 / 25 / 36 / 149 |
| F2 | 2 | global | 1 | 10 | True | 322.50 | - | - | - | model-suggested only (Tier 2) certified | - | 322.50 | - | 0 / 0 / 0 / 0 / 6 |

### T1 FAMILY x SOURCE TABLE (lowest certified design per seed; Tier-1 designs unless marked)

| family | source | tier | lowest certified C per seed s0 / s1 / s2 | blend six bins % (lowest seed design) | other mix values | h vs floor | near passers (n>=1; n>=2, n>=3 beside) | certificates nominal / worst (min of 2 reps) at SD 1.5; SD 1.0 and 2.0 | baseline walk certified C per seed (recorded source) |
|---|---|---|---|---|---|---|---|---|---|
| F0 | recorded | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - / - / - |
| F0 | recorded | Tier 2 model-suggested | 372.75 / 372.75 / 375.25 | 1.3/28.1/35.3/24.5/5.4/5.4 (s0, u289) | w/c 0.2658, CA 1588.7, SP 2.06, sand 0.0, phi 24.36 | 0.565 / min11 0.553 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9035 / 0.9012; SD1.0: 0.9198 / 0.9178; SD2.0: 0.8750 / 0.8721 | - / - / - |
| F0 | global | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - |
| F0 | global | Tier 2 model-suggested | 338.00 / 352.75 / 340.00 | 1.9/1.9/76.7/17.3/1.0/1.1 (s0, g380_s0) | w/c 0.2731, CA 1618.0, SP 1.82, sand 0.0, phi 23.07 | 0.554 / min11 0.549 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9017 / 0.9013; SD1.0: 0.9193 / 0.9178; SD2.0: 0.8742 / 0.8736 | - |
| F1 | recorded | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - / - / - |
| F1 | recorded | Tier 2 model-suggested | 371.70 / 367.85 / 368.20 | 0.0/0.0/34.3/40.2/14.2/11.4 (s1, u358) | w/c 0.2704, CA 1647.5, SP 1.89, sand 0.0, phi 24.24 | 0.412 / min11 0.406 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9090 / 0.9028; SD1.0: 0.9055 / 0.9014; SD2.0: 0.9070 / 0.8999 | - / - / - |
| F1 | global | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - |
| F1 | global | Tier 2 model-suggested | 337.50 / 333.75 / 335.00 | 1.9/11.1/46.9/23.7/7.2/9.2 (s1, g340_s1) | w/c 0.2719, CA 1649.0, SP 1.79, sand 0.0, phi 22.01 | 0.438 / min11 0.430 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9129 / 0.9044; SD1.0: 0.9290 / 0.9244; SD2.0: 0.8861 / 0.8777 | - |
| F2 | recorded | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - / - / - |
| F2 | recorded | Tier 2 model-suggested | 331.00 / 327.75 / 320.00 | 1.4/1.4/1.0/49.9/32.3/14.0 (s2, u39) | w/c 0.2746, CA 1627.2, SP 1.82, sand 0.0, phi 23.14 | 0.312 / min11 0.308 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9084 / 0.9021; SD1.0: 0.9212 / 0.9131; SD2.0: 0.8841 / 0.8788 | - / - / - |
| F2 | global | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - |
| F2 | global | Tier 2 model-suggested | 316.25 / 318.75 / 322.50 | 1.6/1.1/3.5/60.0/16.9/16.8 (s0, g320_s0) | w/c 0.2708, CA 1622.6, SP 1.87, sand 0.0, phi 22.55 | 0.316 / min11 0.311 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9084 / 0.9024; SD1.0: 0.9286 / 0.9203; SD2.0: 0.8763 / 0.8709 | - |

### T2 certified design of every (family, seed, source, tier)

| family | seed | source | tier | parent | C | blend six bins % | other | h | near passers | certificate SD 1.5 nominal / worst | SD checks | fallback index |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 0 | recorded | Tier 2 (model-suggested, never recommended) | u289 | 372.75 | 1.3/28.1/35.3/24.5/5.4/5.4 | w/c 0.2658, CA 1588.7, SP 2.06, sand 0.0, phi 24.36 | 0.565 / min11 0.553 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | 0.9035 / 0.9012 | SD1.5: 0.9035 / 0.9012; SD1.0: 0.9198 / 0.9178; SD2.0: 0.8750 / 0.8721 | 0 |
| F0 | 0 | global | Tier 2 (model-suggested, never recommended) | g380_s0 | 338.00 | 1.9/1.9/76.7/17.3/1.0/1.1 | w/c 0.2731, CA 1618.0, SP 1.82, sand 0.0, phi 23.07 | 0.554 / min11 0.549 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | 0.9017 / 0.9013 | SD1.5: 0.9017 / 0.9013; SD1.0: 0.9193 / 0.9178; SD2.0: 0.8742 / 0.8736 | 0 |
| F1 | 0 | recorded | Tier 2 (model-suggested, never recommended) | u415 | 371.70 | 1.6/1.6/34.7/33.3/22.4/6.3 | w/c 0.2737, CA 1655.1, SP 1.88, sand 0.0, phi 23.98 | 0.438 / min11 0.431 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | 0.9047 / 0.9024 | SD1.5: 0.9047 / 0.9024 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F1 | 0 | global | Tier 2 (model-suggested, never recommended) | g360_s0 | 337.50 | 0.9/19.1/20.7/34.6/15.4/9.3 | w/c 0.2717, CA 1655.3, SP 1.85, sand 0.1, phi 23.54 | 0.408 / min11 0.401 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | 0.9067 / 0.9020 | SD1.5: 0.9067 / 0.9020 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F2 | 0 | recorded | Tier 2 (model-suggested, never recommended) | u63 | 331.00 | 0.1/0.1/0.1/46.0/42.0/11.7 | w/c 0.2724, CA 1588.7, SP 1.94, sand 0.0, phi 25.22 | 0.323 / min11 0.318 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | 0.9059 / 0.9022 | SD1.5: 0.9059 / 0.9022 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F2 | 0 | global | Tier 2 (model-suggested, never recommended) | g320_s0 | 316.25 | 1.6/1.1/3.5/60.0/16.9/16.8 | w/c 0.2708, CA 1622.6, SP 1.87, sand 0.0, phi 22.55 | 0.316 / min11 0.311 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | 0.9084 / 0.9024 | SD1.5: 0.9084 / 0.9024; SD1.0: 0.9286 / 0.9203; SD2.0: 0.8763 / 0.8709 | 0 |
| F0 | 1 | recorded | Tier 2 (model-suggested, never recommended) | u289 | 372.75 | 1.0/27.8/35.9/24.3/5.4/5.5 | w/c 0.2658, CA 1588.7, SP 2.05, sand 0.0, phi 24.26 | 0.564 / min11 0.552 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | 0.9051 / 0.9031 | SD1.5: 0.9051 / 0.9031 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F0 | 1 | global | Tier 2 (model-suggested, never recommended) | g360_s1 | 352.75 | 2.3/2.7/85.2/4.3/2.8/2.8 | w/c 0.2743, CA 1631.6, SP 1.80, sand 0.0, phi 24.14 | 0.577 / min11 0.564 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | 0.9017 / 0.9011 | SD1.5: 0.9017 / 0.9011 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F1 | 1 | recorded | Tier 2 (model-suggested, never recommended) | u358 | 367.85 | 0.0/0.0/34.3/40.2/14.2/11.4 | w/c 0.2704, CA 1647.5, SP 1.89, sand 0.0, phi 24.24 | 0.412 / min11 0.406 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | 0.9090 / 0.9028 | SD1.5: 0.9090 / 0.9028; SD1.0: 0.9055 / 0.9014; SD2.0: 0.9070 / 0.8999 | 0 |
| F1 | 1 | global | Tier 2 (model-suggested, never recommended) | g340_s1 | 333.75 | 1.9/11.1/46.9/23.7/7.2/9.2 | w/c 0.2719, CA 1649.0, SP 1.79, sand 0.0, phi 22.01 | 0.438 / min11 0.430 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | 0.9129 / 0.9044 | SD1.5: 0.9129 / 0.9044; SD1.0: 0.9290 / 0.9244; SD2.0: 0.8861 / 0.8777 | 0 |
| F2 | 1 | recorded | Tier 2 (model-suggested, never recommended) | u290 | 327.75 | 0.0/1.8/0.6/42.0/35.9/19.6 | w/c 0.2736, CA 1639.7, SP 1.91, sand 0.0, phi 23.56 | 0.295 / min11 0.291 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | 0.9197 / 0.9039 | SD1.5: 0.9197 / 0.9039 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F2 | 1 | global | Tier 2 (model-suggested, never recommended) | g320_s1 | 318.75 | 1.8/0.8/2.9/51.8/27.1/15.6 | w/c 0.2784, CA 1625.2, SP 1.96, sand 0.1, phi 22.92 | 0.315 / min11 0.311 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | 0.9067 / 0.9029 | SD1.5: 0.9067 / 0.9029 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F0 | 2 | recorded | Tier 2 (model-suggested, never recommended) | u289 | 375.25 | 0.0/24.8/40.6/28.9/2.9/2.8 | w/c 0.2710, CA 1610.6, SP 1.86, sand 0.0, phi 25.66 | 0.591 / min11 0.578 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | 0.9057 / 0.9030 | SD1.5: 0.9057 / 0.9030 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F0 | 2 | global | Tier 2 (model-suggested, never recommended) | g340_s2 | 340.00 | 3.1/0.0/78.6/14.2/4.0/0.0 | w/c 0.2721, CA 1615.7, SP 1.74, sand 0.0, phi 22.89 | 0.562 / min11 0.549 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | 0.9051 / 0.9032 | SD1.5: 0.9051 / 0.9032 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F1 | 2 | recorded | Tier 2 (model-suggested, never recommended) | u415 | 368.20 | 0.0/0.0/32.1/35.6/23.7/8.6 | w/c 0.2725, CA 1654.2, SP 1.89, sand 0.0, phi 23.92 | 0.409 / min11 0.403 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | 0.9027 / 0.9018 | SD1.5: 0.9027 / 0.9018 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F1 | 2 | global | Tier 2 (model-suggested, never recommended) | g340_s2 | 335.00 | 3.8/15.7/25.1/26.0/19.6/9.8 | w/c 0.2726, CA 1633.0, SP 1.97, sand 0.0, phi 23.30 | 0.407 / min11 0.400 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | 0.9085 / 0.9023 | SD1.5: 0.9085 / 0.9023 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F2 | 2 | recorded | Tier 2 (model-suggested, never recommended) | u39 | 320.00 | 1.4/1.4/1.0/49.9/32.3/14.0 | w/c 0.2746, CA 1627.2, SP 1.82, sand 0.0, phi 23.14 | 0.312 / min11 0.308 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | 0.9084 / 0.9021 | SD1.5: 0.9084 / 0.9021; SD1.0: 0.9212 / 0.9131; SD2.0: 0.8841 / 0.8788 | 0 |
| F2 | 2 | global | Tier 2 (model-suggested, never recommended) | g340_s2 | 322.50 | 1.5/0.7/1.7/55.7/24.3/16.1 | w/c 0.2710, CA 1629.9, SP 1.86, sand 0.0, phi 23.18 | 0.313 / min11 0.309 vs floor 0.2895 | 0 (n>=2: False, n>=3: False) | 0.9131 / 0.9039 | SD1.5: 0.9131 / 0.9039 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |

### T3 CONVERGENCE: do global-seeded and recorded-parent walks end in the same place? (same = cement within 5 kg/m3 AND blend W1 <= 0.02)

Compared per (family, seed): (A) lowest CERTIFIED Tier-1 design of each source; (B) lowest ACCEPTED step of each source (any tier, before certificates).

| family | seed | (A) recorded C | (A) global C | dC | W1 of blends | same? | (B) recorded lowest accepted | (B) global lowest accepted | dC | W1 | same? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 0 | - | - | - | - | n/a (neither source) | 372.75 | 338.00 | -34.75 | 0.1965 | DIFFERENT |
| F0 | 1 | - | - | - | - | n/a (neither source) | 372.75 | 352.75 | -20.00 | 0.2064 | DIFFERENT |
| F0 | 2 | - | - | - | - | n/a (neither source) | 375.25 | 340.00 | -35.25 | 0.1615 | DIFFERENT |
| F1 | 0 | - | - | - | - | n/a (neither source) | 371.70 | 337.50 | -34.20 | 0.0983 | DIFFERENT |
| F1 | 1 | - | - | - | - | n/a (neither source) | 367.85 | 333.75 | -34.10 | 0.1813 | DIFFERENT |
| F1 | 2 | - | - | - | - | n/a (neither source) | 368.20 | 335.00 | -33.20 | 0.1393 | DIFFERENT |
| F2 | 0 | - | - | - | - | n/a (neither source) | 331.00 | 316.25 | -14.75 | 0.1290 | DIFFERENT |
| F2 | 1 | - | - | - | - | n/a (neither source) | 327.75 | 318.75 | -9.00 | 0.0846 | DIFFERENT |
| F2 | 2 | - | - | - | - | n/a (neither source) | 320.00 | 322.50 | +2.50 | 0.0337 | DIFFERENT |

Convergence verdict (A, certified Tier-1 designs): 0 of 0 comparable (family, seed) cases end in the same place. Cases that differ:
- F0 seed 0: (A) n/a (neither source); (B) DIFFERENT dC -34.75 W1 0.1965
- F0 seed 1: (A) n/a (neither source); (B) DIFFERENT dC -20.00 W1 0.2064
- F0 seed 2: (A) n/a (neither source); (B) DIFFERENT dC -35.25 W1 0.1615
- F1 seed 0: (A) n/a (neither source); (B) DIFFERENT dC -34.20 W1 0.0983
- F1 seed 1: (A) n/a (neither source); (B) DIFFERENT dC -34.10 W1 0.1813
- F1 seed 2: (A) n/a (neither source); (B) DIFFERENT dC -33.20 W1 0.1393
- F2 seed 0: (A) n/a (neither source); (B) DIFFERENT dC -14.75 W1 0.1290
- F2 seed 1: (A) n/a (neither source); (B) DIFFERENT dC -9.00 W1 0.0846
- F2 seed 2: (A) n/a (neither source); (B) DIFFERENT dC +2.50 W1 0.0337

### T4 recommended design

none