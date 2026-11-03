**VERDICT (S* 10 MPa, K* 8 mm/s, R* 0.90, integrated global + recorded walk): recommended design = family 0, seed 1, source recorded, parent u431, cement 370.00 kg/m3 (Tier 1: 2 near recorded passing mix(es); certified); blend % 1.9/1.9/93.8/0.8/0.8/0.8; w/c 0.2832, CA 1587.9, SP 0.00, sand 0.0, phi 21.80.**

Target S* 10 MPa, K* 8 mm/s, R* 0.90. Config f4d9b797eaf3393c27a4c81223d9d8cf17e6ea7a99a9e9dfda960e8f9a76bf85. Paste floor 0.160274 mm (unit u83, 138 recorded unit(s) meet the target). Tier rule: Tier 1 only if at least one recorded passing mix is near the design (n_min = 1, frozen near radii, window-clamped mixes never count); everything else Tier 2 model-suggested, never recommended. Baseline = the existing walk of this target (stage\), unchanged.

**FLAG: a global-seeded design is more than 10 kg/m3 cheaper than the best recorded-parent walk design of its family - treat as MODEL-SUGGESTED until checked in the lab:**
- F0 seed 0 global: 261.50 kg/m3 vs best recorded-parent walk design 370.00 (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9048 / 0.9025
- F1 seed 0 global: 240.25 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9032 / 0.9021
- F2 seed 0 global: 230.00 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9375 / 0.9246
- F0 seed 1 global: 271.50 kg/m3 vs best recorded-parent walk design 370.00 (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9028 / 0.9015
- F1 seed 1 global: 246.25 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9045 / 0.9032
- F2 seed 1 global: 230.00 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9267 / 0.9144
- F0 seed 2 global: 268.75 kg/m3 vs best recorded-parent walk design 370.00 (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9110 / 0.9070
- F1 seed 2 global: 237.50 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9057 / 0.9042
- F2 seed 2 global: 230.00 kg/m3 vs best recorded-parent walk design NONE (the recorded-parent walks found no accepted design in this family) (Tier 2); 0 (n>=2: False, n>=3: False) near passers; certificate SD 1.5 nominal / worst 0.9287 / 0.9162

### G0 global search only (best design per family over all seeds and levels; full curve in global/global_scan.csv)

| family | lowest passing level (any seed) | seeds passing there | best min-LB at that level (nominal / worst) | tier of that design | levels scanned | passes at (all levels) |
|---|---|---|---|---|---|---|
| F0 | 275 | 2 of 3 | 0.9473 (0.9485 / 0.9473) | Tier 2 (near passers 0) | 420, 400, 380, 360, 340, 320, 300, 280, 275, 270, 260 | 275, 280, 300, 320, 340, 360, 380, 400, 420 |
| F1 | 250 | 1 of 3 | 0.9368 (0.9375 / 0.9368) | Tier 2 (near passers 0) | 420, 400, 380, 360, 340, 320, 300, 280, 260, 255, 250, 245 | 250, 255, 260, 280, 300, 320, 340, 360, 380, 400, 420 |
| F2 | 245 | 3 of 3 | 0.9524 (0.9536 / 0.9524) | Tier 2 (near passers 0) | 420, 400, 380, 360, 340, 320, 300, 280, 260, 255, 250, 245 | 245, 250, 255, 260, 280, 300, 320, 340, 360, 380, 400, 420 |

### T0 per (family, seed, source): log-derived minima (cement kg/m3) and certificate status

| family | seed | source | parents | attempts | finished | lowest accepted (any tier) | lowest accepted with >= 1 near passer (Tier-1 candidate) | n2 | n3 | status | certified Tier 1 | certified Tier 2 (model-suggested) | baseline walk certified | failing-check counts g/f/s/p/r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 0 | recorded | 12 | 161 | True | 293.00 | - | - | - | model-suggested only (Tier 2) certified | - | 293.00 | - | 44 / 12 / 5 / 0 / 118 |
| F0 | 0 | global | 2 | 25 | True | 261.50 | - | - | - | model-suggested only (Tier 2) certified | - | 261.50 | - | 0 / 0 / 0 / 0 / 12 |
| F0 | 1 | recorded | 12 | 171 | True | 286.00 | 370.00 | 370.00 | 380.00 | certified | 370.00 | 286.00 | 370.00 | 61 / 14 / 9 / 0 / 116 |
| F0 | 1 | global | 2 | 18 | True | 271.50 | - | - | - | model-suggested only (Tier 2) certified | - | 271.50 | - | 5 / 6 / 0 / 0 / 6 |
| F0 | 2 | recorded | 12 | 167 | True | 287.65 | - | - | - | model-suggested only (Tier 2) certified | - | 287.65 | - | 44 / 13 / 3 / 0 / 116 |
| F0 | 2 | global | 1 | 9 | True | 268.75 | - | - | - | model-suggested only (Tier 2) certified | - | 268.75 | - | 0 / 0 / 0 / 0 / 6 |
| F1 | 0 | recorded | 12 | 202 | True | 256.70 | - | - | - | model-suggested only (Tier 2) certified | - | 256.70 | - | 53 / 39 / 0 / 0 / 101 |
| F1 | 0 | global | 1 | 12 | True | 240.25 | - | - | - | model-suggested only (Tier 2) certified | - | 240.25 | - | 0 / 0 / 0 / 0 / 6 |
| F1 | 1 | recorded | 12 | 207 | True | 251.00 | - | - | - | model-suggested only (Tier 2) certified | - | 251.00 | - | 48 / 38 / 1 / 0 / 97 |
| F1 | 1 | global | 1 | 10 | True | 246.25 | - | - | - | model-suggested only (Tier 2) certified | - | 246.25 | - | 0 / 0 / 0 / 0 / 6 |
| F1 | 2 | recorded | 12 | 193 | True | 255.50 | - | - | - | model-suggested only (Tier 2) certified | - | 255.50 | - | 57 / 43 / 0 / 0 / 88 |
| F1 | 2 | global | 2 | 18 | True | 237.50 | - | - | - | model-suggested only (Tier 2) certified | - | 237.50 | - | 0 / 0 / 0 / 0 / 12 |
| F2 | 0 | recorded | 12 | 157 | True | 221.00 | - | - | - | model-suggested only (Tier 2) certified | - | 221.00 | - | 12 / 1 / 13 / 0 / 118 |
| F2 | 0 | global | 2 | 6 | True | 230.00 | - | - | - | model-suggested only (Tier 2) certified | - | 230.00 | - | 0 / 0 / 0 / 0 / 0 |
| F2 | 1 | recorded | 12 | 165 | True | 221.00 | - | - | - | model-suggested only (Tier 2) certified | - | 221.00 | - | 13 / 2 / 11 / 1 / 114 |
| F2 | 1 | global | 2 | 6 | True | 230.00 | - | - | - | model-suggested only (Tier 2) certified | - | 230.00 | - | 0 / 0 / 0 / 0 / 0 |
| F2 | 2 | recorded | 12 | 153 | True | 221.00 | - | - | - | model-suggested only (Tier 2) certified | - | 221.00 | - | 22 / 0 / 11 / 1 / 110 |
| F2 | 2 | global | 2 | 6 | True | 230.00 | - | - | - | model-suggested only (Tier 2) certified | - | 230.00 | - | 0 / 0 / 0 / 0 / 0 |

### T1 FAMILY x SOURCE TABLE (lowest certified design per seed; Tier-1 designs unless marked)

| family | source | tier | lowest certified C per seed s0 / s1 / s2 | blend six bins % (lowest seed design) | other mix values | h vs floor | near passers (n>=1; n>=2, n>=3 beside) | certificates nominal / worst (min of 2 reps) at SD 1.5; SD 1.0 and 2.0 | baseline walk certified C per seed (recorded source) |
|---|---|---|---|---|---|---|---|---|---|
| F0 | recorded | Tier 1 | - / 370.00 / - | 1.9/1.9/93.8/0.8/0.8/0.8 (s1, u431) | w/c 0.2832, CA 1587.9, SP 0.00, sand 0.0, phi 21.80 | 0.677 / min11 0.672 vs floor 0.1603 | 2 (n>=2: True, n>=3: False) | SD1.5: 0.9034 / 0.9019; SD1.0: 0.9157 / 0.9134; SD2.0: 0.8850 / 0.8822 | - / 370.00 / - |
| F0 | recorded | Tier 2 model-suggested | 293.00 / 286.00 / 287.65 | 0.0/52.8/24.7/17.0/2.8/2.7 (s1, u288) | w/c 0.2794, CA 1624.2, SP 1.90, sand 0.0, phi 24.97 | 0.513 / min11 0.501 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9528 / 0.9514; SD1.0: 0.9586 / 0.9574; SD2.0: 0.9446 / 0.9428 | - / 370.00 / - |
| F0 | global | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - |
| F0 | global | Tier 2 model-suggested | 261.50 / 271.50 / 268.75 | 1.3/14.1/67.8/15.7/0.6/0.6 (s0, g320_s0) | w/c 0.2805, CA 1631.4, SP 1.87, sand 0.0, phi 23.93 | 0.454 / min11 0.449 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9048 / 0.9025; SD1.0: 0.9105 / 0.9080; SD2.0: 0.8950 / 0.8925 | - |
| F1 | recorded | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - / - / - |
| F1 | recorded | Tier 2 model-suggested | 256.70 / 251.00 / 255.50 | 0.0/0.0/37.1/29.9/23.3/9.7 (s1, u199) | w/c 0.2790, CA 1644.3, SP 1.89, sand 0.0, phi 26.82 | 0.284 / min11 0.279 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9056 / 0.9030; SD1.0: 0.9096 / 0.9063; SD2.0: 0.8994 / 0.8967 | - / - / - |
| F1 | global | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - |
| F1 | global | Tier 2 model-suggested | 240.25 / 246.25 / 237.50 | 1.6/2.5/31.2/32.7/26.6/5.3 (s2, g250_s2) | w/c 0.2846, CA 1632.1, SP 1.80, sand 0.1, phi 22.50 | 0.286 / min11 0.281 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9057 / 0.9042; SD1.0: 0.9137 / 0.9117; SD2.0: 0.8916 / 0.8904 | - |
| F2 | recorded | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - / - / - |
| F2 | recorded | Tier 2 model-suggested | 221.00 / 221.00 / 221.00 | 0.0/0.0/0.0/51.1/38.0/10.9 (s0, u23) | w/c 0.2937, CA 1625.0, SP 1.92, sand 0.0, phi 23.75 | 0.223 / min11 0.219 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9237 / 0.9198; SD1.0: 0.9360 / 0.9320; SD2.0: 0.9075 / 0.9037 | - / - / - |
| F2 | global | Tier 1 | none certified (model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified, model-suggested only (Tier 2) certified) | - | - | - | - | - | - |
| F2 | global | Tier 2 model-suggested | 230.00 / 230.00 / 230.00 | 0.0/0.8/0.5/36.3/43.0/19.3 (s0, g245_s0) | w/c 0.2841, CA 1627.4, SP 1.84, sand 0.0, phi 24.13 | 0.208 / min11 0.205 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | SD1.5: 0.9375 / 0.9246; SD1.0: 0.9468 / 0.9355; SD2.0: 0.9239 / 0.9104 | - |

### T2 certified design of every (family, seed, source, tier)

| family | seed | source | tier | parent | C | blend six bins % | other | h | near passers | certificate SD 1.5 nominal / worst | SD checks | fallback index |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 0 | recorded | Tier 2 (model-suggested, never recommended) | u288 | 293.00 | 0.0/68.6/18.0/9.1/4.0/0.4 | w/c 0.2773, CA 1617.5, SP 1.92, sand 0.0, phi 24.45 | 0.597 / min11 0.581 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9474 / 0.9438 | SD1.5: 0.9474 / 0.9438 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F0 | 0 | global | Tier 2 (model-suggested, never recommended) | g320_s0 | 261.50 | 1.3/14.1/67.8/15.7/0.6/0.6 | w/c 0.2805, CA 1631.4, SP 1.87, sand 0.0, phi 23.93 | 0.454 / min11 0.449 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9048 / 0.9025 | SD1.5: 0.9048 / 0.9025; SD1.0: 0.9105 / 0.9080; SD2.0: 0.8950 / 0.8925 | 0 |
| F1 | 0 | recorded | Tier 2 (model-suggested, never recommended) | u415 | 256.70 | 0.0/0.0/32.5/36.3/20.8/10.4 | w/c 0.2718, CA 1646.9, SP 1.85, sand 0.0, phi 25.43 | 0.284 / min11 0.279 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9043 / 0.9013 | SD1.5: 0.9043 / 0.9013 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F1 | 0 | global | Tier 2 (model-suggested, never recommended) | g260_s0 | 240.25 | 4.3/1.2/32.0/31.3/26.3/4.9 | w/c 0.2829, CA 1609.2, SP 1.78, sand 0.0, phi 24.24 | 0.298 / min11 0.293 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9032 / 0.9021 | SD1.5: 0.9032 / 0.9021 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F2 | 0 | recorded | Tier 2 (model-suggested, never recommended) | u23 | 221.00 | 0.0/0.0/0.0/51.1/38.0/10.9 | w/c 0.2937, CA 1625.0, SP 1.92, sand 0.0, phi 23.75 | 0.223 / min11 0.219 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9237 / 0.9198 | SD1.5: 0.9237 / 0.9198; SD1.0: 0.9360 / 0.9320; SD2.0: 0.9075 / 0.9037 | 0 |
| F2 | 0 | global | Tier 2 (model-suggested, never recommended) | g245_s0 | 230.00 | 0.0/0.8/0.5/36.3/43.0/19.3 | w/c 0.2841, CA 1627.4, SP 1.84, sand 0.0, phi 24.13 | 0.208 / min11 0.205 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9375 / 0.9246 | SD1.5: 0.9375 / 0.9246; SD1.0: 0.9468 / 0.9355; SD2.0: 0.9239 / 0.9104 | 0 |
| F0 | 1 | recorded | Tier 1 | u431 | 370.00 | 1.9/1.9/93.8/0.8/0.8/0.8 | w/c 0.2832, CA 1587.9, SP 0.00, sand 0.0, phi 21.80 | 0.677 / min11 0.672 vs floor 0.1603 | 2 (n>=2: True, n>=3: False) | 0.9034 / 0.9019 | SD1.5: 0.9034 / 0.9019; SD1.0: 0.9157 / 0.9134; SD2.0: 0.8850 / 0.8822 | 0 |
| F0 | 1 | recorded | Tier 2 (model-suggested, never recommended) | u288 | 286.00 | 0.0/52.8/24.7/17.0/2.8/2.7 | w/c 0.2794, CA 1624.2, SP 1.90, sand 0.0, phi 24.97 | 0.513 / min11 0.501 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9528 / 0.9514 | SD1.5: 0.9528 / 0.9514; SD1.0: 0.9586 / 0.9574; SD2.0: 0.9446 / 0.9428 | 0 |
| F0 | 1 | global | Tier 2 (model-suggested, never recommended) | g275_s1 | 271.50 | 0.5/25.1/43.7/28.1/1.8/0.8 | w/c 0.2774, CA 1495.5, SP 1.79, sand 0.0, phi 25.69 | 0.493 / min11 0.488 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9028 / 0.9015 | SD1.5: 0.9028 / 0.9015 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F1 | 1 | recorded | Tier 2 (model-suggested, never recommended) | u199 | 251.00 | 0.0/0.0/37.1/29.9/23.3/9.7 | w/c 0.2790, CA 1644.3, SP 1.89, sand 0.0, phi 26.82 | 0.284 / min11 0.279 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9056 / 0.9030 | SD1.5: 0.9056 / 0.9030; SD1.0: 0.9096 / 0.9063; SD2.0: 0.8994 / 0.8967 | 0 |
| F1 | 1 | global | Tier 2 (model-suggested, never recommended) | g255_s1 | 246.25 | 0.0/0.0/33.8/30.6/31.4/4.2 | w/c 0.2871, CA 1631.0, SP 1.60, sand 0.5, phi 23.50 | 0.292 / min11 0.287 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9045 / 0.9032 | SD1.5: 0.9045 / 0.9032 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F2 | 1 | recorded | Tier 2 (model-suggested, never recommended) | u23 | 221.00 | 0.0/0.0/0.0/51.2/37.7/11.0 | w/c 0.2937, CA 1626.8, SP 1.95, sand 0.0, phi 23.89 | 0.222 / min11 0.219 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9246 / 0.9206 | SD1.5: 0.9246 / 0.9206 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F2 | 1 | global | Tier 2 (model-suggested, never recommended) | g245_s1 | 230.00 | 0.5/0.7/0.4/38.9/38.9/20.7 | w/c 0.2817, CA 1623.2, SP 1.84, sand 0.0, phi 24.24 | 0.207 / min11 0.205 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9267 / 0.9144 | SD1.5: 0.9267 / 0.9144 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F0 | 2 | recorded | Tier 2 (model-suggested, never recommended) | u400 | 287.65 | 0.0/49.2/27.3/17.5/2.9/3.1 | w/c 0.2800, CA 1631.5, SP 1.88, sand 0.0, phi 24.57 | 0.503 / min11 0.491 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9581 / 0.9571 | SD1.5: 0.9581 / 0.9571 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F0 | 2 | global | Tier 2 (model-suggested, never recommended) | g280_s2 | 268.75 | 3.5/16.1/49.9/26.1/2.3/2.1 | w/c 0.2797, CA 1664.0, SP 2.27, sand 0.1, phi 24.95 | 0.425 / min11 0.415 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9110 / 0.9070 | SD1.5: 0.9110 / 0.9070 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F1 | 2 | recorded | Tier 2 (model-suggested, never recommended) | u362 | 255.50 | 0.0/0.0/34.9/32.5/20.8/11.8 | w/c 0.2710, CA 1653.9, SP 1.87, sand 0.0, phi 25.27 | 0.278 / min11 0.273 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9078 / 0.9021 | SD1.5: 0.9078 / 0.9021 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F1 | 2 | global | Tier 2 (model-suggested, never recommended) | g250_s2 | 237.50 | 1.6/2.5/31.2/32.7/26.6/5.3 | w/c 0.2846, CA 1632.1, SP 1.80, sand 0.1, phi 22.50 | 0.286 / min11 0.281 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9057 / 0.9042 | SD1.5: 0.9057 / 0.9042; SD1.0: 0.9137 / 0.9117; SD2.0: 0.8916 / 0.8904 | 0 |
| F2 | 2 | recorded | Tier 2 (model-suggested, never recommended) | u23 | 221.00 | 1.2/1.2/1.2/52.1/35.5/8.8 | w/c 0.2937, CA 1629.6, SP 1.87, sand 0.0, phi 23.64 | 0.232 / min11 0.228 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9097 / 0.9072 | SD1.5: 0.9097 / 0.9072 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |
| F2 | 2 | global | Tier 2 (model-suggested, never recommended) | g245_s2 | 230.00 | 0.1/0.3/1.0/37.5/40.1/21.0 | w/c 0.2839, CA 1614.7, SP 1.77, sand 0.0, phi 24.29 | 0.208 / min11 0.205 vs floor 0.1603 | 0 (n>=2: False, n>=3: False) | 0.9287 / 0.9162 | SD1.5: 0.9287 / 0.9162 (SD 1.0 / 2.0 only run on the lowest design of the (family, source, tier)) | 0 |

### T3 CONVERGENCE: do global-seeded and recorded-parent walks end in the same place? (same = cement within 5 kg/m3 AND blend W1 <= 0.02)

Compared per (family, seed): (A) lowest CERTIFIED Tier-1 design of each source; (B) lowest ACCEPTED step of each source (any tier, before certificates).

| family | seed | (A) recorded C | (A) global C | dC | W1 of blends | same? | (B) recorded lowest accepted | (B) global lowest accepted | dC | W1 | same? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F0 | 0 | - | - | - | - | n/a (neither source) | 293.00 | 261.50 | -31.50 | 0.2126 | DIFFERENT |
| F0 | 1 | 370.00 | - | - | - | DIFFERENT (only one source has a design) | 286.00 | 271.50 | -14.50 | 0.1435 | DIFFERENT |
| F0 | 2 | - | - | - | - | n/a (neither source) | 287.65 | 268.75 | -18.90 | 0.1489 | DIFFERENT |
| F1 | 0 | - | - | - | - | n/a (neither source) | 256.70 | 240.25 | -16.45 | 0.0779 | DIFFERENT |
| F1 | 1 | - | - | - | - | n/a (neither source) | 251.00 | 246.25 | -4.75 | 0.0474 | DIFFERENT |
| F1 | 2 | - | - | - | - | n/a (neither source) | 255.50 | 237.50 | -18.00 | 0.0555 | DIFFERENT |
| F2 | 0 | - | - | - | - | n/a (neither source) | 221.00 | 230.00 | +9.00 | 0.0778 | DIFFERENT |
| F2 | 1 | - | - | - | - | n/a (neither source) | 221.00 | 230.00 | +9.00 | 0.1029 | DIFFERENT |
| F2 | 2 | - | - | - | - | n/a (neither source) | 221.00 | 230.00 | +9.00 | 0.0933 | DIFFERENT |

Convergence verdict (A, certified Tier-1 designs): 0 of 1 comparable (family, seed) cases end in the same place. Cases that differ:
- F0 seed 0: (A) n/a (neither source); (B) DIFFERENT dC -31.50 W1 0.2126
- F0 seed 1: (A) DIFFERENT (only one source has a design); (B) DIFFERENT dC -14.50 W1 0.1435
- F0 seed 2: (A) n/a (neither source); (B) DIFFERENT dC -18.90 W1 0.1489
- F1 seed 0: (A) n/a (neither source); (B) DIFFERENT dC -16.45 W1 0.0779
- F1 seed 1: (A) n/a (neither source); (B) DIFFERENT dC -4.75 W1 0.0474
- F1 seed 2: (A) n/a (neither source); (B) DIFFERENT dC -18.00 W1 0.0555
- F2 seed 0: (A) n/a (neither source); (B) DIFFERENT dC +9.00 W1 0.0778
- F2 seed 1: (A) n/a (neither source); (B) DIFFERENT dC +9.00 W1 0.1029
- F2 seed 2: (A) n/a (neither source); (B) DIFFERENT dC +9.00 W1 0.0933

### T4 recommended design

Cheapest certified Tier-1 design: family 0, seed 1, source recorded, parent u431, cement 370.00 kg/m3; blend % 1.9/1.9/93.8/0.8/0.8/0.8; w/c 0.2832, CA 1587.9, SP 0.00, sand 0.0, phi 21.80; h 0.677 / min11 0.672 vs floor 0.1603; near recorded passers 2 (n>=2: True, n>=3: False); certificate SD 1.5 nominal / worst 0.9034 / 0.9019; SD1.5: 0.9034 / 0.9019; SD1.0: 0.9157 / 0.9134; SD2.0: 0.8850 / 0.8822.