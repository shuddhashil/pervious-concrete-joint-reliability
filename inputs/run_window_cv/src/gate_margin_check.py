"""Supplementary x-only diagnostic for the three gated-out specials.

Uses recipe coordinates ONLY (plus listing neighbors' observed phis as data).
No density is fit or evaluated here; no selection impact.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_contract as DC
from conditional_density import knn as _knn

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(RUN, 'outputs')

fr = DC.load_frame()
fr = DC.add_recipe_groups(fr)
phi = fr['phi'].to_numpy(float)
groups = fr['recipe_group'].to_numpy()
studies = fr['study'].to_numpy()
Xt_all = DC.transform_xraw(DC.build_xraw_raw(fr)[0])
fall = np.arange(len(fr))
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=999)
fi, ci = next(gss.split(fall, groups=groups))
FIT = fall[fi]
m, s = DC.fit_scaler(Xt_all[FIT])
Xs_all = DC.apply_scaler(Xt_all, m, s)
tcols = DC.XRAW_TCOLS

sealed = json.load(open(os.path.join(OUT, 'candidates_sealed.json')))
# pooled held-out kth-NN distances from the frozen gate calibration
import recipe_space as RS
gthr, held = RS.calibrate_threshold(Xs_all[FIT], groups[FIT], k=10)
print(f'gate threshold k=10: {gthr:.3f} (95th pct, n={len(held)})')

for c in sealed['cases']:
    f = np.asarray(c['bins_pct'], float) / 100.0
    b = c['coarse_agg'] * f
    raw = np.array([[c['water'] / c['cem'], c['cem'], c['sp'],
                     c['sand'], *b]])
    xt_raw = DC.transform_xraw(raw)
    xt_s = DC.apply_scaler(xt_raw, m, s)
    d10 = RS.kth_distances(RS.fit_gate(Xs_all[FIT], k=10), xt_s)[0]
    pct = float((held <= d10).mean() * 100)
    print(f"\n{c['case']}: d10={d10:.3f} vs thr={gthr:.3f} "
          f"(percentile {pct:.1f} of held-out; target={c['target_phi']})")
    idx, dd = _knn(Xs_all[FIT], groups[FIT], xt_s, None, 8)
    idx, dd = idx[0], dd[0]
    print('nearest FIT rows (group-excluded search over FIT):')
    for ii, d in zip(idx, dd):
        if ii < 0:
            continue
        r = FIT[ii]
        gaps = (xt_s[0] - Xs_all[r])  # standardized coord gaps
        top = np.argsort(-np.abs(gaps))[:3]
        gapstr = ','.join(f'{tcols[j]}={gaps[j]:+.2f}' for j in top)
        print(f"  pos={r} {fr['mix_id'].iloc[r]} study={studies[r]} "
              f"group={groups[r]} d={d:.2f} obs_phi={phi[r]:.1f} "
              f"topgaps[{gapstr}]")
