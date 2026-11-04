"""Hyperparameter stability analysis for the selected-method shortlist.

Grouped 5-fold CV (seed 555) on full data. All parameters FIXED at each
perturbation point (no tuning): FIT85/CAL15 grouped split inside train,
fit on FIT, calibrate on CAL, evaluate on test. Measures how density score,
90% coverage (base+conf), HDR width and mode count move around the optimum.

A defendable method must not swing wildly (e.g. 24-28% to 12-36% support)
under small k/bandwidth changes.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conformal_hcds as CH
import conditional_density as CD
import data_contract as DC
import hdr as H
import run_cv as RC

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(RUN, 'outputs')
SEED = 555
GRIDS = json.load(open(os.path.join(RUN, 'config.json')))['grids']


def split_fold(Xt, Xs, phi, groups, tr, te, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15,
                            random_state=int(seed))
    fi, ci = next(gss.split(tr, groups=groups[tr]))
    FIT, CAL = tr[fi], tr[ci]
    return FIT, CAL


def eval_density_variant(name, params, Xt, Xs, phi, groups, grid,
                         fids, seed_base, members_cfg=None):
    """One fixed-config variant -> pooled metrics over outer folds."""
    recs = []
    for fold in range(5):
        te = np.where(fids == fold)[0]
        tr = np.where(fids != fold)[0]
        FIT, CAL = split_fold(Xt, Xs, phi, groups, tr, te,
                              seed_base + fold)
        m_, s_ = DC.fit_scaler(Xt[FIT])
        Xss = DC.apply_scaler(Xt, m_, s_)
        if name == 'B8':
            fitted = {}
            oof, opos = {}, {}
            for mn, mp in members_cfg.items():
                mm = CD.make_density_method(mn, GRIDS)
                # inner OOF for weights (fixed params, no tuning)
                oo, pp = fixed_oof(mm, mn, mp, Xt[FIT], Xss[FIT],
                                   phi[FIT], groups[FIT])
                oof[mn], opos[mn] = oo, pp
                mm.fit(Xt[FIT], Xss[FIT], phi[FIT], groups[FIT], mp,
                       seed=seed_base + fold)
                fitted[mn] = mm
            b8 = CD.B8Stack([fitted[mn] for mn in members_cfg])
            b8.fit_weights(oof, opos)
            meth = b8
        else:
            meth = CD.make_density_method(name, GRIDS)
            meth.fit(Xt[FIT], Xss[FIT], phi[FIT], groups[FIT], params,
                     seed=seed_base + fold)
        pv = meth.point_logpdf(Xt[CAL], Xss[CAL], groups[CAL], phi[CAL])
        thr = CH.calibrate_levels(np.exp(pv), alphas=[0.1])
        lp = meth.point_logpdf(Xt[te], Xss[te], groups[te], phi[te])
        D = meth.grid_density(Xt[te], Xss[te], groups[te], grid)
        cov_b, cov_c, wid_b, wid_c, nmo = [], [], [], [], []
        for i in range(len(te)):
            hb = H.density_to_hdr(D[i], grid, 0.9)
            cov_b.append(H.covers(hb['intervals'], phi[te[i]]))
            wid_b.append(hb['total_measure'])
            nmo.append(hb['n_modes'])
            cb = (D[i] >= thr[0.1])
            ivs, tot = H.mask_to_intervals(cb, grid)
            cov_c.append(H.covers(ivs, phi[te[i]]))
            wid_c.append(tot)
        recs.append({'fold': fold, 'logscore': float(np.mean(lp)),
                     'cov_b90': float(np.mean(cov_b)),
                     'cov_c90': float(np.mean(cov_c)),
                     'wid_b90': float(np.mean(wid_b)),
                     'wid_c90': float(np.mean(wid_c)),
                     'modes': float(np.mean(nmo))})
    df = pd.DataFrame(recs)
    return {'logscore': df['logscore'].mean(),
            'logscore_sd': df['logscore'].std(),
            'cov_b90': df['cov_b90'].mean(),
            'cov_c90': df['cov_c90'].mean(),
            'wid_b90': df['wid_b90'].mean(),
            'wid_c90': df['wid_c90'].mean(),
            'modes': df['modes'].mean()}


def fixed_oof(mm, name, params, Xt_f, Xs_f, phi_f, g_f):
    """Inner GroupKFold OOF log-scores at FIXED params (for B8 weights)."""
    oof, pos = [], []
    for tri, vai in CD.inner_folds(g_f):
        mm.fit(Xt_f[tri], Xs_f[tri], phi_f[tri], g_f[tri], params,
               seed=SEED)
        if name == 'B7':
            sub = vai[::2]
        else:
            sub = vai
        if name == 'B7':
            sc = mm._point_scores(Xs_f[tri], g_f[tri], phi_f[tri],
                                  Xs_f[sub], g_f[sub], phi_f[sub],
                                  params['k'], SEED)
        else:
            sc = mm.point_logpdf(Xt_f[sub], Xs_f[sub], g_f[sub],
                                 phi_f[sub])
        # point_logpdf uses fitted-on-tri state? No: mm just fit on tri.
        # For non-B7 methods point_logpdf reads fitted attrs: correct.
        oof.append(np.asarray(sc, float))
        pos.append(sub)
    return np.concatenate(oof), np.concatenate(pos)


def main():
    fr = DC.load_frame()
    fr = DC.add_recipe_groups(fr)
    grid, _, _ = DC.phi_grid(fr['phi'].to_numpy())
    groups = fr['recipe_group'].to_numpy()
    phi = fr['phi'].to_numpy(float)
    Xt = DC.transform_xraw(DC.build_xraw_raw(fr)[0])
    fids = DC.grouped_folds(groups, 5, SEED)

    rows = []
    # B2 grid around optimum (k=10, h=4.5), probing below min and above max
    for k in [5, 8, 10, 15, 25]:
        for h in ([3.0, 4.5] if k != 10 else [2.0, 3.0, 4.5, 6.0]):
            r = eval_density_variant('B2', {'k': k, 'h_phi': h},
                                     Xt, None, phi, groups, grid, fids,
                                     1000 + k * 10 + int(h * 10))
            r.update({'variant': f'B2 k={k} h={h}'})
            rows.append(r)
            print(r['variant'], {kk: round(v, 3) for kk, v in r.items()
                                 if kk != 'variant'}, flush=True)
    # B8 member-config variants
    modal = {'B2': {'k': 10, 'h_phi': 4.5}, 'B3': {'k': 10, 'h_phi': 4.5},
             'B4': {'mode': 'fixed', 'h_x': 4.0, 'h_phi': 4.0},
             'B5': {'h': 0.4}, 'B6': {'K': 6}, 'B7': {'k': 30}}
    variants = {
        'B8 modal': modal,
        'B8 k+1step': {**modal, 'B2': {'k': 15, 'h_phi': 4.5},
                       'B3': {'k': 15, 'h_phi': 4.5},
                       'B7': {'k': 50}},
        'B8 hphi-1step': {**modal, 'B2': {'k': 10, 'h_phi': 3.0},
                          'B3': {'k': 10, 'h_phi': 3.0},
                          'B4': {'mode': 'fixed', 'h_x': 4.0,
                                 'h_phi': 2.5}},
        'B8 K4': {**modal, 'B6': {'K': 4}},
        'B8 noB5': {k: v for k, v in modal.items() if k != 'B5'},
    }
    for vn, mc in variants.items():
        r = eval_density_variant('B8', None, Xt, None, phi, groups, grid,
                                 fids, 2000 + len(vn), members_cfg=mc)
        r.update({'variant': vn})
        rows.append(r)
        print(vn, {kk: round(v, 3) for kk, v in r.items()
                   if kk != 'variant'}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'stability_grid.csv'),
                              index=False)


if __name__ == '__main__':
    # Xt passed unscaled; scaling done inside per split (arg Xs unused)
    main()
