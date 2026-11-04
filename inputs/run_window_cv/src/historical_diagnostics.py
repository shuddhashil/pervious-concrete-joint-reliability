"""Post-freeze historical diagnostics (runs ONLY after method_freeze.json).

Frozen B8/x_raw pipeline refit on FULL 650 with a grouped FIT/CAL split
(mirrors CV protocol exactly): scaler on FIT85, gate threshold from FIT85,
members at modal configs on FIT85, B8 weights from inner-OOF on FIT85,
conformal thresholds on CAL15.

Evaluates C267, ID-80, ID-90 + 20 randomly selected ordinary recipe groups.
Diagnostic only: cannot alter selection.
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conformal_hcds as CH
import conditional_density as CD
import data_contract as DC
import hdr as H
import recipe_space as RS

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(RUN, 'outputs')
FIG = os.path.join(RUN, 'figures')
LEVELS = [0.8, 0.9, 0.95]
ALPHAS = [round(1.0 - lv, 10) for lv in LEVELS]


def case_to_xraw(case):
    """RAW (untransformed) x_raw row: [w/c, cem, sp, sand, b1..b6]."""
    f = np.asarray(case['bins_pct'], float) / 100.0
    b = case['coarse_agg'] * f
    return np.array([[case['water'] / case['cem'], case['cem'],
                      case['sp'], case['sand'], *b]])


def raw_medoid(fr, med):
    return DC.build_xraw_raw(fr.iloc[[med]])[0]


def main():
    assert os.path.exists(os.path.join(OUT, 'method_freeze.json')), \
        'winner must be frozen before historical diagnostic'
    sel = json.load(open(os.path.join(OUT, 'selected_method.json')))
    sealed = json.load(open(os.path.join(OUT, 'candidates_sealed.json')))
    assert sel['winner'].startswith('B8')
    member_cfg = sel['frozen_pipeline']['member_configs']
    members = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    cfg = json.load(open(os.path.join(RUN, 'config.json')))

    fr = DC.load_frame()
    fr = DC.add_recipe_groups(fr)
    grid = np.load(os.path.join(OUT, '_phi_grid.npy'))
    phi = fr['phi'].to_numpy(float)
    groups = fr['recipe_group'].to_numpy()
    studies = fr['study'].to_numpy()
    Xt_all = DC.transform_xraw(DC.build_xraw_raw(fr)[0])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=999)
    fall = np.arange(len(fr))
    fi, ci = next(gss.split(fall, groups=groups))
    FIT, CAL = fall[fi], fall[ci]
    m, s = DC.fit_scaler(Xt_all[FIT])
    Xs_all = DC.apply_scaler(Xt_all, m, s)

    gate = RS.fit_gate(Xs_all[FIT], k=10)
    gthr, held = RS.calibrate_threshold(Xs_all[FIT], groups[FIT], k=10)

    fitted, oof, opos = {}, {}, {}
    for mn in members:
        mm = CD.make_density_method(mn, cfg['grids'])
        oo, pp = [], []
        for tri, vai in CD.inner_folds(groups[FIT]):
            t3, v3 = FIT[tri], FIT[vai]
            mm.fit(Xt_all[t3], Xs_all[t3], phi[t3], groups[t3],
                   member_cfg[mn], seed=999)
            sub = v3[::2] if mn == 'B7' else v3
            if mn == 'B7':
                sc = mm._point_scores(Xs_all[t3], groups[t3], phi[t3],
                                      Xs_all[sub], groups[sub], phi[sub],
                                      member_cfg[mn]['k'], 999)
            else:
                sc = mm.point_logpdf(Xt_all[sub], Xs_all[sub],
                                     groups[sub], phi[sub])
            oo.append(np.asarray(sc, float))
            pp.append(sub)
        oof[mn] = np.concatenate(oo)
        opos[mn] = np.concatenate(pp)
        mm.fit(Xt_all[FIT], Xs_all[FIT], phi[FIT], groups[FIT],
               member_cfg[mn], seed=999)
        fitted[mn] = mm
    b8 = CD.B8Stack([fitted[mn] for mn in members])
    b8.fit_weights(oof, opos)
    print('final B8 weights:', json.dumps(b8.info['weights'], indent=1),
          flush=True)
    pv = b8.point_logpdf(Xt_all[CAL], Xs_all[CAL], groups[CAL], phi[CAL])
    thr = CH.calibrate_levels(np.exp(pv), alphas=ALPHAS)

    # 20 ordinary groups (exclude specials' nearest groups once known)
    rng = np.random.default_rng(777)
    uniq = np.unique(groups)

    queries = []
    for c in sealed['cases']:
        queries.append({'case': c['case'], 'xraw': case_to_xraw(c),
                        'target_phi': float(c['target_phi'])})
    # nearest-group ids of specials (for exclusion from random draw)
    tmpX = DC.apply_scaler(
        DC.transform_xraw(np.vstack([q['xraw'] for q in queries])), m, s)
    from conditional_density import knn as _knn
    nn0, _ = _knn(Xs_all[FIT], groups[FIT], tmpX, None, 1)
    special_groups = set(groups[FIT][nn0[:, 0][nn0[:, 0] >= 0]].tolist())
    pool = [g for g in uniq if g not in special_groups]
    chosen = rng.choice(np.asarray(pool), size=20, replace=False)
    for gi, g in enumerate(chosen):
        rows = np.where(groups == g)[0]
        # medoid in scaled space as the ordinary-recipe query
        D = ((Xs_all[rows, None, :] - Xs_all[rows][None, :, :]) ** 2).sum(-1)
        med = rows[int(np.argmin(D.sum(1)))]
        queries.append({'case': f'GRP-g{g}-pos{med}',
                        'xraw': raw_medoid(fr, med),
                        'target_phi': float(phi[med]),
                        'own_group': int(g), 'own_pos': int(med)})

    recs = []
    for q in queries:
        Xtq_raw = DC.transform_xraw(q['xraw'])
        Xtq_s = DC.apply_scaler(Xtq_raw, m, s)
        outside, gdist = RS.apply_gate(gate, Xtq_s, gthr)
        gate_status = ('NO_LOCAL_EVIDENCE_FOR_RECIPE' if outside[0]
                       else 'SUPPORTED_RECIPE')
        # nearest FIT rows / groups (own-group excluded from supporters)
        nn_idx, nn_d = _knn(Xs_all[FIT], groups[FIT], Xtq_s, None, 60)
        nn_idx, nn_d = nn_idx[0], nn_d[0]
        ok = nn_idx >= 0
        nn_idx, nn_d = nn_idx[ok], nn_d[ok]
        own_g = int(groups[FIT][nn_idx[0]]) if len(nn_idx) else -1
        if 'own_group' in q:
            own_g = q['own_group']
        seen, supp = set(), []
        for ii, dd in zip(nn_idx, nn_d):
            gg = int(groups[FIT][ii])
            if gg == own_g or gg in seen:
                continue
            seen.add(gg)
            supp.append((gg, ii, dd))
            if len(supp) == 8:
                break
        nb = ';'.join(
            f"{fr['mix_id'].iloc[FIT[ii]]}:{studies[FIT[ii]]}:"
            f"d={dd:.2f}:phi={phi[FIT[ii]]:.1f}" for gg, ii, dd in supp)
        top1 = studies[FIT][nn_idx[:50]]
        tss = float((top1 == top1[0]).mean()) if len(top1) else np.nan
        row = {'case': q['case'], 'target_phi': q['target_phi'],
               'gate_status': gate_status,
               'gate_dist': round(float(gdist[0]), 3),
               'gate_thr': round(float(gthr), 3),
               'gate_pct': round(float((held <= gdist[0]).mean() * 100),
                                 1),
               'nearest_group': own_g,
               'nearest_dist': round(float(nn_d[0]), 3)
               if len(nn_d) else np.nan,
               'topstudy_share50': round(tss, 3),
               'nearest_supporters': nb,
               'b8_weights': json.dumps(
                   {k: round(v, 3) for k, v in b8.info['weights'].items()})}
        if not outside[0]:
            # group=None: novel query; density from full FIT-fitted state
            D = b8.grid_density(Xtq_raw, Xtq_s, None, grid)
            cdf = H.conditional_cdf(D[0], grid)
            cdf_at = float(np.interp(q['target_phi'], grid, cdf))
            f_at = float(np.interp(q['target_phi'], grid, D[0]))
            row['cdf_at_target'] = round(cdf_at, 4)
            row['density_at_target'] = round(f_at, 5)
            for lv, a in zip(LEVELS, ALPHAS):
                hb = H.density_to_hdr(D[0], grid, lv)
                cm = D[0] >= thr[a]
                ivs, tot = H.mask_to_intervals(cm, grid)
                for ver, I, T in (('base', hb['intervals'],
                                   hb['total_measure']),
                                  ('conf', ivs, tot)):
                    row[f'{ver}{int(lv*100)}_hdr'] = ';'.join(
                        f'[{a_:.1f},{b_:.1f}]' for a_, b_ in I) or 'EMPTY'
                    row[f'{ver}{int(lv*100)}_width'] = round(float(T), 2)
                    row[f'{ver}{int(lv*100)}_nmodes'] = len(I)
                    row[f'{ver}{int(lv*100)}_inside'] = bool(
                        H.covers(I, q['target_phi']))
        recs.append(row)
        if q['case'] in ('C267', 'ID-80', 'ID-90') and not outside[0]:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(grid, D[0], lw=1.5, label='stacked f(phi|x)')
            hb90 = H.density_to_hdr(D[0], grid, 0.9)
            for a_, b_ in hb90['intervals']:
                ax.axvspan(a_, b_, color='blue', alpha=0.15)
            cm = D[0] >= thr[0.1]
            ivs, _ = H.mask_to_intervals(cm, grid)
            for a_, b_ in ivs:
                ax.axvspan(a_, b_, color='green', alpha=0.15)
            ax.axvline(q['target_phi'], color='r', lw=2,
                       label=f"target {q['target_phi']}%")
            ax.set_xlabel('hardened porosity (pp)')
            ax.set_ylabel('conditional density')
            ax.set_title(f"{q['case']}: empirical local state support "
                         f"(blue=base90, green=conf90)")
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(
                FIG, f"{q['case'].lower()}_conditional_density.png"),
                dpi=150)
            plt.close(fig)
    pd.DataFrame(recs).to_csv(
        os.path.join(OUT, 'historical_diagnostics.csv'), index=False)
    cols = ['case', 'target_phi', 'gate_status', 'conf90_hdr',
            'conf90_width', 'conf90_inside', 'base90_hdr', 'base90_inside']
    print(pd.DataFrame(recs)[[c for c in cols if c in
                              pd.DataFrame(recs).columns]].to_string(),
          flush=True)


if __name__ == '__main__':
    main()
