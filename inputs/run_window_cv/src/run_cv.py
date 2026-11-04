"""Grouped nested CV for Stage-17 direct local recipe-porosity support.

Per (representation, outer seed, outer fold):
  outer-train -> grouped 85/15 FIT/CAL split (GroupShuffleSplit)
  FIT: recipe gate calibration, method tuning (inner GroupKFold-3 log-score),
       refit, B8 stacking weights from inner-OOF scores only
  CAL: pointwise densities/scores -> conformal/section thresholds (test unseen)
  TEST: gate -> admitted rows get log-score, CRPS, CDE-loss terms, base HDRs,
        conformal HDRs, geometry, local evidence, decoy stress test

Splits never cross recipe groups; test phi never used in fit/tune/calibrate.
Results appended incrementally (resume-safe).
"""

import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conformal_hcds as CH
import conditional_density as CD
import data_contract as DC
import hdr as H
import recipe_space as RS

LEVELS = [0.8, 0.9, 0.95]
ALPHAS = [round(1.0 - lv, 10) for lv in LEVELS]
DMETHODS = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
SMETHODS = ['S1', 'S2', 'S3']
B8_MEMBERS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
DECOY_SHIFTS = {'d5': 5.0, 'd8': 8.0}


def load_all():
    fr = DC.load_frame()
    assert len(fr) == 650, len(fr)
    fr = DC.add_recipe_groups(fr)
    grid, glo, ghi = DC.phi_grid(fr['phi'].to_numpy())
    return fr, grid, glo, ghi


def prep_rep(fr, rep):
    if rep == 'x_raw':
        raw, _ = DC.build_xraw_raw(fr)
        Xt = DC.transform_xraw(raw)
    elif rep == 'x9':
        Xt, _ = DC.build_x9(fr)
    else:
        raise ValueError(rep)
    return Xt


def fit_scaler_fit(Xt_fit):
    m, s = DC.fit_scaler(Xt_fit)
    return m, s


def scale(Xt, m, s):
    return DC.apply_scaler(Xt, m, s)


def fit_nn1_distribution(Xs_fit, g_fit):
    """Leave-one-group-out nearest-neighbor distances within FIT."""
    from conditional_density import euclid2, apply_group_exclusion
    D = apply_group_exclusion(euclid2(Xs_fit, Xs_fit), g_fit, g_fit)
    np.fill_diagonal(D, np.inf)
    return np.sort(D, axis=1, kind='stable')[:, 0]


def top_study_share(Xs_fit, studies_fit, g_fit, Xs_q, g_q, k=50):
    from conditional_density import knn
    idx, _ = knn(Xs_fit, g_fit, Xs_q, g_q, k)
    out = np.empty(len(Xs_q))
    for r, ii in enumerate(idx):
        ii = ii[ii >= 0]
        if len(ii) == 0:
            out[r] = np.nan
            continue
        vals, cnt = np.unique(np.asarray(studies_fit)[ii],
                              return_counts=True)
        out[r] = float(cnt.max() / len(ii))
    return out, idx


def run_fold(fr, grid, rep, seed, fold, fold_ids, outdir, cfg):
    pos = fr['pos'].to_numpy()
    groups = fr['recipe_group'].to_numpy()
    studies = fr['study'].to_numpy()
    phi = fr['phi'].to_numpy(float)
    Xt_all = prep_rep(fr, rep)

    te = np.where(fold_ids == fold)[0]
    tr = np.where(fold_ids != fold)[0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15,
                            random_state=(int(seed) % 100000
                                          + int(fold) * 1000 + 7))
    fit_idx, cal_idx = next(gss.split(tr, groups=groups[tr]))
    FIT, CAL = tr[fit_idx], tr[cal_idx]

    m, s = fit_scaler_fit(Xt_all[FIT])
    Xs_all = scale(Xt_all, m, s)

    Xs_fit, Xt_fit = Xs_all[FIT], Xt_all[FIT]
    g_fit = groups[FIT]
    phi_fit = phi[FIT]

    # ---- recipe gate (x only) ----
    k_gate = int(cfg['recipe_gate']['k'])
    gate = RS.fit_gate(Xs_fit, k=k_gate)
    gthr, held = RS.calibrate_threshold(Xs_fit, g_fit, k=k_gate)
    gate_out_te, gate_d_te = RS.apply_gate(gate, Xs_all[te], gthr)
    gate_out_cal, _ = RS.apply_gate(gate, Xs_all[CAL], gthr)

    nn1_fit = fit_nn1_distribution(Xs_fit, g_fit)
    from conditional_density import knn as _knn
    nn1_te, _ = _knn(Xs_fit, g_fit, Xs_all[te], groups[te], 1)
    nn1_te = nn1_te[:, 0]
    nn1_pct = np.array([float((nn1_fit <= d).mean()) for d in nn1_te])
    gate_pct = np.array([float((held <= d).mean()) for d in gate_d_te])
    tss, _ = top_study_share(Xs_fit, studies[FIT], g_fit, Xs_all[te],
                             groups[te], k=50)

    grids_cfg = cfg['grids']
    # ---- tune + fit density methods ----
    fitted, oof, oof_pos, tuned, infos = {}, {}, {}, {}, {}
    for name in ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']:
        meth = CD.make_density_method(name, grids_cfg)
        t0 = time.time()
        tres = meth.tune(Xt_fit, Xs_fit, phi_fit, g_fit,
                         seed=int(seed) + fold)
        if len(tres) == 3:
            params, oof_sc, oof_p = tres  # B7: subsampled OOF + positions
        else:
            params, oof_sc = tres
            oof_p = np.arange(len(FIT))
        tuned[name] = {'params': params,
                       'tune_time_s': round(time.time() - t0, 1)}
        if name == 'B6':
            tuned[name]['bic_table'] = {
                str(k): v for k, v in meth._last_bic.items()}
        meth.fit(Xt_fit, Xs_fit, phi_fit, g_fit, params,
                 seed=int(seed) + fold)
        fitted[name] = meth
        oof[name] = np.asarray(oof_sc, float)
        oof_pos[name] = np.asarray(oof_p)
        infos[name] = dict(getattr(meth, 'info', {}))
        if hasattr(meth, 'n_unstable'):
            infos[name]['n_unstable_test'] = 0  # filled after test eval
    # ---- B8 stacking (inner-OOF only, common evaluated points) ----
    b8 = CD.B8Stack([fitted[nm] for nm in B8_MEMBERS])
    b8.fit_weights({nm: oof[nm] for nm in B8_MEMBERS},
                   {nm: oof_pos[nm] for nm in B8_MEMBERS})
    fitted['B8'] = b8
    tuned['B8'] = {'params': b8.info, 'tune_time_s': 0.0}
    infos['B8'] = dict(b8.info)

    # ---- CAL pointwise densities -> conformal thresholds ----
    cal_thr = {}
    for name in DMETHODS:
        pv = fitted[name].point_logpdf(Xt_all[CAL], Xs_all[CAL],
                                       groups[CAL], phi[CAL])
        cal_thr[name] = CH.calibrate_levels(np.exp(pv), alphas=ALPHAS)

    # ---- section methods ----
    sec, sec_thr, sec_tuned = {}, {}, {}
    for name in SMETHODS:
        meth = CD.make_section_method(name, grids_cfg)
        t0 = time.time()
        params, tuneres = meth.tune(Xs_fit, phi_fit, g_fit,
                                    seed=int(seed) + fold)
        sec_tuned[name] = {'params': params,
                           'tune_time_s': round(time.time() - t0, 1)}
        meth.fit(Xs_fit, phi_fit, g_fit, params, seed=int(seed) + fold)
        sec[name] = meth
        sc_cal = meth.obs_scores(Xs_all[CAL], phi[CAL])
        sec_thr[name] = CH.calibrate_levels(sc_cal, alphas=ALPHAS)

    # ---- TEST evaluation ----
    Xt_te, Xs_te = Xt_all[te], Xs_all[te]
    g_te, phi_te = groups[te], phi[te]
    adm = ~gate_out_te
    nq = len(te)

    base_rows = []   # per-row per-method base metrics
    hdr_rows = []    # long: row x method x version x level
    dens_cache = {}
    for name in DMETHODS:
        meth = fitted[name]
        lp = np.full(nq, np.nan)
        lp[adm] = meth.point_logpdf(Xt_te[adm], Xs_te[adm], g_te[adm],
                                    phi_te[adm])
        D = np.full((nq, len(grid)), np.nan)
        if adm.any():
            D[adm] = meth.grid_density(Xt_te[adm], Xs_te[adm], g_te[adm],
                                       grid)
        dens_cache[name] = D
        crps = np.full(nq, np.nan)
        intf2 = np.full(nq, np.nan)
        fobs = np.full(nq, np.nan)
        for r in np.where(adm)[0]:
            crps[r] = H.crps_from_density(D[r], grid, phi_te[r])
            a, b = H.cde_ise_terms(D[r], grid, phi_te[r])
            intf2[r], fobs[r] = a, b
        ess = np.full(nq, np.nan)
        if name in ('B1', 'B2', 'B3', 'B7'):
            ess[adm] = float(fitted[name].k)
        elif name == 'B0':
            ess[adm] = len(FIT)
        elif name in ('B4', 'B5'):
            W = (fitted[name]._fit_weights(Xs_te[adm], g_te[adm])
                 if name == 'B4'
                 else fitted[name]._weights(Xs_te[adm], g_te[adm]))
            sw = W.sum(1)
            ess[adm] = np.where(sw > 1e-8, sw ** 2 / (W ** 2).sum(1),
                                np.nan)
        elif name == 'B6':
            W, _, _ = fitted[name]._cond_params(Xs_te[adm])
            ess[adm] = len(FIT) * W.max(1)
        elif name == 'B8':
            acc = np.zeros(int(adm.sum()))
            for nm, w in zip(b8.member_names_, b8.weights_):
                e = ess_base(nm, fitted, Xs_te[adm], g_te[adm], len(FIT))
                acc += w * np.where(np.isfinite(e), e, 0.0)
            ess[adm] = acc
        # decoys at 90%: base + conformal
        hb90 = base_hdr_for(D, grid, 0.9)
        thr90 = cal_thr[name][0.1]
        rej = decoy_rejection(D, grid, phi_te, hb90, thr90, adm)
        for i, r in enumerate(te):
            base_rows.append({
                'rep': rep, 'seed': seed, 'fold': fold, 'pos': int(pos[r]),
                'method': name, 'logscore': float(lp[i]),
                'crps': float(crps[i]), 'int_f2': float(intf2[i]),
                'f_obs': float(fobs[i]), 'ess': float(ess[i]),
                'rej5_b90': rej['b90_5'][i], 'rej8_b90': rej['b90_8'][i],
                'rej5_c90': rej['c90_5'][i], 'rej8_c90': rej['c90_8'][i]})
        if hasattr(meth, 'n_unstable'):
            infos[name]['n_unstable_test'] = int(meth.n_unstable)
        # base + conformal HDR detail at all levels
        conf = CH.apply_levels(np.where(np.isfinite(D), D, 0.0), grid,
                               cal_thr[name])
        for lv, a in zip(LEVELS, ALPHAS):
            for i, r in enumerate(te):
                if not adm[i]:
                    for ver in ('base', 'conf'):
                        hdr_rows.append({
                            'rep': rep, 'seed': seed, 'fold': fold,
                            'pos': int(pos[r]), 'method': name,
                            'version': ver, 'level': lv,
                            'covered': np.nan, 'width': np.nan,
                            'nmodes': np.nan})
                    continue
                hb = H.density_to_hdr(D[i], grid, lv)
                cb = {'intervals': conf[a]['intervals'][i],
                      'total_measure': float(conf[a]['total_measure'][i]),
                      'n_modes': int(conf[a]['n_modes'][i])}
                hdr_rows.append({
                    'rep': rep, 'seed': seed, 'fold': fold,
                    'pos': int(pos[r]), 'method': name,
                    'version': 'base', 'level': lv,
                    'covered': float(H.covers(hb['intervals'], phi_te[i])),
                    'width': hb['total_measure'], 'nmodes': hb['n_modes']})
                hdr_rows.append({
                    'rep': rep, 'seed': seed, 'fold': fold,
                    'pos': int(pos[r]), 'method': name,
                    'version': 'conf', 'level': lv,
                    'covered': float(H.covers(cb['intervals'], phi_te[i])),
                    'width': cb['total_measure'], 'nmodes': cb['n_modes']})
    # ---- section methods on test ----
    for name in SMETHODS:
        meth = sec[name]
        S = np.full((nq, len(grid)), np.nan)
        if adm.any():
            S[adm] = meth.grid_scores(Xs_te[adm], grid)
        for i, r in enumerate(te):
            base_rows.append({
                'rep': rep, 'seed': seed, 'fold': fold, 'pos': int(pos[r]),
                'method': name, 'logscore': np.nan, 'crps': np.nan,
                'int_f2': np.nan, 'f_obs': np.nan, 'ess': np.nan,
                'rej5_b90': np.nan, 'rej8_b90': np.nan, 'rej5_c90': np.nan,
                'rej8_c90': np.nan})
        for lv, a in zip(LEVELS, ALPHAS):
            t = sec_thr[name][a]
            for i, r in enumerate(te):
                if not adm[i]:
                    hdr_rows.append({
                        'rep': rep, 'seed': seed, 'fold': fold,
                        'pos': int(pos[r]), 'method': name,
                        'version': 'conf', 'level': lv,
                        'covered': np.nan, 'width': np.nan,
                        'nmodes': np.nan})
                    continue
                mask = S[i] >= t
                ivs, tot = H.mask_to_intervals(mask, grid)
                hdr_rows.append({
                    'rep': rep, 'seed': seed, 'fold': fold,
                    'pos': int(pos[r]), 'method': name,
                    'version': 'conf', 'level': lv,
                    'covered': float(H.covers(ivs, phi_te[i])),
                    'width': float(tot), 'nmodes': len(ivs)})
        # section decoys at 90%
        t90 = sec_thr[name][0.1]
        for i, r in enumerate(te):
            if not adm[i]:
                continue
            sc = S[i]
            rej5, rej8 = [], []
            for sgn in (1.0, -1.0):
                for tag, sh in (('5', 5.0), ('8', 8.0)):
                    dp = phi_te[i] + sgn * sh
                    if grid[0] - 1e-9 <= dp <= grid[-1] + 1e-9:
                        v = float(np.interp(dp, grid, sc))
                        (rej5 if tag == '5' else rej8).append(v < t90)
            # store into base_rows entry
            for b in base_rows:
                if (b['pos'] == int(pos[r]) and b['method'] == name
                        and b['seed'] == seed and b['fold'] == fold
                        and b['rep'] == rep):
                    b['rej5_c90'] = float(np.mean(rej5)) if rej5 else np.nan
                    b['rej8_c90'] = float(np.mean(rej8)) if rej8 else np.nan
    # ---- gate rows ----
    gate_rows = [{
        'rep': rep, 'seed': seed, 'fold': fold, 'pos': int(pos[r]),
        'phi': float(phi_te[i]), 'study': str(studies[te[i]]),
        'group': int(g_te[i]), 'gate_out': bool(gate_out_te[i]),
        'gate_dist': float(gate_d_te[i]), 'gate_thr': float(gthr),
        'gate_pct': float(gate_pct[i]), 'nn1_dist': float(nn1_te[i])
        if np.isfinite(nn1_te[i]) else np.nan,
        'nn1_pct': float(nn1_pct[i]), 'topstudy_share': float(tss[i])}
        for i, r in enumerate(te)]

    append_csv(os.path.join(outdir, 'row_base.csv'), base_rows)
    append_csv(os.path.join(outdir, 'hdr_detail.csv'), hdr_rows)
    append_csv(os.path.join(outdir, 'gate_detail.csv'), gate_rows)
    foldrec = {'rep': rep, 'seed': seed, 'fold': fold,
               'n_fit': int(len(FIT)), 'n_cal': int(len(CAL)),
               'n_test': int(len(te)),
               'n_gate_out': int(gate_out_te.sum()),
               'gate_thr': float(gthr),
               'scaler_mean': [float(v) for v in m],
               'scaler_sd': [float(v) for v in s],
               'tuned': tuned, 'infos': infos,
               'sec_tuned': sec_tuned}
    with open(os.path.join(outdir, 'fold_params.jsonl'), 'a') as f:
        f.write(json.dumps(foldrec) + '\n')


def ess_base(nm, fitted, Xs_q, g_q, n_fit):
    if nm in ('B1', 'B2', 'B3', 'B7'):
        return np.full(len(Xs_q), float(fitted[nm].k))
    if nm in ('B4', 'B5'):
        W = (fitted[nm]._fit_weights(Xs_q, g_q) if nm == 'B4'
             else fitted[nm]._weights(Xs_q, g_q))
        sw = W.sum(1)
        return np.where(sw > 1e-8, sw ** 2 / (W ** 2).sum(1), np.nan)
    if nm == 'B6':
        W, _, _ = fitted[nm]._cond_params(Xs_q)
        return n_fit * W.max(1)
    if nm == 'B0':
        return np.full(len(Xs_q), float(n_fit))
    return np.full(len(Xs_q), np.nan)


def base_hdr_for(D, grid, level):
    out = []
    for i in range(D.shape[0]):
        if np.isfinite(D[i]).any():
            out.append(H.density_to_hdr(
                np.where(np.isfinite(D[i]), D[i], 0.0), grid, level))
        else:
            out.append({'intervals': [], 'total_measure': np.nan,
                        'n_modes': 0})
    return out


def decoy_rejection(D, grid, phi_te, hb90, thr90, adm):
    glo, ghi = float(grid[0]), float(grid[-1])
    rb5 = np.full(len(phi_te), np.nan)
    rb8 = np.full(len(phi_te), np.nan)
    rc5 = np.full(len(phi_te), np.nan)
    rc8 = np.full(len(phi_te), np.nan)
    for i in range(len(phi_te)):
        if not adm[i]:
            continue
        d = np.where(np.isfinite(D[i]), D[i], 0.0)
        for tag, sh, Rb, Rc in (('5', 5.0, rb5, rc5), ('8', 8.0, rb8, rc8)):
            rej_b, rej_c = [], []
            for sgn in (1.0, -1.0):
                dp = phi_te[i] + sgn * sh
                if glo - 1e-9 <= dp <= ghi + 1e-9:
                    rej_b.append(not H.covers(hb90[i]['intervals'], dp))
                    rej_c.append(float(np.interp(dp, grid, d)) < thr90)
            if rej_b:
                Rb[i] = float(np.mean(rej_b))
                Rc[i] = float(np.mean(rej_c))
    return {'b90_5': rb5, 'b90_8': rb8, 'c90_5': rc5, 'c90_8': rc8}


def append_csv(path, rows):
    if not rows:
        return
    new = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def done_set(outdir):
    p = os.path.join(outdir, 'gate_detail.csv')
    if not os.path.exists(p):
        return set()
    d = pd.read_csv(p, usecols=['rep', 'seed', 'fold'])
    return set(zip(d['rep'], d['seed'], d['fold']))


def main():
    t0 = time.time()
    with open(os.path.join(RUN_DIR, 'config.json')) as f:
        cfg = json.load(f)
    fr, grid, glo, ghi = load_all()
    np.save(os.path.join(RUN_DIR, 'outputs', '_phi_grid.npy'), grid)
    groups = fr['recipe_group'].to_numpy()
    reps = ['x_raw', 'x9']
    done = done_set(os.path.join(RUN_DIR, 'outputs'))
    total = 0
    for rep in reps:
        for seed in cfg['cv']['outer_seeds']:
            fids = DC.grouped_folds(groups, 5, seed)
            for fold in range(5):
                if (rep, seed, fold) in done:
                    continue
                run_fold(fr, grid, rep, seed, fold, fids,
                         os.path.join(RUN_DIR, 'outputs'), cfg)
                total += 1
                print(f'done {rep} seed={seed} fold={fold} '
                      f'({time.time()-t0:.0f}s)', flush=True)
    print(f'new folds: {total} total_time={time.time()-t0:.0f}s', flush=True)


RUN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    main()
