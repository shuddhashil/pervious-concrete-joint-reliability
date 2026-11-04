"""Aggregate row-level CV outputs into summary tables + scorecard.

Reads: row_base.csv, hdr_detail.csv, gate_detail.csv, fold_params.jsonl
Writes: fold_metrics.csv, density_score_summary.csv,
        hdr_calibration_summary.csv, hdr_geometry_summary.csv,
        local_evidence_summary.csv, sensitivity_summary.csv, scorecard.csv
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_contract as DC

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(RUN, 'outputs')
DMETHODS = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
SMETHODS = ['S1', 'S2', 'S3']
LEVELS = [0.8, 0.9, 0.95]


def wilson(p, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def wmean(v, w):
    v = np.asarray(v, float)
    w = np.asarray(w, float)
    m = np.isfinite(v)
    if m.sum() == 0:
        return np.nan
    return float((v[m] * w[m]).sum() / w[m].sum())


def main():
    rb = pd.read_csv(os.path.join(OUT, 'row_base.csv'))
    hd = pd.read_csv(os.path.join(OUT, 'hdr_detail.csv'))
    gd = pd.read_csv(os.path.join(OUT, 'gate_detail.csv'))
    # study weights per test appearance (equal-study sensitivity)
    sw = DC.study_weights(gd['study'].to_numpy())
    gd = gd.copy()
    gd['stw'] = sw
    key = ['rep', 'seed', 'fold', 'pos']
    rb = rb.merge(gd[key + ['stw', 'gate_out', 'nn1_pct', 'gate_pct',
                            'topstudy_share']], on=key, how='left')
    adm = rb[~rb['gate_out'].fillna(True)].copy()

    # ---------------- fold_metrics ----------------
    fm = []
    for (rep, seed, fold, method), g in adm.groupby(
            ['rep', 'seed', 'fold', 'method']):
        fm.append({'rep': rep, 'seed': seed, 'fold': fold,
                   'method': method, 'n_test': len(g),
                   'logscore': g['logscore'].mean(),
                   'logscore_sw': wmean(g['logscore'], g['stw']),
                   'crps': g['crps'].mean(),
                   'cde_ise': (g['int_f2'] - 2 * g['f_obs']).mean()})
    fm = pd.DataFrame(fm)
    gh = gd.groupby(['rep', 'seed', 'fold']).agg(
        n_test=('pos', 'size'), n_gate_out=('gate_out', 'sum')).reset_index()
    fm = fm.merge(gh, on=['rep', 'seed', 'fold'], how='left')
    fm.to_csv(os.path.join(OUT, 'fold_metrics.csv'), index=False)

    # ---------------- density_score_summary ----------------
    ds = []
    for (rep, method), g in adm.groupby(['rep', 'method']):
        per_fold = g.groupby(['seed', 'fold'])['logscore'].mean()
        ds.append({'rep': rep, 'method': method,
                   'n': len(g), 'logscore': g['logscore'].mean(),
                   'logscore_sw': wmean(g['logscore'], g['stw']),
                   'logscore_fold_sd': per_fold.std(),
                   'crps': g['crps'].mean(),
                   'crps_sw': wmean(g['crps'], g['stw']),
                   'cde_ise': (g['int_f2'] - 2 * g['f_obs']).mean()})
    ds = pd.DataFrame(ds)
    ds.to_csv(os.path.join(OUT, 'density_score_summary.csv'), index=False)

    # ---------------- calibration + geometry ----------------
    cal, geo = [], []
    hda = hd.merge(gd[key + ['gate_out']], on=key, how='left')
    hda = hda[~hda['gate_out'].fillna(True)]
    for (rep, method, version, level), g in hda.groupby(
            ['rep', 'method', 'version', 'level']):
        g = g.dropna(subset=['covered'])
        p = float(g['covered'].mean()) if len(g) else np.nan
        lo, hi = wilson(p, len(g))
        folds = g.groupby(['seed', 'fold'])['covered'].mean()
        w = g['width'].to_numpy(float)
        nm = g['nmodes'].to_numpy(float)
        cal.append({'rep': rep, 'method': method, 'version': version,
                    'level': level, 'n': len(g), 'coverage': p,
                    'wilson_lo': lo, 'wilson_hi': hi,
                    'fold_cov_sd': folds.std(),
                    'cal_err': abs(p - level)})
        geo.append({'rep': rep, 'method': method, 'version': version,
                    'level': level, 'n': len(g),
                    'width_mean': np.mean(w), 'width_med': np.median(w),
                    'width_p90': np.percentile(w, 90),
                    'frac_1mode': np.mean(nm == 1),
                    'frac_2mode': np.mean(nm == 2),
                    'frac_3pmode': np.mean(nm >= 3),
                    'frac_empty': np.mean(nm == 0)})
    cal = pd.DataFrame(cal)
    geo = pd.DataFrame(geo)
    cal.to_csv(os.path.join(OUT, 'hdr_calibration_summary.csv'),
               index=False)
    geo.to_csv(os.path.join(OUT, 'hdr_geometry_summary.csv'), index=False)

    # abstention
    abst = gd.groupby('rep').agg(abst_rate=('gate_out', 'mean'),
                                 n=('pos', 'size')).reset_index()

    # ---------------- local evidence ----------------
    le = []
    for (rep, method), g in adm.groupby(['rep', 'method']):
        le.append({'rep': rep, 'method': method,
                   'ess_med': g['ess'].median(),
                   'ess_p10': g['ess'].quantile(0.1),
                   'nn1_pct_med': g['nn1_pct'].median(),
                   'gate_pct_med': g['gate_pct'].median(),
                   'topstudy_med': g['topstudy_share'].median(),
                   'topstudy_p90': g['topstudy_share'].quantile(0.9),
                   'rej5_b90': g['rej5_b90'].mean(),
                   'rej8_b90': g['rej8_b90'].mean(),
                   'rej5_c90': g['rej5_c90'].mean(),
                   'rej8_c90': g['rej8_c90'].mean()})
    le = pd.DataFrame(le).merge(abst, on='rep', how='left')
    # tuned-hyperparameter distributions from fold_params
    tun = []
    for line in open(os.path.join(OUT, 'fold_params.jsonl')):
        r = json.loads(line)
        for mn, t in r['tuned'].items():
            tun.append({'rep': r['rep'], 'method': mn,
                        'params': json.dumps(t['params'], sort_keys=True)})
    tun = pd.DataFrame(tun)
    tdist = tun.groupby(['rep', 'method', 'params']).size().reset_index(
        name='n_folds')
    tdist.to_csv(os.path.join(OUT, 'tuning_distribution.csv'), index=False)
    # B8 weights across folds
    b8w = []
    for line in open(os.path.join(OUT, 'fold_params.jsonl')):
        r = json.loads(line)
        w = r['tuned']['B8']['params']['weights']
        w = dict(w)
        w.update({'rep': r['rep'], 'seed': r['seed'], 'fold': r['fold'],
                  'gain': r['tuned']['B8']['params']['oof_gain_over_best'],
                  'best_single':
                  r['tuned']['B8']['params']['best_single']})
        b8w.append(w)
    pd.DataFrame(b8w).to_csv(os.path.join(OUT, 'b8_weights.csv'),
                             index=False)
    le.to_csv(os.path.join(OUT, 'local_evidence_summary.csv'), index=False)

    # ---------------- sensitivity: study-weighted ----------------
    hdas = hda.merge(gd[key + ['stw']], on=key, how='left')
    sens = []
    for (rep, method, version, level), g in hdas.groupby(
            ['rep', 'method', 'version', 'level']):
        g = g.dropna(subset=['covered'])
        sens.append({'rep': rep, 'method': method, 'version': version,
                     'level': level,
                     'coverage_sw': wmean(g['covered'], g['stw']),
                     'width_sw': wmean(g['width'], g['stw'])})
    sens = pd.DataFrame(sens)
    sens.to_csv(os.path.join(OUT, 'sensitivity_summary.csv'), index=False)

    # ---------------- scorecard (primary rep x_raw, 90%) ----------------
    sc = scorecard(ds, cal, geo, le)
    sc.to_csv(os.path.join(OUT, 'scorecard.csv'), index=False)
    print(sc.to_string())
    print('\nabstention:')
    print(abst.to_string())


def scorecard(ds, cal, geo, le):
    """One row per (method, version): primary-rep 90% roll-up + B0 deltas."""
    rows = []
    b0 = ds[(ds['rep'] == 'x_raw') & (ds['method'] == 'B0')]
    b0_ls = float(b0['logscore'].iloc[0]) if len(b0) else np.nan
    b0w = geo[(geo['rep'] == 'x_raw') & (geo['method'] == 'B0')
              & (geo['version'] == 'base') & (geo['level'] == 0.9)]
    b0_w = float(b0w['width_mean'].iloc[0]) if len(b0w) else np.nan
    for method in DMETHODS + SMETHODS:
        for ver in (['base', 'conf'] if method in DMETHODS else ['conf']):
            d = ds[(ds['rep'] == 'x_raw') & (ds['method'] == method)]
            c = cal[(cal['rep'] == 'x_raw') & (cal['method'] == method)
                    & (cal['version'] == ver) & (cal['level'] == 0.9)]
            gm = geo[(geo['rep'] == 'x_raw') & (geo['method'] == method)
                     & (geo['version'] == ver) & (geo['level'] == 0.9)]
            l = le[(le['rep'] == 'x_raw') & (le['method'] == method)]
            # calibration error across 80/90/95
            cx = cal[(cal['rep'] == 'x_raw') & (cal['method'] == method)
                     & (cal['version'] == ver)]
            rows.append({
                'method': method, 'version': ver,
                'logscore': _f(d, 'logscore'),
                'd_logscore_vs_B0': _f(d, 'logscore') - b0_ls,
                'logscore_sw': _f(d, 'logscore_sw'),
                'logscore_fold_sd': _f(d, 'logscore_fold_sd'),
                'crps': _f(d, 'crps'), 'cde_ise': _f(d, 'cde_ise'),
                'cov90': _f(c, 'coverage'),
                'cov90_lo': _f(c, 'wilson_lo'),
                'cov90_hi': _f(c, 'wilson_hi'),
                'cov_fold_sd': _f(c, 'fold_cov_sd'),
                'cal_err_mean809095': cx['cal_err'].mean()
                if len(cx) else np.nan,
                'width90_mean': _f(gm, 'width_mean'),
                'width90_med': _f(gm, 'width_med'),
                'width90_p90': _f(gm, 'width_p90'),
                'width_ratio_vs_B0': _f(gm, 'width_mean') / b0_w,
                'frac_2mode': _f(gm, 'frac_2mode'),
                'frac_3pmode': _f(gm, 'frac_3pmode'),
                'frac_empty': _f(gm, 'frac_empty'),
                'rej8': _f(l, 'rej8_c90' if ver == 'conf'
                           else 'rej8_b90'),
                'ess_med': _f(l, 'ess_med'),
                'topstudy_med': _f(l, 'topstudy_med')})
    return pd.DataFrame(rows)


def _f(df, col):
    return float(df[col].iloc[0]) if len(df) else np.nan


if __name__ == '__main__':
    main()
