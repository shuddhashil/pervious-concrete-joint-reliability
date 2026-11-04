"""Figures for Stage-17 handoff (matplotlib, Agg)."""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(RUN, 'outputs')
FIG = os.path.join(RUN, 'figures')
DMETHODS = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
COLORS = {'base': '#1f77b4', 'conf': '#2ca02c'}


def main():
    ds = pd.read_csv(os.path.join(OUT, 'density_score_summary.csv'))
    cal = pd.read_csv(os.path.join(OUT, 'hdr_calibration_summary.csv'))
    geo = pd.read_csv(os.path.join(OUT, 'hdr_geometry_summary.csv'))
    stab = pd.read_csv(os.path.join(OUT, 'stability_grid.csv'))

    # 1. density score comparison (x_raw + x9)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, rep in zip(axes, ['x_raw', 'x9']):
        d = ds[ds['rep'] == rep].set_index('method').loc[DMETHODS]
        x = np.arange(len(d))
        ax.bar(x, d['logscore'], yerr=d['logscore_fold_sd'],
               capsize=3, color='#4c78a8')
        ax.set_xticks(x)
        ax.set_xticklabels(d.index, rotation=45)
        ax.set_title(f'mean held-out log-score ({rep})')
        ax.set_ylabel('nats (higher = better)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'density_score_comparison.png'), dpi=150)
    plt.close(fig)

    # 2. coverage calibration (shortlist, x_raw)
    short = ['B0', 'B2', 'B4', 'B8']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, ver in zip(axes, ['base', 'conf']):
        for m in short:
            c = cal[(cal['rep'] == 'x_raw') & (cal['method'] == m)
                    & (cal['version'] == ver)].sort_values('level')
            ax.plot(c['level'], c['coverage'], 'o-', label=m)
        ax.plot([0.8, 0.95], [0.8, 0.95], 'k--', lw=1)
        ax.set_title(f'{ver} HDR calibration (x_raw)')
        ax.set_xlabel('nominal mass')
        ax.set_ylabel('observed coverage')
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'coverage_calibration.png'), dpi=150)
    plt.close(fig)

    # 3. HDR width comparison (90%, x_raw)
    g = geo[(geo['rep'] == 'x_raw') & (geo['level'] == 0.9)]
    gp = g.pivot(index='method', columns='version',
                 values='width_mean').loc[DMETHODS]
    x = np.arange(len(gp))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.2, gp['base'], 0.4, label='base',
           color=COLORS['base'])
    ax.bar(x + 0.2, gp['conf'], 0.4, label='conformal',
           color=COLORS['conf'])
    ax.set_xticks(x)
    ax.set_xticklabels(gp.index, rotation=45)
    ax.set_ylabel('mean total HDR measure (pp)')
    ax.set_title('90% HDR width by method (x_raw)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'hdr_measure_comparison.png'), dpi=150)
    plt.close(fig)

    # 4. mode counts (base90, x_raw)
    g = geo[(geo['rep'] == 'x_raw') & (geo['level'] == 0.9)
            & (geo['version'] == 'base')].set_index('method').loc[DMETHODS]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(g))
    ax.bar(x, g['frac_1mode'], 0.6, label='1 mode', color='#54a24b')
    ax.bar(x, g['frac_2mode'], 0.6, bottom=g['frac_1mode'],
           label='2 modes', color='#eeca3b')
    ax.bar(x, g['frac_3pmode'], 0.6,
           bottom=g['frac_1mode'] + g['frac_2mode'], label='3+ modes',
           color='#e45756')
    ax.set_xticks(x)
    ax.set_xticklabels(g.index, rotation=45)
    ax.set_ylabel('fraction of queries')
    ax.set_title('HDR connected components, base 90% (x_raw)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'mode_count_summary.png'), dpi=150)
    plt.close(fig)

    # 5. hyperparameter stability
    b2 = stab[stab['variant'].str.startswith('B2')].copy()
    b8 = stab[stab['variant'].str.startswith('B8')].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for v in b2['variant'].unique():
        r = b2[b2['variant'] == v].iloc[0]
        axes[0].plot(r['variant'], r['logscore'], 'o', color='#4c78a8')
    axes[0].tick_params(axis='x', rotation=60, labelsize=7)
    axes[0].set_ylabel('log-score (nats)')
    axes[0].set_title('B2 perturbation stability (5-fold CV)')
    axes[1].bar(b8['variant'], b8['logscore'], color='#f58518')
    axes[1].tick_params(axis='x', rotation=25, labelsize=8)
    axes[1].set_ylabel('log-score (nats)')
    axes[1].set_title('B8 member-config variants (5-fold CV)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'hyperparameter_stability.png'),
                dpi=150)
    plt.close(fig)
    print('figures written', flush=True)


if __name__ == '__main__':
    main()
