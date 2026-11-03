"""Reliability-vs-cement plot, built ONLY from a run's own saved
tables/FINAL_DECISIVE_VERIFICATION.csv -- never re-runs or re-computes
anything. Can compare more than one run on the same axes (e.g. old pooled
calibration vs the porosity-conditional calibration) by passing several run
directories.

CLI:
    python plot_reliability_vs_cement.py <run_dir> [<label>] [<run_dir2> [<label2>] ...]

    Each <run_dir> is a run folder containing tables/FINAL_DECISIVE_VERIFICATION.csv
    (e.g. runs/Stage1_Automated_20260922_212440/run_01). An optional label
    after a run_dir names that series in the legend; default is the run's
    own folder name. Output is written next to the FIRST run_dir given, as
    figures/RELIABILITY_VS_CEMENT.{png,pdf,svg} inside that run (or, if run
    standalone against a run outside any current run folder, to ./RELIABILITY_VS_CEMENT.*).

Importable:
    from plot_reliability_vs_cement import plot_reliability_vs_cement
    plot_reliability_vs_cement([(run_dir1, "Old (pooled)"), (run_dir2, "Corrected (porosity-conditional)")],
                                out_path_stem="figures/RELIABILITY_VS_CEMENT", target=0.90)
"""
from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from porosity_analysis import _figure_style, _save_figure  # noqa: E402  (reuse the project's own figure style/save helpers)

REQUIRED_COLS = ["cement_level_kg_m3", "Rhat_joint", "mc_wilson_lower_95",
                  "calib_bootstrap_R_LCB95", "R_LCB95_joint", "PASS_FAIL"]


def _load_series(run_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(run_dir, "tables", "FINAL_DECISIVE_VERIFICATION.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"no saved FINAL_DECISIVE_VERIFICATION.csv at {csv_path!r} "
                                 f"-- this plot only reads existing results, it never runs anything")
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{csv_path} is missing expected column(s) {missing}")
    return df.sort_values("cement_level_kg_m3").reset_index(drop=True)


def plot_reliability_vs_cement(runs, out_path_stem: str, *, target: float = 0.90):
    """runs: list of (run_dir, label) tuples, at least one. Reads only saved
    CSVs; writes out_path_stem.{png,pdf,svg} via the project's own
    _save_figure helper (same style as every other Stage-1 figure)."""
    plt = _figure_style()
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]
    pass_color, fail_color = "#1F8A4C", "#C23B3B"

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for (run_dir, label), color in zip(runs, palette):
        df = _load_series(run_dir)
        ax.plot(df["cement_level_kg_m3"], df["R_LCB95_joint"], "-", color=color, alpha=0.55,
                linewidth=1.5, label=label)
        for pf, marker in (("PASS", "o"), ("FAIL", "x")):
            sub = df[df["PASS_FAIL"] == pf]
            if not len(sub):
                continue
            kwargs = dict(marker=marker, s=46, zorder=5)
            if marker == "o":
                kwargs.update(color=pass_color, edgecolor=color, linewidth=1.2)
            else:
                kwargs.update(color=fail_color, linewidth=1.6)
            ax.scatter(sub["cement_level_kg_m3"], sub["R_LCB95_joint"], **kwargs)

    ax.axhline(float(target), color="black", ls="--", lw=1, label=f"target {float(target):.2f}")
    ax.set_xlabel("Cement (kg/m3)")
    ax.set_ylabel("Joint reliability lower bound (R_LCB95)")
    ax.set_title("Reliability vs. cement -- from saved decisive-verification results")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    _save_figure(fig, out_path_stem)
    return fig


def _default_out_stem(first_run_dir: str) -> str:
    figures_dir = os.path.join(first_run_dir, "figures")
    if os.path.isdir(os.path.dirname(first_run_dir)) or os.path.isdir(first_run_dir):
        return os.path.join(figures_dir, "RELIABILITY_VS_CEMENT")
    return "RELIABILITY_VS_CEMENT"


def _main(argv):
    if not argv:
        print(__doc__)
        raise SystemExit(1)
    runs = []
    i = 0
    while i < len(argv):
        run_dir = argv[i]
        label = None
        if i + 1 < len(argv) and not os.path.isdir(argv[i + 1]):
            label = argv[i + 1]
            i += 2
        else:
            i += 1
        runs.append((run_dir, label or os.path.basename(os.path.normpath(run_dir))))
    out_stem = _default_out_stem(runs[0][0])
    plot_reliability_vs_cement(runs, out_stem)
    print(f"Wrote {out_stem}.png / .pdf / .svg from {len(runs)} run(s):")
    for run_dir, label in runs:
        print(f"  {label}: {run_dir}")


if __name__ == "__main__":
    _main(sys.argv[1:])
