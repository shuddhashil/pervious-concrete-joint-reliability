import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Harmonised PSD feature extraction using the selected global bin system
# from the bootstrap/Pareto bin-selection workflow.
#
# Input  : PSD_master.csv with columns:
#          Paper_ID, Sample_ID, Sieve_mm, Weight_retained
# Output : PSD_global_retained_bins_selected.csv
#          - retained percentage in each selected global bin
#          - weighted average size, WAS_mm
#          - coefficient of uniformity, Cu_D60_over_D10
#          - aggregate surface area proxy, Aagg_per_mm
#          - rows preserve the first-appearance order of Paper_ID/Sample_ID
#            combinations in PSD_master.csv
#
# Units:
#   Retained-bin columns: percent by mass, normalised to sum to 100 within the
#                         selected 2.36-25 mm aggregate range.
#   WAS_mm:              mm
#   D10_mm, D60_mm:      mm
#   Cu_D60_over_D10:     dimensionless
#   Aagg_per_mm:         mm^-1, computed as sum(6*f_i/d_i), where f_i is the
#                         retained fraction in bin i and d_i is the geometric
#                         mean bin diameter.
# -----------------------------------------------------------------------------

MASTER_FILE = "PSD_master.csv"
OUTPUT_FILE = "PSD_global_retained_bins_selected.csv"

# Selected global bin edges from previous bin-selection result:
# B6_25_19_12p5_9p5_6p3_4p75_2p36
GLOBAL_SIEVES = np.array([2.36, 4.75, 6.3, 9.5, 12.5, 19.0, 25.0], dtype=float)
MIN_SIZE = float(GLOBAL_SIEVES.min())
MAX_SIZE = float(GLOBAL_SIEVES.max())
EPS = 1e-12

# Dense grid used only to compute D10 and D60 from the harmonised CDF.
N_DENSE = 500


def display_edge(x):
    return f"{float(x):.2f}".rstrip("0").rstrip(".")


def logw(a, b):
    a = max(float(a), EPS)
    b = max(float(b), EPS)
    return np.log(b) - np.log(a)


def overlap_log_fraction(lo, hi, glo, ghi):
    """Fraction of interval (lo, hi] that overlaps global bin (glo, ghi]
    assuming log-uniform mass distribution within the interval.
    """
    o_lo = max(float(lo), float(glo))
    o_hi = min(float(hi), float(ghi))

    if o_hi <= o_lo:
        return 0.0

    denom = logw(lo, hi)
    if denom <= 0:
        return 0.0

    return logw(o_lo, o_hi) / denom


def normalise_to_fraction(x):
    x = np.asarray(x, dtype=float)
    s = np.nansum(x)
    if not np.isfinite(s) or s <= 0:
        return np.zeros_like(x, dtype=float)
    return x / s


def make_bins(edges):
    """Return ascending and descending bin definitions.

    bins_asc  : [(2.36,4.75), ..., (19,25)]
    bins_desc : [(25,19), ..., (4.75,2.36)] as (upper, lower)
    """
    edges = np.asarray(edges, dtype=float)
    if np.any(np.diff(edges) <= 0):
        raise ValueError(f"GLOBAL_SIEVES must be strictly increasing: {edges}")

    bins_asc = list(zip(edges[:-1], edges[1:]))
    bins_desc = [(hi, lo) for lo, hi in bins_asc[::-1]]
    rep_desc = np.array([np.sqrt(upper * lower) for upper, lower in bins_desc], dtype=float)
    col_desc = [f"R{display_edge(upper)}-{display_edge(lower)}mm" for upper, lower in bins_desc]
    label_desc = [f"{display_edge(upper)}-{display_edge(lower)} mm" for upper, lower in bins_desc]

    return bins_asc, bins_desc, rep_desc, col_desc, label_desc


def raw_intervals_for_sample(g):
    """Convert retained-on-sieve records into particle-size intervals.

    Retained on sieve S is assigned to (S, previous larger sieve]. For the
    largest reported sieve, the upper edge is set to MAX_SIZE.
    """
    g = g.groupby("Sieve_mm", as_index=False)["Weight_retained"].sum()
    g = g.sort_values("Sieve_mm", ascending=False).reset_index(drop=True)

    total = g["Weight_retained"].sum()
    if total <= 0:
        return []

    sieves = g["Sieve_mm"].to_numpy(dtype=float)
    weights = g["Weight_retained"].to_numpy(dtype=float)
    pct_ret = 100.0 * weights / total

    intervals = []
    for i, sieve in enumerate(sieves):
        mass = float(pct_ret[i])
        if mass <= 0:
            continue

        lower = float(sieve)
        upper = MAX_SIZE if i == 0 else float(sieves[i - 1])

        # Restrict to the selected aggregate range. In your dataset this should
        # retain all aggregate mass, but the clipping keeps the code robust.
        lower = max(lower, MIN_SIZE)
        upper = min(upper, MAX_SIZE)

        if upper > lower:
            intervals.append((lower, upper, mass))

    return intervals


def allocate_to_selected_bins(intervals, bins_asc):
    """Allocate raw interval masses into selected global bins.

    Returns retained percentages in descending bin order, normalised to sum to
    100 across the selected global aggregate bins.
    """
    ret_asc = np.zeros(len(bins_asc), dtype=float)

    for lo, hi, mass in intervals:
        for j, (glo, ghi) in enumerate(bins_asc):
            frac = overlap_log_fraction(lo, hi, glo, ghi)
            if frac > 0:
                ret_asc[j] += float(mass) * frac

    ret_asc[ret_asc < 0] = 0.0
    total_in = np.nansum(ret_asc)

    if total_in <= 0:
        return ret_asc[::-1]

    # Normalise to exactly 100% in the selected global-bin representation.
    return 100.0 * ret_asc[::-1] / total_in


def intervals_from_retained_desc(ret_desc, bins_desc):
    """Convert descending retained-bin percentages into intervals (lo, hi, mass)."""
    intervals = []
    for mass, (upper, lower) in zip(ret_desc, bins_desc):
        if mass > 0:
            intervals.append((float(lower), float(upper), float(mass)))
    return intervals


def passing_from_intervals(intervals, thresholds):
    """Percent passing at each threshold using log-uniform mass within bins."""
    total = sum(mass for _, _, mass in intervals)
    if total <= 0:
        return np.full(len(thresholds), np.nan, dtype=float)

    y = []
    for t in thresholds:
        mass_le = 0.0
        for lo, hi, mass in intervals:
            if t <= lo:
                frac = 0.0
            elif t >= hi:
                frac = 1.0
            else:
                frac = logw(lo, t) / logw(lo, hi)
            mass_le += mass * frac
        y.append(100.0 * mass_le / total)

    return np.asarray(y, dtype=float)


def diameter_at_percent_passing(passing, thresholds, pct):
    """Interpolate diameter at a target percent passing value."""
    passing = np.asarray(passing, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    mask = np.isfinite(passing) & np.isfinite(thresholds) & (thresholds > 0)

    if mask.sum() < 2:
        return np.nan

    p = passing[mask]
    x = np.log(thresholds[mask])

    order = np.argsort(p)
    p = p[order]
    x = x[order]

    # Collapse duplicate passing values so interpolation is well-defined.
    unique_p = []
    unique_x = []
    for val in np.unique(p):
        xs = x[p == val]
        unique_p.append(val)
        unique_x.append(np.mean(xs))

    unique_p = np.asarray(unique_p, dtype=float)
    unique_x = np.asarray(unique_x, dtype=float)

    if len(unique_p) < 2 or pct < unique_p.min() or pct > unique_p.max():
        return np.nan

    return float(np.exp(np.interp(pct, unique_p, unique_x)))


def calculate_descriptors(ret_desc, bins_desc, rep_desc):
    """Compute WAS, D10, D60, Cu, and aggregate surface-area proxy."""
    f = normalise_to_fraction(ret_desc)

    if f.sum() <= 0:
        return {
            "WAS_mm": np.nan,
            "D10_mm": np.nan,
            "D60_mm": np.nan,
            "Cu_D60_over_D10": np.nan,
            "Aagg_per_mm": np.nan,
        }

    was = float(np.sum(f * rep_desc))
    aagg = float(np.sum(6.0 * f / rep_desc))

    harm_intervals = intervals_from_retained_desc(ret_desc, bins_desc)
    thresholds = np.geomspace(MIN_SIZE, MAX_SIZE, N_DENSE)
    passing = passing_from_intervals(harm_intervals, thresholds)

    d10 = diameter_at_percent_passing(passing, thresholds, 10.0)
    d60 = diameter_at_percent_passing(passing, thresholds, 60.0)
    cu = float(d60 / d10) if np.isfinite(d10) and np.isfinite(d60) and d10 > 0 else np.nan

    return {
        "WAS_mm": was,
        "D10_mm": d10,
        "D60_mm": d60,
        "Cu_D60_over_D10": cu,
        "Aagg_per_mm": aagg,
    }


def main():
    df = pd.read_csv(MASTER_FILE)

    required = {"Paper_ID", "Sample_ID", "Sieve_mm", "Weight_retained"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {MASTER_FILE}: {sorted(missing)}")

    df = df.copy()
    df["Paper_ID"] = df["Paper_ID"].astype(str).str.strip()
    df["Sample_ID"] = df["Sample_ID"].astype(str).str.strip()
    df["Sieve_mm"] = pd.to_numeric(df["Sieve_mm"], errors="coerce")
    df["Weight_retained"] = pd.to_numeric(df["Weight_retained"], errors="coerce")
    df = df.dropna(subset=["Paper_ID", "Sample_ID", "Sieve_mm", "Weight_retained"])
    df = df[(df["Sieve_mm"] > 0) & (df["Weight_retained"] >= 0)]

    bins_asc, bins_desc, rep_desc, bin_cols_desc, bin_labels_desc = make_bins(GLOBAL_SIEVES)

    records = []

    # Preserve the same Paper_ID/Sample_ID order as the master file.
    # pandas groupby sorts keys alphabetically by default; sort=False prevents that
    # and keeps groups in first-appearance order after cleaning.
    for row_order, ((paper_id, sample_id), g) in enumerate(df.groupby(["Paper_ID", "Sample_ID"], sort=False)):
        intervals = raw_intervals_for_sample(g)
        if not intervals:
            continue

        ret_desc = allocate_to_selected_bins(intervals, bins_asc)
        descriptors = calculate_descriptors(ret_desc, bins_desc, rep_desc)

        rec = {"_Master_order": row_order, "Paper_ID": paper_id, "Sample_ID": sample_id}

        # Retained fractions in descending bin order: coarse to fine.
        for col, val in zip(bin_cols_desc, ret_desc):
            rec[col] = float(val)

        rec.update(descriptors)
        records.append(rec)

    out = pd.DataFrame(records)

    # Keep master-file ordering, then remove the internal helper column.
    if "_Master_order" in out.columns:
        out = out.sort_values("_Master_order", kind="stable").drop(columns=["_Master_order"])
    out = out.reset_index(drop=True)
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved harmonised PSD features to: {OUTPUT_FILE}")
    print("Row order: first-appearance order of Paper_ID/Sample_ID combinations in PSD_master.csv")
    print(f"Selected global sieve edges (mm): {list(GLOBAL_SIEVES)}")
    print("Retained-bin columns, coarse to fine:")
    for label in bin_labels_desc:
        print(f"  {label}")
    print("Descriptor units:")
    print("  WAS_mm: mm")
    print("  D10_mm, D60_mm: mm")
    print("  Cu_D60_over_D10: dimensionless")
    print("  Aagg_per_mm: mm^-1")


if __name__ == "__main__":
    main()
