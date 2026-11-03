import os
import numpy as np
import pandas as pd

MASTER_FILE = "PSD_master.csv"

def calculate_grain_size_parameters(data: pd.DataFrame) -> dict:
    sieves = data['Sieve']
    weight_retained = data['weight retained']

    total_weight = np.sum(weight_retained)
    if total_weight == 0:
        raise ValueError("Total weight is zero, cannot perform analysis.")

    pct_retained = (weight_retained / total_weight) * 100.0
    cum_pct_retained = np.cumsum(pct_retained)
    cum_pct_passing = 100.0 - cum_pct_retained

    sieves_full = sieves
    passing_full = cum_pct_passing

    def interpolate_diameter(target_percent: float) -> float:
        x = passing_full[::-1].values
        y = sieves_full[::-1].values
        y_log = np.log10(y)
        log_d = np.interp(target_percent, x, y_log)
        return float(10.0 ** log_d)

    d10 = interpolate_diameter(10.0)
    d30 = interpolate_diameter(30.0)
    d60 = interpolate_diameter(60.0)

    cu = (d60 / d10) if d10 > 0 else 0.0
    cc = ((d30 ** 2) / (d10 * d60)) if (d10 > 0 and d60 > 0) else 0.0

    mean_sizes = []
    for i in range(len(sieves) - 1):
        mean_size = (sieves.iloc[i] + sieves.iloc[i + 1]) / 2.0
        mean_sizes.append(mean_size)
    was = np.sum(np.array(mean_sizes) * pct_retained.iloc[1:].to_numpy() / 100.0)

    nonzero_indices = np.where(weight_retained.to_numpy() > 0)[0]
    if len(nonzero_indices) > 0:
        first_nonzero = nonzero_indices[0]
        max_size = sieves.iloc[first_nonzero - 1] if first_nonzero > 0 else sieves.iloc[0]
        min_size = sieves.iloc[nonzero_indices[-1]]
    else:
        max_size = np.nan
        min_size = np.nan

    return {
        'Cu': round(cu, 3),
        'Cc': round(cc, 3),
        'WAS': round(was, 3),
        'Max_Size': max_size,
        'Min_Size': min_size
    }

def main():
    print("=== PSD Data Collection (Part 1) ===")
    paper_id = input("Enter Paper ID (e.g., P1): ").strip()

    n = int(input("\nEnter number of sieves used: ").strip())
    sieve_list = []
    print("\nEnter sieve openings (in mm):")
    for i in range(n):
        sieve = float(input(f"Sieve {i+1}: ").strip())
        sieve_list.append(sieve)

    num_samples = int(input("\nEnter number of samples in this paper: ").strip())

    rows = []

    for s in range(num_samples):
        sample_id = input(f"\nEnter Sample ID for sample {s+1}: ").strip()

        weight_list = []
        print(f"Enter weights retained on each sieve for sample '{sample_id}':")
        for i in range(n):
            wt = float(input(f"  Weight retained on sieve {sieve_list[i]} mm: ").strip())
            weight_list.append(wt)

        # ---- NEW: compute and print Cu, Cc, WAS, Max, Min ----
        df_temp = pd.DataFrame({'Sieve': sieve_list, 'weight retained': weight_list})
        grad = calculate_grain_size_parameters(df_temp)

        print("\nGrain-size metrics (not stored, printed only):")
        print(f"  Cu: {grad['Cu']}")
        print(f"  Cc: {grad['Cc']}")
        print(f"  WAS: {grad['WAS']}")
        print(f"  Max_Size: {grad['Max_Size']}")
        print(f"  Min_Size: {grad['Min_Size']}")

        # ---- Store master PSD info (unchanged) ----
        for sieve_mm, wt in zip(sieve_list, weight_list):
            rows.append({
                "Paper_ID": paper_id,
                "Sample_ID": sample_id,
                "Sieve_mm": sieve_mm,
                "Weight_retained": wt
            })

    df_new = pd.DataFrame(rows)
    if os.path.exists(MASTER_FILE):
        df_new.to_csv(MASTER_FILE, mode="a", header=False, index=False)
    else:
        df_new.to_csv(MASTER_FILE, index=False)

    print("\nSaved / appended to PSD_master.csv")
    print("You can now enter another paper or later run PCA (Part 2).")

if __name__ == "__main__":
    main()
