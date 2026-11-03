"""Re-run of the Stage-17 grouped nested CV (B8 ensemble) on the 631-record frame the deployed A3-1 gate was built from.
Reads the original Stage-17 code unchanged; only the frame loader is swapped for the CSV loader of GuidedEngine tools/build_s17.py.
Representation x_raw (the one the deployed bundle uses). Output: hdr_detail.csv with 'covered' per held-out row; zone table by observed porosity."""
import json, os, sys, time
sys.dont_write_bytecode = True
RES = "C:/Users/Shuddhashil/Desktop/Mix Design Paper Rebuild/research"
S17 = RES + "/Stage17_Direct_Local_Recipe_Porosity_Support_20260918"
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "outputs")
sys.path.insert(0, S17 + "/src")
import numpy as np, pandas as pd
import run_cv as R, data_contract as DC
BINS = ['R25-19mm', 'R19-12.5mm', 'R12.5-9.5mm', 'R9.5-6.3mm', 'R6.3-4.75mm', 'R4.75-2.36mm']
CSV = RES + "/GuidedEngine_20261003/base/core/data_inputs/phase1_reconciled_eligible.csv"
def load_frame_from_csv(path):  # copied from GuidedEngine tools/build_s17.py
    d = pd.read_csv(path); G = d[BINS].to_numpy(float) / 100.0; assert np.allclose(G.sum(1), 1.0, atol=1e-9)
    fr = pd.DataFrame({'pos': np.arange(len(d)), 'study': d['_src_ffill'].astype(str).to_numpy(), 'phi': d['porosity'].to_numpy(float), 'w/c': d['w/c'].to_numpy(float),
                       'cem': d['cem (kg/m3)'].to_numpy(float), 'water': d['water (kg/m3)'].to_numpy(float), 'CA': d['CA (kg/m3)'].to_numpy(float),
                       'sand': d['sand (kg/m3)'].to_numpy(float), 'SP': d['SP (kg/m3)'].to_numpy(float)})
    for j in range(6): fr[f'f{j + 1}'] = G[:, j]; fr[f'b{j + 1}'] = fr['CA'].to_numpy() * G[:, j]
    return fr
def main():
    cfg = json.load(open(S17 + "/config.json"))
    fr = DC.add_recipe_groups(load_frame_from_csv(CSV)); assert len(fr) == 631
    grid, glo, ghi = DC.phi_grid(fr['phi'].to_numpy()); fr.to_csv(os.path.join(OUT, "frame_631.csv"), index=False)
    groups = fr['recipe_group'].to_numpy(); done = R.done_set(OUT); t0 = time.time(); only = int(sys.argv[1]) if len(sys.argv) > 1 else None; n = 0
    for seed in cfg['cv']['outer_seeds']:
        fids = DC.grouped_folds(groups, 5, seed)
        for fold in range(5):
            if ('x_raw', seed, fold) in done: continue
            R.run_fold(fr, grid, 'x_raw', seed, fold, fids, OUT, cfg); n += 1
            print(f"done x_raw seed={seed} fold={fold} ({time.time()-t0:.0f}s)", flush=True)
            if only and n >= only: return
    print("all folds done", time.time() - t0, flush=True)
main()
