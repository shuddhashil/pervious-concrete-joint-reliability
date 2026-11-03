"""Verify the 10D production wrapper: determinism, LB ordering, support, joint screen."""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\research\Stage1_CA_Forward_Refit_20260913\run_01\src")
from predict_ca_oracle import predict_ca_oracle_dataframe, joint_screen_from_predictions
RUN = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\research\Stage1_CA_Forward_Refit_20260913\run_01"
PT = os.path.join(RUN, "models", "lcmgp_ca_oracle_full_data.pt")
SUP = os.path.join(RUN, "models", "ocsvm_kde_feasibility_CA10.pkl")
COH = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\data\reconciled_phase2v\phase1_reconciled_eligible.csv"
df = pd.read_csv(COH).head(8)
out1 = predict_ca_oracle_dataframe(PT, df)
out2 = predict_ca_oracle_dataframe(PT, df)
assert np.allclose(out1["strength_pred_mpa"], out2["strength_pred_mpa"]), "non-deterministic"
assert np.all((out1["strength_LB95"] <= out1["strength_pred_mpa"]) & (out1["logperm_LB95"] <= out1["logperm_pred"]))
assert np.all(np.isfinite(out1[["strength_pred_mpa", "logperm_pred", "strength_std_effective_mpa", "D_AD", "AD_ratio"]].to_numpy()))
print(out1[["strength_pred_mpa", "strength_LB95", "perm_pred_mm_s", "perm_LB95_mm_s", "D_AD", "AD_ratio"]].round(3).to_string())
# support scores via 10D artefact
import joblib
art = joblib.load(SUP)
assert len(art["features"]) == 10 and art["x_cont_mean"].shape == (10,)
sys.path.insert(0, r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\code\feasibility")
import ocsvm_kde_feasibility_without_head as F
import LCMGP_without_head as L
L.FEATURES = list(art["features"]); L.CONT_IDXS = list(range(10)); L.TRANSFORM_RECIPE = dict(art["transform_recipe"])
import torch
obj = torch.load(PT, map_location="cpu", weights_only=False)
from predict_ca_oracle import _ensure_i_gap
Xs = _ensure_i_gap(pd.DataFrame(df))[art["features"]].apply(pd.to_numeric, errors="coerce").copy()
# apply transforms like module
Xm = L._apply_feature_transforms(Xs)[art["features"]].to_numpy(dtype=float)
Xss = (Xm - np.asarray(obj["x_cont_mean"])) / np.where(np.asarray(obj["x_cont_std"]) == 0, 1, np.asarray(obj["x_cont_std"]))
sc = F.score_feasibility(Xss, art)
print("support combined:", np.round(sc["combined_score"], 3))
# joint screen demo at 12/5 and 18/13, p=0.9
for tgt in [(12, 5), (18, 13)]:
    print(f"joint pass p=0.9 {tgt}:", joint_screen_from_predictions(out1, PT, tgt[0], tgt[1], p=0.9).tolist())
# 9D refusal check
try:
    from predict_ca_oracle import _activate_10d_contract
    _activate_10d_contract(r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\models\current\lcmgp_without_head_oracle_full_data.pt")
    print("9D REFUSAL FAILED")
except RuntimeError as e:
    print("9D refusal ok:", str(e)[:80])
print("WRAPPER VERIFY DONE")
