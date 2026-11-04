"""Held-out reliability audit (porosity-conditional, as reported in the paper) on a variant engine base (sandbox, 2026-10-21).
CalibrationAudit_A3-1_20261005/heldout_porosity_conditional.py is imported unchanged; only the models folder it reads is redirected.
With a study-grouped bank (variant G) each held-out record's residuals come only from OTHER studies.
usage: python run_audit_variant.py <variant_base_dir> <out_dir>"""
import sys
from pathlib import Path
import pandas as pd
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code" / "03_validation_and_audit" / "audit_code"))
import heldout_porosity_conditional as P
H = P.H; M = Path(sys.argv[1]) / "core" / "models"; out = Path(sys.argv[2])
H.MODELS = M; H.load_data.__defaults__ = (M,)
rc = H.main(["--out-dir", str(out)])
pd.DataFrame({"edge": P._PHI["edges"]}).to_csv(out / "porosity_tercile_edges.csv", index=False)
sys.exit(rc)
