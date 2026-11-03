"""Stage 0 of a per-target run (runs INSIDE the per-target process: env GENGINE_BASE / GENGINE_CONFIG point at targets/<name>/). Computes the paste floor from the recorded data (null + reason when no unit meets the target),
builds the generic CONFIG, freezes it once (refuses to overwrite a different one) and writes paste_floor.json. usage (via run_target.py): python gengine/target_config.py S K R [seeds...]"""
import sys, os, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
from gengine import config, paste as PS, target as TG
from gengine.common import C, BASE
from gengine.data import Data

S, K, R = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]); seeds = [int(x) for x in sys.argv[4:]] or [0, 1, 2]
assert (C.S_STAR, C.K_STAR, C.REL_TARGET) == (S, K, R), f"the imported core carries target {(C.S_STAR, C.K_STAR, C.REL_TARGET)}, requested {(S, K, R)}"
name = TG.name_of(S, K, R); assert os.path.abspath(BASE) == os.path.abspath(os.path.join(TG.TARGETS, name)), "GENGINE_BASE is not this target's copy"
sp = config.shared_parts(); pf = PS.paste_floor(Data(dict(radii=sp["radii"], conflict=sp["conflict"])), S, K)
cfg = config.build_target(S, K, R, name, BASE, pf, seeds)
if os.path.exists(config.CJ):
    old = json.load(open(config.CJ)); assert old["config_sha256"] == cfg["config_sha256"], "per-target CONFIG already frozen with different content (other seeds / code?): use a new target folder or delete it deliberately"; print("CONFIG already frozen", cfg["config_sha256"])
else:
    json.dump(cfg, open(config.CJ, "w"), indent=1); open(config.CS, "w").write(cfg["config_sha256"] + "  CONFIG.json\n"); print("CONFIG frozen", cfg["config_sha256"])
os.makedirs(config.out_dir(cfg), exist_ok=True); json.dump(pf, open(os.path.join(config.out_dir(cfg), "paste_floor.json"), "w"), indent=1)
print(f"target {S:g}/{K:g}/{R:.2f} | paste floor status: {pf['status']} | h_min {pf['h_min']} | units meeting the full target: {pf['n_units_full_target']} | units with CS >= S: {pf['n_units_cs_ge_S']}")
if pf["reason"]: print("PASTE FLOOR NULL:", pf["reason"], "-> every result of this target is labelled 'model-suggested, no paste floor'")
if pf["warning"]: print(pf["warning"])
