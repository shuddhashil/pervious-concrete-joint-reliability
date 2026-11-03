"""Per-target copy of the frozen core + defengine (Phase-2 deviation 8, now built). The frozen core reads its target ONLY from core/TARGET_CONTRACT.json at import time, so one target = one process = one copy
(targets/<name>/core, targets/<name>/defengine). The copy differs from base/ in exactly one file, core/TARGET_CONTRACT.json (and not even that when the target equals base's own). Pure file work: no torch, no core import."""
import os, sys, json, shutil, hashlib
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); BASE0 = os.path.join(ROOT, "base"); TARGETS = os.path.join(ROOT, "targets")
KEYS = ("models/lcmgp_ca_oracle_full_data.pt", "models/ocsvm_kde_feasibility_CA10.pkl", "tables/paired_residual_bank.csv", "state_support_stage17/b8_fulldata_bundle.pkl", "GRADATION_SUPPORT_CONTRACT.json", "UNCERTAINTY_CONTRACT.json", "TARGET_CONTRACT.json")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def name_of(S, K, R):
    return f"S{S:g}_K{K:g}_R{R:.2f}"


def manifest(b):
    """same construction as tools/make_manifest.py, for any folder b holding core/ and defengine/ -> (core dict, defengine dict, manifest text, hashes dict)."""
    core, deng = {}, {}
    for dp, dn, fn in os.walk(os.path.join(b, "core")):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn: core[os.path.relpath(os.path.join(dp, f), os.path.join(b, "core")).replace("\\", "/")] = sha(os.path.join(dp, f))
    for f in os.listdir(os.path.join(b, "defengine")):
        if f.endswith(".py"): deng["defengine/" + f] = sha(os.path.join(b, "defengine", f))
    text = json.dumps(core, indent=0, sort_keys=True); csha = hashlib.sha256(text.encode().replace(b"\n", b"\r\n")).hexdigest()   # hash of the file AS WRITTEN in text mode (tools/make_manifest.py convention: CRLF on Windows)
    tot = hashlib.sha256(json.dumps([core, deng], sort_keys=True).encode()).hexdigest()
    return core, deng, text, dict(core_manifest_sha256=csha, combined_core_and_defengine_sha256=tot, n_core_files=len(core), defengine_files=deng, key={k: core[k] for k in KEYS})


def differences(b0, b1):
    """files (relative, core/ and defengine/*.py) that differ or exist on one side only between two copies."""
    c0, d0, _, _ = manifest(b0); c1, d1, _, _ = manifest(b1)
    return sorted(["core/" + k for k in set(c0) | set(c1) if c0.get(k) != c1.get(k)] + [k for k in set(d0) | set(d1) if d0.get(k) != d1.get(k)])


def make_target(S, K, R):
    """create (or re-verify) targets/<name>/ ; returns dict(dir, name, differs_from_base). Raises if the copy differs from base in anything but core/TARGET_CONTRACT.json."""
    name = name_of(S, K, R); t = os.path.join(TARGETS, name); ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    if not os.path.exists(os.path.join(t, "core")):
        os.makedirs(t, exist_ok=True); shutil.copytree(os.path.join(BASE0, "core"), os.path.join(t, "core"), ignore=ign); shutil.copytree(os.path.join(BASE0, "defengine"), os.path.join(t, "defengine"), ignore=ign)
        tc = os.path.join(t, "core", "TARGET_CONTRACT.json"); c = json.load(open(tc, encoding="utf-8")); want = (float(S), float(K), float(R)); have = (c["strength_target_mpa"], c["permeability_target_mm_s"], c["required_joint_reliability"])
        if want != have:                                                       # rewrite ONLY the three target fields; nothing is touched when the target equals base's own (copy stays byte-identical)
            c.update(strength_target_mpa=want[0], permeability_target_mm_s=want[1], required_joint_reliability=want[2]); open(tc, "w", encoding="utf-8").write(json.dumps(c, indent=2) + "\n")
        _, _, text, hs = manifest(t); open(os.path.join(t, "CORE_MANIFEST.json"), "w").write(text); json.dump(hs, open(os.path.join(t, "CORE_HASH.json"), "w"), indent=1)
    else:                                                                      # re-verify an existing copy against its own manifest
        _, _, text, hs = manifest(t); assert open(os.path.join(t, "CORE_MANIFEST.json")).read() == text and json.load(open(os.path.join(t, "CORE_HASH.json"))) == hs, f"{name}: existing copy no longer matches its manifest"
    diff = differences(BASE0, t); allowed = [] if (float(S), float(K), float(R)) == tuple(json.load(open(os.path.join(BASE0, "core", "TARGET_CONTRACT.json")))[k] for k in ("strength_target_mpa", "permeability_target_mm_s", "required_joint_reliability")) else ["core/TARGET_CONTRACT.json"]
    assert diff == allowed, f"per-target copy differs from base in {diff}, allowed {allowed}"
    c = json.load(open(os.path.join(t, "core", "TARGET_CONTRACT.json"))); assert (c["strength_target_mpa"], c["permeability_target_mm_s"], c["required_joint_reliability"]) == (float(S), float(K), float(R))
    return dict(dir=t, name=name, differs_from_base=diff, hashes=hs)
