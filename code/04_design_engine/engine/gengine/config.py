"""CONFIG freeze (plan section 11), generic target version. The frozen G-3 CONFIG.json (the first full run) stays as it is and `load()` still reads it by default; a per-target run sets GENGINE_CONFIG to its own
targets/<name>/CONFIG.json (written once by gengine/target_config.py, which refuses to overwrite a different one). Everything the walk depends on is inside the hashed file (target, porosity SD, tau, family labels, radii,
conflict constants, walk settings, base hashes of the per-target core copy, paste floor). Family labels are the Phase-1 proposal, k = 3."""
import os, sys, json, hashlib
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CJ = os.environ.get("GENGINE_CONFIG") or os.path.join(ROOT, "CONFIG.json"); CS = os.path.join(os.path.dirname(CJ), "CONFIG.sha256")


def digest(cfg):
    c = {k: v for k, v in cfg.items() if k != "config_sha256"}; return hashlib.sha256(json.dumps(c, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()


def out_dir(cfg):
    """absolute stage folder of this config (cfg['stage'] is relative to the engine root)."""
    return os.path.join(ROOT, *cfg["stage"].split("/"))


def shared_parts():
    """target-independent frozen pieces (families, radii, conflict constants): read from the Phase-1 files."""
    fr = json.load(open(os.path.join(ROOT, "stages", "01_families", "families_radii.json"))); lab = fr["families"]["3"]["labels"]; assert len(lab) == 83 and sorted(set(lab)) == [0, 1, 2]
    rd = fr["radii"]
    families = dict(k=3, features="Ward on standardised q10..q90 of the 83 recorded blends (Phase 1 proposal, k=3, no merging needed)", labels=lab, labels_file="stages/01_families/blend_family_labels.csv (column family_k3)",
                    labels_file_sha256=sha(os.path.join(ROOT, "stages", "01_families", "blend_family_labels.csv")), sizes=fr["families"]["3"]["sizes_final"], r_fam=fr["r_fam_leave_one_blend_out_p95"],
                    rule="design is in family f iff its nearest recorded blend (W1, all 83) has label f and distance <= r_fam; checked at the nominal and all 10 shifted blends")
    radii = dict(m_g=rd["m_g_median_nearest_other_blend"], m_r=rd["m_r_median_nearest_other_unit"], **{"lambda": rd["lambda_rule"]}, lambda_extra=[1.5, 2.0, 2.5, 3.0], R_g=rd["R_g"], R_r=rd["R_r"], scale_std=rd["scale_std"])
    conflict = dict(rmse_S=2.6374, rmse_Q=0.3926, mult_optimistic=1.0, mult_disagree=2.0, residuals="saved cross-fitted bank, mean per unit; clauses (i) and (ii) only")
    return dict(families=families, radii=radii, conflict=conflict)


def build_target(S, K, R, name, base_dir, pf, seeds=(0, 1, 2)):
    """Generic per-target config. base_dir = the per-target copy (holds core/, defengine/, CORE_HASH.json); pf = paste.paste_floor(...) result computed in the per-target process."""
    sp = shared_parts(); hs = json.load(open(os.path.join(base_dir, "CORE_HASH.json"))); gc = json.load(open(os.path.join(base_dir, "core", "GRADATION_SUPPORT_CONTRACT.json")))
    cfg = dict(
        version="G-4 generic target (G-3 rules; null paste floor when no recorded unit meets the target; model-chosen starts only for a family with zero near-target units; 2026-10-04)", stage=f"targets/{name}/stage", target=dict(S=float(S), K=float(K), R=float(R)),
        tau=0.02, porosity_sd_pp=dict(main=1.5, checks=[1.0, 2.0], note="engineering prior, bias 0; certificate-stage checks"),
        base=dict(core_manifest_sha256=hs["core_manifest_sha256"], combined_core_defengine_sha256=hs["combined_core_and_defengine_sha256"],
                  oracle_pt_sha256=sha(os.path.join(base_dir, "core", "models", "lcmgp_ca_oracle_full_data.pt")), bank_csv_sha256=sha(os.path.join(base_dir, "core", "tables", "paired_residual_bank.csv")),
                  data_csv_sha256=sha(os.path.join(base_dir, "core", "data_inputs", "phase1_reconciled_eligible.csv")), w1_epsilon=gc["epsilon_95"],
                  note="forward model A3-1 (Phase 1); no LCMGP fitting or conditioning; per-target copy of core + defengine, only TARGET_CONTRACT.json may differ from base/"),
        families=sp["families"], radii=sp["radii"], conflict=sp["conflict"],
        parents=dict(margin_S=3.0, margin_K=3.0, cap=12, zero_parent_pool_top=24,
                     rule="G-3: candidates = recorded de-duplicated units inside the family with CS >= S* - 3 MPa AND k >= K* - 3 mm/s; parents = ALL deployable recorded passers (porosity inside 20-30 %) + farthest-point sampling (A3 standardised inputs) ONLY inside the candidate set, up to the cap 12; a family with fewer candidates than the cap uses just those (no filling with far-from-target mixes).",
                     zero_parent_rule="generic engine, ONLY for a family with ZERO candidates: model-chosen starts (G-2 style): farthest-point sampling inside the model's top-24 pool of the family's units by point reliability, up to the cap; every result from them is model-suggested (Tier 2 only, never data-backed, never recommended)",
                     passer_rule="G-3: recorded_passer / near-passer evidence require CS >= S*, k >= K* AND recorded porosity inside the 20-30 % deployment window (S17 DEPLOY_LO/HI); window-clamped mixes are logged with phi_clamped and never count as passers or near-passer evidence",
                     failing_checks_logging="G-3: every attempt logs reject_reasons = list of ALL failing checks (order gate, family, support, paste_floor, reliability); reject_reason = first of them"),
        tiers=dict(n_min=1, n_min_also=[2, 3], note="headline n_min = 1 (a real recorded passing unit next to the design); 2 and 3 shown beside. Tier 1 only for a family with recorded deployable passers AND a computed paste floor; everything else is Tier 2 model-suggested; no Tier-1 design -> recommended design is none"),
        walk=dict(N_accept=4096, N_polish=512, m_walk=0.002, tie_rule="stricter (lower) of lowest-index / highest-index tie rule, at nominal and all 10 shifts",
                  m_walk_basis="Phase-1 test B3: sd of LB at N=4096 is 0.0013-0.0017 at designs with LB 0.86-0.88 (0.0032 at LB 0.69), 1 sd ~ 0.002; margin is applied on top of the stricter tie rule",
                  seeds=[int(x) for x in seeds], delta0=10.0, delta_min=1.0, floor=230.0, max_attempts=60, parent_time_cap_s=600, repair_step=10.0, repair_max_steps=12, merge_tol=0.003, K_parents=12,
                  polish=dict(iters=12, lr=0.01, box=0.05, temp=0.35, cand_t=[1.0, 0.5, 0.0], precision="single-precision neighbour maths in the polish only (E-08); acceptance exact")),
        paste_guard=dict(rule="h >= h_min at the nominal blend and all 10 2 percent shifted blends; h = V_paste / (V_agg * A_ag), spheres, six-bin blend, d_k = frozen core DIAM (mm); h_min = null (check skipped) when no recorded unit meets the target",
                         V_paste="cement/3150 + w/c*cement/1000 (no SP, no sand; = PackingPhysics_20261002/phase4.py h_ca_mm)", V_agg="CA/2700 (coarse aggregate only)", A_ag="6*sum(g_k/d_k) per mm", rho_c=3150.0, rho_w=1000.0, rho_ag=2700.0,
                         diam_mm=[21.79449471770337, 15.411035007422441, 10.897247358851684, 7.736278174936576, 5.470374758643141, 3.348133808556641],
                         floor_rule="h_min(S*, K*) = min h over de-duplicated recorded units (631-row A3-1 data) with measured unit-mean CS >= S* AND k >= K* (full target); recomputed from the data at every start for the target and asserted equal to the frozen value. null + reason if no such unit (author-locked option 1).",
                         S_frozen=pf["S"], K_frozen=pf["K"], h_min_mm=pf["h_min"], status=pf["status"], reason=pf["reason"], warning=pf["warning"], h_min_cs_only_mm=pf["h_min_cs_only"], h_p5_mm=pf["h_p5"], h_p10_mm=pf["h_p10"],
                         n_units_cs_ge_S=pf["n_units_cs_ge_S"], n_units_full_target=pf["n_units_full_target"], argmin_unit=pf["argmin_unit"], argmin_unit_cs_only=pf["argmin_unit_cs_only"]),
        bootstrap=dict(unit="study (frozen estimator; no-lab recipe_group and row variants logged only)", n_boot=100000, seed=20261003),
        seeds="sha256(config_sha256|family|round|tag) mod 2^31; banks shared by all parents of a family within a round")
    cfg["config_sha256"] = digest(cfg); return cfg


def load():
    cfg = json.load(open(CJ)); h = open(CS).read().split()[0]; assert digest(cfg) == h == cfg["config_sha256"], "CONFIG hash mismatch"; return cfg, h
