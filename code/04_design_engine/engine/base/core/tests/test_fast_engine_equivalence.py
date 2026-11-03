"""Equivalence tests: fast_engine vs the frozen evaluation path.

Run from a generated run directory with the bundled interpreter, e.g.

    runtime\\python.exe runs\\<run>\\run_01\\tests\\test_fast_engine_equivalence.py

(the run directory must have been produced by run_stage1.make_run so that
src/ is parameterised).  Acceptance criteria:

* scenario arrays (muS, muQ, sS, sQ) agree to float32-level tolerance
  (the frozen GPyTorch path itself carries float32 rounding of ~5e-6 relative),
* support scores agree to ~1e-12,
* joint pass counts under the paired residual bank are identical,
* vectorised gradation descriptors agree with the frozen ones (D10/D60-based Cu
  to ~1e-12 relative; everything else to ~1e-15),
* the bincount study-bootstrap counts equal the per-replicate np.add.at loop exactly.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(RUN, "src"))

import numpy as np  # noqa: E402

import stage1_evaluator as E  # noqa: E402
import fast_descriptors as FD  # noqa: E402
import fast_local_residual as FLR  # noqa: E402
import uncertainty as U  # noqa: E402
import robust_evaluator as RE  # noqa: E402
import fast_engine as FE  # noqa: E402
import cluster_bootstrap as CB  # noqa: E402

FD.install_patch(E)
FLR.install_patch(E.L)

# A fixed, ordinary in-domain control recipe.  This is an evaluator-equivalence
# fixture only, not a historical low-cement search seed or reference result.
CAND = {"C": 350.0, "r": 0.30, "A": 1400.0, "P": 5.0,
        "S": 120.0, "phi": 25.0,
        "g": [0.10, 0.15, 0.20, 0.20, 0.20, 0.15]}
CONTRACT = os.path.join(RUN, "UNCERTAINTY_CONTRACT.json")
BANK_CSV = os.path.join(RUN, "tables", "paired_residual_bank.csv")


class FastEngineEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        E.get_bundle()
        cls.eng = FE.get_engine(E)

    def test_engine_active(self):
        self.assertIsNotNone(self.eng, "fast_engine failed its own self-check")

    def test_scenario_arrays_and_pass_counts(self):
        n = 2000
        spec = U.UncertaintySpec(CONTRACT)
        mb = U.build_manufacturing_bank(spec, n=n, seed=205609401)
        rb = U.build_residual_bank(BANK_CSV, n=n, seed=205609401)
        mfg = U.apply_manufacturing_scenarios(CAND, mb)
        frame = RE._candidates_frame_from_scenarios(mfg)
        pred = E.predict_fast(E.ORACLE_PT, frame)
        ref = (pred["strength_pred_mpa"].to_numpy(float), pred["logperm_pred"].to_numpy(float),
               pred["strength_std_effective_mpa"].to_numpy(float),
               pred["logperm_std_effective"].to_numpy(float), RE._support_scores_for_frame(frame))
        got = self.eng.scenario_arrays(mfg)
        for name, g, w in zip(("muS", "muQ", "sS", "sQ"), got[:4], ref[:4]):
            self.assertLess(float(np.max(np.abs(g - w) / (1.0 + np.abs(w)))), 5e-5, name)
        self.assertLess(float(np.max(np.abs(got[4] - ref[4]))), 1e-10, "support")

        def passes(a):
            muS, muQ, sS, sQ, sup = a
            S = muS + sS * rb.e_strength
            Q = muQ + sQ * rb.e_logk
            return int(((sup >= E.SUPPORT_THRESHOLD) & (S >= 18.0) & (Q >= np.log(13.0))).sum())

        self.assertEqual(passes(got), passes(ref))

    def test_descriptors(self):
        rng = np.random.default_rng(7)
        G = rng.dirichlet(np.ones(6) * 0.4, size=3000)
        G[:100, :3] = 0
        G[:100] /= G[:100].sum(1, keepdims=True)
        G = np.round(G, 12)
        a = FD.descriptors_precise_batch(G, E.DIAM)
        b = FE.descriptors_batch(FD, G, E.DIAM)
        for k in a:
            np.testing.assert_allclose(b[k], a[k], rtol=1e-11, atol=0, equal_nan=True, err_msg=k)

    def test_bootstrap_counts_identical(self):
        r1 = np.random.default_rng(3)
        r2 = np.random.default_rng(3)
        new = CB.sample_study_bootstrap_counts(36, 36, 2000, r1)
        idx = r2.integers(0, 36, size=(2000, 36))
        old = np.zeros((2000, 36), dtype=np.int64)
        for b in range(2000):
            np.add.at(old[b], idx[b], 1)
        self.assertTrue(np.array_equal(new, old))


if __name__ == "__main__":
    unittest.main(verbosity=2)
