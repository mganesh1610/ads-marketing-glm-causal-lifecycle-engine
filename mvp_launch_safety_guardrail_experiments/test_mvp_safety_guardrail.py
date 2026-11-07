import unittest
from mvp_launch_safety_guardrail_experiments.mvp_safety_guardrail_engine import (
    OUTPUT_PATH,
    run_mvp_guardrail_evaluation,
)


class TestMvpSafetyGuardrailEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.res = run_mvp_guardrail_evaluation()

    def test_combined_records_and_gates_pass(self) -> None:
        self.assertEqual(self.res["combined_verified_records"], 109211)
        self.assertEqual(self.res["gates_passed"], 12)
        self.assertEqual(self.res["gates_total"], 12)

    def test_sequential_ramp_checkpoints(self) -> None:
        cps = self.res["sequential_ramp_checkpoints"]
        self.assertEqual(len(cps), 4)
        self.assertTrue(cps[-1]["scale_up_approved"])
        self.assertGreaterEqual(cps[-1]["visit_z_stat"], 20.0)
        self.assertGreaterEqual(cps[-1]["conversion_z_stat"], 6.0)

    def test_sparse_data_proxy_construction(self) -> None:
        proxy = self.res["sparse_data_proxy_construction"]
        self.assertGreaterEqual(proxy["neyman_variance_reduction_pct"], 66.9)
        self.assertEqual(proxy["cold_start_coverage_rate_pct"], 100.0)
        self.assertTrue(OUTPUT_PATH.exists())


if __name__ == "__main__":
    unittest.main()
