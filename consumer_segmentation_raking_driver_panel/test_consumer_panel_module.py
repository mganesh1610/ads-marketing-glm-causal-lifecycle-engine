import unittest

from consumer_segmentation_raking_driver_panel.panel_raking_segmentation_engine import (
    OUTPUT_DIR,
    run_module,
)


class TestConsumerPanelSegmentationModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_module()

    def test_quality_gates_and_scale(self) -> None:
        self.assertEqual(self.report["primary_dataset_rows"], 64000)
        self.assertEqual(self.report["quality_gates_passed"], "10/10")

    def test_ipf_raking_convergence_and_kish_efficiency(self) -> None:
        ipf = self.report["ipf_rim_weighting"]
        self.assertLess(ipf["max_marginal_convergence_drift"], 1e-4)
        self.assertGreaterEqual(ipf["weighting_efficiency_pct"], 85.0)
        self.assertLessEqual(ipf["design_effect_deff"], 1.18)
        self.assertAlmostEqual(ipf["weight_sum"], 64000.0, delta=0.05)

    def test_personas_and_weighted_drivers(self) -> None:
        self.assertEqual(len(self.report["consumer_personas"]), 5)
        self.assertGreater(self.report["segmentation_summary"]["prior_spend_anova_f_stat"], 500.0)
        imp_sum = sum(d["relative_driver_importance_pct"] for d in self.report["weighted_key_drivers"])
        self.assertAlmostEqual(imp_sum, 100.0, delta=0.15)
        self.assertTrue((OUTPUT_DIR / "consumer_personas_k5.csv").exists())
        self.assertTrue((OUTPUT_DIR / "weighted_key_drivers.csv").exists())


if __name__ == "__main__":
    unittest.main()
