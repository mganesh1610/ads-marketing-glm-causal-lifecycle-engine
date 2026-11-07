#!/usr/bin/env python3
"""Unit and boundary tests for Ads Marketing Lifecycle GLM & Causal Inference Engine."""

from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path

from pipeline import (
    HILLSTROM_SHA256,
    OUTPUT_DIR,
    UCI_BANK_SHA256,
    run_pipeline,
)


class TestAdsLifecycleCausalEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_pipeline()

    def test_01_dataset_row_conservation_and_hashes(self) -> None:
        ds = self.report["datasets"]
        self.assertEqual(ds["hillstrom_rows"], 64000)
        self.assertEqual(ds["uci_bank_rows"], 45211)
        self.assertEqual(ds["combined_verified_records"], 109211)
        self.assertEqual(ds["hillstrom_sha256"], HILLSTROM_SHA256)
        self.assertEqual(ds["uci_bank_sha256"], UCI_BANK_SHA256)

    def test_02_sqlite_integrity_and_stage_partitioning(self) -> None:
        self.assertEqual(self.report["sqlite_integrity"], "ok")
        stages = self.report["lifecycle_stages"]
        self.assertEqual(len(stages), 4)
        total_stage_rows = sum(s["total_records"] for s in stages)
        self.assertEqual(total_stage_rows, 109211)

    def test_03_glm_hurdle_discrimination_and_calibration(self) -> None:
        glm = self.report["glm_hurdle_modeling"]
        self.assertGreaterEqual(glm["binomial_logit_roc_auc"], 0.85)
        self.assertLessEqual(glm["binomial_logit_roc_auc"], 0.95)
        self.assertEqual(glm["positive_spend_records"], 5867)
        self.assertGreater(glm["mean_hurdle_expected_revenue_usd"], 1.0)

    def test_04_multivariate_pca_and_kmeans_segmentation(self) -> None:
        mv = self.report["multivariate_and_segmentation"]
        self.assertGreaterEqual(mv["pca_top3_variance_explained"], 0.60)
        self.assertGreaterEqual(mv["kmeans_between_ss_ratio"], 0.35)
        cluster_total = sum(mv["cluster_sizes"].values())
        self.assertEqual(cluster_total, 109211)

    def test_05_neyman_stratified_sampling_variance_reduction(self) -> None:
        samp = self.report["stratified_sampling"]
        self.assertEqual(samp["total_sample_n"], 4000)
        self.assertLess(samp["neyman_estimator_se"], samp["srs_estimator_se"])
        self.assertGreaterEqual(samp["variance_reduction_pct"], 60.0)

    def test_06_causal_incrementality_and_covariate_balance(self) -> None:
        causal = self.report["causal_incrementality_64k_experiment"]
        self.assertEqual(causal["treated_n"] + causal["control_n"], 64000)
        self.assertLess(causal["pre_treatment_smd_history"], 0.02)
        self.assertLess(causal["pre_treatment_smd_recency"], 0.02)
        self.assertAlmostEqual(causal["relative_visit_lift_pct"], 57.35, places=1)
        self.assertAlmostEqual(causal["relative_conversion_lift_pct"], 86.53, places=1)
        self.assertGreater(causal["doubly_robust_aipw_ate_pp"], 0.40)

    def test_07_sql_cte_file_executes_cleanly(self) -> None:
        db_path = OUTPUT_DIR / "ads_lifecycle_analytics.sqlite"
        sql_text = (Path(__file__).resolve().parent / "lifecycle_analytics.sql").read_text(encoding="utf-8")
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(sql_text).fetchall()
        conn.close()
        self.assertEqual(len(rows), 8)


if __name__ == "__main__":
    unittest.main()
