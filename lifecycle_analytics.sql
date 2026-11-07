-- =============================================================================
-- Ads Marketing Lifecycle Funnel, Incrementality & Cohort Analytics (SQL CTEs)
-- Compatible with BigQuery Standard SQL and SQLite 3
-- =============================================================================

WITH lifecycle_base AS (
    SELECT
        record_id,
        source_dataset,
        lifecycle_stage,
        treatment_assigned,
        treatment_arm,
        prior_engagement_score,
        historical_value,
        recency_score,
        engaged_visit,
        converted,
        realized_revenue,
        CASE
            WHEN historical_value >= 350.0 THEN 'High_Value_Tier'
            ELSE 'Core_Value_Tier'
        END AS advertiser_value_segment
    FROM unified_lifecycle_records
),
stage_treatment_summary AS (
    SELECT
        lifecycle_stage,
        advertiser_value_segment,
        treatment_assigned,
        COUNT(*) AS cohort_accounts,
        AVG(engaged_visit) * 100.0 AS visit_rate_pct,
        AVG(converted) * 100.0 AS conversion_rate_pct,
        AVG(realized_revenue) AS mean_revenue_per_account,
        SUM(realized_revenue) AS total_cohort_revenue
    FROM lifecycle_base
    GROUP BY
        lifecycle_stage,
        advertiser_value_segment,
        treatment_assigned
),
counterfactual_incrementality AS (
    SELECT
        t.lifecycle_stage,
        t.advertiser_value_segment,
        t.cohort_accounts AS treated_accounts,
        c.cohort_accounts AS control_accounts,
        ROUND(t.visit_rate_pct - c.visit_rate_pct, 4) AS incremental_visit_lift_pp,
        ROUND(t.conversion_rate_pct - c.conversion_rate_pct, 4) AS incremental_conversion_lift_pp,
        ROUND(t.mean_revenue_per_account - c.mean_revenue_per_account, 4) AS incremental_revenue_per_account,
        ROUND(t.total_cohort_revenue + c.total_cohort_revenue, 2) AS combined_segment_revenue
    FROM stage_treatment_summary t
    INNER JOIN stage_treatment_summary c
        ON t.lifecycle_stage = c.lifecycle_stage
       AND t.advertiser_value_segment = c.advertiser_value_segment
       AND t.treatment_assigned = 1
       AND c.treatment_assigned = 0
)
SELECT *
FROM counterfactual_incrementality
ORDER BY combined_segment_revenue DESC;
