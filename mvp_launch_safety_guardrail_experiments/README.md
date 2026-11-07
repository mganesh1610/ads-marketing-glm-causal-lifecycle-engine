# MVP Feature Launch Experimentation, Sequential Guardrail Gating & Sparse-Data Proxy Module

This module (`mvp_launch_safety_guardrail_experiments/`) extends the **Ads Marketing Lifecycle GLM, Multivariate Segmentation, Stratified Sampling & Counterfactual Causal Incrementality Engine** (`109,211` SHA-256-verified public records) with:

1. **Sequential MVP Rollout & Guardrail Evaluation (`25%`, `50%`, `75%`, `100%` Traffic Ramps):**
   - Evaluates Minimum Viable Product (MVP) feature and model scale-up readiness across **`64,000` randomized user assignments** using **Mixture Sequential Probability Ratio Tests (mSPRT)**, **Holm-Bonferroni multi-metric hypothesis testing**, and **Doubly Robust AIPW (`+0.50 pp` ATE, `+57.35%` visit lift, `+86.53%` conversion lift)**.
2. **Cold-Start / Sparse-Data Proxy Dataset Construction (`109,211` Records):**
   - Constructs a cross-lifecycle marketing and safety suitability proxy dataset when direct ground-truth labels are sparse, combining **Neyman Optimal Stratified Sampling (`n = 4,000`, `66.98%` estimator variance reduction)**, **Binomial Logit + Gamma Log-Link Hurdle GLMs (`0.8862` ROC-AUC)**, and **4 qualitative risk-rubric dimensions**.
3. **12/12 Automated Pre-Scale Launch & Safety Guardrail Gates:**
   - Verifies SHA-256 provenance, pre-treatment covariate balance ($SMD \le 0.0071$), Sample Ratio Mismatch (SRM) absence, mSPRT sequential boundary clearance, and SQLite warehouse integrity (`3/3` unit tests passing).

## Run Locally

```bash
PYTHONPATH=. python3 mvp_launch_safety_guardrail_experiments/mvp_safety_guardrail_engine.py
PYTHONPATH=. python3 -m unittest discover -s mvp_launch_safety_guardrail_experiments -v
```
