# Ads Marketing Lifecycle GLM, Multivariate Segmentation, Stratified Sampling & Counterfactual Causal Incrementality Engine

An end-to-end marketing science and statistical modeling framework evaluating digital advertising campaigns across the four stages of the advertiser and customer lifecycle (**Acquisition**, **Onboarding**, **Growth**, and **Retention**). It combines **Generalized Linear Models (GLMs)**, **Multivariate Principal Component Analysis (PCA)**, **K-Means Behavioral Segmentation**, **Neyman Optimal Stratified Sampling**, and **Doubly Robust Augmented Inverse Probability Weighting (AIPW) Counterfactual Causal Inference** across **109,211 verified public marketing records**.

---

## 1. Primary & Backup Verified Public Datasets

### Primary Executed Datasets (`109,211` Combined Records)
1. **Kevin Hillstrom MineThatData Multi-Channel Randomized Marketing Dataset (`64,000` records × `12` columns)**
   - **Direct Source:** `http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv`
   - **Verified SHA-256:** `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`
   - **Attributes:** `recency`, `history_segment`, `history`, `mens`, `womens`, `zip_code`, `newbie`, `channel`, `segment` (`Mens E-Mail`, `Womens E-Mail`, `No E-Mail`), `visit`, `conversion`, `spend`.
2. **UCI Bank Marketing Campaign Dataset (`45,211` records × `17` columns, DOI `10.24432/C5K306`, CC BY 4.0)**
   - **Direct Source:** `https://archive.ics.uci.edu/static/public/222/bank+marketing.zip`
   - **Verified ZIP SHA-256:** `e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4`
   - **Attributes:** `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `contact`, `day`, `month`, `duration`, `campaign`, `pdays`, `previous`, `poutcome`, `y`.

### Backup Public Datasets
- **Criteo Uplift Prediction Dataset v2 (`13,979,592` randomized treatment/control ad rows):** `https://ailab.criteo.com/criteo-uplift-prediction-dataset/`
- **UCI Online Retail II Multi-Cohort Commercial Dataset (`1,067,371` rows):** `https://archive.ics.uci.edu/dataset/502/online+retail+ii`

---

## 2. Architecture & Statistical Formulations

1. **Two-Part Hurdle Generalized Linear Models (GLMs via IRLS):**
   - **Stage 1 (Conversion Propensity — Binomial Logit GLM):**
     $$\log\left(\frac{\pi_i}{1 - \pi_i}\right) = \mathbf{x}_i^\top \boldsymbol{\beta}_{\text{logit}}$$
   - **Stage 2 (Positive Spend Severity — Gamma Log-Link GLM):**
     $$\mathbb{E}[Y_i \mid Y_i > 0, \mathbf{x}_i] = \mu_i = \exp(\mathbf{x}_i^\top \boldsymbol{\beta}_{\gamma})$$
   - **Expected Lifecycle Revenue:** $\widehat{\text{LTV}}_i = \hat{\pi}_i \cdot \hat{\mu}_i$.
2. **Multivariate Analysis (PCA) & Behavioral Clustering/Segmentation:**
   - Correlation eigendecomposition extracts orthogonal engagement principal components (**67.17%** cumulative variance in Top-3 PCs), paired with deterministic K-Means ($K = 4$) behavioral segmentation (**44.08%** between-cluster variance ratio).
3. **Neyman Optimal Stratified Sampling (`8` Lifecycle × Value-Tier Strata):**
   - Optimal stratum sample allocation minimizing estimator variance for fixed sample budget $n = 4,000$:
     $$n_h = n \cdot \frac{W_h S_h}{\sum_{k=1}^H W_k S_k}, \quad \text{Var}(\bar{y}_{\text{Neyman}}) = \sum_{h=1}^H W_h^2 S_h^2 \left(\frac{1}{n_h} - \frac{1}{N_h}\right)$$
   - Achieves a **66.98% (`67.0%`) reduction in estimator variance** (`SE: 5.034 -> 2.893`) compared to Simple Random Sampling.
4. **Controlled Experiment Design & Doubly Robust AIPW Counterfactual Causal Inference:**
   - Evaluates randomized treatment ($T_i \in \{0,1\}$) with Doubly Robust Augmented Inverse Probability Weighting:
     $$\hat{\tau}_{\text{AIPW}} = \frac{1}{N}\sum_{i=1}^N \left[\hat{\mu}_1(\mathbf{x}_i) + \frac{T_i (Y_i - \hat{\mu}_1(\mathbf{x}_i))}{\hat{e}(\mathbf{x}_i)} - \hat{\mu}_0(\mathbf{x}_i) - \frac{(1 - T_i)(Y_i - \hat{\mu}_0(\mathbf{x}_i))}{1 - \hat{e}(\mathbf{x}_i)}\right]$$

---

## 3. Verified Execution Metrics (`outputs/validation_report.json`)

| Analytical Module | Verified Metric | Value |
| :--- | :--- | :--- |
| **Combined Lifecycle Scale** | Total SHA-256 Verified Campaign Records | **`109,211`** (`64,000` + `45,211`) |
| **Lifecycle Funnel Coverage** | Acquisition / Onboarding / Retention / Growth | `36,868` / `32,230` / `30,792` / `9,321` |
| **Binomial Logit + Gamma GLM** | Conversion Discrimination ROC-AUC | **`0.8862` (`0.89`)** |
| **Multivariate PCA & K-Means** | Top-3 PCA Variance / Between-Cluster Ratio | **`67.17%`** / **`44.08%`** |
| **Neyman Stratified Sampling** | Estimator Variance Reduction ($n = 4,000$) | **`66.98%` (`67.0%`)** (`SE: 5.034 -> 2.893`) |
| **Randomization Balance** | Pre-Treatment Covariate Balance ($SMD$) | **`0.0071`** (History), **`0.0060`** (Recency) |
| **Causal Incrementality (64K RCT)** | Relative Visit / Conversion / Spend Lift | **`+57.35%`** / **`+86.53%`** (`+0.50 pp` AIPW ATE) / **`+91.42%`** |
| **Automated Test Suite** | Unit & Boundary Tests (`test_pipeline.py`) | **`7/7` (`100%`) Passed** |

---

## 4. Repository Structure & Quickstart

- `pipeline.py` — End-to-end zero-dependency Python 3 pipeline (downloads & verifies SHA-256 hashes, populates SQLite, fits Binomial/Gamma GLMs, runs PCA/K-Means, Neyman stratified sampling, and Doubly Robust AIPW causal inference).
- `glm_causal_lifecycle.R` — Companion R statistical script (`stats::glm` Binomial & Gamma families, `prcomp` multivariate analysis, and Doubly Robust AIPW).
- `multivariate_glm_solver.m` — Companion MATLAB numerical solver (IRLS GLM, SVD PCA, and Neyman stratified allocation).
- `lifecycle_analytics.sql` — Reproducible BigQuery / SQLite `SQL Joins & CTEs` lifecycle incrementality queries.
- `test_pipeline.py` — 7 automated unit and boundary tests.
- `outputs/validation_report.json` & `outputs/lifecycle_stage_metrics.csv` — Verified execution outputs.

```bash
python3 pipeline.py
python3 -m unittest discover -v
```
