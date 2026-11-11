# Consumer Survey Rim Weighting (IPF), K-Means Persona Segmentation & Key Driver Analysis Module

This module (`consumer_segmentation_raking_driver_panel/`) extends `ads-marketing-glm-causal-lifecycle-engine` (`109,211` SHA-256-verified public consumer & lifecycle records) with quantitative consumer panel sampling governance, Iterative Proportional Fitting (IPF) rim weighting, needs-based & behavioral K-Means ($K=5$) segmentation, and weighted Pratt Key Driver Analysis across **64,000** retail consumers.

## Verified Execution Summary (`outputs/panel_segmentation_validation.json`)

| Analytical Dimension | Methodology | Verified Result |
| :--- | :--- | :--- |
| **Primary Consumer Dataset** | Kevin Hillstrom Multi-Channel Retail Consumer Dataset (SHA-256 `0e589332...`) | **64,000** consumer accounts $\times$ **12** attributes |
| **Survey Panel Rim Weighting (IPF)** | 2-Margin Iterative Proportional Fitting (`Geography` $\times$ `Shopping Channel`, cap `[0.35, 3.00]`) | **`0.0` marginal drift (`< 1e-6`)**, Kish **$n_{\text{eff}} = 56,356.9$**, **$\text{DEFF} = 1.1356$** (**`88.06%` weighting efficiency**) |
| **Consumer Persona Segmentation** | Standardized $K=5$ Behavioral & Category-Need Clustering + One-Way ANOVA | **64.68% feature variance explained**, prior-spend ANOVA **$F = 5,982.68$** ($p < 10^{-100}$) |
| **Top Consumer Persona Lift** | `Cross_Category_Brand_Champions` (`6,448` shoppers, `11.47%` weighted share, `$529.65` prior spend) | **`1.979%` vs. `0.917%` control CVR** (**`+1.063 pp` / `2.16x` stimuli lift**) |
| **Weighted Key Driver Analysis** | Rim-weighted Standardized Pratt Relative Importance Decomposition | `Dual_Category_Breadth` (**`31.02%`**), `Product_Stimuli_Exposure` & `Purchase_Recency` (`100.0%` conserved) |
| **Automated Sampling & QA Gates** | Weight conservation, DEFF guardrails, cluster sample floors & driver sum checks | **10/10 (`100.0%`) quality gates** & **3/3 (`100.0%`) unit tests passed** |

## Run Locally

```bash
PYTHONPATH=. python3 consumer_segmentation_raking_driver_panel/panel_raking_segmentation_engine.py
PYTHONPATH=. python3 -m unittest discover -s consumer_segmentation_raking_driver_panel -p "test_*.py" -v
```
