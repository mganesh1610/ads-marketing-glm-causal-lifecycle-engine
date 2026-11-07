#!/usr/bin/env python3
"""
Ads Marketing Lifecycle Analytics: Generalized Linear Models (GLMs),
Multivariate Analysis, Stratified Sampling & Counterfactual Causal Inference Engine.

Primary Executed Public Datasets:
1. Kevin Hillstrom MineThatData Multi-Channel Randomized Marketing Dataset (64,000 rows x 12 cols)
   SHA-256: 0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece
2. UCI Bank Marketing Campaign Dataset (45,211 rows x 17 cols, DOI: 10.24432/C5K306)
   ZIP SHA-256: e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4
Combined Verified Scale: 109,211 campaign & lifecycle experiment records.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import sqlite3
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
HILLSTROM_SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"

UCI_BANK_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
UCI_BANK_SHA256 = "e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_verified_bytes(url: str, expected_sha256: str, cache_path: Path) -> bytes:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        raw = cache_path.read_bytes()
        if sha256_bytes(raw) == expected_sha256:
            return raw
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    actual = sha256_bytes(raw)
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual}")
    cache_path.write_bytes(raw)
    return raw


def load_hillstrom_records() -> List[Dict[str, str]]:
    raw = fetch_verified_bytes(
        HILLSTROM_URL,
        HILLSTROM_SHA256,
        DATA_DIR / "hillstrom_experiment_64k.csv",
    )
    text = raw.decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def load_uci_bank_records() -> List[Dict[str, str]]:
    raw_zip = fetch_verified_bytes(
        UCI_BANK_URL,
        UCI_BANK_SHA256,
        DATA_DIR / "uci_bank_marketing.zip",
    )
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as outer_zf:
        inner_zip_bytes = outer_zf.read("bank.zip")
    with zipfile.ZipFile(io.BytesIO(inner_zip_bytes)) as inner_zf:
        csv_bytes = inner_zf.read("bank-full.csv")
    text = csv_bytes.decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text), delimiter=";"))


def assign_hillstrom_lifecycle_stage(recency: int, history: float, newbie: int) -> str:
    if newbie == 1 and recency <= 6:
        return "Onboarding"
    if newbie == 1 and recency > 6:
        return "Acquisition"
    if history >= 250.0 and recency <= 6:
        return "Growth"
    return "Retention"


def assign_uci_lifecycle_stage(campaign: int, previous: int, pdays: int) -> str:
    if previous == 0 and campaign <= 2:
        return "Acquisition"
    if previous == 0 and campaign > 2:
        return "Onboarding"
    if previous > 0 and pdays != -1 and pdays <= 180:
        return "Growth"
    return "Retention"


def solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting."""
    p = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(p):
        pivot = max(range(col, p), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        div = M[col][col] if abs(M[col][col]) > 1e-12 else 1e-12
        for j in range(col, p + 1):
            M[col][j] /= div
        for r in range(p):
            if r != col:
                factor = M[r][col]
                for j in range(col, p + 1):
                    M[r][j] -= factor * M[col][j]
    return [M[i][p] for i in range(p)]


def fit_logistic_glm_irls(
    X: Sequence[Sequence[float]], y: Sequence[float], l2_reg: float = 1e-2, max_iter: int = 18
) -> Tuple[List[float], List[float]]:
    """Fit Binomial Logit GLM via Iteratively Reweighted Least Squares (IRLS)."""
    n = len(X)
    p = len(X[0])
    beta = [0.0] * p
    pos_rate = max(1e-4, min(1.0 - 1e-4, sum(y) / float(n)))
    beta[0] = math.log(pos_rate / (1.0 - pos_rate))

    for _ in range(max_iter):
        grad = [0.0] * p
        hess = [[0.0] * p for _ in range(p)]
        for j in range(1, p):
            grad[j] -= l2_reg * beta[j]
            hess[j][j] += l2_reg

        for i in range(n):
            row = X[i]
            eta = sum(row[j] * beta[j] for j in range(p))
            eta = max(-18.0, min(18.0, eta))
            mu = 1.0 / (1.0 + math.exp(-eta))
            w = max(1e-6, mu * (1.0 - mu))
            err = y[i] - mu
            for j in range(p):
                grad[j] += row[j] * err
                rj_w = row[j] * w
                for k in range(j, p):
                    hess[j][k] += rj_w * row[k]
        for j in range(p):
            for k in range(j):
                hess[j][k] = hess[k][j]
        step = solve_linear_system(hess, grad)
        for j in range(p):
            beta[j] += step[j]
        if max(abs(s) for s in step) < 1e-5:
            break

    probs = []
    for row in X:
        eta = max(-18.0, min(18.0, sum(row[j] * beta[j] for j in range(p))))
        probs.append(1.0 / (1.0 + math.exp(-eta)))
    return beta, probs


def fit_gamma_log_glm_irls(
    X: Sequence[Sequence[float]], y_pos: Sequence[float], l2_reg: float = 5e-2, max_iter: int = 20
) -> Tuple[List[float], List[float]]:
    """Fit Gamma Log-Link GLM on positive spend severity via Fisher scoring IRLS."""
    n = len(X)
    p = len(X[0])
    beta = [0.0] * p
    mean_y = sum(y_pos) / float(max(1, n))
    beta[0] = math.log(max(1.0, mean_y))

    for _ in range(max_iter):
        grad = [0.0] * p
        hess = [[0.0] * p for _ in range(p)]
        for j in range(1, p):
            grad[j] -= l2_reg * beta[j]
            hess[j][j] += l2_reg

        for i in range(n):
            row = X[i]
            eta = max(-8.0, min(10.0, sum(row[j] * beta[j] for j in range(p))))
            mu = math.exp(eta)
            w_resid = (y_pos[i] - mu) / mu
            for j in range(p):
                grad[j] += row[j] * w_resid
                for k in range(j, p):
                    hess[j][k] += row[j] * row[k]
        for j in range(p):
            for k in range(j):
                hess[j][k] = hess[k][j]
        step = solve_linear_system(hess, grad)
        for j in range(p):
            beta[j] += 0.5 * step[j]
        if max(abs(s) for s in step) < 1e-5:
            break

    mu_hat = []
    for row in X:
        eta = max(-8.0, min(10.0, sum(row[j] * beta[j] for j in range(p))))
        mu_hat.append(math.exp(eta))
    return beta, mu_hat


def compute_roc_auc(y_true: Sequence[float], y_score: Sequence[float]) -> float:
    paired = sorted(zip(y_score, y_true), key=lambda t: t[0])
    n_pos = sum(1 for _, y in paired if y == 1.0)
    n_neg = len(paired) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pos_rank_sum = 0.0
    for idx, (_, y) in enumerate(paired, start=1):
        if y == 1.0:
            pos_rank_sum += idx
    return (pos_rank_sum - n_pos * (n_pos + 1.0) / 2.0) / float(n_pos * n_neg)


def mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    n = len(vals)
    m = sum(vals) / float(n)
    var = sum((v - m) ** 2 for v in vals) / float(max(1, n - 1))
    return m, math.sqrt(max(1e-12, var))


def run_multivariate_pca_and_clustering(features: List[List[float]], n_clusters: int = 4) -> Dict[str, object]:
    """Compute correlation covariance eigenvalues (power-deflation) and deterministic K-Means segmentation."""
    n = len(features)
    p = len(features[0])
    col_means = [0.0] * p
    col_stds = [1.0] * p
    for j in range(p):
        col = [features[i][j] for i in range(n)]
        m, s = mean_std(col)
        col_means[j] = m
        col_stds[j] = s

    Z = [[(features[i][j] - col_means[j]) / col_stds[j] for j in range(p)] for i in range(n)]
    cov = [[0.0] * p for _ in range(p)]
    for row in Z:
        for j in range(p):
            rj = row[j]
            for k in range(j, p):
                cov[j][k] += rj * row[k]
    for j in range(p):
        for k in range(j, p):
            cov[j][k] /= float(n - 1)
            cov[k][j] = cov[j][k]

    # Power iteration with deflation for eigenvalues
    mat = [r[:] for r in cov]
    eigvals = []
    eigvecs = []
    for _ in range(p):
        v = [1.0 / math.sqrt(p)] * p
        for _ in range(40):
            wv = [sum(mat[r][c] * v[c] for c in range(p)) for r in range(p)]
            norm = math.sqrt(sum(x * x for x in wv)) or 1.0
            v = [x / norm for x in wv]
        lam = sum(v[r] * sum(mat[r][c] * v[c] for c in range(p)) for r in range(p))
        eigvals.append(max(0.0, lam))
        eigvecs.append(v)
        for r in range(p):
            for c in range(p):
                mat[r][c] -= lam * v[r] * v[c]

    tot_eig = sum(eigvals) or 1.0
    top2_var = sum(eigvals[:2]) / tot_eig
    top3_var = sum(eigvals[:3]) / tot_eig

    # Project onto PC1 for deterministic quantile K-Means initialization
    pc1 = [sum(Z[i][j] * eigvecs[0][j] for j in range(p)) for i in range(n)]
    pc1_sorted = sorted(range(n), key=lambda idx: pc1[idx])
    centroids = []
    for k in range(n_clusters):
        start_idx = int((k / float(n_clusters)) * n)
        end_idx = int(((k + 1) / float(n_clusters)) * n)
        slice_indices = pc1_sorted[start_idx:end_idx]
        c_vec = [
            sum(Z[idx][j] for idx in slice_indices) / float(len(slice_indices))
            for j in range(p)
        ]
        centroids.append(c_vec)

    labels = [0] * n
    for _ in range(15):
        changed = False
        counts = [0] * n_clusters
        new_sums = [[0.0] * p for _ in range(n_clusters)]
        for i in range(n):
            row = Z[i]
            best_k = 0
            best_d = float("inf")
            for k in range(n_clusters):
                cent = centroids[k]
                d = sum((row[j] - cent[j]) ** 2 for j in range(p))
                if d < best_d:
                    best_d = d
                    best_k = k
            if labels[i] != best_k:
                labels[i] = best_k
                changed = True
            counts[best_k] += 1
            for j in range(p):
                new_sums[best_k][j] += row[j]
        for k in range(n_clusters):
            if counts[k] > 0:
                centroids[k] = [new_sums[k][j] / float(counts[k]) for j in range(p)]
        if not changed:
            break

    total_ss = sum(sum(val * val for val in row) for row in Z)
    within_ss = sum(
        sum((Z[i][j] - centroids[labels[i]][j]) ** 2 for j in range(p))
        for i in range(n)
    )
    between_ratio = (total_ss - within_ss) / float(total_ss)
    cluster_sizes = {f"segment_{k+1}": sum(1 for l in labels if l == k) for k in range(n_clusters)}
    return {
        "pca_top2_variance_explained": round(top2_var, 4),
        "pca_top3_variance_explained": round(top3_var, 4),
        "kmeans_between_ss_ratio": round(between_ratio, 4),
        "cluster_sizes": cluster_sizes,
    }


def compute_neyman_stratified_sampling(strata_values: Dict[str, List[float]], total_sample_n: int = 4000) -> Dict[str, object]:
    """Compare Neyman Optimal Stratified Sampling vs. Simple Random Sampling (SRS)."""
    all_vals = [v for arr in strata_values.values() for v in arr]
    N_total = float(len(all_vals))
    _, pop_std = mean_std(all_vals)
    pop_var = pop_std ** 2
    f_srs = total_sample_n / N_total
    var_srs = (1.0 - f_srs) * pop_var / float(total_sample_n)

    weights = {}
    stds = {}
    for stage, arr in strata_values.items():
        weights[stage] = len(arr) / N_total
        _, s_h = mean_std(arr)
        stds[stage] = s_h

    denom = sum(weights[s] * stds[s] for s in strata_values)
    allocations = {}
    var_neyman = 0.0
    for stage, arr in strata_values.items():
        N_h = float(len(arr))
        W_h = weights[stage]
        S_h = stds[stage]
        n_h = max(20.0, total_sample_n * (W_h * S_h) / denom)
        n_h = min(n_h, N_h - 1.0)
        allocations[stage] = int(round(n_h))
        var_neyman += (W_h ** 2) * (S_h ** 2) * (1.0 / n_h - 1.0 / N_h)

    variance_reduction_pct = (1.0 - var_neyman / var_srs) * 100.0
    return {
        "total_sample_n": total_sample_n,
        "srs_estimator_se": round(math.sqrt(var_srs), 6),
        "neyman_estimator_se": round(math.sqrt(var_neyman), 6),
        "variance_reduction_pct": round(variance_reduction_pct, 2),
        "strata_allocations": allocations,
    }


def run_pipeline() -> Dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hillstrom = load_hillstrom_records()
    uci_bank = load_uci_bank_records()

    total_records = len(hillstrom) + len(uci_bank)
    assert len(hillstrom) == 64000, f"Expected 64000 Hillstrom rows, got {len(hillstrom)}"
    assert len(uci_bank) == 45211, f"Expected 45211 UCI Bank rows, got {len(uci_bank)}"
    assert total_records == 109211

    db_path = OUTPUT_DIR / "ads_lifecycle_analytics.sqlite"
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE unified_lifecycle_records (
            record_id INTEGER PRIMARY KEY,
            source_dataset TEXT NOT NULL,
            lifecycle_stage TEXT NOT NULL,
            treatment_assigned INTEGER NOT NULL,
            treatment_arm TEXT NOT NULL,
            prior_engagement_score REAL NOT NULL,
            historical_value REAL NOT NULL,
            recency_score REAL NOT NULL,
            engaged_visit INTEGER NOT NULL,
            converted INTEGER NOT NULL,
            realized_revenue REAL NOT NULL
        )
        """
    )

    unified_rows = []
    rec_id = 1

    h_treat = []
    h_visit = []
    h_conv = []
    h_spend = []
    h_recency = []
    h_history = []
    h_newbie = []
    h_mens = []
    h_womens = []

    for row in hillstrom:
        recency = int(row["recency"])
        history = float(row["history"])
        newbie = int(row["newbie"])
        mens = int(row["mens"])
        womens = int(row["womens"])
        segment = row["segment"].strip()
        visit = int(row["visit"])
        conversion = int(row["conversion"])
        spend = float(row["spend"])
        treated = 0 if segment == "No E-Mail" else 1
        stage = assign_hillstrom_lifecycle_stage(recency, history, newbie)

        h_treat.append(float(treated))
        h_visit.append(float(visit))
        h_conv.append(float(conversion))
        h_spend.append(spend)
        h_recency.append(float(recency))
        h_history.append(history)
        h_newbie.append(float(newbie))
        h_mens.append(float(mens))
        h_womens.append(float(womens))

        unified_rows.append(
            (
                rec_id,
                "Hillstrom_Experiment",
                stage,
                treated,
                segment,
                float(mens + womens),
                history,
                float(recency),
                visit,
                conversion,
                spend,
            )
        )
        rec_id += 1

    for row in uci_bank:
        balance = max(0.0, float(row["balance"]))
        duration = float(row["duration"])
        campaign = int(row["campaign"])
        pdays = int(row["pdays"])
        previous = int(row["previous"])
        subscribed = 1 if row["y"].strip().lower() == "yes" else 0
        contact = row["contact"].strip().lower()
        treated = 1 if contact in ("cellular", "telephone") else 0
        engaged = 1 if duration >= 180.0 else 0
        stage = assign_uci_lifecycle_stage(campaign, previous, pdays)
        est_rev = round(subscribed * (75.0 + 0.025 * min(balance, 30000.0) * (1.0 + 0.15 * campaign)), 2)

        unified_rows.append(
            (
                rec_id,
                "UCI_Bank_Campaign",
                stage,
                treated,
                contact,
                float(previous + 1),
                balance,
                float(campaign),
                engaged,
                subscribed,
                est_rev,
            )
        )
        rec_id += 1

    cur.executemany(
        "INSERT INTO unified_lifecycle_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        unified_rows,
    )
    conn.commit()

    stage_query = """
    WITH base_lifecycle AS (
        SELECT
            lifecycle_stage,
            source_dataset,
            treatment_assigned,
            engaged_visit,
            converted,
            realized_revenue,
            historical_value
        FROM unified_lifecycle_records
    ),
    stage_rollup AS (
        SELECT
            lifecycle_stage,
            COUNT(*) AS total_records,
            SUM(CASE WHEN treatment_assigned = 1 THEN 1 ELSE 0 END) AS treated_records,
            SUM(CASE WHEN treatment_assigned = 0 THEN 1 ELSE 0 END) AS control_records,
            ROUND(AVG(engaged_visit) * 100.0, 3) AS engagement_rate_pct,
            ROUND(AVG(converted) * 100.0, 3) AS conversion_rate_pct,
            ROUND(SUM(realized_revenue), 2) AS total_revenue_usd
        FROM base_lifecycle
        GROUP BY lifecycle_stage
    )
    SELECT * FROM stage_rollup ORDER BY total_records DESC;
    """
    stage_rows = cur.execute(stage_query).fetchall()
    integrity_status = cur.execute("PRAGMA integrity_check;").fetchone()[0]
    conn.close()

    # Build raw & standardized design matrix across 109,211 records
    raw_features = [
        [
            float(r[3]),
            math.log1p(max(0.0, float(r[6]))),
            float(r[7]),
            float(r[5]),
            float(r[8]),
        ]
        for r in unified_rows
    ]
    p_feat = len(raw_features[0])
    f_means = [0.0] * p_feat
    f_stds = [1.0] * p_feat
    for j in range(p_feat):
        m, s = mean_std([row[j] for row in raw_features])
        f_means[j] = m
        f_stds[j] = s

    X_design = [
        [1.0] + [(row[j] - f_means[j]) / f_stds[j] for j in range(p_feat)]
        for row in raw_features
    ]
    all_y_conv = [float(r[9]) for r in unified_rows]
    all_y_rev = [float(r[10]) for r in unified_rows]

    beta_logit, prob_conv = fit_logistic_glm_irls(X_design, all_y_conv)
    glm_roc_auc = compute_roc_auc(all_y_conv, prob_conv)

    X_pos = [X_design[i] for i in range(total_records) if all_y_rev[i] > 0.0]
    y_pos = [all_y_rev[i] for i in range(total_records) if all_y_rev[i] > 0.0]
    beta_gamma, _ = fit_gamma_log_glm_irls(X_pos, y_pos)

    hurdle_expected_rev = []
    for i in range(total_records):
        eta_g = max(-8.0, min(10.0, sum(X_design[i][j] * beta_gamma[j] for j in range(len(beta_gamma)))))
        hurdle_expected_rev.append(prob_conv[i] * math.exp(eta_g))

    mv_metrics = run_multivariate_pca_and_clustering(raw_features, n_clusters=4)

    # Stratified Sampling across 8 Lifecycle Stage x Historical Value Tier strata
    strata_dict: Dict[str, List[float]] = {}
    for idx, r in enumerate(unified_rows):
        tier = "HighTier" if float(r[6]) >= 350.0 else "CoreTier"
        stratum_key = f"{r[2]}_{tier}"
        stage_val = float(r[6]) * 0.15 + float(r[10]) + hurdle_expected_rev[idx]
        strata_dict.setdefault(stratum_key, []).append(stage_val)
    sampling_metrics = compute_neyman_stratified_sampling(strata_dict, total_sample_n=4000)

    # Controlled Experiment & Doubly Robust AIPW Causal Incrementality on 64,000 Hillstrom records
    t_indices = [i for i, t in enumerate(h_treat) if t == 1.0]
    c_indices = [i for i, t in enumerate(h_treat) if t == 0.0]

    m_hist_t, s_hist_t = mean_std([h_history[i] for i in t_indices])
    m_hist_c, s_hist_c = mean_std([h_history[i] for i in c_indices])
    smd_history = abs(m_hist_t - m_hist_c) / math.sqrt(0.5 * (s_hist_t ** 2 + s_hist_c ** 2))

    m_rec_t, s_rec_t = mean_std([h_recency[i] for i in t_indices])
    m_rec_c, s_rec_c = mean_std([h_recency[i] for i in c_indices])
    smd_recency = abs(m_rec_t - m_rec_c) / math.sqrt(0.5 * (s_rec_t ** 2 + s_rec_c ** 2))

    m_rec_all, s_rec_all = mean_std(h_recency)
    log_hist = [math.log1p(v) for v in h_history]
    m_lh_all, s_lh_all = mean_std(log_hist)

    X_cov = [
        [
            1.0,
            (h_recency[i] - m_rec_all) / s_rec_all,
            (log_hist[i] - m_lh_all) / s_lh_all,
            h_newbie[i],
            h_mens[i],
            h_womens[i],
        ]
        for i in range(len(h_treat))
    ]
    _, e_hat = fit_logistic_glm_irls(X_cov, h_treat)
    X_cov_t = [X_cov[i] for i in t_indices]
    y_conv_t = [h_conv[i] for i in t_indices]
    X_cov_c = [X_cov[i] for i in c_indices]
    y_conv_c = [h_conv[i] for i in c_indices]

    beta1_c, _ = fit_logistic_glm_irls(X_cov_t, y_conv_t)
    beta0_c, _ = fit_logistic_glm_irls(X_cov_c, y_conv_c)

    aipw_diffs = []
    for i in range(len(h_treat)):
        row = X_cov[i]
        eh = max(0.05, min(0.95, e_hat[i]))
        eta1 = max(-18.0, min(18.0, sum(row[j] * beta1_c[j] for j in range(len(beta1_c)))))
        eta0 = max(-18.0, min(18.0, sum(row[j] * beta0_c[j] for j in range(len(beta0_c)))))
        mu1 = 1.0 / (1.0 + math.exp(-eta1))
        mu0 = 1.0 / (1.0 + math.exp(-eta0))
        t_i = h_treat[i]
        y_i = h_conv[i]
        psi_1 = mu1 + t_i * (y_i - mu1) / eh
        psi_0 = mu0 + (1.0 - t_i) * (y_i - mu0) / (1.0 - eh)
        aipw_diffs.append(psi_1 - psi_0)

    ate_m, ate_s = mean_std(aipw_diffs)
    ate_conv_pp = ate_m * 100.0
    ate_conv_se_pp = (ate_s / math.sqrt(len(aipw_diffs))) * 100.0

    control_conv_pct = (sum(y_conv_c) / float(len(y_conv_c))) * 100.0
    treated_conv_pct = (sum(y_conv_t) / float(len(y_conv_t))) * 100.0
    relative_conv_lift_pct = ((treated_conv_pct - control_conv_pct) / control_conv_pct) * 100.0

    control_visit_pct = (sum(h_visit[i] for i in c_indices) / float(len(c_indices))) * 100.0
    treated_visit_pct = (sum(h_visit[i] for i in t_indices) / float(len(t_indices))) * 100.0
    relative_visit_lift_pct = ((treated_visit_pct - control_visit_pct) / control_visit_pct) * 100.0

    control_spend_mean = sum(h_spend[i] for i in c_indices) / float(len(c_indices))
    treated_spend_mean = sum(h_spend[i] for i in t_indices) / float(len(t_indices))
    relative_spend_lift_pct = ((treated_spend_mean - control_spend_mean) / control_spend_mean) * 100.0

    stage_csv_path = OUTPUT_DIR / "lifecycle_stage_metrics.csv"
    with stage_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "lifecycle_stage",
                "total_records",
                "treated_records",
                "control_records",
                "engagement_rate_pct",
                "conversion_rate_pct",
                "total_revenue_usd",
            ]
        )
        for s_row in stage_rows:
            writer.writerow(list(s_row))

    report = {
        "project": "Ads Marketing Lifecycle GLM, Multivariate & Causal Incrementality Engine",
        "datasets": {
            "hillstrom_rows": len(hillstrom),
            "hillstrom_sha256": HILLSTROM_SHA256,
            "uci_bank_rows": len(uci_bank),
            "uci_bank_sha256": UCI_BANK_SHA256,
            "combined_verified_records": total_records,
        },
        "sqlite_integrity": integrity_status,
        "glm_hurdle_modeling": {
            "binomial_logit_roc_auc": round(glm_roc_auc, 4),
            "positive_spend_records": len(y_pos),
            "mean_hurdle_expected_revenue_usd": round(sum(hurdle_expected_rev) / float(total_records), 4),
        },
        "multivariate_and_segmentation": mv_metrics,
        "stratified_sampling": sampling_metrics,
        "causal_incrementality_64k_experiment": {
            "treated_n": len(t_indices),
            "control_n": len(c_indices),
            "pre_treatment_smd_history": round(smd_history, 5),
            "pre_treatment_smd_recency": round(smd_recency, 5),
            "control_visit_rate_pct": round(control_visit_pct, 4),
            "treated_visit_rate_pct": round(treated_visit_pct, 4),
            "relative_visit_lift_pct": round(relative_visit_lift_pct, 2),
            "control_conversion_rate_pct": round(control_conv_pct, 4),
            "treated_conversion_rate_pct": round(treated_conv_pct, 4),
            "relative_conversion_lift_pct": round(relative_conv_lift_pct, 2),
            "doubly_robust_aipw_ate_pp": round(ate_conv_pp, 4),
            "doubly_robust_aipw_se_pp": round(ate_conv_se_pp, 4),
            "relative_spend_lift_pct": round(relative_spend_lift_pct, 2),
        },
        "lifecycle_stages": [
            {
                "stage": r[0],
                "total_records": r[1],
                "treated_records": r[2],
                "control_records": r[3],
                "engagement_rate_pct": r[4],
                "conversion_rate_pct": r[5],
                "total_revenue_usd": r[6],
            }
            for r in stage_rows
        ],
    }

    report_path = OUTPUT_DIR / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    results = run_pipeline()
    print(json.dumps(results, indent=2))
