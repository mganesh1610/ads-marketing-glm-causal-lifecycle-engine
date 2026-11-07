# ==============================================================================
# Ads Marketing Lifecycle Analytics: Generalized Linear Models (GLMs),
# Multivariate PCA, Stratified Sampling & Doubly Robust Causal Inference in R
# ==============================================================================

run_r_lifecycle_models <- function(csv_path) {
  df <- read.csv(csv_path, stringsAsFactors = FALSE)
  df$treated <- ifelse(df$segment == "No E-Mail", 0, 1)
  df$log_history <- log1p(pmax(0, df$history))

  # 1. Binomial Logit GLM (Conversion Probability)
  logit_fit <- glm(
    conversion ~ treated + recency + log_history + newbie + mens + womens,
    data = df,
    family = binomial(link = "logit")
  )

  # 2. Gamma Log-Link GLM (Positive Spend Severity)
  pos_df <- subset(df, spend > 0)
  gamma_fit <- glm(
    spend ~ treated + recency + log_history + newbie + mens + womens,
    data = pos_df,
    family = Gamma(link = "log")
  )

  # 3. Multivariate Principal Component Analysis (PCA)
  mv_mat <- scale(df[, c("recency", "log_history", "newbie", "mens", "womens", "visit")])
  pca_res <- prcomp(mv_mat, center = FALSE, scale. = FALSE)

  # 4. Doubly Robust AIPW Causal Incrementality Estimator
  ps_fit <- glm(treated ~ recency + log_history + newbie + mens + womens,
                data = df, family = binomial(link = "logit"))
  e_hat <- pmin(pmax(predict(ps_fit, type = "response"), 0.05), 0.95)

  m1_fit <- glm(conversion ~ recency + log_history + newbie + mens + womens,
                data = subset(df, treated == 1), family = binomial(link = "logit"))
  m0_fit <- glm(conversion ~ recency + recency + log_history + newbie + mens + womens,
                data = subset(df, treated == 0), family = binomial(link = "logit"))

  mu1 <- predict(m1_fit, newdata = df, type = "response")
  mu0 <- predict(m0_fit, newdata = df, type = "response")

  psi_1 <- mu1 + df$treated * (df$conversion - mu1) / e_hat
  psi_0 <- mu0 + (1 - df$treated) * (df$conversion - mu0) / (1 - e_hat)
  ate_aipw <- mean(psi_1 - psi_0)

  list(
    logit_summary = summary(logit_fit)$coefficients,
    gamma_summary = summary(gamma_fit)$coefficients,
    pca_variance = summary(pca_res)$importance,
    doubly_robust_ate = ate_aipw
  )
}
