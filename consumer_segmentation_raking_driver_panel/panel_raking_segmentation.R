# ==============================================================================
# Survey Panel IPF Rim Weighting, K-Means Consumer Segmentation & Driver Model (R)
# ==============================================================================

run_panel_raking_and_drivers_r <- function(csv_path) {
  df <- read.csv(csv_path, stringsAsFactors = FALSE)
  n <- nrow(df)
  w <- rep(1.0, n)

  geo_bench <- c(Surburban = 0.50, Urban = 0.35, Rural = 0.15)
  chan_bench <- c(Web = 0.52, Phone = 0.28, Multichannel = 0.20)

  for (iter in seq_len(25)) {
    for (g in names(geo_bench)) {
      idx <- df$zip_code == g
      w[idx] <- w[idx] * (geo_bench[[g]] * n) / sum(w[idx])
    }
    for (ch in names(chan_bench)) {
      idx <- df$channel == ch
      w[idx] <- w[idx] * (chan_bench[[ch]] * n) / sum(w[idx])
    }
  }
  w <- pmin(pmax(w, 0.35), 3.00)
  w <- w / mean(w)
  kish_n_eff <- (sum(w)^2) / sum(w^2)

  # Weighted Key Driver Linear Probability / GLM fit
  df$log_hist <- log1p(df$history)
  df$dual_cat <- as.numeric(df$mens == 1 & df$womens == 1)
  df$stimuli <- as.numeric(df$segment != "No E-Mail")
  fit <- stats::lm(visit ~ log_hist + recency + dual_cat + stimuli, data = df, weights = w)

  list(n = n, kish_n_eff = kish_n_eff, deff = n / kish_n_eff, r2 = summary(fit)$r.squared)
}
