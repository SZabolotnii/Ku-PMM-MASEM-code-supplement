#!/usr/bin/env Rscript

# Sanity check for EstemPMM PMM3 on centered uniform regression errors.
# This is deliberately not PMM-MASEM density evidence.

suppressPackageStartupMessages(library(EstemPMM))

set.seed(20260524)

n_rep <- 300L
n <- 120L
beta0 <- 0.7
beta1 <- 1.4

ols_b1 <- numeric(n_rep)
pmm3_b1 <- numeric(n_rep)

for (r in seq_len(n_rep)) {
  x <- runif(n, -1, 1)
  eps <- runif(n, -sqrt(3), sqrt(3))
  y <- beta0 + beta1 * x + eps
  dat <- data.frame(y = y, x = x)

  fit_ols <- lm(y ~ x, data = dat)
  fit_pmm3 <- lm_pmm3(y ~ x, data = dat, max_iter = 100, tol = 1e-7)

  ols_b1[r] <- coef(fit_ols)[["x"]]
  pmm3_b1[r] <- coef(fit_pmm3)[["x"]]
}

ratio <- var(pmm3_b1) / var(ols_b1)

cat(sprintf("n_rep=%d n=%d\n", n_rep, n))
cat(sprintf("var(beta1_PMM3) / var(beta1_OLS) = %.4f\n", ratio))
cat("expected_range=[0.30, 0.45]\n")

if (!is.finite(ratio) || ratio < 0.30 || ratio > 0.45) {
  stop("EstemPMM PMM3 uniform regression sanity check outside expected range")
}
