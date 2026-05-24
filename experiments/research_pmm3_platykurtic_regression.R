#!/usr/bin/env Rscript

# Research grid: EstemPMM PMM3 on symmetric platykurtic regression residuals.
# This validates the original regression setting separately from PMM-MASEM
# positive shell-spacing density estimation.

suppressPackageStartupMessages(library(EstemPMM))

root <- normalizePath(getwd(), mustWork = TRUE)
out_path <- file.path(root, "results", "pmm3_platykurtic_regression_research.csv")

make_eps <- function(law, n) {
  if (law == "normal") {
    e <- rnorm(n)
  } else if (law == "triangular") {
    e <- (runif(n) + runif(n) - 1) * sqrt(6)
  } else if (law == "uniform") {
    e <- runif(n, -sqrt(3), sqrt(3))
  } else if (law == "beta_2_2") {
    e <- 2 * rbeta(n, 2, 2) - 1
  } else if (law == "beta_half_half") {
    e <- 2 * rbeta(n, 0.5, 0.5) - 1
  } else if (law == "two_point_jitter") {
    e <- sample(c(-1, 1), n, replace = TRUE) + runif(n, -0.05, 0.05)
  } else {
    stop(sprintf("unknown law: %s", law))
  }
  e <- e - mean(e)
  e / sd(e)
}

moments <- function(x) {
  r <- x - mean(x)
  m2 <- mean(r^2)
  m3 <- mean(r^3)
  m4 <- mean(r^4)
  m6 <- mean(r^6)
  gamma3 <- m3 / (m2^(3/2))
  gamma4 <- m4 / (m2^2) - 3
  gamma6 <- m6 / (m2^3) - 15 * (m4 / (m2^2)) + 30
  denom <- 6 + 9 * gamma4 + gamma6
  g3 <- if (is.finite(denom) && denom > 0) 1 - gamma4^2 / denom else NA_real_
  c(gamma3 = gamma3, gamma4 = gamma4, gamma6 = gamma6, g3 = g3)
}

run_cell <- function(law, n, M, seed) {
  set.seed(seed)
  beta0 <- 0.7
  beta1 <- 1.4
  x <- runif(n, -1, 1)
  ols <- numeric(M)
  pmm3 <- numeric(M)
  conv <- logical(M)
  eps_pool <- numeric(0)

  for (i in seq_len(M)) {
    eps <- make_eps(law, n)
    y <- beta0 + beta1 * x + eps
    dat <- data.frame(y = y, x = x)
    fit_ols <- lm(y ~ x, data = dat)
    fit_pmm3 <- suppressWarnings(lm_pmm3(y ~ x, data = dat, max_iter = 100, tol = 1e-7))
    ols[i] <- coef(fit_ols)[["x"]]
    pmm3[i] <- coef(fit_pmm3)[["x"]]
    conv[i] <- isTRUE(fit_pmm3@convergence)
    eps_pool <- c(eps_pool, eps)
  }

  mom <- moments(eps_pool)
  data.frame(
    law = law,
    n = n,
    M = M,
    method = c("OLS", "PMM3"),
    gamma3 = unname(mom["gamma3"]),
    gamma4 = unname(mom["gamma4"]),
    gamma6 = unname(mom["gamma6"]),
    g3_theory = unname(mom["g3"]),
    bias = c(mean(ols - beta1), mean(pmm3 - beta1)),
    variance = c(var(ols), var(pmm3)),
    mse = c(mean((ols - beta1)^2), mean((pmm3 - beta1)^2)),
    var_ratio_vs_ols = c(1, var(pmm3) / var(ols)),
    are_vs_ols = c(1, var(ols) / var(pmm3)),
    convergence_rate = c(1, mean(conv))
  )
}

main <- function() {
  dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
  laws <- c("normal", "triangular", "uniform", "beta_2_2", "beta_half_half", "two_point_jitter")
  ns <- c(50L, 100L, 200L)
  M <- 300L
  rows <- list()
  idx <- 1L
  for (law in laws) {
    for (n in ns) {
      cat(sprintf("law=%s n=%d\n", law, n))
      rows[[idx]] <- run_cell(law, n, M, seed = 20260524 + idx)
      idx <- idx + 1L
    }
  }
  out <- do.call(rbind, rows)
  write.csv(out, out_path, row.names = FALSE)
  cat(sprintf("wrote %s (%d rows)\n", out_path, nrow(out)))
}

main()
