"""Research experiment: PMM3 for symmetric platykurtic positive spacings.

This script is intentionally separate from the production PMM-MASEM selector.
It studies whether the PMM3 symmetric-platykurtic location equation can be
adapted to positive shell spacings by estimating the per-particle spacing mean
as a location parameter:

    s_ij = mu_i + eps_ij,   E eps = 0,   eps symmetric platykurtic
    rho_i = 1 / mu_i

The PMM3 estimating equation for each row is:

    mean_j eps_ij * (kappa - eps_ij^2) = 0,

with kappa estimated once from pooled centered spacings.  This is a candidate
research adaptation, not a validated production estimator.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pmm3_platykurtic_spacing_research.csv"

SEEDS = [11, 17, 23, 31, 43]
K_GRID = [8, 16, 32, 64, 128]
N_PANELS = 6000


def _sample_positive_law(name: str, rng: np.random.Generator, size: tuple[int, int]) -> np.ndarray:
    """Return positive spacings with population mean near one."""
    if name == "uniform_0_2":
        s = rng.uniform(0.0, 2.0, size=size)
    elif name == "triangular_0_2":
        s = rng.triangular(0.0, 1.0, 2.0, size=size)
    elif name == "beta_2_2":
        s = 2.0 * rng.beta(2.0, 2.0, size=size)
    elif name == "beta_half_half":
        s = 2.0 * rng.beta(0.5, 0.5, size=size)
    elif name == "two_point_jitter":
        base = rng.choice([0.35, 1.65], size=size)
        s = base + rng.uniform(-0.03, 0.03, size=size)
    else:
        raise ValueError(f"unknown law: {name}")
    return s / np.mean(s)


def _moments_pmm3(x: np.ndarray) -> dict[str, float]:
    """Centered PMM3 moments and coefficients."""
    e = np.ravel(x) - float(np.mean(x))
    m2 = float(np.mean(e**2))
    m3 = float(np.mean(e**3))
    m4 = float(np.mean(e**4))
    m6 = float(np.mean(e**6))
    if m2 <= 0:
        return {k: float("nan") for k in ("gamma3", "gamma4", "gamma6", "g3", "kappa")}
    gamma3 = m3 / (m2**1.5)
    gamma4 = m4 / (m2**2) - 3.0
    gamma6 = m6 / (m2**3) - 15.0 * (m4 / (m2**2)) + 30.0
    denom_g3 = 6.0 + 9.0 * gamma4 + gamma6
    g3 = 1.0 - gamma4**2 / denom_g3 if denom_g3 > 0 else float("nan")
    denom_kappa = m4 - 3.0 * m2**2
    kappa = (m6 - 3.0 * m4 * m2) / denom_kappa if abs(denom_kappa) > 1e-14 else float("nan")
    return {
        "gamma3": gamma3,
        "gamma4": gamma4,
        "gamma6": gamma6,
        "g3": g3,
        "kappa": kappa,
    }


def _pmm3_location_mean(
    s: np.ndarray,
    *,
    kappa: float,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized intercept-only PMM3 location solve for rows of ``s``."""
    mu0 = np.mean(s, axis=1)
    if not np.isfinite(kappa) or kappa <= 0:
        return mu0, np.ones_like(mu0, dtype=bool), np.zeros_like(mu0, dtype=int)

    mu = mu0.copy()
    converged = np.zeros_like(mu, dtype=bool)
    n_iter = np.zeros_like(mu, dtype=int)

    for it in range(1, max_iter + 1):
        eps = s - mu[:, None]
        score = np.mean(eps * (kappa - eps**2), axis=1)
        jac = np.mean(3.0 * eps**2 - kappa, axis=1)
        safe = np.isfinite(jac) & (np.abs(jac) > 1e-14)
        step = np.zeros_like(mu)
        step[safe] = score[safe] / jac[safe]

        # Damped update: PMM3 can have multiple roots for tiny k.
        max_step = 0.5 * np.maximum(np.abs(mu), 1e-8)
        step = np.clip(step, -max_step, max_step)
        mu_new = np.maximum(mu - step, 1e-8)

        just_converged = np.abs(step) < tol
        converged |= just_converged
        n_iter[~converged] = it
        mu = mu_new

        if bool(np.all(converged)):
            break

    invalid = ~np.isfinite(mu) | (mu <= 0.0) | (mu > 10.0 * np.maximum(mu0, 1e-8))
    mu[invalid] = mu0[invalid]
    converged[invalid] = False
    return mu, converged, n_iter


def _rows_for(law: str, seed: int, k: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    s = _sample_positive_law(law, rng, (N_PANELS, k))
    mom = _moments_pmm3(s)
    mu_mle = np.mean(s, axis=1)
    mu_pmm3, converged, n_iter = _pmm3_location_mean(s, kappa=mom["kappa"])
    row_min = np.min(s, axis=1)
    row_max = np.max(s, axis=1)
    row_sd = np.std(s, axis=1, ddof=1)
    trust_radius = np.maximum(0.75 * row_sd, 0.25 * np.maximum(mu_mle, 1e-8))
    root_in_range = (mu_pmm3 >= row_min) & (mu_pmm3 <= row_max)
    root_in_trust_region = np.abs(mu_pmm3 - mu_mle) <= trust_radius
    use_pmm3 = converged & root_in_range & root_in_trust_region
    mu_pmm3_guarded = np.where(use_pmm3, mu_pmm3, mu_mle)

    estimators = {
        "mean_mle": mu_mle,
        "pmm3_location_raw": mu_pmm3,
        "pmm3_location_guarded": mu_pmm3_guarded,
    }

    rows: list[dict[str, object]] = []
    for estimator, mu_hat in estimators.items():
        rho_hat = 1.0 / np.maximum(mu_hat, 1e-12)
        rows.append(
            {
                "law": law,
                "seed": seed,
                "n_panels": N_PANELS,
                "k": k,
                "estimator": estimator,
                "gamma3": mom["gamma3"],
                "gamma4": mom["gamma4"],
                "gamma6": mom["gamma6"],
                "g3": mom["g3"],
                "kappa": mom["kappa"],
                "mean_bias": float(np.mean(mu_hat - 1.0)),
                "mean_variance": float(np.var(mu_hat, ddof=1)),
                "mean_mse": float(np.mean((mu_hat - 1.0) ** 2)),
                "density_bias": float(np.mean(rho_hat - 1.0)),
                "density_variance": float(np.var(rho_hat, ddof=1)),
                "density_mse": float(np.mean((rho_hat - 1.0) ** 2)),
                "convergence_rate": float(np.mean(converged)) if estimator.startswith("pmm3") else 1.0,
                "guard_accept_rate": float(np.mean(use_pmm3)) if estimator == "pmm3_location_guarded" else 1.0,
                "mean_iterations": float(np.mean(n_iter)) if estimator.startswith("pmm3") else 0.0,
            }
        )
    return rows


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    laws = ["uniform_0_2", "triangular_0_2", "beta_2_2", "beta_half_half", "two_point_jitter"]
    rows: list[dict[str, object]] = []
    for law in laws:
        for k in K_GRID:
            for seed in SEEDS:
                rows.extend(_rows_for(law, seed, k))

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
