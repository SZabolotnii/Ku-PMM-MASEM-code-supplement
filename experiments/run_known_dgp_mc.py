"""Run the Known-DGP density microbenchmark and write results/known_dgp_mc.csv."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiments.known_dgp_generator import REGIMES, SpacingRegime, generate_spacings
from masem.pmm_module import (  # noqa: E402
    _estimate_cumulants,
    _estimate_pmm3_moments,
    _is_exp1_like,
    _mle_density,
    _pmm2_density,
    _pmm3_density,
    _select_density_estimate,
)


ESTIMATORS = ("Plugin_Estimator", "k_Ensemble", "MLE_Exp", "PMM2/MLE")


def _branch(c3: float, c4: float) -> str:
    """Mirror the PMM selector branch for diagnostics."""
    exp_like = bool(_is_exp1_like(jnp.asarray(c3), jnp.asarray(c4)))
    denom_pmm2 = 2.0 + c4
    g2 = 1.0 - c3**2 / max(denom_pmm2, 1e-10)
    pmm2_valid = denom_pmm2 >= 0.1 and 0.0 <= g2 <= 1.0
    if exp_like:
        return "MLE_fallback"
    if abs(c3) > 0.3 and pmm2_valid:
        return "PMM2"
    if abs(c3) <= 0.3 and c4 < -0.5:
        return "PMM3_disabled"
    return "MLE_fallback"


def _fallback_reason(c3: float, c4: float, branch: str) -> str:
    """Human-readable selector reason for result auditing."""
    if branch == "PMM2":
        return "PMM2_valid_asymmetric_spacing_regime"
    if branch == "PMM3_disabled":
        return "PMM3_density_adaptation_not_derived_fallback_to_MLE"
    if bool(_is_exp1_like(jnp.asarray(c3), jnp.asarray(c4))):
        return "flat_Exp1_MLE_is_reference"
    denom_pmm2 = 2.0 + c4
    g2 = 1.0 - c3**2 / max(denom_pmm2, 1e-10)
    if abs(c3) > 0.3 and not (denom_pmm2 >= 0.1 and 0.0 <= g2 <= 1.0):
        return "PMM2_invalid_g2_or_denominator"
    return "no_valid_PMM_branch"


def _density_estimates(
    s_full: np.ndarray,
    *,
    n_particles: int,
    primary_k: int,
    estimator: str,
) -> np.ndarray:
    """Compute density estimates for one synthetic spacing panel."""
    s = s_full[:, :primary_k]
    delta = jnp.asarray(s / n_particles)
    s_jax = jnp.asarray(s)

    if estimator == "Plugin_Estimator":
        rho = primary_k / np.maximum(np.sum(s, axis=1), 1e-300)
        return np.asarray(rho, dtype=np.float64)
    if estimator == "MLE_Exp":
        return np.asarray(_mle_density(delta, n_particles), dtype=np.float64)
    if estimator == "k_Ensemble":
        k_values = sorted({max(1, primary_k // 2), primary_k, min(s_full.shape[1], 2 * primary_k)})
        rho = np.zeros(n_particles, dtype=np.float64)
        for k_i in k_values:
            rho += k_i / np.maximum(np.sum(s_full[:, :k_i], axis=1), 1e-300)
        return rho / len(k_values)
    if estimator == "PMM2/MLE":
        c3, c4 = _estimate_cumulants(s_jax)
        rho = _select_density_estimate(delta, s_jax, c3, c4, n_particles)
        return np.asarray(rho, dtype=np.float64)
    raise ValueError(f"Unknown estimator: {estimator!r}")


def _candidate_mse(s_full: np.ndarray, n_particles: int, primary_k: int) -> tuple[float, float, float]:
    """Return raw MLE, PMM2, and PMM3 MSEs for diagnostic columns."""
    s = s_full[:, :primary_k]
    delta = jnp.asarray(s / n_particles)
    s_jax = jnp.asarray(s)
    c3, c4 = _estimate_cumulants(s_jax)
    rho_mle = np.asarray(_mle_density(delta, n_particles))
    rho_pmm2 = np.asarray(_pmm2_density(delta, s_jax, c3, c4, n_particles))
    rho_pmm3 = np.asarray(_pmm3_density(delta, s_jax, c4, n_particles))
    return (
        float(np.mean((rho_mle - 1.0) ** 2)),
        float(np.mean((rho_pmm2 - 1.0) ** 2)),
        float(np.mean((rho_pmm3 - 1.0) ** 2)),
    )


def _rows_for_regime(
    regime: SpacingRegime,
    *,
    seed: int,
    n_particles: int,
    primary_k: int,
    k_max: int,
) -> list[dict[str, object]]:
    s_full = generate_spacings(regime, n_particles=n_particles, k_max=k_max, seed=seed)
    s = s_full[:, :primary_k]
    c3_j, c4_j = _estimate_cumulants(jnp.asarray(s))
    pmm3_mom = _estimate_pmm3_moments(jnp.asarray(s))
    c3 = float(c3_j)
    c4 = float(c4_j)
    branch = _branch(c3, c4)
    fallback_reason = _fallback_reason(c3, c4, branch)
    g2 = float(1.0 - c3**2 / max(2.0 + c4, 1e-10))
    gamma6 = float(pmm3_mom.gamma6)
    kappa = float(pmm3_mom.kappa)
    g3 = float(pmm3_mom.g3)
    mle_mse, pmm2_mse, pmm3_mse = _candidate_mse(s_full, n_particles, primary_k)

    rows: list[dict[str, object]] = []
    for estimator in ESTIMATORS:
        rho = _density_estimates(
            s_full,
            n_particles=n_particles,
            primary_k=primary_k,
            estimator=estimator,
        )
        mse = float(np.mean((rho - 1.0) ** 2))
        rows.append(
            {
                "regime": regime.name,
                "regime_label": regime.label,
                "family": regime.family,
                "seed": seed,
                "estimator": estimator,
                "n_particles": n_particles,
                "k": primary_k,
                "curvature_proxy": regime.curvature_proxy,
                "c3": c3,
                "c4": c4,
                "g2": g2,
                "gamma6": gamma6,
                "kappa": kappa,
                "g3_estem": g3,
                "selector_branch": branch,
                "fallback_reason": fallback_reason,
                "density_bias": float(np.mean(rho - 1.0)),
                "density_variance": float(np.var(rho, ddof=1)),
                "density_mse": mse,
                "mse_ratio_vs_mle": float(mse / max(mle_mse, 1e-300)),
                "raw_mle_mse": mle_mse,
                "raw_pmm2_mse": pmm2_mse,
                "raw_pmm3_mse": pmm3_mse,
                "note": regime.note,
            }
        )
    return rows


def main() -> None:
    os.makedirs(ROOT / "results", exist_ok=True)
    out_path = ROOT / "results" / "known_dgp_mc.csv"
    seeds = [11, 17, 23, 31, 43]
    rows: list[dict[str, object]] = []
    for regime in REGIMES:
        if regime.name == "platykurtic_uniform":
            jobs = [
                (n_particles, k)
                for n_particles in (900, 3000)
                for k in (8, 16, 32, 64, 128)
            ]
        else:
            jobs = [(900, 16)]
        for n_particles, primary_k in jobs:
            for seed in seeds:
                rows.extend(
                    _rows_for_regime(
                        regime,
                        seed=seed,
                        n_particles=n_particles,
                        primary_k=primary_k,
                        k_max=max(2 * primary_k, primary_k),
                    )
                )

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
