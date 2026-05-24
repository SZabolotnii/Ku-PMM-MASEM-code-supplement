"""
PMM2 density estimation module for PMM-MASEM.

Implements Algorithm 1 from the paper:
  "Variance-reduced manifold sampling via polynomial-maximization density estimation"

Theory
------
On a flat homogeneous manifold with density ρ(xᵢ):
    s_{i,j} = N · ρ(xᵢ) · Δ_{i,j}  ~  i.i.d. Exp(1)
    (mean=1, c₃=2, c₄=6)

In that flat Exp(1) regime the plugin/MLE estimator is already optimal, so
the switching rule falls back to MLE instead of applying PMM2.

On a curved Riemannian manifold the spacing distribution deviates from Exp(1).
The standardised cumulants c₃ and c₄ capture this deviation.

PMM2 (asymmetric non-Exp errors, |c₃| > threshold):
    ρ̂_PMM2(xᵢ) = 1/(N·mean_Δᵢ) · (1 + c₃·(m₁ᵢ − m₁)/(2 + c₄))

PMM3 status
-----------
EstemPMM PMM3 is a regression-residual estimator with centered residuals,
gamma6, kappa, and the score eps * (kappa - eps^2).  The previous PMM-MASEM
code used a non-centered spacing-ratio correction with a fixed ``+15`` sixth
order surrogate.  That correction has structural bias on Uniform(0, 2)
positive spacings and is not a valid density estimator.  Until a proper
positive-spacing estimating equation is derived, PMM3 density adaptation is
disabled and the selector falls back to MLE in symmetric platykurtic regimes.

Variance reduction coefficients:
    g₂ = 1 − c₃²/(2 + c₄)
    g₃ = 1 − gamma4²/(6 + 9·gamma4 + gamma6)  [diagnostic only]

References
----------
- Kunchenko, Yu. P. (2002). Polynomial Parameter Estimations.
- Zabolotnii, S. et al. (2018–2025). PMM for regression/time-series.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from masem.spacings import knn_distances, shell_spacings, unit_ball_volume
from masem.estimators import _weights_from_density, MLEExpEstimator


# ---------------------------------------------------------------------------
# Switching thresholds
# ---------------------------------------------------------------------------

_THRESHOLD_C3: float = 0.3   # |c₃| > this → PMM2 (asymmetric)
_THRESHOLD_C4: float = -0.5  # c₄ < this → PMM3 (platykurtic, when |c₃| ≤ threshold)
_THRESHOLD_EXP_C3: float = 0.4  # Exp(1) flat-regime tolerance around c₃=2
_THRESHOLD_EXP_C4: float = 1.5  # Exp(1) flat-regime tolerance around c₄=6
_MIN_DENOM: float = 0.1      # 2 + c₄ must exceed this to avoid division by zero


class PMM3Moments(NamedTuple):
    """EstemPMM-compatible centered PMM3 diagnostics."""

    m2: Array
    m4: Array
    m6: Array
    gamma3: Array
    gamma4: Array
    gamma6: Array
    g3: Array
    kappa: Array
    denom_g3: Array
    denom_kappa: Array


# ---------------------------------------------------------------------------
# Cumulant estimation from pooled normalised spacings
# ---------------------------------------------------------------------------

def _estimate_cumulants(s: Array) -> tuple[Array, Array]:
    """
    Estimate standardised cumulants c₃ and c₄ from pooled normalised spacings.

    Parameters
    ----------
    s : Array, shape (N, k)
        Normalised spacings s_{i,j} = N · Δ_{i,j}.
        On a flat manifold s ~ Exp(1) with c₃=2, c₄=6.

    Returns
    -------
    c3 : scalar Array
        Standardised 3rd cumulant (skewness).
    c4 : scalar Array
        Standardised 4th cumulant (excess kurtosis).
    """
    s_flat = s.ravel()  # pool all N*k spacings

    m1 = jnp.mean(s_flat)
    m2 = jnp.mean(s_flat ** 2)
    m3 = jnp.mean(s_flat ** 3)
    m4 = jnp.mean(s_flat ** 4)

    # Central moments
    mu2 = m2 - m1 ** 2
    mu3 = m3 - 3.0 * m2 * m1 + 2.0 * m1 ** 3
    mu4 = m4 - 4.0 * m3 * m1 + 6.0 * m2 * m1 ** 2 - 3.0 * m1 ** 4

    # Standardised cumulants (guard against near-zero variance)
    mu2_safe = jnp.maximum(mu2, 1e-10)
    c3 = mu3 / (mu2_safe ** 1.5)
    c4 = mu4 / (mu2_safe ** 2) - 3.0

    return c3, c4


def _estimate_pmm3_moments(x: Array) -> PMM3Moments:
    """
    Compute EstemPMM PMM3 centered diagnostics up to sixth order.

    This is intentionally a diagnostic utility, not a PMM-MASEM density
    estimator.  EstemPMM PMM3 applies to centered regression residuals; positive
    shell spacings need a separate derivation before PMM3 can be enabled here.
    """
    x_flat = x.ravel()
    eps = x_flat - jnp.mean(x_flat)
    raw_m2 = jnp.mean(eps ** 2)
    eps = eps / jnp.sqrt(jnp.maximum(raw_m2, 1e-12))
    m2 = jnp.mean(eps ** 2)
    m3 = jnp.mean(eps ** 3)
    m4 = jnp.mean(eps ** 4)
    m6 = jnp.mean(eps ** 6)

    m2_safe = jnp.maximum(m2, 1e-12)
    gamma3 = m3 / (m2_safe ** 1.5)
    gamma4 = m4 / (m2_safe ** 2) - 3.0
    gamma6 = m6 / (m2_safe ** 3) - 15.0 * (m4 / (m2_safe ** 2)) + 30.0

    denom_g3 = 6.0 + 9.0 * gamma4 + gamma6
    g3_raw = 1.0 - gamma4 ** 2 / jnp.maximum(denom_g3, 1e-12)
    g3 = jnp.where(denom_g3 > 0.0, g3_raw, jnp.nan)

    denom_kappa = m4 - 3.0 * m2 ** 2
    kappa_raw = (m6 - 3.0 * m4 * m2) / jnp.where(
        jnp.abs(denom_kappa) > 1e-12,
        denom_kappa,
        1.0,
    )
    kappa = jnp.where(jnp.abs(denom_kappa) > 1e-12, kappa_raw, jnp.nan)

    return PMM3Moments(
        m2=m2,
        m4=m4,
        m6=m6,
        gamma3=gamma3,
        gamma4=gamma4,
        gamma6=gamma6,
        g3=g3,
        kappa=kappa,
        denom_g3=denom_g3,
        denom_kappa=denom_kappa,
    )


def _is_exp1_like(
    c3: Array,
    c4: Array,
    threshold_exp_c3: float = _THRESHOLD_EXP_C3,
    threshold_exp_c4: float = _THRESHOLD_EXP_C4,
) -> Array:
    """
    Return True when cumulants are close to the flat Exp(1) spacing regime.

    This is the honest-boundary gate: on a flat homogeneous manifold the plugin
    estimator is the MLE, so PMM should not be selected merely because
    Exp(1) has nonzero skewness c3=2.
    """
    return (
        (jnp.abs(c3 - 2.0) <= threshold_exp_c3)
        & (jnp.abs(c4 - 6.0) <= threshold_exp_c4)
    )


# ---------------------------------------------------------------------------
# Per-particle moments of normalised spacings
# ---------------------------------------------------------------------------

def _particle_moments(s: Array) -> tuple[Array, Array]:
    """
    Compute per-particle first and second moments of normalised spacings.

    Parameters
    ----------
    s : Array, shape (N, k)
        Normalised spacings.

    Returns
    -------
    m1i : Array, shape (N,)
        Per-particle mean of s_{i,j}.
    m2i : Array, shape (N,)
        Per-particle mean of s_{i,j}².
    """
    m1i = jnp.mean(s, axis=1)          # (N,)
    m2i = jnp.mean(s ** 2, axis=1)     # (N,)
    return m1i, m2i


# ---------------------------------------------------------------------------
# PMM2 density estimate
# ---------------------------------------------------------------------------

def _pmm2_density(
    delta: Array,
    s: Array,
    c3: Array,
    c4: Array,
    N: int,
) -> Array:
    """
    PMM2 density estimate for each particle.

    ρ̂_PMM2(xᵢ) = 1/(N·mean_Δᵢ) · (1 + c₃·(m₁ᵢ − m₁)/(2 + c₄))

    where m₁ = mean(s) is the global mean of normalised spacings.

    The correction term uses the deviation of particle i's mean spacing
    from the global mean.  When m₁ᵢ > m₁ (larger spacings → lower MLE
    density), the positive correction raises ρ̂ towards the true density.

    Parameters
    ----------
    delta : Array, shape (N, k)
        Shell spacings Δ_{i,j}.
    s : Array, shape (N, k)
        Normalised spacings s_{i,j} = N · Δ_{i,j}.
    c3 : scalar
        Pooled standardised 3rd cumulant.
    c4 : scalar
        Pooled standardised 4th cumulant.
    N : int
        Number of particles.

    Returns
    -------
    rho : Array, shape (N,)
        PMM2 density estimates.
    """
    mean_delta = jnp.mean(delta, axis=1)                    # (N,)
    rho_mle = 1.0 / (N * jnp.maximum(mean_delta, 1e-37))   # (N,)

    m1i, _ = _particle_moments(s)                           # (N,)
    m1 = jnp.mean(s)                                        # global mean

    denom = 2.0 + c4
    correction = c3 * (m1i - m1) / jnp.maximum(denom, _MIN_DENOM)
    rho_pmm2 = rho_mle * (1.0 + correction)

    return jnp.maximum(rho_pmm2, 1e-37)


# ---------------------------------------------------------------------------
# PMM3 density estimate
# ---------------------------------------------------------------------------

def _pmm3_density(
    delta: Array,
    s: Array,
    c4: Array,
    N: int,
) -> Array:
    """
    Disabled PMM3 density estimate for each particle.

    The previous correction was not EstemPMM-compatible and used a fake sixth
    order surrogate.  Returning MLE here keeps legacy imports/tests finite while
    preventing accidental use of the invalid PMM3 density adaptation.

    Parameters
    ----------
    delta : Array, shape (N, k)
        Shell spacings.
    s : Array, shape (N, k)
        Normalised spacings.
    c4 : scalar
        Pooled standardised 4th cumulant.
    N : int
        Number of particles.

    Returns
    -------
    rho : Array, shape (N,)
        MLE fallback density estimates.
    """
    del s, c4
    return _mle_density(delta, N)


# ---------------------------------------------------------------------------
# MLE fallback density
# ---------------------------------------------------------------------------

def _mle_density(delta: Array, N: int) -> Array:
    """
    MLE (Plugin) density estimate: ρ̂_MLE(xᵢ) = 1/(N·mean_Δᵢ).

    Parameters
    ----------
    delta : Array, shape (N, k)
    N : int

    Returns
    -------
    rho : Array, shape (N,)
    """
    mean_delta = jnp.mean(delta, axis=1)
    return 1.0 / (N * jnp.maximum(mean_delta, 1e-37))


def _select_density_estimate(
    delta: Array,
    s: Array,
    c3: Array,
    c4: Array,
    N: int,
    threshold_c3: float = _THRESHOLD_C3,
    threshold_c4: float = _THRESHOLD_C4,
    threshold_exp_c3: float = _THRESHOLD_EXP_C3,
    threshold_exp_c4: float = _THRESHOLD_EXP_C4,
) -> Array:
    """Select PMM2 or MLE density according to the guarded switching rule."""
    rho_pmm2 = _pmm2_density(delta, s, c3, c4, N)   # (N,)
    rho_mle = _mle_density(delta, N)                 # (N,)

    exp1_like = _is_exp1_like(c3, c4, threshold_exp_c3, threshold_exp_c4)

    # Validity checks for PMM2
    denom_pmm2 = 2.0 + c4
    g2 = 1.0 - c3 ** 2 / jnp.maximum(denom_pmm2, 1e-10)
    pmm2_valid = (denom_pmm2 >= _MIN_DENOM) & (g2 >= 0.0) & (g2 <= 1.0)

    # PMM is only eligible outside the flat Exp(1) regime.
    use_pmm2 = (~exp1_like) & (jnp.abs(c3) > threshold_c3) & pmm2_valid

    del threshold_c4
    return jnp.where(use_pmm2, rho_pmm2, rho_mle)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def pmm_density_weights(
    particles: Array,
    k: int,
    tau: float,
    p: int,
    N: int,
    threshold_c3: float = _THRESHOLD_C3,
    threshold_c4: float = _THRESHOLD_C4,
    threshold_exp_c3: float = _THRESHOLD_EXP_C3,
    threshold_exp_c4: float = _THRESHOLD_EXP_C4,
    **kwargs,
) -> Array:
    """
    PMM2 density estimation with automatic switching rule.

    Algorithm 1 from the paper:
      1. Compute sorted kNN distances εᵢ,₁ ≤ … ≤ εᵢ,ₖ
      2. Compute shell spacings Δⱼ = Vₚ(εᵢ,ⱼᵖ − εᵢ,ⱼ₋₁ᵖ)
      3. Pool spacings; normalise: s_{i,j} = N · Δ_{i,j}
      4. Estimate c₃, c₄ from pooled spacings
      5. if (c₃,c₄) ≈ (2,6): fallback to MLE_Exp (flat Plugin/MLE regime)
         elif |c₃| > threshold_c3: use PMM2
         elif c₃ ≈ 0 and c₄ < threshold_c4: PMM3 disabled, fallback to MLE_Exp
         else: fallback to MLE_Exp
      6. Compute weights: wᵢ = ρ̂ᵢ^{−τ}, normalise

    Parameters
    ----------
    particles : Array, shape (N, p_dim)
        Current particle positions.
    k : int
        Number of nearest neighbours.
    tau : float
        Resampling temperature τ.
    p : int
        Intrinsic dimension of the manifold.
    N : int
        Number of particles (= particles.shape[0]).
    threshold_c3 : float
        |c₃| threshold for PMM2 vs PMM3/fallback switching (default 0.3).
    threshold_c4 : float
        c₄ threshold for PMM3 vs fallback switching (default -0.5).
    threshold_exp_c3 : float
        Tolerance around c₃=2 for detecting the flat Exp(1) regime.
    threshold_exp_c4 : float
        Tolerance around c₄=6 for detecting the flat Exp(1) regime.
    **kwargs
        Ignored (for interface compatibility).

    Returns
    -------
    weights : Array, shape (N,)
        Normalised resampling weights summing to 1.
    """
    # Step 1–2: kNN distances and shell spacings
    eps = knn_distances(particles, k)       # (N, k)
    delta = shell_spacings(eps, p)          # (N, k)

    # Step 3: normalised spacings  s_{i,j} = N · Δ_{i,j}
    s = N * delta                           # (N, k)

    # Step 4: estimate cumulants from pooled spacings
    c3, c4 = _estimate_cumulants(s)

    # Step 5: switching rule (JAX-traceable; no Python data-dependent branch)
    rho_hat = _select_density_estimate(
        delta,
        s,
        c3,
        c4,
        N,
        threshold_c3=threshold_c3,
        threshold_c4=threshold_c4,
        threshold_exp_c3=threshold_exp_c3,
        threshold_exp_c4=threshold_exp_c4,
    )

    return _weights_from_density(rho_hat, tau)
