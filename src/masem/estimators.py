"""
Unified density estimator interface for PMM-MASEM.

Four estimators are compared throughout the paper (Requirement 7.2):

  1. Plugin_Estimator  — k-NN plug-in: q̂(xᵢ) = k / (N · V_p · ε_{i,k}^p)
  2. k_Ensemble        — average over multiple k values (MASEM authors' method)
  3. MLE_Exp           — MLE for Exp(λ) model on shell spacings
  4. PMM_Estimator     — PMM2/PMM3 moment estimator (full impl in pmm_module.py)

Unified interface
-----------------
Every estimator is a callable with signature::

    weights = estimator(particles, k, tau, p, N, **kwargs)

where
  particles : jax.Array, shape (N, p_dim)
  k         : int   — neighbourhood size (primary k for Plugin/MLE/PMM)
  tau       : float — resampling temperature (τ)
  p         : int   — intrinsic dimension of the manifold
  N         : int   — number of particles (= particles.shape[0])

and returns
  weights   : jax.Array, shape (N,) — normalised resampling weights

All estimators are JAX-traceable (no per-particle Python loops).

References
----------
- Braun et al. (2026) arXiv:2605.12338v1  — MASEM, k-ensemble, plug-in
- Loftsgaarden & Quesenberry (1965)        — k-NN density estimation
- Kunchenko (1992); Zabolotnii (2018–2025) — PMM
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array

from masem.spacings import knn_distances, shell_spacings, unit_ball_volume


# ---------------------------------------------------------------------------
# Protocol — unified estimator interface
# ---------------------------------------------------------------------------

@runtime_checkable
class DensityEstimator(Protocol):
    """
    Protocol for all density estimators used in PMM-MASEM.

    Every estimator must be callable with the signature below and return
    normalised resampling weights of shape (N,).
    """

    def __call__(
        self,
        particles: Array,
        k: int,
        tau: float,
        p: int,
        N: int,
        **kwargs,
    ) -> Array:
        """
        Compute normalised resampling weights.

        Parameters
        ----------
        particles : Array, shape (N, p_dim)
            Current particle positions.
        k : int
            Primary neighbourhood size.
        tau : float
            Resampling temperature τ.  Weights ∝ ρ̂(xᵢ)^{−τ}.
        p : int
            Intrinsic dimension of the manifold.
        N : int
            Number of particles (= particles.shape[0]).
        **kwargs
            Estimator-specific keyword arguments.

        Returns
        -------
        weights : Array, shape (N,)
            Normalised resampling weights summing to 1.
        """
        ...


# ---------------------------------------------------------------------------
# Helper: weights from density estimates
# ---------------------------------------------------------------------------

def _weights_from_density(rho_hat: Array, tau: float) -> Array:
    """
    Convert density estimates to normalised resampling weights.

    w_i = ρ̂(xᵢ)^{−τ} / Σ_j ρ̂(xⱼ)^{−τ}

    Numerically stable: divide by max before exponentiation.
    """
    log_rho = jnp.log(jnp.maximum(rho_hat, 1e-300))
    log_w = -tau * log_rho
    log_w = log_w - jnp.max(log_w)          # shift for numerical stability
    w = jnp.exp(log_w)
    return w / jnp.sum(w)


# ---------------------------------------------------------------------------
# Estimator 1: Plugin_Estimator
# ---------------------------------------------------------------------------

class PluginEstimator:
    """
    k-NN plug-in density estimator (baseline MASEM).

    q̂(xᵢ) = k / (N · V_p · ε_{i,k}^p)

    This is the MLE for the Exp(1) spacing model on a flat homogeneous
    manifold.  On such manifolds PMM does NOT outperform this estimator
    (see Spec.md §2 and Theoretical Framework §3(a)).

    Resampling weight: w_i ∝ ε_{i,k}^τ  (equivalently ∝ q̂(xᵢ)^{−τ}).
    """

    name: str = "Plugin_Estimator"

    def __call__(
        self,
        particles: Array,
        k: int,
        tau: float,
        p: int,
        N: int,
        **kwargs,
    ) -> Array:
        """
        Parameters
        ----------
        particles : Array, shape (N, p_dim)
        k         : int
        tau       : float
        p         : int
        N         : int

        Returns
        -------
        weights : Array, shape (N,)
        """
        eps = knn_distances(particles, k)          # (N, k)
        eps_k = eps[:, -1]                         # ε_{i,k}, shape (N,)
        Vp = unit_ball_volume(p)
        rho_hat = k / (N * Vp * eps_k ** p)       # (N,)
        return _weights_from_density(rho_hat, tau)


# ---------------------------------------------------------------------------
# Estimator 2: k_Ensemble
# ---------------------------------------------------------------------------

class kEnsembleEstimator:
    """
    Ensemble over multiple k values — the MASEM authors' ad-hoc method.

    For each k_i in k_values, compute the plug-in density estimate and
    average the resulting density estimates:

        ρ̂_ens(xᵢ) = (1/|K|) Σ_{k∈K} k / (N · V_p · ε_{i,k}^p)

    This reduces variance at the cost of introducing bias (Braun et al.,
    Appendix D.5).  It is the primary baseline against which PMM is compared.

    Parameters
    ----------
    k_values : list[int] | None
        The set of k values to ensemble over.  If None, defaults to
        [k//2, k, 2*k] (centred on the primary k).
    """

    name: str = "k_Ensemble"

    def __init__(self, k_values: list[int] | None = None) -> None:
        self.k_values = k_values

    def __call__(
        self,
        particles: Array,
        k: int,
        tau: float,
        p: int,
        N: int,
        **kwargs,
    ) -> Array:
        """
        Parameters
        ----------
        particles : Array, shape (N, p_dim)
        k         : int   — primary k; k_values defaults to [k//2, k, 2*k]
        tau       : float
        p         : int
        N         : int

        Returns
        -------
        weights : Array, shape (N,)
        """
        k_values = self.k_values
        if k_values is None:
            # Default ensemble: half, primary, double — clamp to at least 1
            k_values = sorted({max(1, k // 2), k, 2 * k})

        k_max = max(k_values)
        eps_all = knn_distances(particles, k_max)  # (N, k_max)
        Vp = unit_ball_volume(p)

        # Accumulate density estimates across ensemble members
        rho_sum = jnp.zeros(N)
        for ki in k_values:
            eps_ki = eps_all[:, ki - 1]            # ε_{i,ki}, shape (N,)
            rho_ki = ki / (N * Vp * eps_ki ** p)
            rho_sum = rho_sum + rho_ki

        rho_hat = rho_sum / len(k_values)          # (N,)
        return _weights_from_density(rho_hat, tau)


# ---------------------------------------------------------------------------
# Estimator 3: MLE_Exp
# ---------------------------------------------------------------------------

class MLEExpEstimator:
    """
    MLE density estimator for the exponential spacing model.

    Model: N · ρ(xᵢ) · Δ_{i,j} ~ i.i.d. Exp(1)
    ⟹ N · ρ(xᵢ) = 1 / mean_j(Δ_{i,j})
    ⟹ ρ̂_MLE(xᵢ) = 1 / (N · mean_j(Δ_{i,j}))

    On a flat homogeneous manifold this coincides with the plug-in estimator
    (both are MLE for the same model).  On curved manifolds the spacing
    distribution deviates from Exp(1), so this estimator differs from the
    plug-in (which uses only the k-th distance).

    Using all k spacings rather than just ε_{i,k} reduces variance by a
    factor of k (sufficient statistic for the rate parameter of Exp).
    """

    name: str = "MLE_Exp"

    def __call__(
        self,
        particles: Array,
        k: int,
        tau: float,
        p: int,
        N: int,
        **kwargs,
    ) -> Array:
        """
        Parameters
        ----------
        particles : Array, shape (N, p_dim)
        k         : int
        tau       : float
        p         : int
        N         : int

        Returns
        -------
        weights : Array, shape (N,)
        """
        eps = knn_distances(particles, k)          # (N, k)
        delta = shell_spacings(eps, p)             # (N, k)

        # MLE for Exp(λ): λ̂ = 1 / mean(Δ)
        # Here λ = N · ρ(xᵢ), so ρ̂ = λ̂ / N
        mean_delta = jnp.mean(delta, axis=1)       # (N,)
        rho_hat = 1.0 / (N * jnp.maximum(mean_delta, 1e-300))  # (N,)
        return _weights_from_density(rho_hat, tau)


# ---------------------------------------------------------------------------
# Estimator 4: PMM_Estimator (stub — full implementation in pmm_module.py)
# ---------------------------------------------------------------------------

class PMMEstimator:
    """
    PMM2/PMM3 density estimator (Kunchenko 1992; Zabolotnii 2018–2025).

    This class provides the unified-interface wrapper.  The closed-form
    PMM2/PMM3 moment expressions and the automatic switching rule are
    implemented in ``masem.pmm_module`` (task 1.2).

    When ``masem.pmm_module`` is not yet available (task 1.2 not complete),
    this estimator falls back to ``MLE_Exp`` with a warning.

    Parameters
    ----------
    fallback_to_mle : bool
        If True (default), fall back to MLE_Exp when the PMM module is
        unavailable.  Set to False to raise ImportError instead.
    """

    name: str = "PMM2/PMM3"

    def __init__(self, fallback_to_mle: bool = True) -> None:
        self.fallback_to_mle = fallback_to_mle
        self._pmm_available = self._check_pmm_module()

    @staticmethod
    def _check_pmm_module() -> bool:
        try:
            from masem import pmm_module  # noqa: F401
            return True
        except ImportError:
            return False

    def __call__(
        self,
        particles: Array,
        k: int,
        tau: float,
        p: int,
        N: int,
        **kwargs,
    ) -> Array:
        """
        Parameters
        ----------
        particles : Array, shape (N, p_dim)
        k         : int
        tau       : float
        p         : int
        N         : int

        Returns
        -------
        weights : Array, shape (N,)
        """
        if self._pmm_available:
            from masem.pmm_module import pmm_density_weights
            return pmm_density_weights(particles, k, tau, p, N, **kwargs)

        if self.fallback_to_mle:
            import warnings
            warnings.warn(
                "masem.pmm_module not found (task 1.2 not yet complete). "
                "PMMEstimator falling back to MLE_Exp.",
                stacklevel=2,
            )
            return MLEExpEstimator()(particles, k, tau, p, N, **kwargs)

        raise ImportError(
            "masem.pmm_module is required for PMMEstimator but was not found. "
            "Complete task 1.2 to enable PMM2/PMM3 estimation."
        )


# ---------------------------------------------------------------------------
# Registry — all four estimators
# ---------------------------------------------------------------------------

#: Canonical registry of all four estimators used in the paper.
#: Keys match the names used in Table E1 and throughout the manuscript.
ESTIMATOR_REGISTRY: dict[str, DensityEstimator] = {
    "Plugin_Estimator": PluginEstimator(),
    "k_Ensemble":       kEnsembleEstimator(),
    "MLE_Exp":          MLEExpEstimator(),
    "PMM2/PMM3":        PMMEstimator(),
}


def get_estimator(name: str) -> DensityEstimator:
    """
    Retrieve an estimator by name from the canonical registry.

    Parameters
    ----------
    name : str
        One of: "Plugin_Estimator", "k_Ensemble", "MLE_Exp", "PMM2/PMM3".

    Returns
    -------
    estimator : DensityEstimator

    Raises
    ------
    KeyError
        If ``name`` is not in the registry.
    """
    if name not in ESTIMATOR_REGISTRY:
        raise KeyError(
            f"Unknown estimator '{name}'. "
            f"Available: {list(ESTIMATOR_REGISTRY.keys())}"
        )
    return ESTIMATOR_REGISTRY[name]
