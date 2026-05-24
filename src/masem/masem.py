"""
MASEM — Manifold Sampling via Entropy Maximization (JAX reimplementation).

This module implements the full MASEM algorithm from:
  Braun C.V., Burghoff T., Toussaint M.
  Manifold Sampling via Entropy Maximization.
  arXiv:2605.12338v1, 2026.

Key components
--------------
1. Importance-resampling step:  wᵢ ∝ εᵢ,ₖ^τ  (equivalently ∝ q̂(xᵢ)^{−τ})
2. Plugin_Estimator:            q̂(xᵢ) = k / (N · Vₚ · εᵢ,ₖᵖ)
3. NHR and OLLA as modular local kernels (see masem.kernels)
4. Slack-penalty from Appendix D.5 of Braun et al. (2026)
5. Fixed seeds for reproducibility

Slack-penalty (Appendix D.5)
-----------------------------
After each rejuvenation step, particles that violate the inequality
constraints g(x) > 0 are penalised by adding a slack term to the weight:

    w_i ← w_i · exp(−β · max(0, g(xᵢ))²)

where β is the slack-penalty coefficient.  This keeps particles near the
feasible region without hard projection.

Usage
-----
    from masem.masem import MASEM, MASEMConfig
    from masem.estimators import PluginEstimator

    config = MASEMConfig(N=500, k=10, tau=0.5, T=20, p=1, seed=0)
    masem = MASEM(config, estimator=PluginEstimator())
    samples = masem.run(h, g=None, x_init=None)

References
----------
Braun et al. (2026) arXiv:2605.12338v1
  - Algorithm 1 (MASEM main loop): Section 3.1
  - NHR kernel: Section 3.2
  - OLLA kernel: Section 3.3
  - Slack-penalty: Appendix D.5
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from masem.estimators import DensityEstimator, PluginEstimator
from masem.spacings import knn_distances, unit_ball_volume


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ConstraintFn = Callable[[Array], Array]   # h: R^d → R^m  (equality)
InequalityFn = Callable[[Array], Array]   # g: R^d → R^q  (inequality, g≤0)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MASEMConfig:
    """
    Configuration for the MASEM algorithm.

    Parameters
    ----------
    N : int
        Number of particles.
    k : int
        Neighbourhood size for k-NN density estimation.
    tau : float
        Resampling temperature τ.  Controls KL compression rate (1 − τ/p).
        Must satisfy 0 < τ < p for convergence.
    T : int
        Number of MASEM iterations (rejuvenation steps).
    p : int
        Intrinsic dimension of the manifold Σ.
    d : int
        Ambient dimension of the space R^d.
    seed : int
        Random seed for reproducibility.
    kernel : str
        Local MCMC kernel: "NHR" or "OLLA".
    n_rejuv : int
        Number of rejuvenation (MCMC) steps per particle per iteration.
    slack_beta : float
        Slack-penalty coefficient β (Appendix D.5).  Set to 0.0 to disable.
    n_newton : int
        Number of Newton projection steps in the local kernel.
    alpha_nhr : float
        Interpolation weight for NHR kernel.
    step_scale_olla : float
        Step scale for OLLA kernel.
    """
    N: int = 500
    k: int = 10
    tau: float = 0.5
    T: int = 20
    p: int = 1
    d: int = 2
    seed: int = 0
    kernel: Literal["NHR", "OLLA"] = "NHR"
    n_rejuv: int = 5
    slack_beta: float = 10.0
    n_newton: int = 5
    alpha_nhr: float = 0.5
    step_scale_olla: float = 1.0


# ---------------------------------------------------------------------------
# Slack-penalty (Appendix D.5)
# ---------------------------------------------------------------------------

def slack_penalty_weights(
    particles: Array,
    weights: Array,
    g: InequalityFn,
    beta: float,
) -> Array:
    """
    Apply slack-penalty to resampling weights (Appendix D.5).

    For each particle xᵢ with inequality constraint g(xᵢ) ≤ 0:

        w_i ← w_i · exp(−β · Σⱼ max(0, gⱼ(xᵢ))²)

    Then renormalise.

    Parameters
    ----------
    particles : Array, shape (N, d)
        Current particle positions.
    weights : Array, shape (N,)
        Current resampling weights.
    g : callable or None
        Inequality constraint g: R^d → R^q.  If None, weights unchanged.
    beta : float
        Slack-penalty coefficient.

    Returns
    -------
    weights_penalised : Array, shape (N,)
        Penalised and renormalised weights.
    """
    if g is None or beta == 0.0:
        return weights

    # Evaluate g for all particles: shape (N, q)
    g_vals = jax.vmap(g)(particles)              # (N, q)
    # Slack: max(0, g(x))² summed over constraints
    slack = jnp.sum(jnp.maximum(0.0, g_vals) ** 2, axis=1)  # (N,)
    penalty = jnp.exp(-beta * slack)             # (N,)
    w_penalised = weights * penalty
    # Renormalise
    w_sum = jnp.sum(w_penalised)
    w_penalised = jnp.where(w_sum > 0, w_penalised / w_sum, weights)
    return w_penalised


# ---------------------------------------------------------------------------
# Systematic resampling
# ---------------------------------------------------------------------------

def systematic_resample(
    weights: Array,
    N: int,
    key: Array,
) -> Array:
    """
    Systematic resampling: draw N indices from a categorical distribution
    defined by weights, using the systematic (low-variance) scheme.

    Parameters
    ----------
    weights : Array, shape (N,)
        Normalised weights summing to 1.
    N : int
        Number of samples to draw.
    key : Array
        JAX PRNG key.

    Returns
    -------
    indices : Array, shape (N,), dtype int32
        Resampled particle indices.
    """
    # Systematic resampling: one uniform draw, then evenly spaced
    u = jax.random.uniform(key, shape=()) / N    # scalar in [0, 1/N)
    positions = u + jnp.arange(N) / N            # (N,) in [0, 1)
    cumsum = jnp.cumsum(weights)                 # (N,)
    # For each position, find the first index where cumsum >= position
    # Using searchsorted for efficiency
    indices = jnp.searchsorted(cumsum, positions, side='left')
    indices = jnp.clip(indices, 0, N - 1)
    return indices.astype(jnp.int32)


# ---------------------------------------------------------------------------
# NHR rejuvenation step (vectorised over particles)
# ---------------------------------------------------------------------------

def _nhr_rejuvenate_single(
    x: Array,
    neighbours: Array,
    h: ConstraintFn,
    key: Array,
    alpha: float,
    n_newton: int,
) -> Array:
    """Single-particle NHR rejuvenation step."""
    from masem.kernels import nhr_kernel
    return nhr_kernel(x, neighbours, h, key, alpha=alpha, n_newton=n_newton)


def _olla_rejuvenate_single(
    x: Array,
    neighbours: Array,
    h: ConstraintFn,
    key: Array,
    step_scale: float,
    n_newton: int,
) -> Array:
    """Single-particle OLLA rejuvenation step."""
    from masem.kernels import olla_kernel
    return olla_kernel(x, neighbours, h, key, step_scale=step_scale, n_newton=n_newton)


# ---------------------------------------------------------------------------
# Main MASEM class
# ---------------------------------------------------------------------------

class MASEM:
    """
    MASEM — Manifold Sampling via Entropy Maximization.

    Implements the full MASEM algorithm (Braun et al., 2026) in JAX with:
    - Modular density estimator (Plugin, k-Ensemble, MLE-Exp, PMM2/MLE)
    - Modular local kernel (NHR or OLLA)
    - Slack-penalty from Appendix D.5
    - Fixed seeds for reproducibility

    Parameters
    ----------
    config : MASEMConfig
        Algorithm configuration.
    estimator : DensityEstimator
        Density estimator for the resampling step.
        Defaults to PluginEstimator (baseline MASEM).
    """

    def __init__(
        self,
        config: MASEMConfig,
        estimator: DensityEstimator | None = None,
    ) -> None:
        self.config = config
        self.estimator = estimator if estimator is not None else PluginEstimator()
        self._key = jax.random.PRNGKey(config.seed)

    def _next_key(self) -> Array:
        """Split off a new subkey from the internal PRNG state."""
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def _get_neighbours(self, particles: Array, k: int) -> Array:
        """
        For each particle, return its k nearest neighbours.

        Returns
        -------
        neighbours : Array, shape (N, k, d)
        """
        N, d = particles.shape
        eps = knn_distances(particles, k)         # (N, k) — distances only
        # We need the actual neighbour positions, not just distances
        # Recompute pairwise distances to get indices
        sq_norms = jnp.sum(particles ** 2, axis=1)
        sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * particles @ particles.T
        sq_dists = jnp.maximum(sq_dists, 0.0)
        dists = jnp.sqrt(sq_dists)
        # Zero out self-distances
        dists = dists.at[jnp.arange(N), jnp.arange(N)].set(jnp.inf)
        # Get k nearest neighbour indices
        nn_indices = jnp.argsort(dists, axis=1)[:, :k]  # (N, k)
        # Gather neighbour positions
        neighbours = particles[nn_indices]        # (N, k, d)
        return neighbours

    def _rejuvenate(
        self,
        particles: Array,
        h: ConstraintFn,
        key: Array,
    ) -> Array:
        """
        Apply n_rejuv steps of the local kernel to each particle.

        Parameters
        ----------
        particles : Array, shape (N, d)
        h : ConstraintFn
        key : Array

        Returns
        -------
        particles_new : Array, shape (N, d)
        """
        cfg = self.config
        N = particles.shape[0]
        neighbours = self._get_neighbours(particles, cfg.k)  # (N, k, d)

        # Split keys: one per particle per rejuvenation step
        keys = jax.random.split(key, N * cfg.n_rejuv).reshape(cfg.n_rejuv, N, 2)

        particles_curr = particles
        for step in range(cfg.n_rejuv):
            step_keys = keys[step]               # (N, 2)
            if cfg.kernel == "NHR":
                # vmap over particles
                particles_curr = jax.vmap(
                    lambda x, nbrs, k_: _nhr_rejuvenate_single(
                        x, nbrs, h, k_, cfg.alpha_nhr, cfg.n_newton
                    )
                )(particles_curr, neighbours, step_keys)
            elif cfg.kernel == "OLLA":
                particles_curr = jax.vmap(
                    lambda x, nbrs, k_: _olla_rejuvenate_single(
                        x, nbrs, h, k_, cfg.step_scale_olla, cfg.n_newton
                    )
                )(particles_curr, neighbours, step_keys)
            else:
                raise ValueError(f"Unknown kernel: {cfg.kernel!r}")

        return particles_curr

    def run(
        self,
        h: ConstraintFn,
        g: InequalityFn = None,
        x_init: Array | None = None,
    ) -> Array:
        """
        Run the full MASEM algorithm.

        Parameters
        ----------
        h : ConstraintFn
            Equality constraint h: R^d → R^m.  The manifold is Σ = {x: h(x)=0}.
        g : InequalityFn, optional
            Inequality constraint g: R^d → R^q.  Feasible region: g(x) ≤ 0.
            Used for slack-penalty (Appendix D.5).
        x_init : Array, shape (N, d), optional
            Initial particle positions.  If None, particles are initialised
            by projecting random points onto the constraint surface.

        Returns
        -------
        particles : Array, shape (N, d)
            Final particle positions approximately uniformly distributed on Σ.
        """
        cfg = self.config
        N, k, tau, T, p, d = cfg.N, cfg.k, cfg.tau, cfg.T, cfg.p, cfg.d

        # --- Initialisation ---
        if x_init is not None:
            particles = jnp.array(x_init)
        else:
            particles = self._init_particles(h, g)

        # --- Main MASEM loop ---
        for t in range(T):
            # 1. Compute resampling weights using the density estimator
            #    wᵢ ∝ εᵢ,ₖ^τ  (Plugin: wᵢ ∝ q̂(xᵢ)^{−τ})
            weights = self.estimator(particles, k=k, tau=tau, p=p, N=N)

            # 2. Apply slack-penalty (Appendix D.5)
            if g is not None and cfg.slack_beta > 0.0:
                weights = slack_penalty_weights(particles, weights, g, cfg.slack_beta)

            # 3. Systematic resampling
            resample_key = self._next_key()
            indices = systematic_resample(weights, N, resample_key)
            particles = particles[indices]

            # 4. Rejuvenation (local MCMC kernel)
            rejuv_key = self._next_key()
            particles = self._rejuvenate(particles, h, rejuv_key)

        return particles

    def _init_particles(
        self,
        h: ConstraintFn,
        g: InequalityFn,
    ) -> Array:
        """
        Initialise particles by projecting random ambient points onto Σ.

        For each particle, sample a random point in R^d and project it
        onto the constraint surface h(x) = 0 via Newton steps.
        """
        cfg = self.config
        key = self._next_key()
        # Sample random points in a reasonable range
        x_rand = jax.random.normal(key, shape=(cfg.N, cfg.d))  # (N, d)

        # Project each particle onto the constraint surface
        from masem.kernels import _newton_project
        particles = jax.vmap(
            lambda x: _newton_project(x, h, n_steps=20)
        )(x_rand)
        return particles
