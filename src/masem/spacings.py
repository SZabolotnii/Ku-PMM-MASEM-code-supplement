"""
kNN shell-spacing utilities for PMM-MASEM.

For a particle xᵢ with sorted kNN distances ε_{i,1} ≤ … ≤ ε_{i,k}, the
j-th shell spacing is:

    Δ_{i,j} = V_p · (ε_{i,j}^p − ε_{i,j−1}^p),   j = 1, …, k

where V_p = π^{p/2} / Γ(p/2 + 1) is the volume of the unit p-ball and
ε_{i,0} = 0 by convention.

On a flat homogeneous manifold with density ρ(xᵢ):
    N · ρ(xᵢ) · Δ_{i,j}  ~  i.i.d. Exp(1)

This module is JAX-vectorised: no per-particle Python loops.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
import math


# ---------------------------------------------------------------------------
# Unit p-ball volume
# ---------------------------------------------------------------------------

def unit_ball_volume(p: int) -> float:
    """Volume of the unit p-ball: V_p = π^{p/2} / Γ(p/2 + 1)."""
    return math.pi ** (p / 2) / math.gamma(p / 2 + 1)


# ---------------------------------------------------------------------------
# kNN distance computation
# ---------------------------------------------------------------------------

def knn_distances(
    particles: Array,
    k: int,
) -> Array:
    """
    Compute sorted kNN distances for every particle.

    Parameters
    ----------
    particles : Array, shape (N, p)
        Particle positions in R^p.
    k : int
        Number of nearest neighbours (excluding self).

    Returns
    -------
    eps : Array, shape (N, k)
        Sorted distances ε_{i,1} ≤ … ≤ ε_{i,k} for each particle i.
    """
    N, p = particles.shape
    # Pairwise squared distances via ||a-b||² = ||a||² + ||b||² - 2 a·b
    sq_norms = jnp.sum(particles ** 2, axis=1)  # (N,)
    sq_dists = (
        sq_norms[:, None] + sq_norms[None, :] - 2.0 * particles @ particles.T
    )  # (N, N)
    # Clamp numerical negatives to zero before sqrt
    sq_dists = jnp.maximum(sq_dists, 0.0)
    dists = jnp.sqrt(sq_dists)  # (N, N)

    # Zero out self-distances so they don't appear in top-k
    dists = dists.at[jnp.arange(N), jnp.arange(N)].set(jnp.inf)

    # Take k smallest distances (sorted ascending)
    eps = jnp.sort(dists, axis=1)[:, :k]  # (N, k)
    return eps


# ---------------------------------------------------------------------------
# Shell spacings
# ---------------------------------------------------------------------------

def shell_spacings(
    eps: Array,
    p: int,
) -> Array:
    """
    Compute shell spacings from sorted kNN distances.

    Δ_{i,j} = V_p · (ε_{i,j}^p − ε_{i,j−1}^p),  j = 1, …, k

    Parameters
    ----------
    eps : Array, shape (N, k)
        Sorted kNN distances (ε_{i,1} ≤ … ≤ ε_{i,k}).
    p : int
        Intrinsic dimension of the manifold.

    Returns
    -------
    delta : Array, shape (N, k)
        Shell spacings Δ_{i,j}.
    """
    Vp = unit_ball_volume(p)
    eps_p = eps ** p  # (N, k)
    # Prepend zeros for ε_{i,0} = 0
    eps_p_prev = jnp.concatenate(
        [jnp.zeros((eps.shape[0], 1)), eps_p[:, :-1]], axis=1
    )  # (N, k)
    delta = Vp * (eps_p - eps_p_prev)  # (N, k)
    return delta


# ---------------------------------------------------------------------------
# Convenience: compute both distances and spacings in one call
# ---------------------------------------------------------------------------

def knn_spacings(
    particles: Array,
    k: int,
    p: int,
) -> tuple[Array, Array]:
    """
    Compute kNN distances and shell spacings for all particles.

    Parameters
    ----------
    particles : Array, shape (N, p_dim)
        Particle positions.
    k : int
        Number of nearest neighbours.
    p : int
        Intrinsic dimension (used for spacing formula).

    Returns
    -------
    eps : Array, shape (N, k)
        Sorted kNN distances.
    delta : Array, shape (N, k)
        Shell spacings.
    """
    eps = knn_distances(particles, k)
    delta = shell_spacings(eps, p)
    return eps, delta
