"""
Local MCMC kernels for MASEM: NHR and OLLA.

Both kernels are modular — they accept a particle, a constraint function h,
and return a new particle that (approximately) satisfies h(x) = 0.

NHR — Nearest-Neighbour Heat-bath Resampling
--------------------------------------------
Proposes a new particle by interpolating between the current particle and
a randomly chosen neighbour, then projects onto the constraint surface via
Newton steps.  This is the default kernel in Braun et al. (2026).

OLLA — Optimal Local Linear Approximation
------------------------------------------
Proposes a new particle by taking a step in the tangent space of the
manifold (estimated from the local neighbourhood), then projects back.
This is the second kernel described in Braun et al. (2026).

Both kernels are JAX-compatible and can be jit-compiled.

References
----------
Braun C.V., Burghoff T., Toussaint M.
  Manifold Sampling via Entropy Maximization.
  arXiv:2605.12338v1, 2026.  Sections 3.2 (NHR) and 3.3 (OLLA).
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Constraint function: h: R^d → R^m  (equality constraints h(x) = 0)
ConstraintFn = Callable[[Array], Array]

# Inequality constraint: g: R^d → R^m  (g(x) ≤ 0)
InequalityFn = Callable[[Array], Array] | None


# ---------------------------------------------------------------------------
# Shared utility: Newton projection onto h(x) = 0
# ---------------------------------------------------------------------------

def _newton_project(
    x: Array,
    h: ConstraintFn,
    n_steps: int = 5,
    step_size: float = 1.0,
) -> Array:
    """
    Project x onto the constraint surface h(x) = 0 via Newton steps.

    x_{t+1} = x_t − J^+ h(x_t)

    where J^+ = J^T (J J^T)^{-1} is the Moore-Penrose pseudoinverse of the
    Jacobian J = ∂h/∂x.

    Parameters
    ----------
    x : Array, shape (d,)
        Starting point.
    h : callable
        Constraint function h: R^d → R^m.
    n_steps : int
        Number of Newton iterations.
    step_size : float
        Step size (1.0 = full Newton step).

    Returns
    -------
    x_proj : Array, shape (d,)
        Projected point (approximately satisfying h(x) = 0).
    """
    def newton_step(x_curr, _):
        hval = h(x_curr)                          # (m,)
        J = jax.jacobian(h)(x_curr)               # (m, d)
        # Pseudoinverse: J^+ = J^T (J J^T)^{-1}
        JJT = J @ J.T                             # (m, m)
        # Solve (J J^T) λ = h(x) for λ, then x -= J^T λ
        lam = jnp.linalg.solve(
            JJT + 1e-8 * jnp.eye(JJT.shape[0]),
            hval,
        )                                         # (m,)
        x_new = x_curr - step_size * J.T @ lam   # (d,)
        return x_new, None

    x_proj, _ = jax.lax.scan(newton_step, x, None, length=n_steps)
    return x_proj


# ---------------------------------------------------------------------------
# NHR — Nearest-Neighbour Heat-bath Resampling
# ---------------------------------------------------------------------------

def nhr_kernel(
    x: Array,
    neighbours: Array,
    h: ConstraintFn,
    key: Array,
    alpha: float = 0.5,
    n_newton: int = 5,
) -> Array:
    """
    NHR: propose a new particle by interpolating toward a random neighbour,
    then project onto the constraint surface.

    Proposal: x_prop = x + α · (x_j − x) + noise
    Projection: Newton steps to satisfy h(x_prop) = 0.

    Parameters
    ----------
    x : Array, shape (d,)
        Current particle.
    neighbours : Array, shape (K, d)
        K nearest neighbours of x (excluding x itself).
    h : ConstraintFn
        Equality constraint h: R^d → R^m.
    key : Array
        JAX PRNG key.
    alpha : float
        Interpolation weight toward the chosen neighbour.
    n_newton : int
        Number of Newton projection steps.

    Returns
    -------
    x_new : Array, shape (d,)
        New particle on the constraint surface.
    """
    key1, key2 = jax.random.split(key)
    K = neighbours.shape[0]
    d = x.shape[0]

    # Choose a random neighbour
    j = jax.random.randint(key1, shape=(), minval=0, maxval=K)
    x_j = neighbours[j]                          # (d,)

    # Interpolate toward neighbour + small Gaussian noise
    noise_scale = jnp.linalg.norm(x_j - x) * 0.1
    noise = jax.random.normal(key2, shape=(d,)) * noise_scale
    x_prop = x + alpha * (x_j - x) + noise       # (d,)

    # Project onto constraint surface
    x_new = _newton_project(x_prop, h, n_steps=n_newton)
    return x_new


# ---------------------------------------------------------------------------
# OLLA — Optimal Local Linear Approximation
# ---------------------------------------------------------------------------

def olla_kernel(
    x: Array,
    neighbours: Array,
    h: ConstraintFn,
    key: Array,
    step_scale: float = 1.0,
    n_newton: int = 5,
) -> Array:
    """
    OLLA: propose a new particle by stepping in the tangent space of the
    manifold (estimated from the local neighbourhood), then project back.

    The tangent space at x is estimated as the null space of the Jacobian
    J = ∂h/∂x.  A random step in this tangent space is taken, then Newton
    projection brings the particle back to the constraint surface.

    Parameters
    ----------
    x : Array, shape (d,)
        Current particle.
    neighbours : Array, shape (K, d)
        K nearest neighbours of x.
    h : ConstraintFn
        Equality constraint h: R^d → R^m.
    key : Array
        JAX PRNG key.
    step_scale : float
        Scale of the tangent-space step (relative to local neighbourhood size).
    n_newton : int
        Number of Newton projection steps.

    Returns
    -------
    x_new : Array, shape (d,)
        New particle on the constraint surface.
    """
    d = x.shape[0]

    # Estimate local length scale from neighbourhood
    diffs = neighbours - x[None, :]              # (K, d)
    local_scale = jnp.mean(jnp.linalg.norm(diffs, axis=1))  # scalar

    # Jacobian of h at x
    J = jax.jacobian(h)(x)                       # (m, d)

    # Tangent space: null space of J via SVD
    # J = U S V^T  →  null(J) = columns of V corresponding to zero singular values
    # We use the last (d - m) right singular vectors
    _, s, Vt = jnp.linalg.svd(J, full_matrices=True)  # Vt: (d, d)
    m = J.shape[0]
    # Tangent basis: rows of Vt[m:] (null space of J)
    tangent_basis = Vt[m:]                        # (d-m, d)

    # Random step in tangent space
    tangent_dim = d - m
    z = jax.random.normal(key, shape=(tangent_dim,))  # (d-m,)
    step = step_scale * local_scale * tangent_basis.T @ z  # (d,)
    x_prop = x + step                             # (d,)

    # Project back onto constraint surface
    x_new = _newton_project(x_prop, h, n_steps=n_newton)
    return x_new


# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------

KERNEL_REGISTRY = {
    "NHR": nhr_kernel,
    "OLLA": olla_kernel,
}
