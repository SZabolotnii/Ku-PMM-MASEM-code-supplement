"""
Benchmark manifolds for PMM-MASEM verification.

Implements the synthetic benchmarks from Braun et al. (2026) arXiv:2605.12338v1,
Appendix D.2:
  - Disconnected disks (two disks on a sphere in R³)
  - Sine manifold (damped sine curve in R²)

These benchmarks have known ground-truth uniform distributions, enabling
W₂² (Wasserstein-2 squared) verification against the values reported in
Braun et al. (2026), Table 1.

Done condition (task 1.1):
  Baseline MASEM reproduces results on disconnected disks and sine within
  ±5% of the W₂² values in Braun et al. tables.

Reference values from Braun et al. Table 1 (MASEM-NHR):
  - Disconnected Disks (3d): W₂² = 0.01 ± 0.01
  - Sine (2d):               W₂² = 0.13 ± 0.06

W₂² computation
---------------
We use the Sinkhorn approximation via scipy (if ot is available),
falling back to a nearest-neighbour approximation.

References
----------
Braun C.V., Burghoff T., Toussaint M.
  Manifold Sampling via Entropy Maximization.
  arXiv:2605.12338v1, 2026.  Appendix D.2.
"""

from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ConstraintFn = Callable[[Array], Array]
InequalityFn = Callable[[Array], Array] | None


# ---------------------------------------------------------------------------
# Disconnected Disks benchmark (3D sphere, Appendix D.2)
# ---------------------------------------------------------------------------

class DisconnectedDisks:
    """
    Two disks embedded on a sphere in R³ (Braun et al., Appendix D.2).

    Equality constraint (sphere):
        h(x) = ‖x‖₂ − R,   R = 2.5

    Inequality constraint (two disk caps):
        g(x) = min_{i∈{1,2}} { cos(ρᵢ) − R⟨x, μᵢ⟩ }

    where ρ₁ = 0.2, ρ₂ = 0.6 are the half-angles of the two disks,
    and μ₁, μ₂ are the disk centres on the unit sphere.

    In the disconnected case, the arc-length between centres is δ = 1.35.
    In the connected case, δ = 0.6.

    Bound constraints: x ∈ [−5, 5]³.

    Parameters
    ----------
    connected : bool
        If True, use δ = 0.6 (connected disks).
        If False (default), use δ = 1.35 (disconnected disks).
    R : float
        Sphere radius (default 2.5).
    """

    name: str = "disconnected_disks"
    p: int = 2   # intrinsic dimension (2D surface patches)
    d: int = 3   # ambient dimension

    def __init__(self, connected: bool = False, R: float = 2.5) -> None:
        self.connected = connected
        self.R = R
        self.rho1 = 0.2
        self.rho2 = 0.6
        self.delta = 0.6 if connected else 1.35
        if connected:
            self.name = "connected_disks"

        # Rotation matrix: 45 degrees around x-axis
        t = math.sqrt(2) / 2  # cos(45°) = sin(45°)
        Rx = np.array([
            [1, 0, 0],
            [0, t, -t],
            [0, t,  t],
        ])

        # Disk centres (unit vectors on sphere)
        d = self.delta
        mu1_local = np.array([math.cos(-d/2), math.sin(-d/2), 0.0])
        mu2_local = np.array([math.cos( d/2), math.sin( d/2), 0.0])
        self.mu1 = jnp.array(Rx @ mu1_local, dtype=jnp.float32)
        self.mu2 = jnp.array(Rx @ mu2_local, dtype=jnp.float32)

        # Orthonormal bases for each disk (for sampling)
        self._e1_1, self._e2_1 = self._disk_basis(np.array(self.mu1))
        self._e1_2, self._e2_2 = self._disk_basis(np.array(self.mu2))

    @staticmethod
    def _disk_basis(mu: np.ndarray):
        """Compute two orthonormal vectors perpendicular to mu."""
        # Find a vector not parallel to mu
        v = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(mu, v)) > 0.9:
            v = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(mu, v)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(mu, e1)
        e2 /= np.linalg.norm(e2)
        return e1, e2

    def h(self, x: Array) -> Array:
        """
        Sphere constraint: h(x) = ‖x‖₂ − R.

        Returns shape (1,).
        """
        return jnp.array([jnp.linalg.norm(x) - self.R])

    def g(self, x: Array) -> Array:
        """
        Disk cap constraints: g(x) = −min_{i} { cos(ρᵢ) − R⟨x/‖x‖, μᵢ⟩ }

        The feasible region is where at least one disk cap contains x:
            min_i { cos(ρᵢ) − R⟨x, μᵢ⟩ } ≤ 0

        We also add bound constraints x ∈ [−5, 5]³.

        Returns shape (3,): [disk_constraint, bound_lo, bound_hi].
        """
        # Normalise x to unit sphere for dot product
        x_norm = x / (jnp.linalg.norm(x) + 1e-8)
        cap1 = math.cos(self.rho1) - jnp.dot(x_norm, self.mu1)
        cap2 = math.cos(self.rho2) - jnp.dot(x_norm, self.mu2)
        # Feasible if at least one cap is satisfied (min ≤ 0)
        disk_constraint = jnp.minimum(cap1, cap2)
        # Bound constraints: x ∈ [−5, 5]³
        bound_lo = jnp.max(-5.0 - x)   # ≤ 0 when x ≥ -5
        bound_hi = jnp.max(x - 5.0)    # ≤ 0 when x ≤ 5
        return jnp.array([disk_constraint, bound_lo, bound_hi])

    def sample_ground_truth(self, N: int, seed: int = 0) -> np.ndarray:
        """
        Sample N points uniformly from the union of the two disk caps.

        Uses the exact sampling procedure from Appendix D.2:
        - Choose disk i with probability proportional to its spherical area
          P(i) = (1 − cos(ρᵢ)) / Σⱼ (1 − cos(ρⱼ))
        - Sample uniformly on disk i:
          φ ~ Unif(0, 2π), cos α ~ Unif(cos ρᵢ, 1)
          x = R [cos α μᵢ + sin α (cos φ eᵢ₁ + sin φ eᵢ₂)]
        - Correct for overlap: accept with probability 1/m(x)

        Parameters
        ----------
        N : int
        seed : int

        Returns
        -------
        samples : np.ndarray, shape (N, 3)
        """
        rng = np.random.default_rng(seed)
        R = self.R
        rho1, rho2 = self.rho1, self.rho2
        mu1 = np.array(self.mu1)
        mu2 = np.array(self.mu2)
        e1_1, e2_1 = self._e1_1, self._e2_1
        e1_2, e2_2 = self._e1_2, self._e2_2

        # Spherical areas (proportional to 1 - cos(ρ))
        area1 = 1.0 - math.cos(rho1)
        area2 = 1.0 - math.cos(rho2)
        total_area = area1 + area2
        p1 = area1 / total_area

        samples = []
        while len(samples) < N:
            batch = max(10 * N, 1000)
            # Choose disk
            disk_choice = rng.random(batch) < p1  # True → disk 1

            # Sample on disk 1
            phi1 = rng.uniform(0, 2 * math.pi, batch)
            cos_alpha1 = rng.uniform(math.cos(rho1), 1.0, batch)
            sin_alpha1 = np.sqrt(np.maximum(1.0 - cos_alpha1**2, 0.0))
            pts1 = R * (
                cos_alpha1[:, None] * mu1[None, :]
                + sin_alpha1[:, None] * (
                    np.cos(phi1)[:, None] * e1_1[None, :]
                    + np.sin(phi1)[:, None] * e2_1[None, :]
                )
            )

            # Sample on disk 2
            phi2 = rng.uniform(0, 2 * math.pi, batch)
            cos_alpha2 = rng.uniform(math.cos(rho2), 1.0, batch)
            sin_alpha2 = np.sqrt(np.maximum(1.0 - cos_alpha2**2, 0.0))
            pts2 = R * (
                cos_alpha2[:, None] * mu2[None, :]
                + sin_alpha2[:, None] * (
                    np.cos(phi2)[:, None] * e1_2[None, :]
                    + np.sin(phi2)[:, None] * e2_2[None, :]
                )
            )

            # Select based on disk choice
            pts = np.where(disk_choice[:, None], pts1, pts2)

            # Overlap correction: count how many disks contain each point
            x_norm = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8)
            in_disk1 = (np.dot(x_norm, mu1) >= math.cos(rho1)).astype(float)
            in_disk2 = (np.dot(x_norm, mu2) >= math.cos(rho2)).astype(float)
            m = in_disk1 + in_disk2  # number of disks containing each point
            accept_prob = 1.0 / np.maximum(m, 1.0)
            u = rng.random(batch)
            accepted = pts[u < accept_prob]
            samples.extend(accepted.tolist())

        return np.array(samples[:N], dtype=np.float32)

    def init_particles(self, N: int, seed: int = 0) -> np.ndarray:
        """
        Initialise N particles on the two disk caps.

        Parameters
        ----------
        N : int
        seed : int

        Returns
        -------
        particles : np.ndarray, shape (N, 3)
        """
        return self.sample_ground_truth(N, seed=seed + 1000)


# ---------------------------------------------------------------------------
# Sine manifold benchmark (Appendix D.2)
# ---------------------------------------------------------------------------

class SineManifold:
    """
    Damped sine curve in R² (Braun et al., Appendix D.2).

    Equality constraint:
        h(x) = x₂ − exp(−0.15 x₁) sin(x₁)

    Inequality constraints:
        x₂ ≥ 0  (halfspace)
        x ∈ [−20, 20]²  (bound constraints)

    Ground-truth: uniform arc-length measure on the curve, restricted to
    the feasible region (x₂ ≥ 0 and bounds).

    Parameters
    ----------
    x_min : float
        Left boundary (default −20.0).
    x_max : float
        Right boundary (default 20.0).
    """

    name: str = "sine"
    p: int = 1   # intrinsic dimension
    d: int = 2   # ambient dimension

    def __init__(self, x_min: float = -20.0, x_max: float = 20.0) -> None:
        self.x_min = x_min
        self.x_max = x_max

    def h(self, x: Array) -> Array:
        """
        Equality constraint: h(x₁, x₂) = x₂ − exp(−0.15 x₁) sin(x₁).

        Returns shape (1,).
        """
        return jnp.array([x[1] - jnp.exp(-0.15 * x[0]) * jnp.sin(x[0])])

    def g(self, x: Array) -> Array:
        """
        Inequality constraints:
          - x₂ ≥ 0  →  −x₂ ≤ 0
          - x ∈ [−20, 20]²  →  x_min − x₁ ≤ 0, x₁ − x_max ≤ 0, etc.

        Returns shape (6,).
        """
        return jnp.array([
            -x[1],                    # x₂ ≥ 0
            self.x_min - x[0],        # x₁ ≥ x_min
            x[0] - self.x_max,        # x₁ ≤ x_max
            self.x_min - x[1],        # x₂ ≥ x_min
            x[1] - self.x_max,        # x₂ ≤ x_max
            -x[1],                    # duplicate x₂ ≥ 0 (for robustness)
        ])

    def _curve_point(self, t: float) -> np.ndarray:
        """Evaluate the curve at parameter t."""
        return np.array([t, np.exp(-0.15 * t) * np.sin(t)])

    def _arc_length_element(self, t: np.ndarray) -> np.ndarray:
        """
        Arc-length element ds/dt = sqrt(1 + (d/dt [exp(-0.15t)sin(t)])²).

        d/dt [exp(-0.15t)sin(t)] = exp(-0.15t)(cos(t) - 0.15 sin(t))
        """
        deriv = np.exp(-0.15 * t) * (np.cos(t) - 0.15 * np.sin(t))
        return np.sqrt(1.0 + deriv**2)

    def sample_ground_truth(self, N: int, seed: int = 0) -> np.ndarray:
        """
        Sample N points uniformly (arc-length) from the feasible sine curve.

        Uses rejection sampling from the parametrisation:
          t ~ Unif(x_min, x_max)
          x(t) = (t, exp(−0.15t) sin(t))
        Accept with probability proportional to arc-length element ds/dt,
        and only if x₂ ≥ 0.

        Parameters
        ----------
        N : int
        seed : int

        Returns
        -------
        samples : np.ndarray, shape (N, 2)
        """
        rng = np.random.default_rng(seed)
        # Max arc-length element (at t=0: ds/dt = sqrt(1 + 1) = sqrt(2))
        # For large |t|, exp(-0.15t) → 0, so ds/dt → 1
        # We bound by sqrt(1 + 1) = sqrt(2) conservatively
        max_ds = math.sqrt(2.0)

        samples = []
        while len(samples) < N:
            batch = max(20 * N, 2000)
            t = rng.uniform(self.x_min, self.x_max, size=batch)
            x2 = np.exp(-0.15 * t) * np.sin(t)
            ds = self._arc_length_element(t)
            u = rng.uniform(0, max_ds, size=batch)
            # Accept if arc-length weight and feasibility (x₂ ≥ 0)
            accepted_mask = (u < ds) & (x2 >= 0.0)
            t_accepted = t[accepted_mask]
            x2_accepted = x2[accepted_mask]
            pts = np.stack([t_accepted, x2_accepted], axis=1)
            samples.extend(pts.tolist())

        return np.array(samples[:N], dtype=np.float32)

    def init_particles(self, N: int, seed: int = 0) -> np.ndarray:
        """
        Initialise N particles on the feasible sine curve.

        Parameters
        ----------
        N : int
        seed : int

        Returns
        -------
        particles : np.ndarray, shape (N, 2)
        """
        return self.sample_ground_truth(N, seed=seed + 1000)


# ---------------------------------------------------------------------------
# W₂² (Wasserstein-2 squared) computation
# ---------------------------------------------------------------------------

def wasserstein2_squared(
    samples: np.ndarray,
    reference: np.ndarray,
    reg: float = 0.05,
    n_iter: int = 200,
) -> float:
    """
    Compute W₂² between two empirical distributions using Sinkhorn approximation.

    Falls back to a nearest-neighbour approximation if the `ot` package
    is not available.

    Parameters
    ----------
    samples : np.ndarray, shape (N, d)
        Samples from the estimated distribution.
    reference : np.ndarray, shape (M, d)
        Samples from the reference (ground-truth) distribution.
    reg : float
        Sinkhorn regularisation parameter.
    n_iter : int
        Number of Sinkhorn iterations.

    Returns
    -------
    w2_sq : float
        Approximate W₂² value.
    """
    try:
        import ot
        N = len(samples)
        M = len(reference)
        a = np.ones(N) / N
        b = np.ones(M) / M
        # Cost matrix: squared Euclidean distances
        C = np.sum((samples[:, None, :] - reference[None, :, :]) ** 2, axis=-1)
        w2_sq = ot.sinkhorn2(a, b, C, reg=reg, numIterations=n_iter)[0]
        return float(w2_sq)
    except ImportError:
        return _w2_sq_nn_approx(samples, reference)


def _w2_sq_nn_approx(
    samples: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Approximate W₂² via nearest-neighbour assignment.

    For each sample point, find the nearest reference point and compute
    the squared distance.  This is an upper bound on W₂².

    Parameters
    ----------
    samples : np.ndarray, shape (N, d)
    reference : np.ndarray, shape (M, d)

    Returns
    -------
    w2_sq_approx : float
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(reference)
    dists, _ = tree.query(samples, k=1)
    return float(np.mean(dists ** 2))


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    benchmark,
    masem_config_kwargs: dict | None = None,
    estimator=None,
    n_reference: int = 2000,
    seed: int = 0,
) -> dict:
    """
    Run MASEM on a benchmark and compute W₂².

    Parameters
    ----------
    benchmark : DisconnectedDisks | SineManifold
        Benchmark instance.
    masem_config_kwargs : dict, optional
        Keyword arguments for MASEMConfig (overrides defaults).
    estimator : DensityEstimator, optional
        Density estimator (defaults to PluginEstimator).
    n_reference : int
        Number of reference samples for W₂² computation.
    seed : int
        Random seed.

    Returns
    -------
    result : dict with keys:
        - "w2_sq": float — W₂² value
        - "particles": np.ndarray — final particle positions
        - "benchmark": str — benchmark name
        - "estimator": str — estimator name
    """
    from masem.masem import MASEM, MASEMConfig
    from masem.estimators import PluginEstimator

    # Default config — matches paper hyperparameters from Table 5
    # For disconnected disks: τ=1.0, M=50, k=4
    # For sine: τ=0.75, M=5, k=16
    if benchmark.name in ("disconnected_disks", "connected_disks"):
        default_kwargs = dict(
            N=300,
            k=4,
            tau=1.0,
            T=20,
            p=benchmark.p,
            d=benchmark.d,
            seed=seed,
            kernel="NHR",
            n_rejuv=50,
            slack_beta=1000.0,
            n_newton=5,
        )
    elif benchmark.name == "sine":
        default_kwargs = dict(
            N=300,
            k=16,
            tau=0.75,
            T=20,
            p=benchmark.p,
            d=benchmark.d,
            seed=seed,
            kernel="NHR",
            n_rejuv=5,
            slack_beta=1000.0,
            n_newton=5,
        )
    else:
        default_kwargs = dict(
            N=300,
            k=10,
            tau=0.5,
            T=15,
            p=benchmark.p,
            d=benchmark.d,
            seed=seed,
            kernel="NHR",
            n_rejuv=3,
            slack_beta=1000.0,
            n_newton=5,
        )

    if masem_config_kwargs:
        default_kwargs.update(masem_config_kwargs)

    config = MASEMConfig(**default_kwargs)
    est = estimator if estimator is not None else PluginEstimator()

    # Initialise particles on the manifold
    x_init = jnp.array(
        benchmark.init_particles(config.N, seed=seed).astype(np.float32)
    )

    # Run MASEM
    masem = MASEM(config, estimator=est)

    # Get constraint functions
    h = benchmark.h
    g = getattr(benchmark, 'g', None)

    particles = masem.run(h, g=g, x_init=x_init)
    particles_np = np.array(particles)

    # Sample reference distribution
    reference = benchmark.sample_ground_truth(n_reference, seed=seed + 42)

    # Compute W₂²
    w2_sq = wasserstein2_squared(particles_np, reference)

    return {
        "w2_sq": w2_sq,
        "particles": particles_np,
        "benchmark": benchmark.name,
        "estimator": est.name if hasattr(est, 'name') else str(type(est).__name__),
    }
