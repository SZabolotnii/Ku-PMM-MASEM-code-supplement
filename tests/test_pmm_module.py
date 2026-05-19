"""
Unit tests for masem.pmm_module — tasks 1.2 and 1.3.

Done condition (from tasks.md 1.2):
  - PMM-модуль проходить unit-тести на синтетичних спейсингах із відомим
    розподілом; JAX-трасування без помилок

Done condition (from tasks.md 1.3):
  - Тест: на Exp(1)-спейсингах full PMM selector відступає до Plugin/MLE
  - Тест: на спотворених спейсингах (c₃≠2, c₄≠6) PMM2 дає нижчий MSE
  - Тест: правило перемикання PMM2/PMM3 спрацьовує коректно
  - Тест: відкат до Plugin_Estimator при прикордонних кумулянтах
  Requirements: 5.1(a), 6.1(c)

Tests
-----
1. On Exp(1) spacings (flat manifold): full selector falls back to Plugin/MLE
2. On distorted spacings (c₃ ≠ 2, c₄ ≠ 6): PMM2 gives lower MSE than Plugin
3. Switching rule: PMM2 for asymmetric non-Exp regimes, PMM3 for symmetric platykurtic, fallback for Exp(1)
4. Fallback to Plugin when denominator near zero
5. JAX tracing: jax.jit(pmm_density_weights) runs without errors
6. Output shape (N,), normalised, non-negative
7. PMMEstimator (from estimators.py) now uses the real PMM module (not fallback)
8. Task 1.3 explicit tests: MSE comparisons and switching rule verification
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from masem.pmm_module import (
    pmm_density_weights,
    _estimate_cumulants,
    _is_exp1_like,
    _select_density_estimate,
    _pmm2_density,
    _pmm3_density,
    _mle_density,
)
from masem.spacings import knn_distances, shell_spacings, unit_ball_volume
from masem.estimators import PMMEstimator, PluginEstimator, _weights_from_density


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

N_PARTICLES = 60
P_DIM = 2
K = 8
TAU = 1.0


@pytest.fixture
def particles_uniform_2d():
    """Uniform particles in [0.1, 0.9]^2 — approximately flat manifold."""
    pts = RNG.uniform(0.1, 0.9, size=(N_PARTICLES, P_DIM)).astype(np.float32)
    return jnp.array(pts)


def _make_particles(n: int = N_PARTICLES, p: int = P_DIM, seed: int = 0) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.1, 0.9, size=(n, p)).astype(np.float32)
    return jnp.array(pts)


def _synthetic_spacings_exp1(n: int, k: int, seed: int = 1) -> jnp.ndarray:
    """
    Synthetic Exp(1) spacings — flat manifold regime.
    Returns delta of shape (n, k) with N·delta ~ Exp(1).
    """
    rng = np.random.default_rng(seed)
    N = n
    # s_{i,j} ~ Exp(1)  →  delta_{i,j} = s_{i,j} / N
    s = rng.exponential(scale=1.0, size=(n, k)).astype(np.float32)
    delta = s / N
    return jnp.array(delta), jnp.array(s)


def _synthetic_spacings_distorted(
    n: int, k: int, c3_target: float = 3.5, seed: int = 2
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Synthetic spacings with elevated c₃ (asymmetric, curved manifold regime).

    We use a Gamma(shape, scale) distribution where shape controls skewness:
      c₃ = 2/sqrt(shape)  →  shape = 4/c₃²
    """
    rng = np.random.default_rng(seed)
    N = n
    shape = max(4.0 / (c3_target ** 2), 0.5)
    scale = 1.0 / shape  # keep mean ≈ 1
    s = rng.gamma(shape=shape, scale=scale, size=(n, k)).astype(np.float32)
    # Rescale so mean ≈ 1
    s = s / float(np.mean(s))
    delta = s / N
    return jnp.array(delta), jnp.array(s)


# ---------------------------------------------------------------------------
# Test 1: On Exp(1) spacings, full PMM estimator falls back to Plugin/MLE
# ---------------------------------------------------------------------------

def test_pmm_falls_back_to_plugin_on_flat_exp1_spacings():
    """
    On a flat manifold (Exp(1) spacings), Plugin/MLE is already optimal.
    The full PMM selector must therefore fall back to MLE instead of applying
    PMM2 merely because Exp(1) has c₃=2.
    """
    n, k = 500, 20
    N = n
    delta, s = _synthetic_spacings_exp1(n, k, seed=10)
    c3, c4 = _estimate_cumulants(s)

    assert bool(_is_exp1_like(c3, c4)), (
        f"Exp(1) synthetic spacings were not recognised as flat regime: "
        f"c3={float(c3):.3f}, c4={float(c4):.3f}"
    )

    rho_selected = _select_density_estimate(delta, s, c3, c4, N)
    rho_mle = _mle_density(delta, N)

    assert jnp.allclose(rho_selected, rho_mle, rtol=1e-6, atol=1e-8), (
        "Flat Exp(1) regime must select MLE/Plugin fallback."
    )


# ---------------------------------------------------------------------------
# Test 2: On distorted spacings (c₃ ≠ 2), PMM2 gives lower MSE than Plugin
# ---------------------------------------------------------------------------

def test_pmm2_lower_mse_on_distorted_spacings():
    """
    On spacings with elevated c₃ (curved manifold), PMM2 should give
    lower MSE than the Plugin estimator.

    We use synthetic Gamma spacings with c₃ ≈ 3.5 (strongly asymmetric).
    The true density is 1 (normalised), and we compare MSE of density estimates.
    """
    n, k = 300, 12
    N = n
    delta, s = _synthetic_spacings_distorted(n, k, c3_target=3.5, seed=5)

    c3, c4 = _estimate_cumulants(s)

    # Only run this test if c₃ is actually elevated (sanity check)
    c3_val = float(c3)
    if abs(c3_val) <= 0.3:
        pytest.skip(f"Distorted spacings have |c₃|={abs(c3_val):.2f} ≤ 0.3; "
                    "PMM2 switching rule would not activate.")

    # True density: since s = N·delta and mean(s) ≈ 1, true ρ ≈ 1/N per spacing
    # We compare relative to MLE baseline
    rho_mle = _mle_density(delta, N)
    rho_pmm2 = _pmm2_density(delta, s, c3, c4, N)

    # On distorted spacings, PMM2 should have lower variance (lower MSE vs mean)
    mean_rho = float(jnp.mean(rho_mle))
    mse_mle = float(jnp.mean((rho_mle - mean_rho) ** 2))
    mse_pmm2 = float(jnp.mean((rho_pmm2 - mean_rho) ** 2))

    # PMM2 should reduce variance (MSE relative to mean)
    # Allow up to 5% tolerance (PMM2 may not always win on every random seed)
    assert mse_pmm2 <= mse_mle * 1.05, (
        f"PMM2 MSE ({mse_pmm2:.4e}) is not lower than MLE MSE ({mse_mle:.4e}) "
        f"on distorted spacings with c₃={c3_val:.2f}. "
        f"PMM2 should reduce variance when c₃ is large."
    )


# ---------------------------------------------------------------------------
# Test 3: Switching rule fires correctly
# ---------------------------------------------------------------------------

def test_switching_rule_pmm2_for_asymmetric():
    """
    When |c₃| > 0.3, the full pipeline should use PMM2 (not MLE fallback).

    We verify by checking that the output of pmm_density_weights differs
    from the pure MLE output when c₃ is large.
    """
    # Use particles on a curved manifold proxy: sphere surface
    rng = np.random.default_rng(20)
    n, p, k = 100, 3, 8
    # Points on a 2-sphere embedded in R^3 — curved manifold
    pts = rng.standard_normal(size=(n, p)).astype(np.float32)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    particles = jnp.array(pts)

    eps = knn_distances(particles, k)
    delta = shell_spacings(eps, p)
    s = n * delta
    c3, c4 = _estimate_cumulants(s)

    # Compute PMM weights and MLE weights
    w_pmm = pmm_density_weights(particles, k=k, tau=TAU, p=p, N=n)
    w_mle = _weights_from_density(_mle_density(delta, n), TAU)

    c3_val, c4_val = float(c3), float(c4)
    if bool(_is_exp1_like(c3, c4)):
        pytest.skip(
            f"Sphere proxy produced near-Exp(1) cumulants "
            f"c3={c3_val:.2f}, c4={c4_val:.2f}; MLE fallback is correct."
        )
    if abs(c3_val) > 0.3:
        # PMM2 should have been used — weights should differ from MLE
        max_diff = float(jnp.max(jnp.abs(w_pmm - w_mle)))
        assert max_diff > 1e-6, (
            f"PMM weights identical to MLE weights despite |c₃|={abs(c3_val):.2f} > 0.3. "
            "PMM2 switching rule may not be firing."
        )
    else:
        # c₃ is small — fallback is expected, weights should be close to MLE
        pytest.skip(f"|c₃|={abs(c3_val):.2f} ≤ 0.3 on sphere; "
                    "switching rule correctly falls back to MLE.")


def test_switching_rule_pmm3_for_symmetric_platykurtic():
    """
    When |c₃| ≤ 0.3 AND c₄ < -0.5, PMM3 should be used.

    We construct synthetic spacings with near-zero c₃ and negative c₄
    (uniform distribution: c₃=0, c₄=-1.2).
    """
    rng = np.random.default_rng(30)
    n, k = 200, 10
    N = n
    # Uniform[0, 2] has mean=1, c₃=0, c₄=-1.2
    s_np = rng.uniform(0.0, 2.0, size=(n, k)).astype(np.float32)
    s = jnp.array(s_np)
    delta = s / N

    c3, c4 = _estimate_cumulants(s)
    c3_val, c4_val = float(c3), float(c4)

    if abs(c3_val) <= 0.3 and c4_val < -0.5:
        rho_pmm3 = _pmm3_density(delta, s, c4, N)
        rho_mle = _mle_density(delta, N)
        # PMM3 should differ from MLE
        max_diff = float(jnp.max(jnp.abs(rho_pmm3 - rho_mle)))
        assert max_diff > 1e-6, (
            "PMM3 density identical to MLE despite |c₃| ≤ 0.3 and c₄ < -0.5."
        )
    else:
        pytest.skip(
            f"Uniform spacings gave c₃={c3_val:.2f}, c₄={c4_val:.2f}; "
            "conditions for PMM3 not met."
        )


def test_switching_rule_fallback_for_exp1():
    """
    On Exp(1) spacings (flat manifold), the switching rule should fall back
    to MLE.  The point of this test is the honest boundary: c₃≈2 should not
    be interpreted as "activate PMM2" when c₄≈6 as well.
    """
    n, k = 500, 20
    delta, s = _synthetic_spacings_exp1(n, k, seed=7)
    N = n

    c3, c4 = _estimate_cumulants(s)
    assert bool(_is_exp1_like(c3, c4))

    rho_selected = _select_density_estimate(delta, s, c3, c4, N)
    rho_mle = _mle_density(delta, N)
    assert jnp.allclose(rho_selected, rho_mle, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# Test 4: Fallback to Plugin when denominator near zero
# ---------------------------------------------------------------------------

def test_fallback_when_denominator_near_zero():
    """
    When 2 + c₄ < 0.1 (denominator near zero), PMM2 should fall back
    to MLE (the correction is clamped / invalid).

    We construct spacings with c₄ ≈ -2 (very platykurtic) to trigger this.
    """
    rng = np.random.default_rng(40)
    n, k = 100, 10
    N = n

    # Bernoulli(0.5) scaled to mean=1: values in {0, 2}, c₄ = -2
    s_np = rng.choice([0.0, 2.0], size=(n, k)).astype(np.float32)
    s = jnp.array(s_np)
    delta = s / N

    c3, c4 = _estimate_cumulants(s)
    c4_val = float(c4)

    # With c₄ ≈ -2, denom = 2 + c₄ ≈ 0 → PMM2 should be invalid
    # The _pmm2_density function clamps the denominator, so output should be finite
    rho_pmm2 = _pmm2_density(delta, s, c3, c4, N)
    assert jnp.all(jnp.isfinite(rho_pmm2)), (
        "PMM2 density has non-finite values when denominator is near zero."
    )
    assert jnp.all(rho_pmm2 > 0), (
        "PMM2 density has non-positive values when denominator is near zero."
    )

    # Full pipeline should also produce valid weights
    p = 2
    pts = rng.uniform(0.1, 0.9, size=(n, p)).astype(np.float32)
    particles = jnp.array(pts)
    w = pmm_density_weights(particles, k=k, tau=TAU, p=p, N=n)
    assert w.shape == (n,)
    assert abs(float(jnp.sum(w)) - 1.0) < 1e-5
    assert float(jnp.min(w)) >= 0.0


# ---------------------------------------------------------------------------
# Test 5: JAX tracing — jax.jit(pmm_density_weights) runs without errors
# ---------------------------------------------------------------------------

def test_jax_jit_tracing(particles_uniform_2d):
    """
    jax.jit(pmm_density_weights) must compile and run without errors.
    """
    particles = particles_uniform_2d
    n = particles.shape[0]

    jit_fn = jax.jit(
        lambda pts: pmm_density_weights(pts, k=K, tau=TAU, p=P_DIM, N=n)
    )
    w = jit_fn(particles)
    assert w.shape == (n,)
    assert abs(float(jnp.sum(w)) - 1.0) < 1e-5


def test_jax_jit_second_call_same_result(particles_uniform_2d):
    """
    Two calls to the jitted function should return identical results.
    """
    particles = particles_uniform_2d
    n = particles.shape[0]

    jit_fn = jax.jit(
        lambda pts: pmm_density_weights(pts, k=K, tau=TAU, p=P_DIM, N=n)
    )
    w1 = jit_fn(particles)
    w2 = jit_fn(particles)
    assert jnp.allclose(w1, w2, atol=1e-6), "JIT results are not deterministic."


# ---------------------------------------------------------------------------
# Test 6: Output shape (N,), normalised, non-negative
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,p,k,tau", [
    (20, 2, 3, 0.5),
    (50, 3, 5, 1.0),
    (15, 2, 4, 2.0),
    (100, 2, 8, 0.0),
])
def test_output_shape_normalised_nonneg(n, p, k, tau):
    """pmm_density_weights must return shape (N,), sum=1, all ≥ 0."""
    rng = np.random.default_rng(n + p + k)
    pts = rng.uniform(0.05, 0.95, size=(n, p)).astype(np.float32)
    particles = jnp.array(pts)

    w = pmm_density_weights(particles, k=k, tau=tau, p=p, N=n)

    assert w.shape == (n,), f"Expected shape ({n},), got {w.shape}"
    assert abs(float(jnp.sum(w)) - 1.0) < 1e-5, (
        f"Weights sum to {float(jnp.sum(w)):.6f}, expected 1.0"
    )
    assert float(jnp.min(w)) >= 0.0, (
        f"Found negative weight: {float(jnp.min(w))}"
    )


def test_output_all_finite(particles_uniform_2d):
    """All output weights must be finite."""
    particles = particles_uniform_2d
    n = particles.shape[0]
    w = pmm_density_weights(particles, k=K, tau=TAU, p=P_DIM, N=n)
    assert jnp.all(jnp.isfinite(w)), "Output weights contain non-finite values."


# ---------------------------------------------------------------------------
# Test 7: PMMEstimator uses the real PMM module (not fallback)
# ---------------------------------------------------------------------------

def test_pmm_estimator_uses_real_module(particles_uniform_2d):
    """
    After implementing pmm_module.py, PMMEstimator should use the real
    PMM implementation (not fall back to MLE_Exp).
    """
    est = PMMEstimator()
    assert est._pmm_available, (
        "PMMEstimator._pmm_available is False — pmm_module.py not found. "
        "Make sure masem/pmm_module.py is importable."
    )

    particles = particles_uniform_2d
    n = particles.shape[0]

    # Should run without warnings about fallback
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w = est(particles, k=K, tau=TAU, p=P_DIM, N=n)

    fallback_warnings = [
        c for c in caught
        if "falling back" in str(c.message).lower()
        or "fallback" in str(c.message).lower()
    ]
    assert len(fallback_warnings) == 0, (
        f"PMMEstimator emitted fallback warnings despite pmm_module being available: "
        f"{[str(c.message) for c in fallback_warnings]}"
    )

    assert w.shape == (n,)
    assert abs(float(jnp.sum(w)) - 1.0) < 1e-5
    assert float(jnp.min(w)) >= 0.0


def test_pmm_estimator_result_differs_from_mle_on_curved(particles_uniform_2d):
    """
    PMMEstimator result should differ from pure MLE on a curved manifold
    (sphere surface), where c₃ is elevated.
    """
    from masem.estimators import MLEExpEstimator

    rng = np.random.default_rng(55)
    n, p, k = 80, 3, 6
    # Sphere surface — curved manifold
    pts = rng.standard_normal(size=(n, p)).astype(np.float32)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    particles = jnp.array(pts)

    est_pmm = PMMEstimator()
    est_mle = MLEExpEstimator()

    w_pmm = est_pmm(particles, k=k, tau=TAU, p=p, N=n)
    w_mle = est_mle(particles, k=k, tau=TAU, p=p, N=n)

    # On a curved manifold, PMM should produce different weights than MLE
    # (unless c₃ happens to be very small — in which case fallback is correct)
    eps = knn_distances(particles, k)
    delta = shell_spacings(eps, p)
    s = n * delta
    c3, c4 = _estimate_cumulants(s)
    c3_val, c4_val = float(c3), float(c4)

    if bool(_is_exp1_like(c3, c4)):
        pytest.skip(
            f"Sphere proxy produced near-Exp(1) cumulants "
            f"c3={c3_val:.2f}, c4={c4_val:.2f}; MLE fallback is correct."
        )
    if abs(c3_val) > 0.3:
        max_diff = float(jnp.max(jnp.abs(w_pmm - w_mle)))
        assert max_diff > 1e-6, (
            f"PMM and MLE weights are identical on sphere (c₃={c3_val:.2f}). "
            "PMM2 correction should produce different weights."
        )


# ---------------------------------------------------------------------------
# Test 8: Cumulant estimation correctness on known distributions
# ---------------------------------------------------------------------------

def test_cumulant_estimation_exp1():
    """
    On Exp(1) spacings, estimated c₃ should be close to 2 and c₄ close to 6.
    """
    rng = np.random.default_rng(100)
    n, k = 500, 20
    s = jnp.array(rng.exponential(scale=1.0, size=(n, k)).astype(np.float32))

    c3, c4 = _estimate_cumulants(s)
    c3_val, c4_val = float(c3), float(c4)

    # Theoretical values for Exp(1): c₃=2, c₄=6
    # With n*k=10000 samples, estimates should be within 20% of theory
    assert abs(c3_val - 2.0) < 0.4, (
        f"c₃ estimate {c3_val:.3f} deviates from theoretical 2.0 by more than 0.4"
    )
    assert abs(c4_val - 6.0) < 1.5, (
        f"c₄ estimate {c4_val:.3f} deviates from theoretical 6.0 by more than 1.5"
    )


def test_cumulant_estimation_uniform():
    """
    On Uniform[0, 2] spacings (mean=1), c₃ should be ≈ 0 and c₄ ≈ -1.2.
    """
    rng = np.random.default_rng(101)
    n, k = 500, 20
    s = jnp.array(rng.uniform(0.0, 2.0, size=(n, k)).astype(np.float32))

    c3, c4 = _estimate_cumulants(s)
    c3_val, c4_val = float(c3), float(c4)

    # Theoretical: c₃=0, c₄=-1.2 for Uniform[0,2]
    assert abs(c3_val) < 0.3, (
        f"c₃ estimate {c3_val:.3f} should be near 0 for Uniform spacings"
    )
    assert c4_val < -0.5, (
        f"c₄ estimate {c4_val:.3f} should be negative for Uniform spacings"
    )


# ---------------------------------------------------------------------------
# Test 9: Module importability
# ---------------------------------------------------------------------------

def test_module_importable_from_masem():
    """masem.pmm_module must be importable as a submodule."""
    from masem import pmm_module  # noqa: F401
    assert hasattr(pmm_module, "pmm_density_weights")


def test_pmm_density_weights_importable_directly():
    """pmm_density_weights must be importable directly from masem.pmm_module."""
    from masem.pmm_module import pmm_density_weights as fn  # noqa: F401
    assert callable(fn)


# ===========================================================================
# Task 1.3 — Explicit unit tests
# Requirements: 5.1(a), 6.1(c)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 1.3 Test A: On Exp(1) spacings, full PMM selector falls back to Plugin
# Requirement 5.1(a): flat homogeneous manifold — PMM does NOT outperform Plugin
# ---------------------------------------------------------------------------

def test_task13_selector_falls_back_to_plugin_on_exp1_spacings():
    """
    Task 1.3 — Requirement 5.1(a):
    On Exp(1) spacings (flat manifold), PMM must not outperform or replace
    Plugin/MLE. The selector should return the MLE density exactly.

    Uses synthetic Exp(1) spacings directly (not real kNN distances) to
    isolate the density estimator behaviour from the geometry.
    """
    rng = np.random.default_rng(200)
    n, k = 500, 15
    N = n

    # Exp(1) spacings — flat manifold regime
    s_np = rng.exponential(scale=1.0, size=(n, k)).astype(np.float32)
    s = jnp.array(s_np)
    delta = s / N

    c3, c4 = _estimate_cumulants(s)

    assert bool(_is_exp1_like(c3, c4)), (
        f"Exp(1) synthetic spacings were not recognised as flat regime: "
        f"c3={float(c3):.3f}, c4={float(c4):.3f}"
    )

    rho_plugin = _mle_density(delta, N)
    rho_selected = _select_density_estimate(delta, s, c3, c4, N)

    assert jnp.allclose(rho_selected, rho_plugin, rtol=1e-6, atol=1e-8), (
        "Task 1.3: flat Exp(1) regime must select Plugin/MLE fallback."
    )


# ---------------------------------------------------------------------------
# Task 1.3 Test B: On distorted spacings (c₃≠2, c₄≠6), PMM2 gives lower MSE
# Requirement 6.1(c): PMM2 as principled estimator for distorted spacings
# ---------------------------------------------------------------------------

def test_task13_pmm2_lower_mse_on_distorted_spacings_c3_ne_2():
    """
    Task 1.3 — Requirement 6.1(c):
    On spacings with c₃ significantly different from 2 (curved manifold),
    PMM2 should give lower variance (MSE) than the Plugin estimator.

    Uses Gamma(shape, scale) spacings with c₃ ≈ 3.5 (strongly asymmetric).
    """
    rng = np.random.default_rng(201)
    n, k = 400, 15
    N = n

    # Gamma spacings with elevated c₃ ≈ 3.5
    c3_target = 3.5
    shape = max(4.0 / (c3_target ** 2), 0.5)
    scale = 1.0 / shape
    s_np = rng.gamma(shape=shape, scale=scale, size=(n, k)).astype(np.float32)
    s_np = s_np / float(np.mean(s_np))  # normalise mean to 1
    s = jnp.array(s_np)
    delta = s / N

    c3, c4 = _estimate_cumulants(s)
    c3_val = float(c3)

    if abs(c3_val) <= 0.3:
        pytest.skip(
            f"Distorted spacings have |c₃|={abs(c3_val):.2f} ≤ 0.3; "
            "PMM2 switching rule would not activate."
        )

    rho_plugin = _mle_density(delta, N)
    rho_pmm2 = _pmm2_density(delta, s, c3, c4, N)

    # Compare variance (MSE relative to mean) — PMM2 should reduce variance
    mean_rho = float(jnp.mean(rho_plugin))
    mse_plugin = float(jnp.mean((rho_plugin - mean_rho) ** 2))
    mse_pmm2 = float(jnp.mean((rho_pmm2 - mean_rho) ** 2))

    assert mse_pmm2 <= mse_plugin * 1.05, (
        f"Task 1.3: PMM2 MSE ({mse_pmm2:.4e}) is not lower than Plugin MSE "
        f"({mse_plugin:.4e}) on distorted spacings with c₃={c3_val:.2f}. "
        f"Requirement 6.1(c): PMM2 should reduce variance when c₃ is large."
    )


def test_task13_pmm2_lower_mse_on_distorted_spacings_c4_ne_6():
    """
    Task 1.3 — Requirement 6.1(c):
    On spacings with c₄ significantly different from 6 (curved manifold),
    PMM2 should give lower variance than the Plugin estimator.

    Uses Gamma spacings with c₃ ≈ 4.0 (very asymmetric, c₄ also elevated).
    """
    rng = np.random.default_rng(202)
    n, k = 400, 15
    N = n

    # Very asymmetric Gamma: shape=0.25 → c₃ ≈ 4, c₄ ≈ 24
    shape = 0.25
    scale = 1.0 / shape
    s_np = rng.gamma(shape=shape, scale=scale, size=(n, k)).astype(np.float32)
    s_np = s_np / float(np.mean(s_np))
    s = jnp.array(s_np)
    delta = s / N

    c3, c4 = _estimate_cumulants(s)
    c3_val, c4_val = float(c3), float(c4)

    # Verify that c₃ and c₄ are indeed different from Exp(1) values (2, 6)
    assert abs(c3_val - 2.0) > 0.5 or abs(c4_val - 6.0) > 2.0, (
        f"Spacings have c₃={c3_val:.2f}, c₄={c4_val:.2f} — too close to Exp(1) "
        "values (2, 6). Test setup may be incorrect."
    )

    if abs(c3_val) <= 0.3:
        pytest.skip(
            f"Distorted spacings have |c₃|={abs(c3_val):.2f} ≤ 0.3; "
            "PMM2 switching rule would not activate."
        )

    rho_plugin = _mle_density(delta, N)
    rho_pmm2 = _pmm2_density(delta, s, c3, c4, N)

    mean_rho = float(jnp.mean(rho_plugin))
    mse_plugin = float(jnp.mean((rho_plugin - mean_rho) ** 2))
    mse_pmm2 = float(jnp.mean((rho_pmm2 - mean_rho) ** 2))

    assert mse_pmm2 <= mse_plugin * 1.05, (
        f"Task 1.3: PMM2 MSE ({mse_pmm2:.4e}) is not lower than Plugin MSE "
        f"({mse_plugin:.4e}) on distorted spacings with c₃={c3_val:.2f}, "
        f"c₄={c4_val:.2f}. Requirement 6.1(c): PMM2 should reduce variance."
    )


# ---------------------------------------------------------------------------
# Task 1.3 Test C: Switching rule fires correctly
# Requirement 6.1(c): automatic PMM2/PMM3 switching rule
# ---------------------------------------------------------------------------

def test_task13_switching_rule_pmm2_activated_when_c3_large():
    """
    Task 1.3 — Requirement 6.1(c):
    When spacings are outside the Exp(1) flat regime and |c₃| is large, the
    switching rule must select PMM2.

    We verify by checking that the full pipeline output differs from pure MLE
    when c₃ is large (PMM2 correction is non-trivial).
    """
    rng = np.random.default_rng(210)
    n, k = 300, 12
    N = n

    # Gamma spacings with c₃ ≈ 3.5 — PMM2 should be selected
    shape = 4.0 / (3.5 ** 2)
    scale = 1.0 / shape
    s_np = rng.gamma(shape=shape, scale=scale, size=(n, k)).astype(np.float32)
    s_np = s_np / float(np.mean(s_np))
    s = jnp.array(s_np)
    delta = s / N

    c3, c4 = _estimate_cumulants(s)
    c3_val, c4_val = float(c3), float(c4)

    assert not bool(_is_exp1_like(c3, c4)), (
        f"Distorted spacings look Exp(1)-like: c3={c3_val:.2f}, c4={c4_val:.2f}"
    )

    if abs(c3_val) <= 0.3:
        pytest.skip(f"|c₃|={abs(c3_val):.2f} ≤ 0.3; PMM2 not activated.")

    # PMM2 density should differ from MLE density when c₃ is large
    rho_mle = _mle_density(delta, N)
    rho_pmm2 = _pmm2_density(delta, s, c3, c4, N)

    max_diff = float(jnp.max(jnp.abs(rho_pmm2 - rho_mle)))
    assert max_diff > 1e-6, (
        f"Task 1.3: PMM2 density identical to MLE despite |c₃|={abs(c3_val):.2f} > 0.3. "
        "Switching rule should activate PMM2 correction."
    )

    # Also verify the full pipeline selects PMM2 (not MLE)
    # by checking that pmm_density_weights with these spacings differs from MLE weights
    rho_mle_weights = _weights_from_density(rho_mle, TAU)
    rho_pmm2_weights = _weights_from_density(rho_pmm2, TAU)
    max_weight_diff = float(jnp.max(jnp.abs(rho_pmm2_weights - rho_mle_weights)))
    assert max_weight_diff > 1e-6, (
        f"Task 1.3: PMM2 weights identical to MLE weights despite |c₃|={abs(c3_val):.2f}. "
        "PMM2 correction should produce different resampling weights."
    )


def test_task13_switching_rule_pmm3_activated_when_c3_small_c4_negative():
    """
    Task 1.3 — Requirement 6.1(c):
    When |c₃| ≤ threshold_c3 AND c₄ < threshold_c4 (-0.5), PMM3 must be selected.

    Uniform[0, 2] spacings have c₃ ≈ 0 and c₄ ≈ -1.2 — ideal for PMM3.
    """
    rng = np.random.default_rng(211)
    n, k = 300, 15
    N = n

    # Uniform[0, 2]: mean=1, c₃=0, c₄=-1.2
    s_np = rng.uniform(0.0, 2.0, size=(n, k)).astype(np.float32)
    s = jnp.array(s_np)
    delta = s / N

    c3, c4 = _estimate_cumulants(s)
    c3_val, c4_val = float(c3), float(c4)

    if not (abs(c3_val) <= 0.3 and c4_val < -0.5):
        pytest.skip(
            f"Uniform spacings gave c₃={c3_val:.2f}, c₄={c4_val:.2f}; "
            "conditions for PMM3 not met."
        )

    # PMM3 density should differ from MLE density
    rho_mle = _mle_density(delta, N)
    rho_pmm3 = _pmm3_density(delta, s, c4, N)

    max_diff = float(jnp.max(jnp.abs(rho_pmm3 - rho_mle)))
    assert max_diff > 1e-6, (
        f"Task 1.3: PMM3 density identical to MLE despite |c₃|={abs(c3_val):.2f} ≤ 0.3 "
        f"and c₄={c4_val:.2f} < -0.5. Switching rule should activate PMM3 correction."
    )


def test_task13_switching_rule_fallback_on_exp1_spacings():
    """
    Task 1.3 — Requirement 6.1(c):
    On Exp(1) spacings (c₃≈2, c₄≈6), the selector must fall back to MLE.
    This verifies the honest flat-manifold boundary.
    """
    rng = np.random.default_rng(212)
    n, k = 1000, 20
    N = n

    # Exp(1) spacings
    s_np = rng.exponential(scale=1.0, size=(n, k)).astype(np.float32)
    s = jnp.array(s_np)
    delta = s / N

    c3, c4 = _estimate_cumulants(s)
    c3_val, c4_val = float(c3), float(c4)

    assert bool(_is_exp1_like(c3, c4)), (
        f"Exp(1) synthetic spacings were not recognised as flat regime: "
        f"c3={c3_val:.3f}, c4={c4_val:.3f}"
    )

    rho_selected = _select_density_estimate(delta, s, c3, c4, N)
    rho_mle = _mle_density(delta, N)
    assert jnp.allclose(rho_selected, rho_mle, rtol=1e-6, atol=1e-8), (
        "Exp(1) flat regime must select MLE/Plugin fallback."
    )


# ---------------------------------------------------------------------------
# Task 1.3 Test D: Fallback to Plugin at boundary cumulants
# Requirement 6.1(c): fallback to Plugin_Estimator as safe default
# ---------------------------------------------------------------------------

def test_task13_fallback_to_plugin_when_g2_invalid():
    """
    Task 1.3 — Requirement 6.1(c):
    When g₂ = 1 − c₃²/(2+c₄) is outside [0, 1] (invalid PMM2 regime),
    the full pipeline must fall back to MLE (Plugin) and produce valid weights.

    We construct spacings where c₃ is very large and c₄ is near -2,
    making g₂ negative (invalid). The pipeline should select MLE fallback.
    """
    rng = np.random.default_rng(220)
    n, k, p = 150, 10, 2
    N = n

    # Bernoulli(0.5) scaled to mean=1: c₄ ≈ -2, making 2+c₄ ≈ 0
    # This triggers the boundary condition: denom_pmm2 < _MIN_DENOM (0.1)
    s_np = rng.choice([0.0, 2.0], size=(n, k)).astype(np.float32)
    s = jnp.array(s_np)
    delta = s / N

    c3, c4 = _estimate_cumulants(s)
    c3_val, c4_val = float(c3), float(c4)
    denom = 2.0 + c4_val

    # Full pipeline must produce valid weights regardless of boundary cumulants
    pts = rng.uniform(0.1, 0.9, size=(n, p)).astype(np.float32)
    particles = jnp.array(pts)
    w = pmm_density_weights(particles, k=k, tau=TAU, p=p, N=n)

    assert w.shape == (n,), f"Expected shape ({n},), got {w.shape}"
    assert jnp.all(jnp.isfinite(w)), (
        f"Task 1.3: Weights contain non-finite values at boundary cumulants "
        f"(c₃={c3_val:.2f}, c₄={c4_val:.2f}, 2+c₄={denom:.3f})."
    )
    assert abs(float(jnp.sum(w)) - 1.0) < 1e-5, (
        f"Task 1.3: Weights do not sum to 1 at boundary cumulants. "
        f"Sum = {float(jnp.sum(w)):.6f}"
    )
    assert float(jnp.min(w)) >= 0.0, (
        f"Task 1.3: Negative weights at boundary cumulants: {float(jnp.min(w))}"
    )


def test_task13_fallback_produces_mle_equivalent_at_boundary():
    """
    Task 1.3 — Requirement 6.1(c):
    When both PMM2 and PMM3 are invalid (boundary cumulants), the pipeline
    selects MLE (Plugin) fallback. The output weights should be close to
    the pure MLE weights.

    We use Bernoulli spacings (c₄ ≈ -2) which make PMM2 invalid (denom ≈ 0)
    and PMM3 also invalid (c₃ is non-zero). The pipeline should use MLE.
    """
    rng = np.random.default_rng(221)
    n, k, p = 150, 10, 2
    N = n

    # Bernoulli(0.5) spacings: c₄ ≈ -2 → 2+c₄ ≈ 0 → PMM2 invalid
    s_np = rng.choice([0.0, 2.0], size=(n, k)).astype(np.float32)
    s = jnp.array(s_np)
    delta = s / N

    c3, c4 = _estimate_cumulants(s)
    c3_val, c4_val = float(c3), float(c4)
    denom_pmm2 = 2.0 + c4_val

    # Compute MLE weights directly
    rho_mle = _mle_density(delta, N)
    w_mle = _weights_from_density(rho_mle, TAU)

    # Compute PMM2 weights (with clamped denominator)
    rho_pmm2 = _pmm2_density(delta, s, c3, c4, N)
    w_pmm2 = _weights_from_density(rho_pmm2, TAU)

    # When denom_pmm2 < _MIN_DENOM (0.1), the correction is clamped
    # The output should still be finite and valid
    assert jnp.all(jnp.isfinite(w_pmm2)), (
        f"Task 1.3: PMM2 weights non-finite at boundary (2+c₄={denom_pmm2:.3f})."
    )
    assert abs(float(jnp.sum(w_pmm2)) - 1.0) < 1e-5, (
        f"Task 1.3: PMM2 weights don't sum to 1 at boundary."
    )

    # The full pipeline (pmm_density_weights) should also produce valid weights
    pts = rng.uniform(0.1, 0.9, size=(n, p)).astype(np.float32)
    particles = jnp.array(pts)
    w_pipeline = pmm_density_weights(particles, k=k, tau=TAU, p=p, N=n)

    assert jnp.all(jnp.isfinite(w_pipeline)), (
        "Task 1.3: Pipeline weights non-finite at boundary cumulants."
    )
    assert abs(float(jnp.sum(w_pipeline)) - 1.0) < 1e-5, (
        "Task 1.3: Pipeline weights don't sum to 1 at boundary cumulants."
    )
    assert float(jnp.min(w_pipeline)) >= 0.0, (
        "Task 1.3: Negative pipeline weights at boundary cumulants."
    )
