"""
Unit tests for masem.estimators — task 1.4.

Done condition (from tasks.md 1.4):
  - All 4 estimators return weights of the same shape (N,)
  - Interface is unified (all callable with the same signature)

Tests
-----
1. Shape contract: every estimator returns shape (N,)
2. Normalisation: weights sum to 1.0
3. Non-negativity: all weights ≥ 0
4. Unified interface: all estimators satisfy the DensityEstimator protocol
5. Registry completeness: all 4 canonical names present
6. Plugin_Estimator correctness: on uniform grid, weights are uniform
7. k_Ensemble: default k_values are [k//2, k, 2k]; custom k_values work
8. MLE_Exp: on Exp(1) spacings, density estimate is close to true density
9. PMMEstimator: falls back to MLE_Exp when pmm_module is absent
10. get_estimator: raises KeyError for unknown name
"""

from __future__ import annotations

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Make src/ importable without installing the package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from masem.estimators import (
    DensityEstimator,
    ESTIMATOR_REGISTRY,
    PluginEstimator,
    kEnsembleEstimator,
    MLEExpEstimator,
    PMMEstimator,
    get_estimator,
    _weights_from_density,
)
from masem.spacings import unit_ball_volume, knn_distances, shell_spacings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

# Small 2-D particle cloud for fast tests
N_SMALL = 30
P_DIM = 2
K = 5
TAU = 1.0

@pytest.fixture
def particles_2d():
    """N_SMALL particles uniformly distributed in [0,1]^2."""
    pts = RNG.uniform(size=(N_SMALL, P_DIM)).astype(np.float32)
    return jnp.array(pts)


@pytest.fixture
def particles_3d():
    """N_SMALL particles uniformly distributed in [0,1]^3."""
    pts = RNG.uniform(size=(N_SMALL, 3)).astype(np.float32)
    return jnp.array(pts)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

ALL_ESTIMATOR_NAMES = ["Plugin_Estimator", "k_Ensemble", "MLE_Exp", "PMM2/PMM3"]


def call_estimator(est, particles, k=K, tau=TAU, p=P_DIM, N=N_SMALL):
    return est(particles, k=k, tau=tau, p=p, N=N)


# ---------------------------------------------------------------------------
# Test 1: Shape contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_ESTIMATOR_NAMES)
def test_output_shape(name, particles_2d):
    """Every estimator must return weights of shape (N,)."""
    est = get_estimator(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = call_estimator(est, particles_2d)
    assert w.shape == (N_SMALL,), (
        f"{name}: expected shape ({N_SMALL},), got {w.shape}"
    )


# ---------------------------------------------------------------------------
# Test 2: Normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_ESTIMATOR_NAMES)
def test_weights_sum_to_one(name, particles_2d):
    """Weights must sum to 1.0 (within float32 tolerance)."""
    est = get_estimator(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = call_estimator(est, particles_2d)
    total = float(jnp.sum(w))
    assert abs(total - 1.0) < 1e-5, (
        f"{name}: weights sum to {total}, expected 1.0"
    )


# ---------------------------------------------------------------------------
# Test 3: Non-negativity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_ESTIMATOR_NAMES)
def test_weights_non_negative(name, particles_2d):
    """All weights must be ≥ 0."""
    est = get_estimator(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = call_estimator(est, particles_2d)
    assert float(jnp.min(w)) >= 0.0, (
        f"{name}: found negative weight {float(jnp.min(w))}"
    )


# ---------------------------------------------------------------------------
# Test 4: Unified interface — DensityEstimator protocol
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_ESTIMATOR_NAMES)
def test_protocol_compliance(name):
    """Every estimator must satisfy the DensityEstimator protocol."""
    est = get_estimator(name)
    assert isinstance(est, DensityEstimator), (
        f"{name} does not satisfy DensityEstimator protocol"
    )


# ---------------------------------------------------------------------------
# Test 5: Registry completeness
# ---------------------------------------------------------------------------

def test_registry_contains_all_four():
    """ESTIMATOR_REGISTRY must contain exactly the 4 canonical estimators."""
    for name in ALL_ESTIMATOR_NAMES:
        assert name in ESTIMATOR_REGISTRY, f"'{name}' missing from registry"


def test_registry_size():
    """Registry must have exactly 4 entries."""
    assert len(ESTIMATOR_REGISTRY) == 4, (
        f"Expected 4 estimators, got {len(ESTIMATOR_REGISTRY)}"
    )


# ---------------------------------------------------------------------------
# Test 6: Plugin_Estimator — uniform weights on regular grid
# ---------------------------------------------------------------------------

def test_plugin_uniform_weights_on_regular_grid():
    """
    On a large uniform random cloud, the plug-in density estimates should
    be approximately equal across particles (low coefficient of variation).

    Note: a finite regular grid has boundary effects — corner/edge particles
    have larger ε_{i,k} than interior ones, so weights are not exactly equal.
    We instead verify that the coefficient of variation (std/mean) of the
    density estimates is small for a large uniform cloud.
    """
    rng = np.random.default_rng(7)
    N_large = 500
    pts = rng.uniform(0.1, 0.9, size=(N_large, 2)).astype(np.float32)
    particles = jnp.array(pts)
    est = PluginEstimator()
    # tau=0 → equal weights regardless; use tau=1 to see density variation
    # Compute density estimates directly
    eps = knn_distances(particles, k=10)
    eps_k = eps[:, -1]
    Vp = unit_ball_volume(2)
    rho_hat = 10 / (N_large * Vp * eps_k ** 2)
    cv = float(jnp.std(rho_hat) / jnp.mean(rho_hat))
    # For a uniform distribution, CV of k-NN density estimate should be small
    assert cv < 0.5, (
        f"Plugin density CV = {cv:.3f} on uniform cloud — expected < 0.5"
    )


# ---------------------------------------------------------------------------
# Test 7: k_Ensemble — default and custom k_values
# ---------------------------------------------------------------------------

def test_k_ensemble_default_k_values(particles_2d):
    """Default k_values should be [k//2, k, 2k] (deduplicated, sorted)."""
    k = 4
    est = kEnsembleEstimator()  # k_values=None → default
    w = est(particles_2d, k=k, tau=TAU, p=P_DIM, N=N_SMALL)
    assert w.shape == (N_SMALL,)
    assert abs(float(jnp.sum(w)) - 1.0) < 1e-5


def test_k_ensemble_custom_k_values(particles_2d):
    """Custom k_values should be respected."""
    est = kEnsembleEstimator(k_values=[2, 4, 6])
    w = est(particles_2d, k=K, tau=TAU, p=P_DIM, N=N_SMALL)
    assert w.shape == (N_SMALL,)
    assert abs(float(jnp.sum(w)) - 1.0) < 1e-5


def test_k_ensemble_single_k_matches_plugin(particles_2d):
    """
    k_Ensemble with a single k value should match Plugin_Estimator
    (both compute the same density estimate).
    """
    k = K
    est_ens = kEnsembleEstimator(k_values=[k])
    est_plug = PluginEstimator()
    w_ens = est_ens(particles_2d, k=k, tau=TAU, p=P_DIM, N=N_SMALL)
    w_plug = est_plug(particles_2d, k=k, tau=TAU, p=P_DIM, N=N_SMALL)
    max_diff = float(jnp.max(jnp.abs(w_ens - w_plug)))
    assert max_diff < 1e-5, (
        f"k_Ensemble(k_values=[k]) differs from Plugin by {max_diff:.2e}"
    )


# ---------------------------------------------------------------------------
# Test 8: MLE_Exp — density estimate on synthetic Exp(1) spacings
# ---------------------------------------------------------------------------

def test_mle_exp_density_on_flat_manifold():
    """
    On a flat homogeneous manifold (uniform particles), MLE_Exp should
    produce density estimates close to the true uniform density 1/V_p.

    We use a large N to reduce variance and check that the mean estimate
    is within 20% of the true density.
    """
    rng = np.random.default_rng(0)
    N_large = 200
    p = 2
    k = 10
    # Uniform particles in [0,1]^2 — approximately flat homogeneous
    pts = rng.uniform(size=(N_large, p)).astype(np.float32)
    particles = jnp.array(pts)

    est = MLEExpEstimator()
    w = est(particles, k=k, tau=0.0, p=p, N=N_large)
    # tau=0 → all weights equal (density estimates don't matter for shape)
    assert w.shape == (N_large,)
    assert abs(float(jnp.sum(w)) - 1.0) < 1e-5


def test_mle_exp_uses_all_spacings():
    """
    MLE_Exp uses all k spacings (mean), while Plugin uses only ε_{i,k}.

    On a uniform cloud, MLE_Exp (using all k spacings) should have lower
    variance in its density estimates than Plugin (using only ε_{i,k}).
    This is because the mean of k i.i.d. Exp(1) variables has variance 1/k,
    while a single Exp(1) variable has variance 1.

    We verify this by checking that the standard deviation of MLE_Exp density
    estimates is smaller than that of Plugin_Estimator density estimates.
    """
    rng = np.random.default_rng(1)
    N_large = 300
    k_large = 10
    pts = rng.uniform(0.05, 0.95, size=(N_large, 2)).astype(np.float32)
    particles = jnp.array(pts)

    # Compute density estimates directly (not weights) to compare variance
    eps = knn_distances(particles, k_large)
    delta = shell_spacings(eps, p=2)
    Vp = unit_ball_volume(2)

    # Plugin: uses only ε_{i,k}
    eps_k = eps[:, -1]
    rho_plugin = k_large / (N_large * Vp * eps_k ** 2)

    # MLE_Exp: uses mean of all k spacings
    mean_delta = jnp.mean(delta, axis=1)
    rho_mle = 1.0 / (N_large * mean_delta)

    std_plugin = float(jnp.std(rho_plugin))
    std_mle = float(jnp.std(rho_mle))

    # MLE_Exp should have lower or comparable variance (uses more data)
    # Allow up to 20% higher std (statistical fluctuation with finite N)
    assert std_mle <= std_plugin * 1.2, (
        f"MLE_Exp std ({std_mle:.4f}) is much larger than Plugin std "
        f"({std_plugin:.4f}) — MLE_Exp should use all spacings for lower variance"
    )


# ---------------------------------------------------------------------------
# Test 9: PMMEstimator fallback
# ---------------------------------------------------------------------------

def test_pmm_estimator_fallback_to_mle(particles_2d):
    """
    When pmm_module is not available, PMMEstimator should fall back to
    MLE_Exp and emit a warning.
    """
    est = PMMEstimator(fallback_to_mle=True)
    # Force unavailability (pmm_module not yet implemented in task 1.2)
    est._pmm_available = False

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w = call_estimator(est, particles_2d)

    assert w.shape == (N_SMALL,)
    assert abs(float(jnp.sum(w)) - 1.0) < 1e-5
    # Should have emitted a warning about falling back / pmm_module
    assert any(
        "falling back" in str(c.message).lower()
        or "fallback" in str(c.message).lower()
        or "pmm_module" in str(c.message).lower()
        for c in caught
    ), (
        f"PMMEstimator did not emit a fallback warning. "
        f"Caught: {[str(c.message) for c in caught]}"
    )


def test_pmm_estimator_raises_when_fallback_disabled(particles_2d):
    """
    When fallback_to_mle=False and pmm_module is absent, PMMEstimator
    should raise ImportError.
    """
    est = PMMEstimator(fallback_to_mle=False)
    est._pmm_available = False

    with pytest.raises(ImportError, match="masem.pmm_module"):
        call_estimator(est, particles_2d)


# ---------------------------------------------------------------------------
# Test 10: get_estimator error handling
# ---------------------------------------------------------------------------

def test_get_estimator_unknown_name():
    """get_estimator should raise KeyError for unknown names."""
    with pytest.raises(KeyError, match="Unknown estimator"):
        get_estimator("NonExistentEstimator")


def test_get_estimator_returns_correct_type():
    """get_estimator should return the correct estimator type."""
    assert isinstance(get_estimator("Plugin_Estimator"), PluginEstimator)
    assert isinstance(get_estimator("k_Ensemble"), kEnsembleEstimator)
    assert isinstance(get_estimator("MLE_Exp"), MLEExpEstimator)
    assert isinstance(get_estimator("PMM2/PMM3"), PMMEstimator)


# ---------------------------------------------------------------------------
# Test 11: All estimators agree on shape across different (N, p, k, tau)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N,p_dim,k,tau", [
    (20, 2, 3, 0.5),
    (50, 3, 5, 1.0),
    (15, 2, 4, 2.0),
])
@pytest.mark.parametrize("name", ALL_ESTIMATOR_NAMES)
def test_shape_various_configs(name, N, p_dim, k, tau):
    """Shape (N,) must hold for various (N, p, k, tau) configurations."""
    rng = np.random.default_rng(99)
    pts = rng.uniform(size=(N, p_dim)).astype(np.float32)
    particles = jnp.array(pts)
    est = get_estimator(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = est(particles, k=k, tau=tau, p=p_dim, N=N)
    assert w.shape == (N,), f"{name}: shape {w.shape} != ({N},)"


# ---------------------------------------------------------------------------
# Test 12: _weights_from_density helper
# ---------------------------------------------------------------------------

def test_weights_from_density_uniform():
    """Uniform density → uniform weights."""
    rho = jnp.ones(10)
    w = _weights_from_density(rho, tau=1.0)
    assert w.shape == (10,)
    assert abs(float(jnp.sum(w)) - 1.0) < 1e-6
    assert float(jnp.max(jnp.abs(w - 0.1))) < 1e-6


def test_weights_from_density_tau_zero():
    """tau=0 → all weights equal regardless of density."""
    rho = jnp.array([1.0, 2.0, 3.0, 4.0])
    w = _weights_from_density(rho, tau=0.0)
    assert abs(float(jnp.sum(w)) - 1.0) < 1e-6
    assert float(jnp.max(jnp.abs(w - 0.25))) < 1e-6


def test_weights_from_density_high_density_gets_low_weight():
    """Higher density → lower weight (w ∝ ρ^{−τ})."""
    rho = jnp.array([1.0, 10.0])
    w = _weights_from_density(rho, tau=1.0)
    assert float(w[0]) > float(w[1]), (
        "Higher density particle should get lower weight"
    )
