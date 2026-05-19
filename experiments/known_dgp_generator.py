"""Known-DGP spacing generators for the PMM-MASEM density microbenchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpacingRegime:
    """Specification for a synthetic shell-spacing law."""

    name: str
    label: str
    family: str
    curvature_proxy: float
    note: str


REGIMES: tuple[SpacingRegime, ...] = (
    SpacingRegime(
        name="flat_exp",
        label="Flat Exp(1)",
        family="exponential",
        curvature_proxy=0.0,
        note="flat homogeneous control; Plugin/MLE is the target selector",
    ),
    SpacingRegime(
        name="mild_curved_gamma",
        label="Mild curved gamma",
        family="gamma_mild",
        curvature_proxy=0.35,
        note="moderate asymmetric misspecification",
    ),
    SpacingRegime(
        name="strong_curved_gamma",
        label="Strong curved gamma",
        family="gamma_strong",
        curvature_proxy=0.70,
        note="strong asymmetric misspecification",
    ),
    SpacingRegime(
        name="boundary_mixture",
        label="Boundary mixture",
        family="boundary_mixture",
        curvature_proxy=0.90,
        note="truncated/boundary-heavy spacing proxy",
    ),
    SpacingRegime(
        name="platykurtic_uniform",
        label="Platykurtic uniform",
        family="uniform",
        curvature_proxy=0.50,
        note="symmetric platykurtic PMM3 stress case",
    ),
)


def generate_spacings(
    regime: SpacingRegime,
    *,
    n_particles: int,
    k_max: int,
    seed: int,
) -> np.ndarray:
    """Generate normalized spacings ``s = N * rho * Delta`` with mean one."""
    rng = np.random.default_rng(seed)
    size = (n_particles, k_max)

    if regime.family == "exponential":
        s = rng.exponential(scale=1.0, size=size)
    elif regime.family == "gamma_mild":
        shape = 0.70
        s = rng.gamma(shape=shape, scale=1.0 / shape, size=size)
    elif regime.family == "gamma_strong":
        shape = 0.42
        s = rng.gamma(shape=shape, scale=1.0 / shape, size=size)
    elif regime.family == "boundary_mixture":
        base = rng.gamma(shape=0.55, scale=1.0 / 0.55, size=size)
        mask = rng.random(size=size) < 0.18
        inflated = rng.lognormal(mean=1.0, sigma=0.55, size=size)
        s = np.where(mask, inflated, base)
    elif regime.family == "uniform":
        s = rng.uniform(0.0, 2.0, size=size)
    else:
        raise ValueError(f"Unknown spacing family: {regime.family!r}")

    s = np.asarray(s, dtype=np.float64)
    return s / np.mean(s)

