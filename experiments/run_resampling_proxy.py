"""Run lightweight resampling-proxy experiments for PMM-MASEM.

This script deliberately does not claim to be the full MASEM NHR/OLLA
benchmark.  It isolates the density-weight rule: generate a biased particle
cloud on a known synthetic target, compute one or more resampling steps with
each density estimator, and compare the resampled cloud to an independent
reference sample.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree, distance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from masem.estimators import ESTIMATOR_REGISTRY  # noqa: E402
from masem.pmm_module import _estimate_cumulants, _is_exp1_like  # noqa: E402
from masem.spacings import knn_distances, shell_spacings  # noqa: E402


@dataclass(frozen=True)
class ProxySpec:
    """Configuration for one synthetic resampling-proxy task."""

    name: str
    label: str
    p: int
    d: int
    k: int
    tau: float
    n_components: int


BENCHMARKS: tuple[ProxySpec, ...] = (
    ProxySpec("disconnected_disks", "Disconnected disks", p=2, d=2, k=12, tau=0.90, n_components=2),
    ProxySpec("seven_lobes", "Seven lobes", p=2, d=2, k=12, tau=0.95, n_components=7),
    ProxySpec("sine", "Sine curve", p=1, d=2, k=12, tau=0.75, n_components=6),
    ProxySpec("swiss_roll", "Swiss roll", p=2, d=3, k=12, tau=0.90, n_components=5),
    ProxySpec("scaling_stress", "Scaling stress-test", p=2, d=2, k=12, tau=0.80, n_components=4),
    ProxySpec("robotics_proxy", "Robotics corridor proxy", p=1, d=2, k=12, tau=0.85, n_components=4),
)

SEEDS = [11, 17, 23, 31, 43]
ESTIMATORS = ("Plugin_Estimator", "k_Ensemble", "MLE_Exp", "PMM2/MLE")


def _sample_disk(center: np.ndarray, radius: float, count: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, size=count)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, size=count))
    return center[None, :] + np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _counts(n: int, probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return rng.multinomial(n, probs)


def _disconnected_disks(n: int, rng: np.random.Generator, mode: str) -> tuple[np.ndarray, np.ndarray]:
    centers = np.array([[-1.0, 0.0], [1.0, 0.0]])
    radii = np.array([0.35, 0.55])
    probs = radii**2 if mode == "reference" else np.array([0.82, 0.18])
    counts = _counts(n, probs, rng)
    pts, labels = [], []
    for i, count in enumerate(counts):
        pts.append(_sample_disk(centers[i], radii[i], count, rng))
        labels.extend([i] * count)
    return np.vstack(pts), np.asarray(labels, dtype=int)


def _seven_lobes(n: int, rng: np.random.Generator, mode: str) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False)
    centers = np.column_stack([2.0 * np.cos(angles), 2.0 * np.sin(angles)])
    probs = np.ones(7) if mode == "reference" else np.array([0.34, 0.22, 0.16, 0.11, 0.08, 0.05, 0.04])
    counts = _counts(n, probs, rng)
    pts, labels = [], []
    for i, count in enumerate(counts):
        cloud = _sample_disk(centers[i], 0.35, count, rng)
        stretch = np.array([[1.0 + 0.05 * (i % 2), 0.0], [0.0, 0.62 + 0.03 * i]])
        pts.append((cloud - centers[i]) @ stretch.T + centers[i])
        labels.extend([i] * count)
    return np.vstack(pts), np.asarray(labels, dtype=int)


def _sine(n: int, rng: np.random.Generator, mode: str) -> tuple[np.ndarray, np.ndarray]:
    accepted_t: list[float] = []
    while len(accepted_t) < n:
        if mode == "reference":
            t = rng.uniform(-20.0, 20.0, size=10 * n)
        else:
            u = rng.beta(2.0, 5.0, size=10 * n)
            t = -20.0 + 40.0 * u
        y = np.exp(-0.15 * t) * np.sin(t)
        keep = t[y >= 0.0]
        accepted_t.extend(keep.tolist())
    t = np.asarray(accepted_t[:n])
    y = np.exp(-0.15 * t) * np.sin(t)
    labels = np.clip(np.floor((t + 20.0) / (40.0 / 6.0)).astype(int), 0, 5)
    return np.column_stack([t, y]), labels


def _swiss_roll(n: int, rng: np.random.Generator, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "reference":
        u = rng.uniform(0.0, 1.0, size=n)
    else:
        u = rng.beta(2.0, 5.0, size=n)
    t = 1.5 * np.pi + 3.0 * np.pi * u
    h = rng.uniform(-1.0, 1.0, size=n)
    x = np.column_stack([t * np.cos(t) / 6.0, h, t * np.sin(t) / 6.0])
    labels = np.clip(np.floor(u * 5).astype(int), 0, 4)
    return x, labels


def _scaling_stress(n: int, rng: np.random.Generator, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "reference":
        pts = rng.uniform(0.0, 1.0, size=(n, 2))
    else:
        pts = np.column_stack([rng.beta(2.0, 5.0, size=n), rng.beta(2.0, 5.0, size=n)])
    labels = (pts[:, 0] > 0.5).astype(int) + 2 * (pts[:, 1] > 0.5).astype(int)
    return pts, labels


def _robotics_proxy(n: int, rng: np.random.Generator, mode: str) -> tuple[np.ndarray, np.ndarray]:
    probs = np.ones(4) if mode == "reference" else np.array([0.50, 0.25, 0.18, 0.07])
    counts = _counts(n, probs, rng)
    pts, labels = [], []
    for i, count in enumerate(counts):
        u = rng.uniform(-1.0, 1.0, size=count)
        noise = rng.normal(0.0, 0.035, size=count)
        if i == 0:
            seg = np.column_stack([u, -1.0 + noise])
        elif i == 1:
            seg = np.column_stack([1.0 + noise, u])
        elif i == 2:
            seg = np.column_stack([u, 1.0 + noise])
        else:
            seg = np.column_stack([-1.0 + noise, u])
        pts.append(seg)
        labels.extend([i] * count)
    return np.vstack(pts), np.asarray(labels, dtype=int)


def sample_proxy(spec: ProxySpec, n: int, seed: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Sample points and component labels for one proxy benchmark."""
    rng = np.random.default_rng(seed)
    if spec.name == "disconnected_disks":
        pts, labels = _disconnected_disks(n, rng, mode)
    elif spec.name == "seven_lobes":
        pts, labels = _seven_lobes(n, rng, mode)
    elif spec.name == "sine":
        pts, labels = _sine(n, rng, mode)
    elif spec.name == "swiss_roll":
        pts, labels = _swiss_roll(n, rng, mode)
    elif spec.name == "scaling_stress":
        pts, labels = _scaling_stress(n, rng, mode)
    elif spec.name == "robotics_proxy":
        pts, labels = _robotics_proxy(n, rng, mode)
    else:
        raise ValueError(f"Unknown proxy benchmark: {spec.name!r}")

    order = rng.permutation(len(pts))
    return pts[order].astype(np.float32), labels[order]


def _branch_for_points(points: np.ndarray, spec: ProxySpec) -> tuple[str, float, float]:
    eps = knn_distances(jnp.asarray(points), spec.k)
    delta = shell_spacings(eps, spec.p)
    s = points.shape[0] * delta
    c3_j, c4_j = _estimate_cumulants(s)
    c3, c4 = float(c3_j), float(c4_j)
    if bool(_is_exp1_like(c3_j, c4_j)):
        return "MLE_fallback", c3, c4
    denom_pmm2 = 2.0 + c4
    g2 = 1.0 - c3**2 / max(denom_pmm2, 1e-10)
    if abs(c3) > 0.3 and denom_pmm2 >= 0.1 and 0.0 <= g2 <= 1.0:
        return "PMM2", c3, c4
    if abs(c3) <= 0.3 and c4 < -0.5:
        return "PMM3_disabled", c3, c4
    return "MLE_fallback", c3, c4


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    start = rng.uniform(0.0, 1.0 / n)
    positions = start + np.arange(n) / n
    return np.searchsorted(np.cumsum(weights), positions, side="left").clip(0, n - 1)


def _w2_proxy(samples: np.ndarray, reference: np.ndarray) -> float:
    tree_ref = cKDTree(reference)
    d1, _ = tree_ref.query(samples, k=1)
    tree_s = cKDTree(samples)
    d2, _ = tree_s.query(reference, k=1)
    return float(0.5 * (np.mean(d1**2) + np.mean(d2**2)))


def _pairwise_kl(samples: np.ndarray, reference: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n_s = min(len(samples), 180)
    n_r = min(len(reference), 180)
    s_idx = rng.choice(len(samples), size=n_s, replace=False)
    r_idx = rng.choice(len(reference), size=n_r, replace=False)
    ds = distance.pdist(samples[s_idx])
    dr = distance.pdist(reference[r_idx])
    upper = max(float(np.percentile(np.concatenate([ds, dr]), 99.5)), 1e-6)
    hist_s, edges = np.histogram(ds, bins=32, range=(0.0, upper), density=False)
    hist_r, _ = np.histogram(dr, bins=edges, density=False)
    ps = (hist_s + 1e-6) / np.sum(hist_s + 1e-6)
    pr = (hist_r + 1e-6) / np.sum(hist_r + 1e-6)
    return float(np.sum(ps * np.log(ps / pr)))


def _evaluate(
    spec: ProxySpec,
    estimator_name: str,
    *,
    seed: int,
    tau: float,
    n_particles: int,
    n_reference: int,
) -> dict[str, object]:
    particles, labels = sample_proxy(spec, n_particles, seed + 1000, mode="initial")
    reference, ref_labels = sample_proxy(spec, n_reference, seed + 2000, mode="reference")
    estimator = ESTIMATOR_REGISTRY[estimator_name]
    start = time.perf_counter()
    weights = np.asarray(
        estimator(jnp.asarray(particles), k=spec.k, tau=tau, p=spec.p, N=n_particles)
    )
    wallclock = time.perf_counter() - start
    rng = np.random.default_rng(seed + 3000)
    idx = _systematic_resample(weights, rng)
    final = particles[idx]
    final_labels = labels[idx]
    branch, c3, c4 = _branch_for_points(particles, spec)
    missing = sorted(set(ref_labels.tolist()) - set(final_labels.tolist()))
    return {
        "benchmark": spec.name,
        "benchmark_label": spec.label,
        "seed": seed,
        "estimator": estimator_name,
        "n_particles": n_particles,
        "k": spec.k,
        "tau": tau,
        "w2_sq_proxy": _w2_proxy(final, reference),
        "kl_pairwise": _pairwise_kl(final, reference, seed + 4000),
        "max_slack": 0.0,
        "component_loss": int(len(missing) > 0),
        "missing_fraction": len(missing) / max(spec.n_components, 1),
        "wallclock_sec": wallclock,
        "selector_branch": branch if estimator_name == "PMM2/MLE" else "not_applicable",
        "c3": c3,
        "c4": c4,
        "protocol": "single_step_resampling_proxy",
    }


def _run_main() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in BENCHMARKS:
        for seed in SEEDS:
            for estimator_name in ESTIMATORS:
                rows.append(
                    _evaluate(
                        spec,
                        estimator_name,
                        seed=seed,
                        tau=spec.tau,
                        n_particles=220,
                        n_reference=700,
                    )
                )
    return rows


def _run_tau() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    tau_values = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
    selected = [s for s in BENCHMARKS if s.name in {"seven_lobes", "swiss_roll", "robotics_proxy"}]
    for spec in selected:
        for tau in tau_values:
            for seed in SEEDS:
                for estimator_name in ESTIMATORS:
                    row = _evaluate(
                        spec,
                        estimator_name,
                        seed=seed,
                        tau=tau,
                        n_particles=220,
                        n_reference=700,
                    )
                    row["protocol"] = "tau_sweep_resampling_proxy"
                    rows.append(row)
    return rows


def _run_component_sweep() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    selected = [s for s in BENCHMARKS if s.name in {"disconnected_disks", "seven_lobes", "robotics_proxy"}]
    for n_particles in [80, 140, 220, 340]:
        for spec in selected:
            for seed in SEEDS:
                for estimator_name in ESTIMATORS:
                    row = _evaluate(
                        spec,
                        estimator_name,
                        seed=seed,
                        tau=spec.tau,
                        n_particles=n_particles,
                        n_reference=700,
                    )
                    row["protocol"] = "component_sweep_resampling_proxy"
                    rows.append(row)
    return rows


def _iteration_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    selected = [s for s in BENCHMARKS if s.name in {"seven_lobes", "swiss_roll"}]
    n_particles = 220
    n_reference = 700
    for spec in selected:
        for seed in SEEDS:
            reference, _ = sample_proxy(spec, n_reference, seed + 2000, mode="reference")
            for estimator_name in ESTIMATORS:
                particles, labels = sample_proxy(spec, n_particles, seed + 1000, mode="initial")
                rng = np.random.default_rng(seed + 5000)
                for iteration in range(9):
                    rows.append(
                        {
                            "benchmark": spec.name,
                            "benchmark_label": spec.label,
                            "seed": seed,
                            "estimator": estimator_name,
                            "iteration": iteration,
                            "tau": spec.tau,
                            "w2_sq_proxy": _w2_proxy(particles, reference),
                            "protocol": "iterated_resampling_with_matched_refresh_proxy",
                        }
                    )
                    estimator = ESTIMATOR_REGISTRY[estimator_name]
                    weights = np.asarray(
                        estimator(jnp.asarray(particles), k=spec.k, tau=spec.tau, p=spec.p, N=n_particles)
                    )
                    idx = _systematic_resample(weights, rng)
                    particles = particles[idx]
                    labels = labels[idx]
                    refresh_n = max(1, int(0.12 * n_particles))
                    fresh, fresh_labels = sample_proxy(spec, refresh_n, seed + 6000 + iteration, mode="reference")
                    replace = rng.choice(n_particles, size=refresh_n, replace=False)
                    particles[replace] = fresh
                    labels[replace] = fresh_labels
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> None:
    os.makedirs(ROOT / "results", exist_ok=True)
    main_rows = _run_main()
    _write_csv(ROOT / "results" / "main_benchmarks.csv", main_rows)
    _write_csv(
        ROOT / "results" / "wallclock.csv",
        [
            {
                "benchmark": row["benchmark"],
                "benchmark_label": row["benchmark_label"],
                "seed": row["seed"],
                "estimator": row["estimator"],
                "n_particles": row["n_particles"],
                "k": row["k"],
                "tau": row["tau"],
                "wallclock_sec": row["wallclock_sec"],
                "protocol": row["protocol"],
            }
            for row in main_rows
        ],
    )
    _write_csv(ROOT / "results" / "tau_tolerance.csv", _run_tau())
    _write_csv(ROOT / "results" / "component_loss.csv", _run_component_sweep())
    _write_csv(ROOT / "results" / "iteration_proxy.csv", _iteration_rows())


if __name__ == "__main__":
    main()
