"""Small statistical helpers used by the PMM-MASEM experiment scripts."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Summary:
    """Mean, standard error, and two-sided 95% confidence interval."""

    n: int
    mean: float
    se: float
    ci95: float


def summarize(values: Iterable[float]) -> Summary:
    """Return mean +/- 95% CI using the Student t critical value."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return Summary(n=0, mean=float("nan"), se=float("nan"), ci95=float("nan"))
    if arr.size == 1:
        return Summary(n=1, mean=float(arr[0]), se=0.0, ci95=0.0)
    se = float(np.std(arr, ddof=1) / sqrt(arr.size))
    ci = float(stats.t.ppf(0.975, arr.size - 1) * se)
    return Summary(n=int(arr.size), mean=float(np.mean(arr)), se=se, ci95=ci)


def paired_ttest(x: Iterable[float], y: Iterable[float]) -> float:
    """Two-sided paired t-test p-value with finite-value filtering."""
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() < 2:
        return float("nan")
    res = stats.ttest_rel(xa[mask], ya[mask])
    return float(res.pvalue)


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Return Holm-Bonferroni adjusted p-values for a keyed p-value map."""
    finite_items = [(k, p) for k, p in p_values.items() if np.isfinite(p)]
    adjusted = {k: float("nan") for k in p_values}
    if not finite_items:
        return adjusted

    ordered = sorted(finite_items, key=lambda item: item[1])
    m = len(ordered)
    running_max = 0.0
    for rank, (key, p) in enumerate(ordered, start=1):
        raw = min((m - rank + 1) * p, 1.0)
        running_max = max(running_max, raw)
        adjusted[key] = float(running_max)
    return adjusted


def format_mean_ci(values: Iterable[float], digits: int = 3) -> str:
    """Format values as mean +/- 95% CI for LaTeX tables."""
    s = summarize(values)
    if s.n == 0:
        return "--"
    return f"{s.mean:.{digits}f} $\\pm$ {s.ci95:.{digits}f}"

