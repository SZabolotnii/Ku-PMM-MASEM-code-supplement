"""Generate LaTeX tables and PDF figures from saved PMM-MASEM results."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.stats_protocol import format_mean_ci, holm_bonferroni, paired_ttest, summarize


ESTIMATORS = ["Plugin_Estimator", "k_Ensemble", "MLE_Exp", "PMM2/PMM3"]
MARKERS = ["o", "s", "^", "D"]
LINESTYLES = ["-", "--", "-.", ":"]
COLORS = ["0.00", "0.25", "0.55", "0.00"]
OFFSETS = {
    "Plugin_Estimator": -0.060,
    "k_Ensemble": -0.020,
    "MLE_Exp": 0.020,
    "PMM2/PMM3": 0.060,
}


def _offset_for(estimator: str, scale: float = 1.0) -> float:
    """Small horizontal offset used only to reveal overlapping curves."""
    return OFFSETS[estimator] * scale


def _tex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")


def _table_tab1(known: pd.DataFrame) -> None:
    pmm = known[known["estimator"] == "PMM2/PMM3"].copy()
    rows = []
    order = [
        "flat_exp",
        "mild_curved_gamma",
        "strong_curved_gamma",
        "boundary_mixture",
        "platykurtic_uniform",
    ]
    for regime in order:
        sub = pmm[pmm["regime"] == regime]
        c3 = summarize(sub["c3"])
        c4 = summarize(sub["c4"])
        g2 = summarize(sub["g2"])
        action = sub["selector_branch"].mode().iloc[0]
        label = sub["regime_label"].iloc[0]
        rows.append(
            f"{_tex_escape(label)} & {c3.mean:.3f} & {c4.mean:.3f} & "
            f"{g2.mean:.3f} & {_tex_escape(action)} \\\\"
        )

    body = "\n".join(rows)
    text = rf"""\begin{{table}}[t]
\centering
\caption{{Empirical kNN-spacing regimes in the Known-DGP Monte Carlo.  Values are means over five seeds; $g_2=1-c_3^2/(2+c_4)$ is diagnostic and is not interpreted as an end-to-end gain.}}
\label{{tab:spacing_regimes}}
\begin{{tabular}}{{lrrrr}}
\toprule
Regime & $c_3$ & $c_4$ & $g_2$ & selector \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    _write(ROOT / "tables" / "tab1.tex", text)


def _table_tab_e1(main: pd.DataFrame) -> None:
    rows = []
    for benchmark, bdf in main.groupby("benchmark_label", sort=False):
        p_raw: dict[str, float] = {}
        plugin = (
            bdf[bdf["estimator"] == "Plugin_Estimator"]
            .sort_values("seed")["w2_sq_proxy"]
            .to_numpy()
        )
        for estimator in ESTIMATORS:
            if estimator == "Plugin_Estimator":
                continue
            vals = (
                bdf[bdf["estimator"] == estimator]
                .sort_values("seed")["w2_sq_proxy"]
                .to_numpy()
            )
            p_raw[estimator] = paired_ttest(vals, plugin)
        p_adj = holm_bonferroni(p_raw)

        cells = []
        for estimator in ESTIMATORS:
            vals = bdf[bdf["estimator"] == estimator]["w2_sq_proxy"]
            cell = format_mean_ci(vals, digits=3)
            if estimator != "Plugin_Estimator":
                p = p_adj.get(estimator, float("nan"))
                cell += f" ({p:.3f})" if np.isfinite(p) else " (--)"
            cells.append(cell)
        rows.append(f"{_tex_escape(benchmark)} & " + " & ".join(cells) + r" \\")

    body = "\n".join(rows)
    header = "Benchmark & Plugin & $k$-Ens. & MLE-Exp & PMM2/PMM3 \\\\"
    text = rf"""\begin{{table*}}[t]
\centering
\caption{{Single-step resampling-proxy $W_2^2$ results.  Entries are mean $\pm$ 95\% CI over five seeds; parentheses report Holm-adjusted paired $p$-values against Plugin.  This table is weight-rule evidence, not a full NHR/OLLA MASEM benchmark.}}
\label{{tab:main_proxy_w2}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrr}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table*}}
"""
    _write(ROOT / "tables" / "tab_e1.tex", text)


def _table_tab_e2(tau: pd.DataFrame) -> None:
    rows = []
    for benchmark, bdf in tau.groupby("benchmark_label", sort=False):
        cells = []
        for estimator in ESTIMATORS:
            edf = bdf[bdf["estimator"] == estimator]
            means = edf.groupby("tau")["w2_sq_proxy"].mean().sort_index()
            if 0.50 in means.index:
                threshold = 1.10 * means.loc[0.50]
            else:
                threshold = 1.10 * means.iloc[0]
            eligible = means[means <= threshold]
            max_tau = float(eligible.index.max()) if not eligible.empty else float(means.index.min())
            cells.append(f"{max_tau:.2f}")
        rows.append(f"{_tex_escape(benchmark)} & " + " & ".join(cells) + r" \\")

    body = "\n".join(rows)
    text = rf"""\begin{{table}}[t]
\centering
\caption{{Proxy $\tau$ tolerance: largest tested $\tau$ whose mean $W_2^2$ stays within 10\% of the estimator's $\tau=0.50$ value.}}
\label{{tab:tau_tolerance}}
\begin{{tabular}}{{lrrrr}}
\toprule
Benchmark & Plugin & $k$-Ens. & MLE-Exp & PMM2/PMM3 \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    _write(ROOT / "tables" / "tab_e2.tex", text)


def _figure_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    ax.axis("off")
    labels = ["Particles", "kNN radii", "Density rule", "Weights", "Resampling"]
    x = np.linspace(0.08, 0.92, len(labels))
    for i, (xi, label) in enumerate(zip(x, labels)):
        face = "0.90" if label != "Density rule" else "0.78"
        ax.add_patch(plt.Rectangle((xi - 0.075, 0.38), 0.15, 0.24, facecolor=face, edgecolor="black", lw=1.1))
        ax.text(xi, 0.50, label, ha="center", va="center", fontsize=9)
        if i < len(labels) - 1:
            ax.annotate("", xy=(x[i + 1] - 0.085, 0.50), xytext=(xi + 0.085, 0.50), arrowprops=dict(arrowstyle="->", lw=1))
    ax.text(0.50, 0.22, "PMM gate replaces only the density rule; MASEM kernels stay unchanged.", ha="center", fontsize=9)
    _savefig(ROOT / "figures" / "fig1_pipeline.pdf")


def _figure_e1(known: pd.DataFrame) -> None:
    order = [
        "Flat Exp(1)",
        "Mild curved gamma",
        "Strong curved gamma",
        "Boundary mixture",
        "Platykurtic uniform",
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    base_x = np.arange(len(order), dtype=float)
    for estimator, marker, ls, color in zip(ESTIMATORS, MARKERS, LINESTYLES, COLORS):
        edf = known[known["estimator"] == estimator]
        means = edf.groupby("regime_label")["density_mse"].mean().reindex(order)
        x = base_x + _offset_for(estimator)
        ax.plot(
            x,
            means,
            marker=marker,
            linestyle=ls,
            color=color,
            markerfacecolor="white" if estimator == "MLE_Exp" else color,
            markeredgecolor="black",
            linewidth=1.5,
            markersize=5.5,
            label=estimator,
        )
    ax.set_yscale("log")
    ax.set_xticks(base_x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Density MSE (log scale)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(
        0.01,
        0.02,
        "small x-offsets reveal overlapping curves",
        transform=ax.transAxes,
        fontsize=7,
        color="0.35",
    )
    _savefig(ROOT / "figures" / "fig_e1.pdf")


def _figure_e2(iteration: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=False)
    for ax, benchmark in zip(axes, ["seven_lobes", "swiss_roll"]):
        bdf = iteration[iteration["benchmark"] == benchmark]
        for estimator, marker, ls, color in zip(ESTIMATORS, MARKERS, LINESTYLES, COLORS):
            edf = bdf[bdf["estimator"] == estimator]
            mean = edf.groupby("iteration")["w2_sq_proxy"].mean()
            x = mean.index.to_numpy(dtype=float) + _offset_for(estimator, scale=0.55)
            ax.plot(
                x,
                mean.values,
                marker=marker,
                linestyle=ls,
                color=color,
                markerfacecolor="white" if estimator == "MLE_Exp" else color,
                markeredgecolor="black",
                linewidth=1.4,
                markersize=4.8,
                label=estimator,
            )
        ax.set_title(bdf["benchmark_label"].iloc[0])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("$W_2^2$ proxy")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    _savefig(ROOT / "figures" / "fig_e2.pdf")


def _figure_e3(tau: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for estimator, marker, ls, color in zip(ESTIMATORS, MARKERS, LINESTYLES, COLORS):
        edf = tau[tau["estimator"] == estimator]
        mean = edf.groupby("tau")["w2_sq_proxy"].mean()
        x = mean.index.to_numpy(dtype=float) + _offset_for(estimator, scale=0.35)
        ax.plot(
            x,
            mean.values,
            marker=marker,
            linestyle=ls,
            color=color,
            markerfacecolor="white" if estimator == "MLE_Exp" else color,
            markeredgecolor="black",
            linewidth=1.4,
            markersize=5.0,
            label=estimator,
        )
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("Mean $W_2^2$ proxy")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    _savefig(ROOT / "figures" / "fig_e3.pdf")


def _figure_e4(component: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for estimator, marker, ls, color in zip(ESTIMATORS, MARKERS, LINESTYLES, COLORS):
        edf = component[component["estimator"] == estimator]
        mean = edf.groupby("n_particles")["component_loss"].mean()
        x = mean.index.to_numpy(dtype=float) + _offset_for(estimator, scale=95.0)
        ax.plot(
            x,
            mean.values,
            marker=marker,
            linestyle=ls,
            color=color,
            markerfacecolor="white" if estimator == "MLE_Exp" else color,
            markeredgecolor="black",
            linewidth=1.4,
            markersize=5.0,
            label=estimator,
        )
    ax.set_xlabel("Particles")
    ax.set_ylabel("Component-loss frequency")
    ax.set_yscale("symlog", linthresh=0.02, linscale=0.6)
    ax.set_ylim(-0.002, 1.02)
    ax.set_yticks([0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00])
    ax.set_yticklabels(["0", "0.02", "0.05", "0.10", "0.20", "0.50", "1.00"])
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    _savefig(ROOT / "figures" / "fig_e4.pdf")


def _figure_e5(main: pd.DataFrame) -> None:
    mean = main.groupby("estimator")["wallclock_sec"].mean().reindex(ESTIMATORS)
    base = max(mean.loc["Plugin_Estimator"], 1e-12)
    rel = mean / base
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.bar(range(len(rel)), rel.values, color="0.75", edgecolor="black")
    ax.set_xticks(range(len(rel)))
    ax.set_xticklabels(["Plugin", "$k$-Ens.", "MLE", "PMM"], rotation=0)
    ax.set_ylabel("Relative wall-clock")
    ax.axhline(1.0, color="black", lw=0.8, linestyle=":")
    ax.grid(True, axis="y", alpha=0.25)
    _savefig(ROOT / "figures" / "fig_e5.pdf")


def main() -> None:
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "tables").mkdir(exist_ok=True)

    known = pd.read_csv(ROOT / "results" / "known_dgp_mc.csv")
    main_df = pd.read_csv(ROOT / "results" / "main_benchmarks.csv")
    tau = pd.read_csv(ROOT / "results" / "tau_tolerance.csv")
    component = pd.read_csv(ROOT / "results" / "component_loss.csv")
    iteration = pd.read_csv(ROOT / "results" / "iteration_proxy.csv")

    _table_tab1(known)
    _table_tab_e1(main_df)
    _table_tab_e2(tau)
    _figure_pipeline()
    _figure_e1(known)
    _figure_e2(iteration)
    _figure_e3(tau)
    _figure_e4(component)
    _figure_e5(main_df)


if __name__ == "__main__":
    main()
