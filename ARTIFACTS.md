# Artifact Map

This file maps repository artifacts to the manuscript evidence.

## Known-DGP Density Monte Carlo

Driver:

- `experiments/run_known_dgp_mc.py`

Generated output:

- `results/known_dgp_mc.csv`
- `tables/tab1.tex`
- `figures/fig_e1.pdf`

Manuscript role:

- Verifies the flat `Exp(1)` MLE fallback.
- Shows PMM2 density-MSE reduction in asymmetric gamma and boundary-spacing
  regimes.
- Shows PMM3 degradation in the platykurtic uniform stress case.

## Resampling Proxy

Driver:

- `experiments/run_resampling_proxy.py`

Generated output:

- `results/main_benchmarks.csv`
- `results/tau_tolerance.csv`
- `results/component_loss.csv`
- `results/iteration_proxy.csv`
- `results/wallclock.csv`
- `tables/tab_e1.tex`
- `tables/tab_e2.tex`
- `figures/fig_e2.pdf`
- `figures/fig_e3.pdf`
- `figures/fig_e4.pdf`
- `figures/fig_e5.pdf`

Manuscript role:

- Provides diagnostic operational evidence for the density weight rule.
- Supports the seven-lobes positive proxy result.
- Documents failure modes on sine and swiss-roll proxies.

## PMM Density Module

Implementation:

- `src/masem/pmm_module.py`
- `src/masem/estimators.py`
- `src/masem/spacings.py`

Tests:

- `tests/test_pmm_module.py`
- `tests/test_estimators.py`

Manuscript role:

- Implements the gated PMM2/PMM3 density rule.
- Confirms that flat spacing regimes fall back to MLE/Plugin.
- Confirms shape, normalization, and JAX tracing behaviour.
