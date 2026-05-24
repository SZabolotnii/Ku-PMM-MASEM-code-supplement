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
- Separates symmetric platykurtic panels into the PMM3-location diagnostic
  branch rather than the production density selector.

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

- Implements the gated PMM2/MLE production density rule with PMM3-location
  diagnostics kept separate.
- Confirms that flat spacing regimes fall back to MLE/Plugin.
- Confirms shape, normalization, and JAX tracing behaviour.

## PMM3-Location Diagnostics

Primary Python driver:

- `experiments/research_pmm3_platykurtic_spacing.py`

Generated output:

- `results/pmm3_platykurtic_spacing_research.csv`

Optional R sanity-check drivers:

- `experiments/research_pmm3_platykurtic_regression.R`
- `experiments/verify_estempmm_pmm3_uniform.R`

Generated optional output:

- `results/pmm3_platykurtic_regression_research.csv`

Manuscript role:

- Checks the centered PMM3 location equation under symmetric platykurtic
  residual laws.
- Tests whether the location equation can be adapted to positive shell-spacing
  panels under strict guards.
- Supports PMM3 only as conditional diagnostic evidence, not as an end-to-end
  production MASEM claim.
