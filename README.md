# PMM-MASEM Code Supplement

This repository contains the public verification package for the manuscript
**"Variance-Reduced Manifold Sampling via Polynomial-Maximization Density
Estimation"**.

The package is intentionally narrow.  It reproduces the evidence used in the
paper: a Known-DGP density Monte Carlo, a lightweight resampling-proxy layer,
generated tables, generated figures, and unit tests for the PMM density
module.  It does **not** claim to be a full end-to-end MASEM NHR/OLLA benchmark
suite.

The PMM2/PMM3 density module used in the manuscript is implemented directly in
Python/JAX in `src/masem/pmm_module.py`.  This supplement is self-contained for
the experiments reported in the manuscript.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python -m experiments.run_all
```

The CI workflow runs the test suite on Python 3.12.  The package was also
verified locally on Python 3.14.

Expected test result:

```text
70 passed
```

The reproduction driver regenerates:

- `results/known_dgp_mc.csv`
- `results/main_benchmarks.csv`
- `results/tau_tolerance.csv`
- `results/component_loss.csv`
- `results/iteration_proxy.csv`
- `results/wallclock.csv`
- `tables/tab1.tex`
- `tables/tab_e1.tex`
- `tables/tab_e2.tex`
- `figures/fig1_pipeline.pdf`
- `figures/fig_e1.pdf` ... `figures/fig_e5.pdf`

## Main Claims Supported by This Package

The code supports the following bounded claims:

1. On flat `Exp(1)` shell spacings, the selector falls back to the
   plug-in/MLE density rule.
2. PMM2 reduces local density MSE in selected asymmetric gamma and
   boundary-spacing Known-DGP regimes.
3. PMM3 is not supported by the platykurtic stress case.
4. The resampling-proxy layer is mixed: it improves the seven-lobes proxy but
   degrades the sine and swiss-roll proxies.

The code does not support a universal PMM-MASEM superiority claim.

## Repository Layout

- `src/masem/` - PMM-MASEM prototype modules.
- `experiments/` - reproduction drivers and statistical protocol.
- `tests/` - unit tests for estimators and PMM module behaviour.
- `results/` - committed CSV outputs used by the manuscript.
- `figures/` - committed generated PDF figures.
- `tables/` - committed generated LaTeX tables.
- `paper/PMM-MASEM-preprint.pdf` - current preprint PDF snapshot.
- `REPRODUCIBILITY.md` - detailed reproduction notes.
- `ARTIFACTS.md` - mapping between files and manuscript evidence.
- `SHA256SUMS.md` - hashes for committed result, figure, and table artifacts.

## Citation

If you use this code supplement, please cite the manuscript/preprint and this
repository.  A machine-readable citation file is provided in `CITATION.cff`.

## License

Code is released under the MIT License.  The manuscript PDF in `paper/` is
provided as a scholarly preprint snapshot; cite the paper when reusing results
or text.
