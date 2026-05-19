# Reproducibility Notes

This supplement is designed to reproduce the evidence bundle for the
PMM-MASEM boundary paper.

## Environment

The reference environment used during package preparation was:

- Python 3.14
- JAX/JAXLIB as pinned in `requirements.txt`
- NumPy, SciPy, pandas, matplotlib, pytest, hypothesis as pinned in
  `requirements.txt`

The manuscript experiments use the self-contained Python/JAX PMM2/PMM3
implementation in `src/masem/pmm_module.py`.

Create a clean environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Verification Commands

Run unit tests:

```bash
python -m pytest -q
```

Run the full local reproduction pipeline:

```bash
python -m experiments.run_all
```

The driver runs:

1. `experiments.run_known_dgp_mc`
2. `experiments.run_resampling_proxy`
3. `experiments.generate_tables_figures`

## Random Seeds and Statistical Protocol

Experiments use five fixed seeds.  Tables report mean plus/minus 95% confidence
intervals.  Pairwise tests use paired t-tests with Holm-Bonferroni correction
at the manuscript threshold `p < 0.01`.

## Evidence Boundary

The resampling-proxy experiments are diagnostic.  They test whether the density
weight rule can have an operational effect under matched particle clouds and
resampling, but they do not include the full MASEM NHR/OLLA local-rejuvenation
loop.  The manuscript therefore treats this package as evidence for a
regime-aware density module, not as proof of a general sampler improvement.

## Expected Outputs

After running `python -m experiments.run_all`, the following files should be
created or updated:

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
- `figures/fig_e1.pdf`
- `figures/fig_e2.pdf`
- `figures/fig_e3.pdf`
- `figures/fig_e4.pdf`
- `figures/fig_e5.pdf`

Because wall-clock timing is machine-dependent, `results/wallclock.csv` may
vary across hardware.  The qualitative manuscript claim is only that the PMM
gate is not a dominant cost in these small unjitted proxy runs.

For the committed artifact snapshot, see `SHA256SUMS.md`.
