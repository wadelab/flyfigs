# flyfigs

Contrast-response, effect-size and climbing analyses for *Drosophila* SSVEP and
climbing data. These are the analyses behind Stretch et al. Figures 2 and 5: Hedges' g
against control across age, contrast-response function (CRF) fits with bootstrap and
permutation tests of c50, and a climbing summary.

The code runs on any dataset in the same format. A TOML config file describes each
dataset: where its files are, which genotypes to compare and which ages to use. The code
itself does not change between datasets.

## Install

Needs Python 3.11 or later and [uv](https://docs.astral.sh/uv/) (one-line install on
the linked page).

```bash
git clone <this repository> flyfigs
cd flyfigs
uv sync                     # makes .venv with numpy, scipy, pandas, matplotlib, statsmodels
uv run pytest               # optional: a few seconds of checks
```

## Run

```bash
uv run flyfigs examples/gal80_hsnca/gal80_hsnca.toml --quick     # fast check, about a minute
uv run flyfigs examples/gal80_hsnca/gal80_hsnca.toml             # full run, about five minutes
uv run flyfigs my_dataset.toml crf_combined hedges_g             # only these figures
uv run flyfigs my_dataset.toml --out /somewhere/else
```

`--quick` caps resamples at 100 bootstraps and 1,000 permutations and writes to
`<output_dir>/quick/`, so it never overwrites the full results. The run begins by
printing a table of flies per genotype × age for each measure; a zero row means a
genotype code in the config does not match the data.

Figures are written as PDF, EPS and PNG, with a CSV of the statistics beside each.

## A new dataset

1. Copy `examples/gal80_hsnca/gal80_hsnca.toml` next to where the figures should go.
   Paths in the config are relative to the config file.
2. Point `[ssvep]` at one CSV per harmonic and `[climbing]` at the climbing CSV. Delete
   either section if the dataset doesn't have it; the figures that need it are skipped.
3. List the genotypes in plotting order. Mark one with `control = true`. `codes` gives
   each genotype's value in the Genotype column of the SSVEP and climbing files, since
   these often differ (`GAL80LACZ` vs `GAL80LacZ`).
4. Run with `--quick` first and check the flies-per-genotype table.

The example config is annotated and lists every option.

### Data format

- **SSVEP**: one row per fly, with a genotype column, an age column and one column per
  contrast (`x5, x10, x30, x70, x100` by default; set `columns` if they differ).
- **Climbing**: one row per fly, with a genotype column, an age column and the measure
  (the TrackMate export; `Total distance traveled` by default).

Column names are set in the config. An optional `query` (pandas syntax, for example
`"Drug == 'no'"`) selects rows before analysis.

The code plots whatever the SSVEP columns contain and labels the axis `response (units)`
with `units` from the config. **The example SSVEP files hold signal-to-noise ratios, not
amplitudes**: the MATLAB export divides the FFT magnitude at each harmonic by the RMS of
the five bins either side. Because those neighbouring bins pick up response energy at
high contrast, SNR can flatten or fall where the amplitude itself keeps rising. Say which
quantity a dataset holds when describing its figures, and set `units` to match.

## Figures

| Name | Outputs | What |
|---|---|---|
| `hedges_g` | `hedges_g_vision_vs_climbing{,_box}`, `hedges_g_values.csv` | Hedges' g vs control per measure and age; bootstrap 95% CI, permutation p (Holm) |
| `climbing_summary` | `climbing_summary`, `climbing_summary_tests.csv` | Climbing per fly by genotype and age; Mann-Whitney U vs control (Holm) |
| `crf_fits` | `crf_fits_<model>`, `crf_fits_<model>_params.csv` | CRFs with fits for each model in `[crf_fits] models` |
| `crf_param_boxes` | `crf_params_{full,fixed_c50,fixed_Rmax}`, `crf_param_bootstrap.csv` | Bootstrap distributions of Rmax and c50 |
| `crf_combined` | `crf_combined_fixRmax`, `crf_c50_tests.csv` | Fixed-Rmax CRFs with bootstrap c50 and permutation tests (Figure 2 A–D) |

Each module's docstring (`flyfigs/figures/*.py`) gives its statistics in full. Tables
identify genotypes by the `name` given in the config.

### CRF models (`flyfigs/crf.py`)

| Model | Form | Free |
|---|---|---|
| `hr` | Rmax c² / (c50² + c²) | Rmax, c50 |
| `ss` | Rmax c² / (c50^q + c^q), q ≥ 2 | Rmax, c50, q |
| `fixc50` | `hr` with c50 fixed at the pooled fit | Rmax |
| `fixRmax` | `hr` with Rmax fixed at the pooled fit | c50 |

The pooled values are fitted to the mean over every fly of every genotype at the ages
plotted, so they follow the data rather than being hard-coded. c50 is bounded to
[lowest contrast / 5, 3 × highest contrast] and starts at the median contrast.

### What adapts to the data

- **Ages** are plotted evenly spaced, so irregular ages such as 1, 7, 14 DPE stay
  legible. Set `ages` at the top of the config or per figure, or `"all"` for every age
  in the data.
- **Genotypes and measures**: box widths and offsets follow their number. The Hedges' g
  figure wraps to three panels per row.
- **Small groups** with fewer than two flies are skipped with a warning.

## Layout

```
flyfigs/            the package: cli, config, crf (models and fits), stats, plotting
flyfigs/figures/    one module per figure, each with a run(dataset) function
flyfigs/wadelab.mplstyle   figure style (journal column sizes, embedded fonts)
examples/gal80_hsnca/      the GAL80 hSNCA dataset and its config
tests/              pytest checks of the fits, effect sizes and the example
```

Reproducibility: every bootstrap and permutation uses `seed` from the config, so a rerun
on the same data gives the same tables. `uv.lock` pins the library versions, and
`uv sync` installs exactly those. Across scipy versions the only change seen so far is
in the last decimal of Rmax for the `ss` model, in groups where that fit is
unconstrained; every other table is identical.
