"""Bootstrap distributions of hyperbolic ratio parameters, R = Rmax*c^2/(c50^2 + c^2).

For each harmonic x genotype x age, flies are resampled with replacement and the
model refitted to the resampled group mean (n_boot times).

crf_params_full: both parameters free.
crf_params_fixed_c50: c50 fixed, Rmax free, for every harmonic.
crf_params_fixed_Rmax: Rmax fixed, c50 free, for every harmonic.
  The fixed value is the full-model fit to all flies of all genotypes and ages for
  that harmonic, so it favours no genotype. (Harmonics differ in response scale, so
  each gets its own fixed value of the same parameter.)

Boxes: median and IQR; whiskers 2.5th-97.5th percentiles (95% CI).

Outputs: crf_params_{full,fixed_c50,fixed_Rmax}.{pdf,eps,png}, crf_param_bootstrap.csv
"""
import string
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from ..crf import CRF
from ..plotting import MM, age_axis, dodge, letter, log_axis, save

CONSTRAINTS = {'fixc50': 'Rmax', 'fixRmax': 'c50'}   # constraint: its free parameter
SPACING = 7   # x distance between ages, in the units of the widths below


def run(ds):
    rng = ds.rng()
    crf = CRF(ds.contrasts)
    ages = ds.ages('crf_param_boxes', ds.harmonics)
    n_boot = ds.option('crf_param_boxes', 'n_boot')

    # Fixed values (Rmax, c50) from the pooled fit for each harmonic
    pooled = {h.key: crf.fit('hr', h.pooled_mean(ages)) for h in ds.harmonics}
    print('Pooled fits (Rmax, c50):', {k: p.round(1).tolist() for k, p in pooled.items()})

    def fit_all(y, h):
        """Full fit plus both constrained fits -> flat dict of parameters."""
        out = dict(zip(['Rmax_full', 'c50_full'], crf.fit('hr', y)))
        for name in CONSTRAINTS:
            out.update(zip([f'Rmax_{name}', f'c50_{name}'], crf.fit(name, y, pooled[h.key])))
        return out

    rows, boots = [], []
    for h in ds.harmonics:
        for g in ds.genotypes:
            for age in ages:
                y_fly = h.flies(g.name, age)
                if len(y_fly) < 2:
                    warnings.warn(f'crf_param_boxes: {h.key} {g.name} {age} DPE skipped ({len(y_fly)} flies)')
                    continue
                y = y_fly.mean(axis=0)
                row = dict(harmonic=h.key, genotype=g.name, age_dpe=age, n=len(y_fly), **fit_all(y, h))
                for name in ['full', *CONSTRAINTS]:
                    row[f'R2_{name}'] = crf.r_squared('hr', y, (row[f'Rmax_{name}'], row[f'c50_{name}']))
                rows.append(row)
                for _ in range(n_boot):
                    yb = y_fly[rng.integers(0, len(y_fly), len(y_fly))].mean(axis=0)
                    try:
                        boots.append(dict(harmonic=h.key, genotype=g.name, age_dpe=age, **fit_all(yb, h)))
                    except RuntimeError:
                        continue
    fits = pd.DataFrame(rows)
    boots = pd.DataFrame(boots)

    # Percentile summaries for the CSV
    keys = ['harmonic', 'genotype', 'age_dpe']
    summ = boots.groupby(keys).quantile([0.025, 0.5, 0.975]).unstack()
    summ.columns = [f'{p}_q{int(q * 1000):03d}' for p, q in summ.columns]
    fits.merge(summ.reset_index(), on=keys).to_csv(ds.output_dir / 'crf_param_bootstrap.csv', index=False,
                                                   float_format='%.3f')

    n = len(ds.genotypes)
    step = 6 / max(n, 4)

    def box_panel(ax, h, col):
        """Boxes of boots[col] for harmonic h: ages on x, genotypes dodged."""
        for i, g in enumerate(ds.genotypes):
            sub = boots[(boots.harmonic == h.key) & (boots.genotype == g.name)]
            present = [k for k, a in enumerate(ages) if (sub.age_dpe == a).any()]
            ax.boxplot([sub[sub.age_dpe == ages[k]][col].to_numpy() for k in present],
                       positions=[k * SPACING + dodge(i, n, step) for k in present], widths=step - 0.2,
                       whis=(2.5, 97.5), showfliers=False, patch_artist=True, manage_ticks=False,
                       boxprops=dict(facecolor=g.colour, edgecolor=g.colour, lw=0.8),
                       medianprops=dict(color='white', lw=1.1),
                       whiskerprops=dict(color=g.colour, lw=0.8), capprops=dict(color=g.colour, lw=0.8))
        age_axis(ax, ages, SPACING, 4)
        if col.startswith('c50'):
            log_axis(ax, 3 * crf.c50_bounds[0], crf.c50_bounds[1])

    def legend(ax):
        handles = [plt.matplotlib.patches.Patch(facecolor=g.colour, edgecolor=g.colour, label=g.label)
                   for g in ds.genotypes]
        ax.legend(handles=handles, loc='upper right', fontsize=6)

    labels = {'Rmax': rf'$R_{{\max}}$ ({ds.units})', 'c50': r'$c_{50}$ (% contrast)'}
    symbol = {'Rmax': r'$R_{\max}$', 'c50': r'$c_{50}$'}

    # ---- Full model ------------------------------------------------------------------
    nh = len(ds.harmonics)
    fig, axs = plt.subplots(nh, 2, figsize=(183 * MM, 55 * MM * nh), sharex=True, squeeze=False)
    for row, h in enumerate(ds.harmonics):
        for colidx, p in enumerate(['Rmax', 'c50']):
            ax = axs[row, colidx]
            box_panel(ax, h, f'{p}_full')
            ax.set_ylabel(labels[p])
            ax.set_title(h.title)
            letter(ax, string.ascii_uppercase[row * 2 + colidx], -0.16)
    for ax in axs[-1]:
        ax.set_xlabel('Age (DPE)')
    legend(axs[0, 0])
    save(fig, ds.output_dir, 'crf_params_full')

    # ---- The same parameter fixed for every harmonic -----------------------------------
    for name, free in CONSTRAINTS.items():
        fixed = 'c50' if free == 'Rmax' else 'Rmax'
        fig, axs = plt.subplots(1, nh, figsize=(183 * MM, 62 * MM), sharex=True, squeeze=False)
        for ax, h, lt in zip(axs[0], ds.harmonics, string.ascii_uppercase):
            box_panel(ax, h, f'{free}_{name}')
            value = pooled[h.key][0 if fixed == 'Rmax' else 1]
            ax.set_title(f'{h.title}: {symbol[fixed]} fixed at {value:.0f}' + ('%' if fixed == 'c50' else ''))
            ax.set_ylabel(labels[free])
            ax.set_xlabel('Age (DPE)')
            letter(ax, lt, -0.14)
        legend(axs[0, -1])
        save(fig, ds.output_dir, f'crf_params_fixed_{fixed}')

    print(fits.groupby('harmonic')[['R2_full', 'R2_fixc50', 'R2_fixRmax']].agg(['min', 'median']).round(2))
