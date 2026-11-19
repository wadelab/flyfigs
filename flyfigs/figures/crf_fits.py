"""Contrast-response functions with model fits, one figure per model ([crf_fits] models).

Points: mean +/- SEM across flies. Curves: fit to the group mean.
Bands: 95% bootstrap interval of the fitted curve (flies resampled within group).
fixc50 / fixRmax hold that parameter at the hyperbolic ratio fit to every fly of
every genotype and age, for each harmonic. Models are described in crf.py.

Outputs: crf_fits_<model>.{pdf,eps,png}, crf_fits_<model>_params.csv
"""
import string
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..crf import CRF
from ..plotting import MM, letter, save


def run(ds):
    rng = ds.rng()
    crf = CRF(ds.contrasts)
    ages = ds.ages('crf_fits', ds.harmonics)
    n_boot = ds.option('crf_fits', 'n_boot')
    pooled = {h.key: crf.fit('hr', h.pooled_mean(ages)) for h in ds.harmonics}   # (Rmax, c50)

    for model in ds.option('crf_fits', 'models'):
        params = []
        fig, axs = plt.subplots(len(ds.harmonics), len(ages), sharex=True, sharey='row', squeeze=False,
                                figsize=(183 * MM, 50 * MM * len(ds.harmonics)))
        for row, h in enumerate(ds.harmonics):
            for col, age in enumerate(ages):
                ax = axs[row, col]
                for g in ds.genotypes:
                    y_fly = h.flies(g.name, age)
                    if len(y_fly) < 2:
                        warnings.warn(f'crf_fits: {h.key} {g.name} {age} DPE skipped ({len(y_fly)} flies)')
                        continue
                    y = y_fly.mean(axis=0)
                    sem = y_fly.std(axis=0, ddof=1) / np.sqrt(len(y_fly))
                    p = crf.fit(model, y, pooled[h.key])
                    boots = []
                    for _ in range(n_boot):
                        yb = y_fly[rng.integers(0, len(y_fly), len(y_fly))].mean(axis=0)
                        try:
                            boots.append(crf.curve(model, crf.fit(model, yb, pooled[h.key])))
                        except RuntimeError:
                            pass
                    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
                    ax.fill_between(crf.c_fine, lo, hi, color=g.colour, alpha=0.15, lw=0, zorder=1,
                                    rasterized=True)
                    ax.plot(crf.c_fine, crf.curve(model, p), color=g.colour, lw=1.1, zorder=2)
                    ax.errorbar(crf.c, y, yerr=sem, fmt=g.marker, color=g.colour, ms=3.5,
                                elinewidth=0.7, mfc=g.colour, mec='white', mew=0.4, zorder=3)
                    params.append(dict(harmonic=h.key, genotype=g.name, age_dpe=age, n=len(y_fly),
                                       Rmax=p[0], c50=p[1], **({'q': p[2]} if model == 'ss' else {}),
                                       R2=crf.r_squared(model, y, p)))
                ax.set_xscale('log')
                ax.set_xticks(ds.contrast_ticks, [f'{t:g}' for t in ds.contrast_ticks])
                ax.minorticks_off()
                ax.set_xlim(*crf.xlim)
                if row == 0:
                    ax.set_title(f'{age:g} DPE')
                if row == len(ds.harmonics) - 1 and (col == 0 or len(ages) <= 4):   # one label if panels are narrow
                    ax.set_xlabel('Contrast (%)')
            axs[row, 0].set_ylabel(f'{h.title}\n{h.ylabel}')
            if model.startswith('fix'):
                rmax, c50 = pooled[h.key]
                note = f'$c_{{50}}$ fixed at {c50:.0f}%' if model == 'fixc50' else f'$R_{{\\max}}$ fixed at {rmax:.0f}'
                axs[row, -1].text(0.04, 0.96, note, transform=axs[row, -1].transAxes, va='top', fontsize=6.5)
            letter(axs[row, 0], string.ascii_uppercase[row], -0.42, 1.06)
        handles = [plt.Line2D([], [], color=g.colour, marker=g.marker, mec='white', mew=0.4, ms=4, label=g.label)
                   for g in ds.genotypes]
        axs[0, 0].legend(handles=handles, loc='upper left', fontsize=6)
        save(fig, ds.output_dir, f'crf_fits_{model}')
        pd.DataFrame(params).to_csv(ds.output_dir / f'crf_fits_{model}_params.csv', index=False,
                                    float_format='%.3f')
