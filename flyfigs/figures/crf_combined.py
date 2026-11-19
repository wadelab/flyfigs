"""Combined figure: fixed-Rmax CRF fits (left) and bootstrap c50 (right), one row per harmonic.

Model: R = Rmax*c^2/(c50^2 + c^2), Rmax fixed at the pooled fit for each harmonic
(all genotypes and ages); c50 free. Same model as crf_fits_fixRmax and
crf_params_fixed_Rmax.

CRF panels: points mean +/- SEM across flies; curve fitted to the group mean;
band 95% bootstrap interval of the fitted curve.
c50 panel: bootstrap distribution of c50 (box = IQR, line = median,
whiskers = 2.5th-97.5th percentiles). Bands and boxes come from the same resamples.
Stars: two-sided permutation test of log c50 (each genotype vs Control, c50 refitted
to each permuted group mean), Holm-corrected across all harmonic x genotype x age tests.

Outputs: crf_combined_fixRmax.{pdf,eps,png}, crf_c50_tests.csv
"""
import string
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..crf import CRF, hr
from ..plotting import MM, age_axis, dodge, letter, log_axis, save
from ..stats import holm, stars

SPACING = 7   # x distance between ages, in the units of the widths below


def run(ds):
    rng = ds.rng()
    crf = CRF(ds.contrasts)
    ages = ds.ages('crf_combined', ds.harmonics)
    n_boot, n_perm = ds.option('crf_combined', 'n_boot'), ds.option('crf_combined', 'n_perm')
    pooled = {h.key: crf.fit('hr', h.pooled_mean(ages)) for h in ds.harmonics}

    tests = []
    for h in ds.harmonics:
        for g in ds.tests:
            for age in ages:
                a, b = h.flies(g.name, age), h.flies(ds.control.name, age)
                if min(len(a), len(b)) < 2:
                    warnings.warn(f'crf_combined: {h.key} {g.name} {age} DPE not tested '
                                  f'({len(a)} vs {len(b)} control flies)')
                    continue
                log_ratio, p = crf.perm_test_c50(a, b, pooled[h.key][0], rng, n_perm)
                tests.append(dict(harmonic=h.key, genotype=g.name, age_dpe=age, n_genotype=len(a),
                                  n_control=len(b), c50_ratio_genotype_over_control=np.exp(log_ratio),
                                  p_perm=p))
    tests = pd.DataFrame(tests)
    tests['p_holm'] = holm(tests.p_perm)
    tests.to_csv(ds.output_dir / 'crf_c50_tests.csv', index=False, float_format='%.4g')
    p_holm = tests.set_index(['harmonic', 'genotype', 'age_dpe']).p_holm

    na, nh, n = len(ages), len(ds.harmonics), len(ds.genotypes)
    step = 6 / max(n, 4)
    fig, axs = plt.subplots(nh, na + 2, figsize=(183 * MM, 50 * MM * nh), squeeze=False,
                            gridspec_kw=dict(width_ratios=[1] * na + [0.12, 1.9]))
    for ax in axs[:, na]:
        ax.set_visible(False)   # spacer between the CRFs and the c50 panel

    for row, h in enumerate(ds.harmonics):
        rmax = pooled[h.key][0]
        crf_axs, box_ax = axs[row, :na], axs[row, -1]
        for i, g in enumerate(ds.genotypes):
            c50_boot, pos, present = [], [], []
            for col, age in enumerate(ages):
                ax = crf_axs[col]
                y_fly = h.flies(g.name, age)
                if len(y_fly) < 2:
                    warnings.warn(f'crf_combined: {h.key} {g.name} {age} DPE skipped ({len(y_fly)} flies)')
                    continue
                y = y_fly.mean(axis=0)
                sem = y_fly.std(axis=0, ddof=1) / np.sqrt(len(y_fly))
                boots = []
                for _ in range(n_boot):
                    yb = y_fly[rng.integers(0, len(y_fly), len(y_fly))].mean(axis=0)
                    try:
                        boots.append(crf.fit('fixRmax', yb, pooled[h.key])[1])
                    except RuntimeError:
                        pass
                boots = np.array(boots)
                c50_boot.append(boots)
                pos.append(col * SPACING + dodge(i, n, step))
                present.append(age)
                curves = hr(crf.c_fine[None, :], rmax, boots[:, None])
                lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
                ax.fill_between(crf.c_fine, lo, hi, color=g.colour, alpha=0.15, lw=0, zorder=1, rasterized=True)
                ax.plot(crf.c_fine, crf.curve('fixRmax', crf.fit('fixRmax', y, pooled[h.key])),
                        color=g.colour, lw=1.1, zorder=2)
                ax.errorbar(crf.c, y, yerr=sem, fmt=g.marker, color=g.colour, ms=3.5,
                            elinewidth=0.7, mfc=g.colour, mec='white', mew=0.4, zorder=3)
            # c50 boxes: ages on x, genotypes dodged
            box_ax.boxplot(c50_boot, positions=pos, widths=step - 0.2, whis=(2.5, 97.5), showfliers=False,
                           patch_artist=True, manage_ticks=False,
                           boxprops=dict(facecolor=g.colour, edgecolor=g.colour, lw=0.8),
                           medianprops=dict(color='white', lw=1.1),
                           whiskerprops=dict(color=g.colour, lw=0.8), capprops=dict(color=g.colour, lw=0.8))
            if g is not ds.control:   # stars above each box
                for x, b, age in zip(pos, c50_boot, present):
                    if (h.key, g.name, age) in p_holm.index:
                        # stacked vertically so neighbouring labels never collide
                        box_ax.text(x, np.percentile(b, 97.5) * 1.05, '\n'.join(stars(p_holm[(h.key, g.name, age)])),
                                    ha='center', va='bottom', fontsize=6, color=g.colour, linespacing=0.55)

        # CRF axes
        ymax = max(ax.get_ylim()[1] for ax in crf_axs)
        for col, (ax, age) in enumerate(zip(crf_axs, ages)):
            ax.set_xscale('log')
            ax.set_xticks(ds.contrast_ticks, [f'{t:g}' for t in ds.contrast_ticks])
            ax.minorticks_off()
            ax.set_xlim(*crf.xlim)
            ax.set_ylim(bottom=min(0, ax.get_ylim()[0]), top=ymax)
            if col:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.set_title(f'{age:g} DPE')
            if row == nh - 1 and (col == 0 or na <= 4):   # one label when panels are too narrow for one each
                ax.set_xlabel('Contrast (%)')
        crf_axs[0].set_ylabel(f'{h.title}\n{h.ylabel}')
        crf_axs[-1].text(0.04, 0.96, rf'$R_{{\max}}$ fixed at {rmax:.0f}', transform=crf_axs[-1].transAxes,
                         va='top', fontsize=6.5)

        # c50 axis
        log_axis(box_ax, 3 * crf.c50_bounds[0], crf.c50_bounds[1])
        age_axis(box_ax, ages, SPACING, 4)
        box_ax.set_ylabel(r'$c_{50}$ (% contrast)')
        if row == nh - 1:
            box_ax.set_xlabel('Age (DPE)')

        letter(crf_axs[0], string.ascii_uppercase[2 * row], -0.42, 1.06)
        letter(box_ax, string.ascii_uppercase[2 * row + 1], -0.3, 1.06)

    handles = [plt.Line2D([], [], color=g.colour, marker=g.marker, mec='white', mew=0.4, ms=4, label=g.label)
               for g in ds.genotypes]
    axs[-1, -1].legend(handles=handles, loc='lower center', ncol=2, fontsize=6, columnspacing=0.8)
    save(fig, ds.output_dir, 'crf_combined_fixRmax')
