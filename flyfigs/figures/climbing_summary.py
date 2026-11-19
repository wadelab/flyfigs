"""Climbing summary: the climbing measure per fly across age, by genotype.

Boxes: median and IQR, whiskers 1.5 x IQR; dots are individual flies.
Stars: each genotype vs Control at the same age, two-sided Mann-Whitney U,
Holm-corrected across all genotype x age comparisons.

Outputs: climbing_summary.{pdf,eps,png}, climbing_summary_tests.csv
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..plotting import MM, age_axis, dodge, positions, save
from ..stats import holm, stars

SPACING = 5.0   # x distance between ages, in the units of the widths below


def run(ds):
    rng = ds.rng()
    m, ctrl = ds.climbing, ds.control.name
    ages = ds.ages('climbing_summary', [m])

    tests = []
    for g in ds.tests:
        for age in ages:
            a, b = m.flies(g.name, age), m.flies(ctrl, age)
            if not (len(a) and len(b)):
                warnings.warn(f'climbing_summary: {g.name} {age} DPE not tested ({len(a)} vs {len(b)} control flies)')
                continue
            p = stats.mannwhitneyu(a, b, alternative='two-sided').pvalue
            tests.append(dict(genotype=g.name, age_dpe=age, n_genotype=len(a), n_control=len(b), p_mwu=p))
    tests = pd.DataFrame(tests)
    tests['p_holm'] = holm(tests.p_mwu)
    tests.to_csv(ds.output_dir / 'climbing_summary_tests.csv', index=False, float_format='%.4g')
    p_holm = tests.set_index(['genotype', 'age_dpe']).p_holm

    n = len(ds.genotypes)
    vmax = m.data[m.values].max()
    fig, ax = plt.subplots(figsize=(np.clip(5 * len(ages) * n, 89, 183) * MM, 62 * MM))
    xs = positions(ages, SPACING)
    step = 4.4 / max(n, 4)
    width = step - 0.15
    for i, g in enumerate(ds.genotypes):
        pos = xs + dodge(i, n, step)
        vals = [m.flies(g.name, a) for a in ages]
        ax.boxplot(vals, positions=pos, widths=width, showfliers=False, patch_artist=True,
                   manage_ticks=False,
                   boxprops=dict(facecolor='white', edgecolor=g.colour, lw=0.9),
                   medianprops=dict(color=g.colour, lw=1.4),
                   whiskerprops=dict(color=g.colour, lw=0.8), capprops=dict(color=g.colour, lw=0.8))
        for x, v in zip(pos, vals):
            ax.scatter(x + rng.uniform(-0.3 * width, 0.3 * width, len(v)), v, s=4, color=g.colour,
                       alpha=0.55, edgecolors='none', zorder=3, rasterized=True)   # EPS has no transparency
        if g is not ds.control:
            for x, a, v in zip(pos, ages, vals):
                if (g.name, a) in p_holm.index:
                    ax.text(x, v.max() + 0.035 * vmax, stars(p_holm[(g.name, a)]), ha='center',
                            va='bottom', fontsize=6, color=g.colour)

    age_axis(ax, ages, SPACING, 3)
    ax.set_ylim(bottom=-0.03 * vmax)
    ax.set_xlabel('Age (DPE)')
    ax.set_ylabel(m.ylabel)
    handles = [plt.matplotlib.patches.Patch(facecolor='white', edgecolor=g.colour, lw=0.9, label=g.label)
               for g in ds.genotypes]
    ax.legend(handles=handles, loc='upper right', ncol=2, fontsize=6, columnspacing=0.8)
    save(fig, ds.output_dir, 'climbing_summary')
    print(tests.round(4).to_string(index=False))
