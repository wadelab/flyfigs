"""Hedges' g (each genotype vs Control) across age, for each SSVEP harmonic and climbing.

SSVEP: response at one contrast ([hedges_g] contrast). Climbing: the configured measure.
Sign: positive g = change in the phenotypic direction set for each measure
(`phenotype` in the dataset config):
  increase   g = (genotype - Control) / s_pooled * J
  decrease   g = (Control - genotype) / s_pooled * J
J is the Hedges small-sample correction. 95% CIs are bootstrap percentile intervals
(flies resampled with replacement within each group). p: two-sided permutation test of
g = 0, Holm-corrected across all genotype x measure x age tests.

Outputs: hedges_g_values.csv, hedges_g_vision_vs_climbing{,_box}.{pdf,eps,png}
"""
import string
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from ..plotting import MM, age_axis, dodge, letter, positions, save
from ..stats import g_with_ci, holm, stars

SPACING = 7   # x distance between ages, in the units of the offsets and widths below
INK = '#262626'
# SSVEP harmonics take these (marker, filled) in config order, so 1F1 is open and 2F1
# filled; climbing is always dashed with filled squares, and hatched in the box version.
SSVEP_STYLES = [('o', False), ('o', True), ('^', False), ('^', True), ('D', False), ('D', True)]


def styles(measures, harmonics):
    """measure label -> (line style, marker, filled, box hatch), in plot order."""
    out = {}
    for m in measures:
        if m.kind == 'climbing':
            out[m.label] = ('--', 's', True, '//////')
        else:
            out[m.label] = ('-', *SSVEP_STYLES[harmonics.index(m) % len(SSVEP_STYLES)], None)
    return out


def panels(n):
    """One panel per genotype, three to a row; returns the figure and the n axes."""
    ncols = min(n, 3)
    nrows = -(-n // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(183 * MM, 62 * MM * nrows),
                            sharex=True, sharey=True, squeeze=False)
    axs = axs.ravel()
    for ax in axs[n:]:
        ax.set_visible(False)
    for i, ax in enumerate(axs[:n]):
        if i + ncols >= n:   # nothing below: label the x axis
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel('Age (DPE)')
        if i % ncols == 0:
            ax.set_ylabel("Effect size vs Control\n(Hedges' $g$)")
    return fig, axs[:n]


def run(ds):
    rng = ds.rng()
    measures = ds.measures()
    ages = ds.ages('hedges_g', measures)
    ci = ds.contrast_index(ds.option('hedges_g', 'contrast')) if ds.harmonics else None
    n_boot, n_perm = ds.option('hedges_g', 'n_boot'), ds.option('hedges_g', 'n_perm')

    rows = []
    for g in ds.tests:
        for age in ages:
            for m in measures:
                a, b = m.flies(g.name, age), m.flies(ds.control.name, age)
                if m.kind == 'ssvep':
                    a, b = a[:, ci], b[:, ci]
                if min(len(a), len(b)) < 2:
                    warnings.warn(f'hedges_g: {g.name} {m.label} {age} DPE skipped '
                                  f'({len(a)} vs {len(b)} control flies)')
                    continue
                s, boots = g_with_ci(*((a, b) if m.phenotype == 'increase' else (b, a)), rng, n_boot, n_perm)
                rows.append(dict(genotype=g.name, measure=m.label, age_dpe=age, n_genotype=len(a),
                                 n_control=len(b), **s, boots=boots))
    res = pd.DataFrame(rows)
    res['p_perm_holm'] = holm(res.p_perm)
    res['p_welch_holm'] = holm(res.p_welch)
    res.drop(columns='boots').to_csv(ds.output_dir / 'hedges_g_values.csv', index=False, float_format='%.3f')
    print(res.pivot_table(index=['genotype', 'measure'], columns='age_dpe', values='hedges_g').round(2))

    order = ds.option('hedges_g', 'measures') or [m.key for m in measures]
    by_key = {m.key: m for m in measures}
    unknown = set(order) - set(by_key)
    if unknown:
        raise ValueError(f'[hedges_g] measures: {sorted(unknown)} not among {list(by_key)}')
    plotted = [by_key[k] for k in order]
    style = styles(plotted, ds.harmonics)
    x_age = dict(zip(ages, positions(ages, SPACING)))

    def rows_for(g, m):
        r = res[(res.genotype == g.name) & (res.measure == m.label)]
        return r, r.age_dpe.map(x_age).to_numpy()

    # ---- Points: g with bootstrap 95% CI ------------------------------------------
    fig, axs = panels(len(ds.tests))
    for ax, g, lt in zip(axs, ds.tests, string.ascii_lowercase):
        ax.axhline(0, color=INK, lw=0.6, ls=':', zorder=1)
        for j, m in enumerate(plotted):
            ls, mk, filled, _ = style[m.label]
            r, x = rows_for(g, m)
            ax.errorbar(x + dodge(j, len(plotted), 0.5), r.hedges_g,
                        yerr=[r.hedges_g - r.ci95_lo, r.ci95_hi - r.hedges_g],
                        ls=ls, marker=mk, color=g.colour, ecolor=g.colour, elinewidth=0.7, zorder=3,
                        mfc=g.colour if filled else 'white', mec=g.colour, mew=1.0, ms=4, label=m.label)
        ax.set_title(g.label, color=g.colour)
        age_axis(ax, ages, SPACING, 2.5)
        letter(ax, lt, -0.14)
    handles = [plt.Line2D([], [], ls=ls, marker=mk, color=INK, mfc=INK if f else 'white',
                          mec=INK, mew=1.0, ms=4, label=label) for label, (ls, mk, f, _) in style.items()]
    axs[0].legend(handles=handles, loc='upper right', handlelength=2.2)
    save(fig, ds.output_dir, 'hedges_g_vision_vs_climbing')

    # ---- Boxes: bootstrap distribution of g -----------------------------------------
    # Box = IQR, line = median, whiskers = 2.5-97.5th percentiles (the 95% CI above).
    step = 5.1 / max(len(plotted), 3)
    fig, axs = panels(len(ds.tests))
    for ax, g, lt in zip(axs, ds.tests, string.ascii_lowercase):
        ax.axhline(0, color=INK, lw=0.6, ls=':', zorder=1)
        for j, m in enumerate(plotted):
            _, _, filled, hatch = style[m.label]
            fc, med = (g.colour, 'white') if filled and not hatch else ('white', g.colour)
            r, x = rows_for(g, m)
            if r.empty:
                continue
            x = x + dodge(j, len(plotted), step)
            bp = ax.boxplot(list(r.boots), positions=x, widths=step - 0.2, whis=(2.5, 97.5),
                            showfliers=False, patch_artist=True, manage_ticks=False,
                            boxprops=dict(facecolor=fc, edgecolor=g.colour, lw=0.8, hatch=hatch),
                            medianprops=dict(color=med, lw=1.2),
                            whiskerprops=dict(color=g.colour, lw=0.8), capprops=dict(color=g.colour, lw=0.8))
            for b in bp['boxes']:
                b.set_zorder(3)
            for xi, top, p in zip(x, r.ci95_hi, r.p_perm_holm):
                ax.text(xi, top + 0.1, stars(p), ha='center', va='bottom', fontsize=6, color=g.colour)
        ax.set_title(g.label, color=g.colour)
        age_axis(ax, ages, SPACING, 3.5)
        letter(ax, lt, -0.14)
    plt.rcParams['hatch.color'] = INK
    handles = [plt.matplotlib.patches.Patch(facecolor=INK if f and not h else 'white', edgecolor=INK,
                                            hatch=h, lw=0.8, label=label)
               for label, (_, _, f, h) in style.items()]
    axs[0].legend(handles=handles, loc='upper right')
    save(fig, ds.output_dir, 'hedges_g_vision_vs_climbing_box')
