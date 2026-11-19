"""Figure helpers shared by the figure modules."""
import numpy as np
import matplotlib.pyplot as plt

MM = 1 / 25.4


def positions(ages, spacing):
    """x position of each age. Ages are evenly spaced, so irregular ages (e.g. 1, 7, 14 DPE) stay legible."""
    return np.arange(len(ages)) * spacing


def dodge(i, n, step):
    """Offset of item i of n, centred on 0."""
    return (i - (n - 1) / 2) * step


def age_axis(ax, ages, spacing, margin):
    x = positions(ages, spacing)
    ax.set_xticks(x, [f'{a:g}' for a in ages])
    ax.set_xlim(x[0] - margin, x[-1] + margin)


def log_axis(ax, lo, hi, which='y'):
    """Log axis with ticks at 1, 3, 10, 30, ... between lo and hi."""
    ticks = [m * 10**e for e in range(-3, 6) for m in (1, 3) if lo <= m * 10**e <= hi]
    getattr(ax, f'set_{which}scale')('log')
    getattr(ax, f'set_{which}ticks')(ticks, [f'{t:g}' for t in ticks])
    ax.minorticks_off()


def letter(ax, text, x, y=1.04):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')


def save(fig, out_dir, stem):
    """PDF, EPS and PNG. EPS stores rasterised layers uncompressed, so its raster resolution stays modest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (('pdf', 600), ('eps', 200), ('png', 300)):
        fig.savefig(out_dir / f'{stem}.{ext}', dpi=dpi)
    plt.close(fig)
