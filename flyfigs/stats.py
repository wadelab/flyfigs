"""Effect sizes and tests shared by the figures."""
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def hedges_g(a, b):
    """g for mean(a) - mean(b), pooled SD, small-sample corrected."""
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (n1 + n2) - 9))


def g_with_ci(a, b, rng, n_boot, n_perm):
    """Hedges' g of a - b with a bootstrap 95% CI and two-sided permutation and Welch p values.

    Returns (summary dict, bootstrap distribution of g).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    boots = np.array([hedges_g(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # Two-sided permutation test of g = 0: shuffle fly labels between the two groups
    g, pooled = hedges_g(a, b), np.concatenate([a, b])
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(pooled)
        null[i] = hedges_g(p[:len(a)], p[len(a):])
    p_perm = (np.sum(np.abs(null) >= abs(g)) + 1) / (n_perm + 1)
    p_welch = stats.ttest_ind(a, b, equal_var=False).pvalue
    return dict(hedges_g=g, ci95_lo=lo, ci95_hi=hi, p_perm=p_perm, p_welch=p_welch), boots


def holm(p):
    return multipletests(p, method='holm')[1]


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
