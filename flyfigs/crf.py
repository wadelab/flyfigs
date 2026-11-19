"""Contrast-response functions: hyperbolic ratio models and their fits.

Contrasts are in % (e.g. 5-100). Bounds and starting values follow the contrasts tested:
c50 is bounded to [lowest contrast / 5, 3 x highest contrast] (beyond ~3x the highest
contrast c50 is not identifiable) and starts at the median contrast.

Models:
  hr      : Rmax * c^2 / (c50^2 + c^2)          -- hyperbolic ratio
  ss      : Rmax * c^2 / (c50^q + c^q), q >= 2  -- supersaturating form (Peirce 2007),
            which lets the response fall at high contrast
  fixc50  : hr with c50 fixed (Rmax free)
  fixRmax : hr with Rmax fixed (c50 free)
"""
import numpy as np
from scipy.optimize import curve_fit

MODELS = ('hr', 'ss', 'fixc50', 'fixRmax')
RMAX_MAX = 1e4
MAXFEV = 20000


def hr(c, rmax, c50):
    return rmax * c**2 / (c50**2 + c**2)


def ss(c, rmax, c50, q):
    return rmax * c**2 / (c50**q + c**q)


class CRF:
    def __init__(self, contrasts):
        self.c = np.asarray(contrasts, float)
        lo, hi = self.c.min(), self.c.max()
        self.c50_bounds = (lo / 5, 3 * hi)
        self.c50_start = float(np.median(self.c))
        self.ss_c50_max = 10 * hi
        self.c_fine = np.geomspace(lo * 4 / 5, hi * 11 / 10, 200)    # for drawing curves
        self.xlim = (lo * 4 / 5, hi * 6 / 5)
        self.c50_grid = np.geomspace(*self.c50_bounds, 3000)         # for grid_c50

    def fit(self, model, y, pooled=None):
        """Least-squares fit of `model` to y (one response per contrast).

        Returns every parameter, fixed ones included: (Rmax, c50), or (Rmax, c50, q) for
        'ss'. For 'fixc50' and 'fixRmax', `pooled` = (Rmax, c50) supplies the fixed value.
        Raises RuntimeError if the fit does not converge.
        """
        lo, hi = self.c50_bounds
        if model == 'hr':
            p, _ = curve_fit(hr, self.c, y, p0=[y.max(), self.c50_start],
                             bounds=([0, lo], [RMAX_MAX, hi]), maxfev=MAXFEV)
            return p
        if model == 'ss':
            p, _ = curve_fit(ss, self.c, y, p0=[y.max(), self.c50_start, 2.5],
                             bounds=([0, lo, 2], [1e6, self.ss_c50_max, 6]), maxfev=MAXFEV)
            return p
        rmax, c50 = pooled
        if model == 'fixc50':
            p, _ = curve_fit(lambda c, r: hr(c, r, c50), self.c, y, p0=[y.max()],
                             bounds=([0], [RMAX_MAX]), maxfev=MAXFEV)
            return np.array([p[0], c50])
        if model == 'fixRmax':
            p, _ = curve_fit(lambda c, x: hr(c, rmax, x), self.c, y, p0=[self.c50_start],
                             bounds=([lo], [hi]), maxfev=MAXFEV)
            return np.array([rmax, p[0]])
        raise ValueError(f'unknown CRF model {model!r}; choose from {MODELS}')

    def curve(self, model, p, c=None):
        """Model response at contrasts c (default: the fine grid for drawing)."""
        return (ss if model == 'ss' else hr)(self.c_fine if c is None else c, *p)

    def r_squared(self, model, y, p):
        return 1 - np.sum((y - self.curve(model, p, self.c))**2) / np.sum((y - y.mean())**2)

    # ---- Permutation test of c50 with Rmax fixed ---------------------------------
    # With Rmax fixed the fit has one parameter, so fit by vectorised grid search over
    # log-spaced c50 values; this fits thousands of permuted group means at once.

    def grid_c50(self, Y, rmax):
        """Least-squares c50 for each row of Y (n x contrasts group means), Rmax fixed."""
        G = hr(self.c[None, :], rmax, self.c50_grid[:, None])                 # grid x contrasts
        sse = (G**2).sum(1)[:, None] - 2 * G @ Y.T + (Y**2).sum(1)[None, :]   # grid x n
        return self.c50_grid[np.argmin(sse, axis=0)]

    def perm_test_c50(self, a, b, rmax, rng, n_perm, chunk=5000):
        """Two-sided permutation test of log c50(a) - log c50(b); a, b are flies x contrasts.

        Returns (observed log ratio, p).
        """
        obs = np.log(self.grid_c50(a.mean(0, keepdims=True), rmax)
                     / self.grid_c50(b.mean(0, keepdims=True), rmax))[0]
        pooled, na = np.vstack([a, b]), len(a)
        null = []
        for start in range(0, n_perm, chunk):
            m = min(chunk, n_perm - start)
            idx = np.argsort(rng.random((m, len(pooled))), axis=1)
            grp = pooled[idx]                                                 # m x N x contrasts
            null.append(np.log(self.grid_c50(grp[:, :na].mean(1), rmax)
                               / self.grid_c50(grp[:, na:].mean(1), rmax)))
        null = np.concatenate(null)
        return obs, (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (n_perm + 1)
