"""Checks that the fits, effect sizes and the example dataset behave as documented.

Run with `uv run pytest`. The last test makes one figure from the example dataset, so
it takes about ten seconds.
"""
from pathlib import Path

import numpy as np
import pytest

from flyfigs.cli import main
from flyfigs.config import Dataset, spaced_ticks
from flyfigs.crf import CRF, hr
from flyfigs.stats import hedges_g, holm, stars

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'gal80_hsnca' / 'gal80_hsnca.toml'
CONTRASTS = [5, 10, 30, 70, 100]


def test_hr_fit_recovers_known_parameters():
    crf = CRF(CONTRASTS)
    y = hr(crf.c, 120.0, 25.0)
    rmax, c50 = crf.fit('hr', y)
    assert rmax == pytest.approx(120.0, rel=1e-4)
    assert c50 == pytest.approx(25.0, rel=1e-4)


def test_fixed_parameter_fits_keep_the_fixed_value():
    crf = CRF(CONTRASTS)
    y = hr(crf.c, 120.0, 25.0)
    rmax, c50 = crf.fit('fixRmax', y, pooled=(100.0, 25.0))
    assert rmax == 100.0
    rmax, c50 = crf.fit('fixc50', y, pooled=(120.0, 40.0))
    assert c50 == 40.0


def test_grid_c50_matches_curve_fit():
    crf = CRF(CONTRASTS)
    y = hr(crf.c, 120.0, 25.0)
    grid = crf.grid_c50(y[None, :], 120.0)[0]
    assert grid == pytest.approx(25.0, rel=1e-2)   # grid spacing limits the precision


def test_hedges_g_known_value():
    # Two groups with means 1 and 0, pooled SD 1, n = 10 each: d = 1, J = 1 - 3/(4*20-9) = 0.9577
    a = np.array([0.5, 1.5] * 5)
    b = a - 1.0
    g = hedges_g(a, b)
    sp = np.sqrt(a.var(ddof=1))
    assert g == pytest.approx(1 / sp * (1 - 3 / (4 * 20 - 9)))


def test_holm_and_stars():
    assert stars(0.0005) == '***' and stars(0.005) == '**' and stars(0.04) == '*' and stars(0.2) == ''
    adjusted = holm([0.01, 0.04, 0.03])
    assert adjusted[0] == pytest.approx(0.03)        # 0.01 * 3
    assert all(adjusted >= [0.01, 0.04, 0.03])


def test_spaced_ticks():
    assert spaced_ticks([5, 10, 30, 70, 100]) == [5, 10, 30, 100]


def test_example_dataset_loads_with_all_genotypes():
    ds = Dataset(EXAMPLE)
    assert [g.name for g in ds.genotypes] == ['LacZ', 'SNCAWT', 'A30P', 'A53T']
    assert ds.control.name == 'LacZ'
    assert [m.key for m in ds.measures()] == ['1F1', '2F1', 'climbing']
    for m in ds.measures():
        counts = m.data.groupby(['genotype', 'age']).size()
        assert (counts > 0).all()
        assert set(m.data.genotype) == {'LacZ', 'SNCAWT', 'A30P', 'A53T'}


def test_quick_run_writes_figure_and_table(tmp_path):
    main([str(EXAMPLE), 'climbing_summary', '--quick', '--out', str(tmp_path)])
    for ext in ('pdf', 'eps', 'png'):
        assert (tmp_path / f'climbing_summary.{ext}').stat().st_size > 0
    table = (tmp_path / 'climbing_summary_tests.csv').read_text()
    assert 'A53T' in table
