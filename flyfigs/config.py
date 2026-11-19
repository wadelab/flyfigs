"""Load a dataset description (TOML) and the per-fly data it points to.

A dataset config names the data files, the genotypes to analyse (one of them the
control), the ages, and per-figure options. See
examples/gal80_hsnca/gal80_hsnca.toml for an annotated example.
"""
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Per-figure options and their defaults. Every figure section may also set `ages`.
DEFAULTS = {
    'hedges_g': dict(contrast=30, n_boot=5000, n_perm=10000, measures=None),
    'climbing_summary': {},
    'crf_fits': dict(n_boot=300, models=['hr', 'ss', 'fixc50', 'fixRmax']),
    'crf_param_boxes': dict(n_boot=1000),
    'crf_combined': dict(n_boot=1000, n_perm=50000),
}
QUICK = dict(n_boot=100, n_perm=1000)   # caps applied by --quick

# Keys each part of the config may use, to catch typos
KEYS = {
    None: {'output_dir', 'ages', 'seed', 'style', 'ssvep', 'climbing', 'genotypes', *DEFAULTS},
    'ssvep': {'contrasts', 'columns', 'contrast_ticks', 'units', 'phenotype', 'genotype_column',
              'age_column', 'query', 'harmonics'},
    'harmonic': {'code', 'label', 'file'},
    'climbing': {'file', 'measure', 'label', 'ylabel', 'phenotype', 'genotype_column', 'age_column',
                 'query'},
    'genotype': {'name', 'label', 'colour', 'marker', 'control', 'codes'},
    **{s: {'ages', *d} for s, d in DEFAULTS.items()},
}


@dataclass
class Genotype:
    name: str      # identifier used in the output tables
    label: str     # legend and title text (matplotlib mathtext allowed)
    colour: str
    marker: str    # CRF marker
    codes: dict    # data kind ('ssvep', 'climbing') -> this genotype's code in that kind's files


@dataclass
class Measure:
    """Per-fly data for one measure: an SSVEP harmonic or climbing.

    `data` has columns 'genotype' (config names), 'age' and the value column(s):
    one per contrast for SSVEP (`values` is a list), one for climbing (a str).
    """
    key: str           # '1F1', '2F1', ... or 'climbing'
    label: str         # e.g. 'Photoreceptor', 'Climbing'
    kind: str          # 'ssvep' or 'climbing'
    phenotype: str     # direction of the disease phenotype: 'increase' or 'decrease'
    ylabel: str
    data: pd.DataFrame
    values: list | str

    @property
    def title(self):
        return f'{self.label} ({self.key})' if self.kind == 'ssvep' else self.label

    def flies(self, genotype, age):
        """One genotype x age: flies x contrasts (SSVEP) or one value per fly (climbing)."""
        d = self.data
        return d[(d.genotype == genotype) & (d.age == age)][self.values].to_numpy()

    def pooled_mean(self, ages):
        """Mean response at each contrast over every fly of every genotype at these ages."""
        return self.data[self.data.age.isin(ages)][self.values].to_numpy().mean(axis=0)


def spaced_ticks(contrasts, ratio=1.5):
    """Contrasts at least `ratio` apart on a log axis, keeping the highest: 5-100% -> 5, 10, 30, 100."""
    ticks = []
    for c in sorted(contrasts, reverse=True):
        if not ticks or c * ratio <= ticks[-1]:
            ticks.append(c)
    return ticks[::-1]


def _check_keys(section, table, where):
    for key in set(table) - KEYS[section]:
        warnings.warn(f'{where}: unknown key {key!r} ignored', stacklevel=3)


class Dataset:
    def __init__(self, path, quick=False, output_dir=None):
        self.path = Path(path).resolve()
        with open(self.path, 'rb') as f:
            cfg = self.cfg = tomllib.load(f)
        here = self.path.parent
        _check_keys(None, cfg, self.path.name)
        for section in DEFAULTS:
            _check_keys(section, cfg.get(section, {}), f'[{section}]')

        self.quick = quick
        out = Path(output_dir) if output_dir else here / cfg.get('output_dir', '.')
        self.output_dir = out / 'quick' if quick and not output_dir else out
        self.style = here / cfg['style'] if 'style' in cfg else Path(__file__).with_name('wadelab.mplstyle')
        self.seed = cfg.get('seed', 1)

        self.genotypes, controls = [], []
        for i, g in enumerate(cfg['genotypes']):
            _check_keys('genotype', g, f'genotype {g.get("name", i + 1)!r}')
            self.genotypes.append(Genotype(g['name'], g.get('label', g['name']), g.get('colour', f'C{i}'),
                                           g.get('marker', 'o'), g.get('codes', {})))
            if g.get('control'):
                controls.append(self.genotypes[-1])
        if len(controls) != 1:
            raise ValueError(f'{self.path.name}: exactly one genotype needs control = true')
        self.control = controls[0]
        self.tests = [g for g in self.genotypes if g is not self.control]

        self.harmonics = []
        if 'ssvep' in cfg:
            s = cfg['ssvep']
            _check_keys('ssvep', s, '[ssvep]')
            self.contrasts = np.array(s['contrasts'], float)
            self.columns = s.get('columns', [f'x{c:g}' for c in s['contrasts']])
            self.contrast_ticks = s.get('contrast_ticks', spaced_ticks(s['contrasts']))
            self.units = s.get('units', 'AU')
            for h in s['harmonics']:
                _check_keys('harmonic', h, f'[[ssvep.harmonics]] {h.get("code")!r}')
                d = self._load(h['file'], 'ssvep', s.get('genotype_column', 'Genotype'),
                               s.get('age_column', 'Age'), s.get('query'), self.columns)
                self.harmonics.append(Measure(h['code'], h.get('label', h['code']), 'ssvep',
                                              s.get('phenotype', 'increase'), f'response ({self.units})',
                                              d, self.columns))
        self.climbing = None
        if 'climbing' in cfg:
            c = cfg['climbing']
            _check_keys('climbing', c, '[climbing]')
            d = self._load(c['file'], 'climbing', c.get('genotype_column', 'Genotype'),
                           c.get('age_column', 'Age dpe'), c.get('query'), [c['measure']])
            self.climbing = Measure('climbing', c.get('label', 'Climbing'), 'climbing',
                                    c.get('phenotype', 'decrease'), c.get('ylabel', c['measure']),
                                    d, c['measure'])
        for m in self.measures():
            if m.phenotype not in ('increase', 'decrease'):
                raise ValueError(f'{m.title}: phenotype must be "increase" or "decrease"')

    def _load(self, file, kind, genotype_column, age_column, query, values):
        path = self.path.parent / file
        d = pd.read_csv(path, encoding='utf-8-sig')
        missing = [c for c in [genotype_column, age_column, *values] if c not in d.columns]
        if missing:
            raise KeyError(f'{path.name} has no column(s) {missing}; it has {list(d.columns)}')
        if query:
            d = d.query(query)
        names = {g.codes.get(kind, g.name): g.name for g in self.genotypes}
        d = d[d[genotype_column].isin(list(names))].copy()
        d['genotype'] = d[genotype_column].map(names)
        d['age'] = d[age_column]
        for code, name in names.items():
            if name not in d.genotype.values:
                warnings.warn(f'{path.name}: no flies with {genotype_column} == {code!r} '
                              f'(genotype {name!r})', stacklevel=2)
        return d

    def measures(self):
        return self.harmonics + ([self.climbing] if self.climbing else [])

    def option(self, section, key):
        value = self.cfg.get(section, {}).get(key, DEFAULTS[section][key])
        return min(value, QUICK[key]) if self.quick and key in QUICK else value

    def ages(self, section, measures):
        """Ages for a figure: the section's `ages`, else the top-level `ages`, else every age in the data."""
        ages = self.cfg.get(section, {}).get('ages', self.cfg.get('ages', 'all'))
        present = sorted(set().union(*(m.data.age.unique().tolist() for m in measures)))
        if ages == 'all':
            return present
        absent = [a for a in ages if a not in present]
        if absent:
            warnings.warn(f'[{section}]: no data at ages {absent}', stacklevel=2)
        return [a for a in ages if a in present]

    def contrast_index(self, contrast):
        if contrast not in self.contrasts:
            raise ValueError(f'contrast {contrast} is not one of {self.contrasts.tolist()}')
        return int(np.flatnonzero(self.contrasts == contrast)[0])

    def rng(self):
        return np.random.default_rng(self.seed)

    def describe(self):
        """Print flies per genotype x age for each measure, to check the config matched the data."""
        print(f'{self.path.name} -> {self.output_dir}')
        names = [g.name for g in self.genotypes]
        for m in self.measures():
            n = pd.crosstab(m.data.genotype, m.data.age).reindex(names, fill_value=0)
            print(f'\n{m.title}: flies per genotype x age\n{n.to_string()}')
        print()
