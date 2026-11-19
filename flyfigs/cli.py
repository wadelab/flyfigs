"""Make the analysis figures for one fly dataset.

    flyfigs DATASET.toml [FIGURE ...] [--quick] [--out DIR]

FIGURE is any of the names below (default: every figure the dataset has data for).
"""
import argparse
import warnings

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

from .config import QUICK, Dataset   # noqa: E402
from .figures import climbing_summary, crf_combined, crf_fits, crf_param_boxes, hedges_g   # noqa: E402

# name: (module, does the dataset have the data it needs?)
FIGURES = {
    'hedges_g': (hedges_g, lambda ds: bool(ds.measures())),
    'climbing_summary': (climbing_summary, lambda ds: ds.climbing is not None),
    'crf_fits': (crf_fits, lambda ds: bool(ds.harmonics)),
    'crf_param_boxes': (crf_param_boxes, lambda ds: bool(ds.harmonics)),
    'crf_combined': (crf_combined, lambda ds: bool(ds.harmonics)),
}


def main(argv=None):
    warnings.formatwarning = lambda message, *args, **kwargs: f'warning: {message}\n'
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', help='dataset config (.toml)')
    parser.add_argument('figures', nargs='*', metavar='FIGURE', help=', '.join(FIGURES))
    parser.add_argument('--quick', action='store_true',
                        help=f'cap resamples at {QUICK["n_boot"]} bootstraps and {QUICK["n_perm"]} permutations '
                             'and write to <output_dir>/quick, to check a new config')
    parser.add_argument('--out', help='output directory (default: output_dir in the config)')
    args = parser.parse_intermixed_args(argv)   # figure names may follow options
    unknown = set(args.figures) - set(FIGURES)
    if unknown:
        parser.error(f'unknown figure(s) {sorted(unknown)}; choose from {list(FIGURES)}')

    ds = Dataset(args.config, quick=args.quick, output_dir=args.out)
    ds.describe()
    ds.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.figures or FIGURES:
        module, has_data = FIGURES[name]
        if not has_data(ds):
            print(f'--- {name}: skipped, no data for it in {ds.path.name}')
            continue
        print(f'--- {name}')
        with plt.rc_context():
            plt.style.use(ds.style)
            module.run(ds)
