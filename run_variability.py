"""
Fit every emission line in every spectrum under spectrum/, attach observation
JDs from jd.xlsx (SWP sheet), and compute F_avg, F_var and R_max per line per calendar year.

Methods
  refine      (default) weighted least squares, Gaussian + local continuum  -> outputs/
  manuscript  method of the 3C 390.3 manuscript (src/manuscript_fit.py)      -> outputs/manuscript/

Outputs (in the output folder)
  line_fluxes.csv            one row per (spectrum, line)
  light_curves.csv           detected fluxes: JD, JD - JD_min, F, F_err
  fvar_by_year.csv           one row per (line, year)
  variability_results.xlsx   organised workbook (src/excel_export.py)
  campaign/*.json            (refine method) full per-spectrum results for the /v5 page

Usage: python run_variability.py [--method manuscript] [--workers N] [--reuse]
"""
import argparse
import glob
import os

import pandas as pd

from src.campaign import run_batch, default_workers

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIRS = {'refine': os.path.join(ROOT, 'outputs'), 'manuscript': os.path.join(ROOT, 'outputs', 'manuscript')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=sorted(OUT_DIRS), default='refine')
    ap.add_argument('--workers', type=int, default=default_workers())
    ap.add_argument('--spectra', default=os.path.join(ROOT, 'spectrum'))
    ap.add_argument('--jd', default=os.path.join(ROOT, 'jd.xlsx'))
    ap.add_argument('--reuse', action='store_true',
                    help='skip fitting; recompute F_var from the existing line_fluxes.csv')
    args = ap.parse_args()
    out = OUT_DIRS[args.method]

    fits = None
    if args.reuse:
        fits = pd.read_csv(os.path.join(out, 'line_fluxes.csv'))
        fits = fits.drop(columns=[c for c in ('jd', 'date', 'year') if c in fits])
    files = sorted(glob.glob(os.path.join(args.spectra, '*.txt')))
    if fits is None:
        print(f'Fitting {len(files)} spectra ({args.method} method) with {args.workers} worker processes ...')

    def progress(i, n):
        if i % 25 == 0 or i == n:
            print(f'  {i}/{n} spectra', flush=True)

    json_dir = os.path.join(out, 'campaign') if args.method == 'refine' else None
    res = run_batch(files, out, args.method, args.workers, args.jd, json_dir, fits, progress)
    undated = sorted(res['fits'].loc[res['fits']['jd'].isna(), 'spectrum'].unique())
    if undated:
        print(f'{len(undated)} spectra have no observation date in jd.xlsx and are excluded from F_var: '
              + ', '.join(undated))
    print(f"Wrote line_fluxes.csv, light_curves.csv, fvar_by_year.csv, variability_results.xlsx in {out}")


if __name__ == '__main__':
    main()
