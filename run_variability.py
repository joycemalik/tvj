"""
Fit every emission line in every spectrum under spectrum/, attach observation
JDs from jd.xlsx (SWP sheet), and compute F_var per line per calendar year.

Outputs:
  outputs/line_fluxes.csv   one row per (spectrum, line)
  outputs/fvar_by_year.csv  one row per (line, year)

Usage: python run_variability.py [--workers 3]
"""
import argparse
import contextlib
import glob
import io
import os
from multiprocessing import Pool

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'outputs')


def fit_one(path):
    from src.pipeline import run_single_spectrum_pipeline
    with contextlib.redirect_stdout(io.StringIO()):
        res = run_single_spectrum_pipeline(path)
    rows = []
    for ln in res['lines']:
        rows.append(dict(
            spectrum=os.path.basename(path), line=ln['line_name'],
            rest_wavelength=ln.get('rest_wavelength'), detected=bool(ln.get('detected')),
            refined=bool(ln.get('refined')), status=ln.get('fit_status', ''),
            center=ln.get('center'), center_err=ln.get('center_err'),
            sigma=ln.get('sigma'), sigma_err=ln.get('sigma_err'),
            fwhm_kms=ln.get('fwhm_kms'), fwhm_kms_intrinsic=ln.get('fwhm_kms_intrinsic'),
            flux=ln.get('flux'), flux_err=ln.get('flux_err'), significance=ln.get('snr'),
            reduced_chi2=ln.get('reduced_chi2'), at_bound=ln.get('at_bound'),
            flux_window=ln.get('flux_window'), wing_window=ln.get('wing_window'),
            ew=ln.get('ew'), ew_err=ln.get('ew_err'),
            verification=(ln.get('verification') or {}).get('status'),
            failed_checks='; '.join((ln.get('verification') or {}).get('failed', [])),
            min_wavelength=ln.get('min_wavelength'), max_wavelength=ln.get('max_wavelength'),
            spectral_index=res.get('spectral_index'), spectral_index_err=res.get('spectral_index_err'),
            continuum_amplitude=res.get('continuum_amplitude'),
            continuum_amplitude_err=res.get('continuum_amplitude_err')))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--spectra', default=os.path.join(ROOT, 'spectrum'))
    ap.add_argument('--jd', default=os.path.join(ROOT, 'jd.xlsx'))
    ap.add_argument('--reuse', action='store_true',
                    help='skip fitting; recompute F_var from the existing outputs/line_fluxes.csv')
    args = ap.parse_args()

    from src.variability import load_jd, fvar_by_year, light_curve_points

    if args.reuse:
        fits = pd.read_csv(os.path.join(OUT, 'line_fluxes.csv'))
        fits = fits.drop(columns=[c for c in ('jd', 'date', 'year') if c in fits])
    else:
        files = sorted(glob.glob(os.path.join(args.spectra, '*.txt')))
        print(f'Fitting {len(files)} spectra with {args.workers} workers ...')
        with Pool(args.workers) as pool:
            rows = [r for chunk in pool.imap_unordered(fit_one, files, chunksize=4) for r in chunk]
        fits = pd.DataFrame(rows)

    jd = load_jd(args.jd)
    fits = fits.merge(jd[['spectrum', 'jd', 'date', 'year']], on='spectrum', how='left')
    undated = sorted(fits.loc[fits['jd'].isna(), 'spectrum'].unique())
    if undated:
        print(f'{len(undated)} spectra have no observation date in jd.xlsx and are excluded from F_var: '
              + ', '.join(undated))

    os.makedirs(OUT, exist_ok=True)
    fits = fits.sort_values(['line', 'jd'])
    fits.to_csv(os.path.join(OUT, 'line_fluxes.csv'), index=False)

    used = fits[fits['detected'] & fits['jd'].notna()]
    curves = light_curve_points(used)
    curves.to_csv(os.path.join(OUT, 'light_curves.csv'), index=False)
    table = fvar_by_year(used)
    table.to_csv(os.path.join(OUT, 'fvar_by_year.csv'), index=False)

    xlsx = os.path.join(OUT, 'variability_results.xlsx')
    with pd.ExcelWriter(xlsx) as xw:
        table.to_excel(xw, sheet_name='Fvar by year', index=False)
        curves.to_excel(xw, sheet_name='Light curves', index=False)
        fits.assign(date=fits['date'].dt.strftime('%Y-%m-%d')).to_excel(xw, sheet_name='Line fits', index=False)

    with pd.option_context('display.width', 200, 'display.max_rows', 500):
        print(table.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print(f"\nWrote line_fluxes.csv, light_curves.csv, fvar_by_year.csv, variability_results.xlsx in {OUT}")


if __name__ == '__main__':
    main()
