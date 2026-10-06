"""
campaign.py
-----------
Batch processing of many spectra: fit every line in every spectrum in parallel
(CPU processes), save each spectrum's full result as JSON for the /v5 page,
join observation dates, and write the per-line / per-year tables and workbook.
"""
import contextlib
import functools
import io
import json
import os
from multiprocessing import Pool
from typing import Dict, List, Optional

import pandas as pd


def default_workers() -> int:
    """All but two logical cores, capped at 10 so memory stays well below the limit."""
    return max(1, min((os.cpu_count() or 2) - 2, 10))


def fit_one(path: str, method: str = 'refine', json_dir: Optional[str] = None) -> List[Dict]:
    from src.pipeline import run_single_spectrum_pipeline
    from src.serialize import serialize_record
    name = os.path.basename(path)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_single_spectrum_pipeline(path, method=method)
    except Exception as e:                       # keep the batch going; report the failure
        return [dict(spectrum=name, line='(failed)', error=str(e), detected=False)]
    if json_dir:
        rec = serialize_record(res)
        with open(os.path.join(json_dir, name + '.json'), 'w', encoding='utf-8') as fh:
            json.dump(rec, fh, separators=(',', ':'))
    rows = []
    for ln in res['lines']:
        v = ln.get('verification') or {}
        rows.append(dict(
            spectrum=name, line=ln['line_name'],
            rest_wavelength=ln.get('rest_wavelength'), detected=bool(ln.get('detected')),
            refined=bool(ln.get('refined')), status=ln.get('fit_status', ''),
            center=ln.get('center'), center_err=ln.get('center_err'),
            sigma=ln.get('sigma'), sigma_err=ln.get('sigma_err'),
            fwhm_kms=ln.get('fwhm_kms'), fwhm_kms_intrinsic=ln.get('fwhm_kms_intrinsic'),
            fwhm_ang_err=ln.get('fwhm_ang_err'), fwhm_kms_err=ln.get('fwhm_kms_err'),
            flux=ln.get('flux'), flux_err=ln.get('flux_err'), significance=ln.get('snr'),
            reduced_chi2=ln.get('reduced_chi2'), at_bound=ln.get('at_bound'),
            flux_window=ln.get('flux_window'), wing_window=ln.get('wing_window'),
            ew=ln.get('ew'), ew_err=ln.get('ew_err'),
            verification=v.get('status'), failed_checks='; '.join(v.get('failed', [])),
            min_wavelength=ln.get('min_wavelength'), max_wavelength=ln.get('max_wavelength'),
            spectral_index=res.get('spectral_index'), spectral_index_err=res.get('spectral_index_err'),
            continuum_amplitude=res.get('continuum_amplitude'),
            continuum_amplitude_err=res.get('continuum_amplitude_err')))
    return rows


def run_batch(files: List[str], out_dir: str, method: str = 'refine', workers: Optional[int] = None,
              jd_path: Optional[str] = None, json_dir: Optional[str] = None,
              fits: Optional[pd.DataFrame] = None, progress=None) -> Dict[str, pd.DataFrame]:
    """
    Fit `files` (or reuse `fits`), date them from jd.xlsx, compute F_avg / F_var / R_max per line per year,
    and write line_fluxes.csv, light_curves.csv, fvar_by_year.csv and variability_results.xlsx to out_dir.
    `progress(done, total)` is called as spectra finish.
    """
    from src.variability import load_jd, fvar_by_year, light_curve_points
    from src.excel_export import write_variability_workbook

    os.makedirs(out_dir, exist_ok=True)
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)
    if fits is None:
        workers = workers or default_workers()
        rows = []
        work = functools.partial(fit_one, method=method, json_dir=json_dir)
        with Pool(workers) as pool:
            for i, chunk in enumerate(pool.imap_unordered(work, files, chunksize=2), start=1):
                rows.extend(chunk)
                if progress:
                    progress(i, len(files))
        fits = pd.DataFrame(rows)
        fits = fits[fits['line'] != '(failed)'] if 'error' in fits else fits

    if jd_path and os.path.exists(jd_path):
        # match on the SWP image number so renamed files still get their observation date
        key = lambda s: s.str.lower().str.extract(r'(swp\d+)', expand=False).fillna(s.str.lower())
        jd = load_jd(jd_path)
        jd = jd.assign(_swp=key(jd['spectrum']))[['_swp', 'jd', 'date', 'year']].drop_duplicates('_swp')
        fits = fits.assign(_swp=key(fits['spectrum'])).merge(jd, on='_swp', how='left').drop(columns='_swp')
    else:
        fits = fits.assign(jd=float('nan'), date=pd.NaT, year=float('nan'))

    fits = fits.sort_values(['line', 'jd', 'spectrum'])
    fits.to_csv(os.path.join(out_dir, 'line_fluxes.csv'), index=False)
    used = fits[fits['detected'].astype(bool) & fits['jd'].notna()]
    curves = light_curve_points(used) if len(used) else used
    curves.to_csv(os.path.join(out_dir, 'light_curves.csv'), index=False)
    table = (fvar_by_year(used, err_mode='combined' if method == 'manuscript' else 'piecewise')
             if len(used) else pd.DataFrame())
    table.to_csv(os.path.join(out_dir, 'fvar_by_year.csv'), index=False)
    write_variability_workbook(os.path.join(out_dir, 'variability_results.xlsx'), fits, table, curves, method)
    return {'fits': fits, 'table': table, 'curves': curves}
