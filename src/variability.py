"""
variability.py
--------------
Fractional variability amplitude F_var of emission-line light curves,
computed separately for every line and every calendar year.

    F̄       = (1/N) Σ F_i
    σ²_F     = 1/(N-1) Σ (F_i - F̄)²          (sample variance)
    ΔF̄²     = (1/N) Σ ΔF_i²                  (mean square measurement error)
    F_var    = √(σ²_F - ΔF̄²) / F̄

Error (Vaughan et al. 2003, MNRAS 345, 1271; Sukanya et al. 2018), piecewise:

    σ²_F / ΔF̄² <  cutoff  (σ²_F ≈ ΔF̄²):  err = √(1/2N) · ΔF̄² / (F̄² F_var)
    σ²_F / ΔF̄² >= cutoff  (σ²_F ≫ ΔF̄²):  err = √(ΔF̄²/N) / F̄

When σ²_F <= ΔF̄² the scatter is consistent with measurement noise alone and
F_var is undefined (reported as NaN, "not variable").
"""

import csv
import math
import os
from typing import Dict, List, Optional

import numpy as np

FVAR_ERR_CUTOFF = 10.0


def fvar(flux, err, cutoff: float = FVAR_ERR_CUTOFF) -> Dict[str, float]:
    F = np.asarray(flux, dtype=float)
    E = np.asarray(err, dtype=float)
    ok = np.isfinite(F) & np.isfinite(E)
    F, E = F[ok], E[ok]
    n = len(F)
    nan = float('nan')
    out = dict(n=n, f_mean=nan, err_rms=nan, variance=nan, mse_err=nan, ratio=nan,
               fvar=nan, fvar_err=nan, regime='')
    if n < 2:
        out['regime'] = 'N < 2'
        return out
    f_mean = float(F.mean())
    variance = float(F.var(ddof=1))
    mse = float(np.mean(E ** 2))
    ratio = variance / mse if mse > 0 else float('inf')
    out.update(f_mean=f_mean, err_rms=math.sqrt(mse), variance=variance, mse_err=mse, ratio=ratio)
    excess = variance - mse
    if excess <= 0 or f_mean <= 0:
        out['regime'] = 'not variable'
        return out
    fv = math.sqrt(excess) / f_mean
    if ratio < cutoff:
        fv_err = math.sqrt(1.0 / (2 * n)) * mse / (f_mean ** 2 * fv)
        regime = 'S² ≈ σ²err'
    else:
        fv_err = math.sqrt(mse / n) / f_mean
        regime = 'S² ≫ σ²err'
    out.update(fvar=fv, fvar_err=fv_err, regime=regime)
    return out


def load_jd(path: str = 'jd.xlsx'):
    """SWP spectra (Sheet2) -> spectrum, jd, date, year. Rows without a date are dropped."""
    import pandas as pd
    raw = pd.read_excel(path, sheet_name='Sheet2', header=2)
    raw = raw.rename(columns=lambda c: str(c).strip())
    df = pd.DataFrame({
        'spectrum': raw['Spectrum'].astype(str).str.strip(),
        'jd': pd.to_numeric(raw['LJD-OBS'], errors='coerce'),
        'date': pd.to_datetime(raw['LDATEOBS'], errors='coerce'),
    })
    df = df[df['spectrum'].str.contains('dcdr', case=False, na=False)]
    df = df.dropna(subset=['jd', 'date']).copy()
    df['year'] = df['date'].dt.year.astype(int)
    return df.reset_index(drop=True)


def fvar_by_year(line_fluxes, cutoff: float = FVAR_ERR_CUTOFF):
    """
    `line_fluxes` columns: line, year, jd, flux, flux_err (detected measurements only),
    optionally rest_wavelength (rows are then ordered by wavelength).
    One row per (line, year) with F_var ± err and the inputs used.
    """
    rows: List[Dict] = []
    if 'rest_wavelength' in line_fluxes:
        rest = line_fluxes.groupby('line')['rest_wavelength'].first()
        line_order = list(rest.sort_values().index)
    else:
        line_order = sorted(line_fluxes['line'].unique())
    for (line, year), g in sorted(line_fluxes.groupby(['line', 'year']),
                                  key=lambda kv: (line_order.index(kv[0][0]), kv[0][1])):
        r = fvar(g['flux'], g['flux_err'], cutoff)
        jd0 = float(g['jd'].min())
        rows.append(dict(line=line, year=int(year), n=r['n'],
                         jd_min=jd0, jd_span=float(g['jd'].max() - jd0),
                         f_mean=r['f_mean'], err_rms=r['err_rms'],
                         ratio=r['ratio'], fvar=r['fvar'], fvar_err=r['fvar_err'],
                         regime=r['regime']))
    import pandas as pd
    return pd.DataFrame(rows)


def light_curve_points(line_fluxes):
    """(JD − JD_min, F, F_err) per line, JD_min taken over the whole campaign for that line."""
    lf = line_fluxes.copy()
    lf['jd_minus_min'] = lf['jd'] - lf.groupby('line')['jd'].transform('min')
    return lf[['line', 'year', 'spectrum', 'jd', 'jd_minus_min', 'flux', 'flux_err']]


def spectrum_context(spectrum: str, outputs_dir: Optional[str] = None) -> Dict:
    """
    Place one spectrum in the campaign light curves: its observation date (from the
    batch run joined with jd.xlsx), and for every line the detected fluxes of the same
    calendar year plus that year's F_var row. Matching is by SWP image number
    (e.g. 'swp35476'), so renamed uploads of a campaign file still match.
    """
    import re
    outputs_dir = outputs_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    path = os.path.join(outputs_dir, 'line_fluxes.csv')
    m = re.search(r'swp\d+', spectrum.lower())
    key = m.group(0) if m else spectrum.lower()
    if not os.path.exists(path):
        return {'matched': False, 'reason': 'outputs/line_fluxes.csv not found — run `python run_variability.py`'}

    def num(v):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        return x if math.isfinite(x) else None

    with open(path, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    own = [r for r in rows if key in r['spectrum'].lower()]
    if not own or num(own[0]['jd']) is None:
        return {'matched': bool(own), 'spectrum': own[0]['spectrum'] if own else spectrum,
                'reason': 'no observation date for this spectrum in jd.xlsx' if own
                else 'spectrum not in the campaign (SWP number not found in the SWP log)'}
    jd, year = num(own[0]['jd']), int(float(own[0]['year']))
    fv = {(r['line'], r['year']): r for r in (load_fvar_table(os.path.join(outputs_dir, 'fvar_by_year.csv')) or [])}

    lines = []
    for o in sorted(own, key=lambda r: num(r['rest_wavelength']) or 0):
        pts = [{'spectrum': r['spectrum'], 'jd': num(r['jd']), 'flux': num(r['flux']), 'flux_err': num(r['flux_err'])}
               for r in rows
               if r['line'] == o['line'] and r['detected'] == 'True' and r['year'] not in ('', None)
               and int(float(r['year'])) == year and num(r['jd']) is not None]
        lines.append({'line': o['line'], 'rest_wavelength': num(o['rest_wavelength']),
                      'detected': o['detected'] == 'True', 'flux': num(o['flux']), 'flux_err': num(o['flux_err']),
                      'year_points': sorted(pts, key=lambda p: p['jd']),
                      'fvar': fv.get((o['line'], year))})
    return {'matched': True, 'spectrum': own[0]['spectrum'], 'jd': jd, 'date': own[0]['date'][:10],
            'year': year, 'lines': lines}


def load_fvar_table(path: Optional[str] = None) -> Optional[List[Dict]]:
    """Rows of outputs/fvar_by_year.csv as dicts (numbers parsed; empty/NaN -> None). No pandas needed."""
    path = path or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'fvar_by_year.csv')
    if not os.path.exists(path):
        return None

    def num(v):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        return x if math.isfinite(x) else None

    rows = []
    with open(path, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            rows.append({'line': r['line'], 'year': int(r['year']), 'n': int(r['n']),
                         'jd_min': num(r['jd_min']), 'jd_span': num(r['jd_span']),
                         'f_mean': num(r['f_mean']), 'err_rms': num(r['err_rms']), 'ratio': num(r['ratio']),
                         'fvar': num(r['fvar']), 'fvar_err': num(r['fvar_err']), 'regime': r.get('regime') or ''})
    return rows
