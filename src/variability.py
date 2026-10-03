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

import math
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

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


def load_jd(path: str = 'jd.xlsx') -> pd.DataFrame:
    """SWP spectra (Sheet2) -> spectrum, jd, date, year. Rows without a date are dropped."""
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


def fvar_by_year(line_fluxes: pd.DataFrame, cutoff: float = FVAR_ERR_CUTOFF) -> pd.DataFrame:
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
    return pd.DataFrame(rows)


def light_curve_points(line_fluxes: pd.DataFrame) -> pd.DataFrame:
    """(JD − JD_min, F, F_err) per line, JD_min taken over the whole campaign for that line."""
    lf = line_fluxes.copy()
    lf['jd_minus_min'] = lf['jd'] - lf.groupby('line')['jd'].transform('min')
    return lf[['line', 'year', 'spectrum', 'jd', 'jd_minus_min', 'flux', 'flux_err']]


def load_fvar_table(path: Optional[str] = None) -> Optional[pd.DataFrame]:
    path = path or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'fvar_by_year.csv')
    return pd.read_csv(path) if os.path.exists(path) else None
