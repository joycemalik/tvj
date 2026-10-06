"""
manuscript_fit.py
-----------------
Emission-line measurement exactly as described in the 3C 390.3 manuscript
(response to reviewers, item 2). Used by the /v4 page and
`run_variability.py --method manuscript`; the default pipeline is unchanged.

(a) Continuum: for each line, a power law F_λ = A λ^α is fitted by linear least
    squares on ln F = ln A + α ln λ in nearly flat, line-free windows on both sides
    of the line (per-line YAML `manuscript.continuum_windows`); A = e^{ln A};
    standard errors from the residual variance. F_sub = F_obs − F_λ.
(b) Lines: F_sub between the windows (λmin = blue window end, λmax = red window
    start) is modelled by a sum of Gaussians G(λ) = a exp[−(λ − b)²/(2c²)].
(c) Uncertainties:
    flux     F = ∫_{λmin}^{λmax} G dλ by trapezoidal integration of the fitted Gaussian
             (analytic a c √(2π) as fallback); reported as |F|;
             σ_F = √N · σ_c · Δλ · F_λ(b)  (IRAF noise prescription), with
             N = pixels within b ± 3c, σ_c = rms of the sigma-clipped normalised
             residuals in the continuum windows (MAD if needed), Δλ = Å per pixel,
             F_λ(b) = continuum at b;
    EW       W = F / F_λ(b),  σ_W = W · σ_F / F;
    FWHM     2.3548 c,  σ(FWHM) = 2.3548 σ_c(fit), σ_c(fit) from the fit covariance;
             FWHM_v = FWHM_λ / b · c_light;
    χ²_red   χ² / (N_pix − 3k), k Gaussians, σ_pix = σ_c F_λ.
"""

import math
from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import curve_fit

from config import SPEED_OF_LIGHT_KMS, INSTRUMENTAL_FWHM_REST, DEFAULT_CONTINUUM_WINDOWS
from src.continuum import parse_window_ranges
from src.refine import apply_masks, pixel_correlation, verify_fit, SIGMA_INSTR

UNIT = 1.0e-13
_trapz = getattr(np, 'trapezoid', None) or np.trapz


def clipped_rms(r: np.ndarray, n_iter: int = 4, clip: float = 3.0) -> float:
    """rms after iterative 3σ clipping; MAD-based estimate if clipping leaves too few points."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return float('nan')
    keep = np.ones_like(r, dtype=bool)
    for _ in range(n_iter):
        mu, sd = r[keep].mean(), r[keep].std(ddof=1)
        new = np.abs(r - mu) <= clip * sd
        if new.sum() < 5 or np.array_equal(new, keep):
            break
        keep = new
    rms = float(r[keep].std(ddof=1))
    if not np.isfinite(rms) or rms <= 0:
        rms = float(1.4826 * np.median(np.abs(r - np.median(r))))
    return rms


def normalised_continuum_rms(wavelength: np.ndarray, flux: np.ndarray, continuum: np.ndarray,
                             windows=DEFAULT_CONTINUUM_WINDOWS) -> float:
    """σ_c: rms of (F − F_c)/F_c in the continuum windows after 3σ clipping."""
    ranges = parse_window_ranges(windows) if isinstance(windows, str) else windows
    m = np.zeros_like(wavelength, dtype=bool)
    for lo, hi in ranges:
        m |= (wavelength >= lo) & (wavelength <= hi) & (flux != 0) & (continuum > 0)
    return clipped_rms((flux[m] - continuum[m]) / continuum[m])


def local_power_law(wavelength: np.ndarray, flux: np.ndarray, windows) -> Optional[Dict[str, float]]:
    """OLS on ln F = ln A + α ln λ in the given windows; returns A, α and their standard errors."""
    m = np.zeros_like(wavelength, dtype=bool)
    for lo, hi in windows:
        m |= (wavelength >= lo) & (wavelength <= hi) & (flux > 0)
    if m.sum() < 4:
        return None
    x, y = np.log(wavelength[m]), np.log(flux[m])
    n = len(x)
    alpha, ln_a = np.polyfit(x, y, 1)
    s2 = float(np.sum((y - alpha * x - ln_a) ** 2) / max(n - 2, 1))
    sxx = float(np.sum((x - x.mean()) ** 2))
    return {'A': float(np.exp(ln_a)), 'alpha': float(alpha),
            'alpha_err': math.sqrt(s2 / sxx) if sxx > 0 else float('nan'),
            'A_err': float(np.exp(ln_a)) * math.sqrt(s2 * (1.0 / n + x.mean() ** 2 / sxx)) if sxx > 0 else float('nan'),
            'mask': m}


def _gauss_sum(x, *p):
    y = np.zeros_like(x, dtype=float)
    for k in range(len(p) // 3):
        a, b, c = p[3 * k:3 * k + 3]
        y = y + a * np.exp(-((x - b) ** 2) / (2.0 * c ** 2))
    return y


def fit_line_manuscript(wavelength: np.ndarray, flux: np.ndarray, continuum: np.ndarray,
                        rest_wl: float, cfg: Dict[str, Any], seed: Dict[str, float],
                        sigma_c_global: Optional[float] = None) -> Optional[Dict[str, Any]]:
    mcfg = cfg.get('manuscript') or {}
    rcfg = cfg.get('refine') or {}
    sig_cfg = cfg.get('sigma') or {}
    tol = float(rcfg.get('center_tol', 8.0))
    good = apply_masks(wavelength, cfg.get('mask_ranges')) & (flux != 0)

    windows = mcfg.get('continuum_windows')
    if windows:
        blue, red = windows[0], windows[-1]
        pl = local_power_law(wavelength, np.where(good, flux, 0.0), windows)
        if pl is None:
            return None
        cont_full = pl['A'] * wavelength ** pl['alpha']
        lmin, lmax = float(blue[1]), float(red[0])
        sigma_c = clipped_rms(((flux - cont_full) / cont_full)[pl['mask'] & good])
    else:   # fallback: global continuum, refine window
        cont_full = continuum
        lmin, lmax = rcfg.get('window', [rest_wl - 30.0, rest_wl + 30.0])
        sigma_c = sigma_c_global
        pl = None
    m = (wavelength >= lmin) & (wavelength <= lmax) & good
    x = wavelength[m]
    fc = cont_full[m]
    if len(x) < 8 or sigma_c is None or not np.isfinite(sigma_c) or sigma_c <= 0:
        return None
    y = (flux[m] - fc) / UNIT                       # F_sub
    sig_pix = sigma_c * fc / UNIT

    comps = [dict(rest=rest_wl, smin=float(sig_cfg.get('min', SIGMA_INSTR)), smax=float(sig_cfg.get('max', 12.0)),
                  b0=float(seed.get('center', rest_wl)), c0=float(seed.get('sigma', 5.0)),
                  a0=float(seed.get('amplitude', 0.0)) / UNIT)]
    for c in mcfg.get('companions', rcfg.get('companions', [])):
        comps.append(dict(rest=float(c['rest']), smin=float(c.get('sigma_min', SIGMA_INSTR)),
                          smax=float(c.get('sigma_max', 12.0)), b0=float(c['rest']),
                          c0=float(c.get('sigma_seed', 5.0)), a0=None))
    p0, plo, phi = [], [], []
    for c in comps:
        k = int(np.argmin(np.abs(x - c['b0'])))
        a0 = c['a0'] if c['a0'] and c['a0'] > 0 else max(float(y[k]), float(np.median(sig_pix)))
        blo, bhi = max(c['rest'] - tol, lmin), min(c['rest'] + tol, lmax)
        p0 += [a0, float(np.clip(c['b0'], blo + 1e-6, bhi - 1e-6)),
               float(np.clip(c['c0'], c['smin'] + 1e-6, c['smax'] - 1e-6))]
        plo += [0.0, blo, c['smin']]
        phi += [np.inf, bhi, c['smax']]
    try:
        popt, pcov = curve_fit(_gauss_sum, x, y, p0=p0, bounds=(plo, phi), sigma=sig_pix,
                               absolute_sigma=True, x_scale='jac', maxfev=20000)
    except Exception:
        return None

    a, b, c = popt[0:3]
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    a_err, b_err, c_err = perr[0:3]
    k = len(comps)

    # Flux: trapezoidal integral of the fitted Gaussian from λmin to λmax; analytic fallback
    grid = np.linspace(lmin, lmax, 4001)
    F = float(_trapz(a * np.exp(-((grid - b) ** 2) / (2 * c ** 2)), grid))
    if not np.isfinite(F):
        F = math.sqrt(2 * math.pi) * a * c
    F = abs(F) * UNIT

    # IRAF noise prescription
    dlam = float(np.median(np.diff(wavelength[wavelength > 0])))
    N = int(np.sum(np.abs(x - b) <= 3 * c))
    fc_b = float(np.interp(b, x, fc))
    F_err = math.sqrt(max(N, 1)) * sigma_c * dlam * fc_b

    model = _gauss_sum(x, *popt)
    resid = (y - model) / sig_pix
    dof = max(len(x) - 3 * k, 1)
    chi2r = float(np.sum(resid ** 2) / dof)

    fwhm = 2.3548 * c
    fwhm_err = 2.3548 * c_err
    fwhm_int = math.sqrt(max(fwhm ** 2 - INSTRUMENTAL_FWHM_REST ** 2, 0.0))
    ew = F / fc_b
    ew_err = ew * F_err / F if F > 0 else float('nan')
    at_bound = bool(np.isclose(c, comps[0]['smin'], rtol=1e-3) or np.isclose(c, comps[0]['smax'], rtol=1e-3)
                    or np.isclose(b, plo[1], atol=1e-3) or np.isclose(b, phi[1], atol=1e-3))

    cont_units = fc / UNIT
    gauss = [popt[3 * j] * np.exp(-((x - popt[3 * j + 1]) ** 2) / (2 * popt[3 * j + 2] ** 2)) for j in range(k)]
    noise_mean = float(np.median(sig_pix))
    verification = verify_fit(x, y + cont_units, model + cont_units, cont_units, gauss, resid, noise_mean,
                              chi2r, dof, b, c, F / UNIT, F_err / UNIT, fwhm, at_bound,
                              pixel_correlation(wavelength, flux))
    labels = [cfg.get('line_name', f'{rest_wl:g} Å')] + [f"companion {cc['rest']:g} Å" for cc in comps[1:]]

    # Plot over the windows too, so the continuum fit is visible
    span = (wavelength >= (windows[0][0] if windows else lmin)) & (wavelength <= (windows[-1][1] if windows else lmax)) & good
    xs = wavelength[span]
    cs = cont_full[span] / UNIT
    gs = [popt[3 * j] * np.exp(-((xs - popt[3 * j + 1]) ** 2) / (2 * popt[3 * j + 2] ** 2)) for j in range(k)]
    ms = cs + sum(gs)
    rs = (flux[span] / UNIT - ms) / (sigma_c * cs)

    out = {
        'refined': True,
        'method': 'manuscript',
        'amplitude': a * UNIT, 'amplitude_err': a_err * UNIT,
        'center': float(b), 'center_err': float(b_err),
        'sigma': float(c), 'sigma_err': float(c_err),
        'flux': F, 'flux_err': F_err, 'snr': F / F_err if F_err > 0 else 0.0,
        'ew': ew, 'ew_err': ew_err,
        'fwhm_ang': fwhm, 'fwhm_ang_err': fwhm_err,
        'fwhm_kms': fwhm / b * SPEED_OF_LIGHT_KMS, 'fwhm_kms_err': fwhm_err / b * SPEED_OF_LIGHT_KMS,
        'fwhm_kms_intrinsic': fwhm_int / b * SPEED_OF_LIGHT_KMS,
        'reduced_chi2': chi2r, 'reduced_chi2_indep': chi2r, 'dof': int(dof),
        'pixel_noise': noise_mean * UNIT, 'sigma_c': sigma_c, 'n_pix_line': N, 'dispersion': dlam,
        'continuum_at_center': fc_b, 'lambda_min': lmin, 'lambda_max': lmax,
        'refine_window': [lmin, lmax], 'at_bound': at_bound, 'n_pix': int(len(x)),
        'plot': {
            'x': xs.tolist(), 'y': (flux[span]).tolist(), 'yerr': float(noise_mean * UNIT),
            'model': (ms * UNIT).tolist(), 'continuum': (cs * UNIT).tolist(),
            'components': [{'label': lab, 'y': (g * UNIT).tolist()} for lab, g in zip(labels, gs)],
            'norm_resid': rs.tolist(), 'mask_ranges': cfg.get('mask_ranges') or [],
        },
        'verification': verification,
    }
    if pl is not None:
        out.update(local_alpha=pl['alpha'], local_alpha_err=pl['alpha_err'],
                   local_A=pl['A'], local_A_err=pl['A_err'],
                   continuum_windows_local=[list(w) for w in windows])
    return out
