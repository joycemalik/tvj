"""
refine.py
---------
Continuous weighted least-squares refinement of an emission-line fit.

Model over a per-line window (observed flux, not continuum-subtracted):

    F(λ) = Σ_k A_k exp[-(λ-μ_k)² / 2σ_k²] + c0 + c1 (λ - λ_mid)

The primary line plus any overlapping companions (e.g. Lyα with N V) and a
local linear continuum are fitted together, so the continuum under the line is
not fixed in advance. Pixel noise is the DER_SNR estimate (Stoehr et al. 2008,
ASPC 394, 505). Parameter errors come from the covariance (JᵀJ)⁻¹, scaled by
√χ²_red when χ²_red > 1. Line flux is the full Gaussian integral √(2π)·A·σ.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist, norm, shapiro

from config import SPEED_OF_LIGHT_KMS, INSTRUMENTAL_FWHM_REST

UNIT = 1.0e-13          # fit in units of 1e-13 so all parameters are O(1)-O(1e3)
_trapz = getattr(np, 'trapezoid', None) or np.trapz
SIGMA_INSTR = INSTRUMENTAL_FWHM_REST / 2.3548
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def der_snr_noise(flux: np.ndarray) -> float:
    """Per-pixel noise via DER_SNR (Stoehr et al. 2008)."""
    f = np.asarray(flux, dtype=float)
    f = f[np.isfinite(f) & (f != 0)]
    if len(f) < 5:
        return float('nan')
    return 1.482602 / math.sqrt(6.0) * float(np.median(np.abs(2.0 * f[2:-2] - f[:-4] - f[4:])))


def pixel_correlation(wavelength: np.ndarray, flux: np.ndarray,
                      windows: Optional[List[Tuple[float, float]]] = None) -> float:
    """
    Lag-1 autocorrelation of the pixel noise, measured in line-free continuum
    windows (residuals from a straight line in each window, pooled). Resampling
    of IUE MXLO spectra makes neighbouring pixels correlated.
    """
    from config import DEFAULT_CONTINUUM_WINDOWS
    from src.continuum import parse_window_ranges
    windows = windows or parse_window_ranges(DEFAULT_CONTINUUM_WINDOWS)
    num = den = 0.0
    for lo, hi in windows:
        m = (wavelength >= lo) & (wavelength <= hi) & (flux != 0)
        if m.sum() < 6:
            continue
        r = flux[m] - np.polyval(np.polyfit(wavelength[m], flux[m], 1), wavelength[m])
        num += float(np.sum(r[1:] * r[:-1]))
        den += float(np.sum(r * r))
    rho = num / den if den > 0 else 0.0
    return float(np.clip(rho, 0.0, 0.95))


def apply_masks(wavelength: np.ndarray, mask_ranges: Optional[List[List[float]]]) -> np.ndarray:
    """Boolean array, True where the pixel is usable (outside every mask range)."""
    good = np.ones_like(wavelength, dtype=bool)
    for lo, hi in (mask_ranges or []):
        good &= ~((wavelength >= lo) & (wavelength <= hi))
    return good


def _model(x: np.ndarray, p: np.ndarray, n_g: int, x_mid: float) -> np.ndarray:
    y = p[3 * n_g] + p[3 * n_g + 1] * (x - x_mid)
    for k in range(n_g):
        A, mu, s = p[3 * k:3 * k + 3]
        y = y + A * np.exp(-0.5 * ((x - mu) / s) ** 2)
    return y


def refine_line(
    wavelength: np.ndarray,
    flux: np.ndarray,
    rest_wl: float,
    cfg: Dict[str, Any],
    seed: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    """
    Refine one line. `cfg` is the per-line YAML; its `refine` block gives
    window: [lo, hi], companions: [{rest, sigma_min, sigma_max}], center_tol.
    `seed` holds the grid-search center/sigma/amplitude. Returns None on failure.
    """
    rcfg = cfg.get('refine') or {}
    sig_cfg = cfg.get('sigma') or {}
    lo, hi = rcfg.get('window', [rest_wl - 30.0, rest_wl + 30.0])
    tol = float(rcfg.get('center_tol', 8.0))

    good = apply_masks(wavelength, cfg.get('mask_ranges'))
    m = (wavelength >= lo) & (wavelength <= hi) & good & (flux != 0)
    x, y = wavelength[m], flux[m] / UNIT
    noise_region = (wavelength >= lo - 20) & (wavelength <= hi + 20) & good
    noise = der_snr_noise(flux[noise_region]) / UNIT
    if len(x) < 8 or not np.isfinite(noise) or noise <= 0:
        return None

    comps = [dict(rest=rest_wl,
                  smin=float(sig_cfg.get('min', SIGMA_INSTR)),
                  smax=float(sig_cfg.get('max', 12.0)),
                  mu0=float(seed.get('center', rest_wl)),
                  s0=float(seed.get('sigma', 5.0)),
                  a0=float(seed.get('amplitude', 0.0)) / UNIT)]
    for c in rcfg.get('companions', []):
        comps.append(dict(rest=float(c['rest']),
                          smin=float(c.get('sigma_min', SIGMA_INSTR)),
                          smax=float(c.get('sigma_max', 12.0)),
                          mu0=float(c['rest']), s0=float(c.get('sigma_seed', 5.0)), a0=None))

    base = float(np.percentile(y, 10))
    p0, plo, phi = [], [], []
    for c in comps:
        k = int(np.argmin(np.abs(x - c['mu0'])))
        a0 = c['a0'] if c['a0'] and c['a0'] > 0 else max(y[k] - base, noise)
        mlo, mhi = c['rest'] - tol, c['rest'] + tol
        p0 += [a0, float(np.clip(c['mu0'], mlo, mhi)), float(np.clip(c['s0'], c['smin'], c['smax']))]
        plo += [0.0, mlo, c['smin']]
        phi += [np.inf, mhi, c['smax']]
    p0 += [base, 0.0]
    plo += [-np.inf, -np.inf]
    phi += [np.inf, np.inf]
    plo, phi = np.array(plo), np.array(phi)
    # least_squares needs a strictly interior start
    eps = 1e-6 * np.where(np.isfinite(phi - plo), phi - plo, 1.0)
    p0 = np.minimum(np.maximum(np.array(p0, dtype=float), plo + eps), phi - eps)

    n_g = len(comps)
    x_mid = float(np.mean(x))

    def resid(p):
        return (y - _model(x, p, n_g, x_mid)) / noise

    try:
        res = least_squares(resid, p0, bounds=(plo, phi), method='trf', x_scale='jac',
                            max_nfev=20000)
    except Exception:
        return None

    p = res.x
    # Weights are uniform, so the best-fit parameters do not depend on `noise`.
    # Re-estimate it from the residuals: DER_SNR on the raw spectrum is inflated
    # by the line's own curvature; on (data - model) it measures only the
    # pixel-to-pixel noise, while smooth misfit still shows up in chi2.
    resid_noise = der_snr_noise(y - _model(x, p, n_g, x_mid))
    if np.isfinite(resid_noise) and resid_noise > 0:
        scale = noise / resid_noise
        noise = resid_noise
    else:
        scale = 1.0
    fun = res.fun * scale
    jac = res.jac * scale
    dof = max(len(x) - len(p), 1)
    chi2r = float(np.sum(fun ** 2) / dof)
    try:
        cov = np.linalg.pinv(jac.T @ jac) * max(1.0, chi2r)
    except np.linalg.LinAlgError:
        return None

    A, mu, s = p[0:3]
    vA, vmu, vs, cAs = cov[0, 0], cov[1, 1], cov[2, 2], cov[0, 2]
    F = math.sqrt(2.0 * math.pi) * A * s
    varF = 2.0 * math.pi * (s * s * vA + A * A * vs + 2.0 * A * s * cAs)
    F_err = math.sqrt(max(varF, 0.0))

    cont_at_mu = p[3 * n_g] + p[3 * n_g + 1] * (mu - x_mid)
    fwhm_ang = FWHM_PER_SIGMA * s
    fwhm_int = math.sqrt(max(fwhm_ang ** 2 - INSTRUMENTAL_FWHM_REST ** 2, 0.0))
    at_bound = bool(np.isclose(s, comps[0]['smin'], rtol=1e-3) or np.isclose(s, comps[0]['smax'], rtol=1e-3)
                    or np.isclose(abs(mu - rest_wl), tol, rtol=1e-3))

    model = _model(x, p, n_g, x_mid)
    continuum = p[3 * n_g] + p[3 * n_g + 1] * (x - x_mid)
    gauss = [p[3 * k] * np.exp(-0.5 * ((x - p[3 * k + 1]) / p[3 * k + 2]) ** 2) for k in range(n_g)]
    norm_resid = (y - model) / noise

    rho = pixel_correlation(wavelength, flux)
    verification = verify_fit(x, y, model, continuum, gauss, norm_resid, noise, chi2r, dof,
                              mu, s, F, F_err, fwhm_ang, at_bound, rho)
    labels = [cfg.get('line_name', f'{rest_wl:g} Å')] + [f"companion {c['rest']:g} Å" for c in comps[1:]]
    plot = {
        'x': x.tolist(),
        'y': (y * UNIT).tolist(),
        'yerr': float(noise * UNIT),
        'model': (model * UNIT).tolist(),
        'continuum': (continuum * UNIT).tolist(),
        'components': [{'label': lab, 'y': (g * UNIT).tolist()} for lab, g in zip(labels, gauss)],
        'norm_resid': norm_resid.tolist(),
        'mask_ranges': cfg.get('mask_ranges') or [],
    }

    return {
        'refined': True,
        'amplitude': A * UNIT,
        'amplitude_err': math.sqrt(max(vA, 0.0)) * UNIT,
        'center': float(mu),
        'center_err': math.sqrt(max(vmu, 0.0)),
        'sigma': float(s),
        'sigma_err': math.sqrt(max(vs, 0.0)),
        'flux': F * UNIT,
        'flux_err': F_err * UNIT,
        'snr': F / F_err if F_err > 0 else 0.0,
        'ew': (F / cont_at_mu) if cont_at_mu > 0 else float('nan'),
        'ew_err': (F_err / cont_at_mu) if cont_at_mu > 0 else float('nan'),
        'fwhm_ang': fwhm_ang,
        'fwhm_kms': fwhm_ang / mu * SPEED_OF_LIGHT_KMS,
        'fwhm_kms_intrinsic': fwhm_int / mu * SPEED_OF_LIGHT_KMS,
        'reduced_chi2': chi2r,
        'reduced_chi2_indep': chi2r,
        'pixel_noise': noise * UNIT,
        'refine_window': [float(lo), float(hi)],
        'at_bound': at_bound,
        'n_pix': int(len(x)),
        'dof': int(dof),
        'plot': plot,
        'verification': verification,
    }


def runs_test(signs: np.ndarray) -> Tuple[int, float, float]:
    """Wald–Wolfowitz runs test on residual signs. Returns (runs, z, two-sided p)."""
    s = signs[signs != 0]
    n1, n2 = int(np.sum(s > 0)), int(np.sum(s < 0))
    n = n1 + n2
    if n1 == 0 or n2 == 0 or n < 3:
        return 0, float('nan'), float('nan')
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    mean = 2.0 * n1 * n2 / n + 1.0
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1.0))
    z = (runs - mean) / math.sqrt(var) if var > 0 else float('nan')
    return runs, z, float(2.0 * norm.sf(abs(z))) if np.isfinite(z) else float('nan')


def verify_fit(x, y, model, continuum, gauss, norm_resid, noise, chi2r, dof,
               mu, s, F, F_err, fwhm_ang, at_bound, rho=0.0) -> Dict[str, Any]:
    """
    Statistical checks of a refined fit (all quantities in fit units).

    1. Goodness of fit: p = P(χ² ≥ χ²_obs). With lag-1 pixel correlation ρ, the
       sum of squared residuals has variance inflated by c = (1+ρ²)/(1−ρ²) (AR(1)
       noise); χ²/c is compared with a χ² distribution of dof/c degrees of
       freedom (Satterthwaite 1946 moment matching).
    2. Residual randomness: Wald–Wolfowitz runs test on residual signs (Wald & Wolfowitz 1940).
    3. Residual normality: Shapiro–Wilk test of (data − model)/σ (Shapiro & Wilk 1965).
    4. Model-independent flux: direct integration of data − continuum − companions over
       μ ± 3σ, compared with the fitted Gaussian over the same pixels.
    5. Detection: F/σ_F ≥ 3.   6. Parameters inside their allowed ranges.
    7. Resolution: FWHM above the instrumental FWHM (informational).
    """
    checks = []

    def add(name, value, ok, criterion, ref, required=True):
        checks.append({'name': name, 'value': value, 'pass': bool(ok), 'criterion': criterion,
                       'reference': ref, 'required': required})

    c = (1.0 + rho ** 2) / (1.0 - rho ** 2)
    p_chi2 = float(chi2_dist.sf(chi2r * dof / c, dof / c))
    add('Goodness of fit', f'χ²_red = {chi2r:.2f} (dof {dof}, pixel ρ = {rho:.2f}), p = {p_chi2:.3g}',
        p_chi2 >= 0.01, 'p(χ²) ≥ 0.01, corrected for pixel correlation',
        'Bevington & Robinson 2003; Satterthwaite 1946')

    runs, z_runs, p_runs = runs_test(np.sign(norm_resid))
    add('Residual randomness', f'{runs} runs, z = {z_runs:.2f}, p = {p_runs:.3g}',
        np.isfinite(p_runs) and p_runs >= 0.01, 'runs-test p ≥ 0.01', 'Wald & Wolfowitz 1940')

    if len(norm_resid) >= 3:
        p_sw = float(shapiro(norm_resid).pvalue)
        add('Residual normality', f'Shapiro–Wilk p = {p_sw:.3g}', p_sw >= 0.01, 'p ≥ 0.01',
            'Shapiro & Wilk 1965')

    core = np.abs(x - mu) <= 3.0 * s
    if np.sum(core) >= 3:
        primary = gauss[0]
        others = sum(gauss[1:]) if len(gauss) > 1 else 0.0
        net = (y - continuum - others)[core]
        xc = x[core]
        f_direct = float(_trapz(net, xc))
        f_model = float(_trapz(primary[core], xc))
        dlam = float(np.median(np.diff(xc))) if len(xc) > 1 else 1.0
        e_direct = noise * dlam * math.sqrt(len(xc))
        z_flux = (f_direct - f_model) / e_direct if e_direct > 0 else float('nan')
        add('Model-independent flux', f'direct {f_direct * UNIT:.3e} vs Gaussian {f_model * UNIT:.3e} '
            f'(z = {z_flux:.2f})', np.isfinite(z_flux) and abs(z_flux) <= 2.0,
            '|z| ≤ 2 over μ ± 3σ', 'direct numerical integration')

    sig = F / F_err if F_err > 0 else 0.0
    add('Detection', f'F/σ_F = {sig:.1f}', sig >= 3.0, 'F/σ_F ≥ 3', 'standard 3σ criterion')
    add('Parameter bounds', 'at bound' if at_bound else 'inside allowed ranges', not at_bound,
        'σ and μ not at configured limits', 'config ranges')
    add('Resolved by instrument', f'FWHM {fwhm_ang:.2f} Å vs instrumental {INSTRUMENTAL_FWHM_REST:.2f} Å',
        fwhm_ang > INSTRUMENTAL_FWHM_REST, 'FWHM > instrumental FWHM', 'IUE NEWSIPS manual §2.3.2.1',
        required=False)

    failed = [c['name'] for c in checks if c['required'] and not c['pass']]
    return {'status': 'VERIFIED' if not failed else 'CHECK', 'failed': failed, 'checks': checks}
