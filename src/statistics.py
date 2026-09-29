import math
import numpy as np
from typing import Dict, Any, Tuple, Optional
from config import (SPEED_OF_LIGHT_KMS, SIGMA_CLIP_ITERS, SIGMA_CLIP_THRESHOLD, LOW_RES_FACTOR,
                    INSTRUMENTAL_FWHM_REST)

# Half-distance (Å) from line center at which the local linear baseline is sampled.
BL_HW = 11.0


def local_baseline(wavelength: np.ndarray, obs_flux: np.ndarray, center: float,
                   wl_win: np.ndarray) -> Optional[np.ndarray]:
    """Linear baseline through mean obs flux in center ± BL_HW (±3 Å bands); None if a band is empty."""
    lo = (wavelength >= center - BL_HW - 3) & (wavelength <= center - BL_HW + 3)
    hi = (wavelength >= center + BL_HW - 3) & (wavelength <= center + BL_HW + 3)
    if not (np.any(lo) and np.any(hi)):
        return None
    bl_lo, wl_lo = float(np.mean(obs_flux[lo])), float(np.mean(wavelength[lo]))
    bl_hi, wl_hi = float(np.mean(obs_flux[hi])), float(np.mean(wavelength[hi]))
    slope = (bl_hi - bl_lo) / (wl_hi - wl_lo) if wl_hi > wl_lo else 0.0
    return bl_lo + slope * (wl_win - wl_lo)


def fractional_variability(flux, err) -> Dict[str, float]:
    """
    Fractional rms variability amplitude and its error, Vaughan et al. 2003
    (MNRAS 345, 1271) eqs. 10 and B2, plus Rmax = Fmax/Fmin with propagated error.
    Fvar is NaN when the excess variance is not positive (no intrinsic variability detected).
    """
    F = np.asarray(flux, dtype=float)
    E = np.asarray(err, dtype=float)
    ok = np.isfinite(F) & np.isfinite(E)
    F, E = F[ok], E[ok]
    n = len(F)
    nan = float('nan')
    if n < 2:
        return dict(n=n, mean=nan, excess_var=nan, fvar=nan, fvar_err=nan, rmax=nan, rmax_err=nan)
    xbar = float(np.mean(F))
    s2 = float(np.var(F, ddof=1))
    mse = float(np.mean(E ** 2))
    excess = s2 - mse
    if excess > 0 and xbar != 0:
        fvar = math.sqrt(excess) / xbar
        fvar_err = math.sqrt((math.sqrt(1.0 / (2 * n)) * mse / (xbar ** 2 * fvar)) ** 2
                             + (math.sqrt(mse / n) / xbar) ** 2)
    else:
        fvar = fvar_err = nan
    i_max, i_min = int(np.argmax(F)), int(np.argmin(F))
    rmax = float(F[i_max] / F[i_min]) if F[i_min] != 0 else nan
    rmax_err = rmax * math.sqrt((E[i_min] / F[i_min]) ** 2 + (E[i_max] / F[i_max]) ** 2)
    return dict(n=n, mean=xbar, mean_err=float(np.mean(E)), excess_var=excess,
                fvar=fvar, fvar_err=fvar_err, rmax=rmax, rmax_err=rmax_err)

def estimate_noise(residuals: np.ndarray, n_iters: int = SIGMA_CLIP_ITERS, threshold: float = SIGMA_CLIP_THRESHOLD) -> float:
    """
    Estimates noise std via Median Absolute Deviation (MAD), scaled to match a normal distribution.
    This provides a robust noise estimate that isn't heavily inflated by the line wings in the residuals.
    """
    if len(residuals) < 2:
        return 1.0e-20
    
    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))
    robust_std = float(mad * 1.4826)
    
    if robust_std <= 1.0e-30:
        return 1.0e-20
    return robust_std

def compute_statistics(
    wavelength: np.ndarray,
    subtracted_y: np.ndarray,
    continuum_fit: np.ndarray,
    amplitude: float,
    center: float,
    sigma: float,
    wing_window: float,
    manual_min_wl: float = None,
    manual_max_wl: float = None,
    rest_wavelength: float = 1216.0,
    noise: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculates scientific metrics exactly as performed by the reference Spectrum-Analysis app:
    - Reduced Chi-Squared (χ²_red)
    - Flux & Flux Error
    - SNR
    - Equivalent Width (EW) & EW Error
    - FWHM (Å and km/s)
    - Wing Wavelengths (Blue Wing, Red Wing, Δλ)
    """
    # wing_window is the FULL fitting width; half is the radius
    if manual_min_wl is None:
        manual_min_wl = center - wing_window / 2.0
    if manual_max_wl is None:
        manual_max_wl = center + wing_window / 2.0
        
    # Fit window mask
    mask = (wavelength >= manual_min_wl) & (wavelength <= manual_max_wl)
    wl_win = wavelength[mask]
    sub_win = subtracted_y[mask]
    cont_win = continuum_fit[mask]
    
    # Gaussian model — amplitude is the plain OLS value from net_win
    g_model = amplitude * np.exp(-((wl_win - center) ** 2) / (2.0 * (sigma ** 2)))

    # Same local linear baseline the candidate engine used for the amplitude
    # (sampled on the full spectrum, outside the fit window).
    obs_win = sub_win + cont_win
    local_bl = local_baseline(wavelength, subtracted_y + continuum_fit, center, wl_win)
    if local_bl is None:
        local_bl = cont_win

    net_win = obs_win - local_bl   # baseline-corrected signal

    # Residuals: narrow Gaussian vs baseline-corrected signal
    res_win = net_win - g_model

    mean_cont = np.mean(continuum_fit) if len(continuum_fit) > 0 else 1.0e-13

    # 1. Noise estimate from baseline-corrected residuals
    noise_sigma = estimate_noise(res_win)
    if noise_sigma <= 1.0e-30:
        norm_res  = sub_win / (mean_cont if mean_cont != 0 else 1.0e-30)
        std_norm  = estimate_noise(norm_res)
        noise_sigma = std_norm * abs(mean_cont)
    if noise_sigma <= 1.0e-30:
        noise_sigma = 1.0e-20
        
    # 2. Reduced Chi-Squared. `reduced_chi2` normalises by the fit's own residual
    # scatter (reference-tool convention, ~1 by construction). `reduced_chi2_indep`
    # uses independent line-free continuum noise and is the real goodness-of-fit.
    dof = max(len(wl_win) - 3, 1)
    reduced_chi2 = np.sum((res_win / noise_sigma) ** 2) / dof
    if noise is not None and noise > 1.0e-30:
        reduced_chi2_indep = float(np.sum((res_win / noise) ** 2) / dof)
    else:
        reduced_chi2_indep = float('nan')
    
    # 3. Flux via Trapezoidal integration of the GAUSSIAN MODEL
    # This matches the reference JS calculator which calculates flux from the fitted model.
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    flux = float(_trapz(g_model, wl_win)) if len(wl_win) > 1 else 0.0
    
    # Flux Error: sqrt(N) * noise_per_pixel * dispersion
    # noise_sigma is already in flux units (erg/s/cm²/Å); multiply by pixel width
    n_pts = len(wl_win)
    flux_err = float(np.sqrt(max(n_pts, 1)) * noise_sigma * LOW_RES_FACTOR)
    
    # 4. SNR
    snr = abs(flux) / flux_err if flux_err > 0 else 0.0
    
    # 5. Equivalent Width (EW) via Trapezoidal integration of (F_narrow / F_cont)
    #    net_win = obs - local_bl is the narrow-core signal above local background.
    ratio = np.where(cont_win > 0, net_win / cont_win, 0.0)
    ew = float(_trapz(ratio, wl_win)) if len(wl_win) > 1 else 0.0
    ew_err = abs(ew) * (flux_err / abs(flux)) if abs(flux) > 0 else 0.0
    
    # 6. FWHM
    fwhm_ang = 2.3548 * sigma
    fwhm_kms = (fwhm_ang / center) * SPEED_OF_LIGHT_KMS if center > 0 else 0.0
    # Intrinsic width: remove instrumental broadening in quadrature
    fwhm_int_ang = math.sqrt(max(fwhm_ang ** 2 - INSTRUMENTAL_FWHM_REST ** 2, 0.0))
    fwhm_kms_intrinsic = (fwhm_int_ang / center) * SPEED_OF_LIGHT_KMS if center > 0 else 0.0
    # Full Gaussian line flux (window-independent); `flux` above is the window-truncated integral
    flux_total = float(math.sqrt(2.0 * math.pi) * amplitude * sigma)

    # 7. Wing Wavelengths
    wing_delta_lambda = rest_wavelength * (fwhm_kms / 2.0) / SPEED_OF_LIGHT_KMS
    blue_wing = center - wing_delta_lambda
    red_wing = center + wing_delta_lambda
    
    return {
        'amplitude': amplitude,
        'center': center,
        'sigma': sigma,
        'wing_window': wing_window,
        'min_wavelength': manual_min_wl,
        'max_wavelength': manual_max_wl,
        'reduced_chi2': float(reduced_chi2),
        'reduced_chi2_indep': reduced_chi2_indep,
        'flux': flux,
        'flux_err': flux_err,
        'snr': float(snr),
        'ew': ew,
        'ew_err': ew_err,
        'fwhm_ang': float(fwhm_ang),
        'fwhm_kms': float(fwhm_kms),
        'fwhm_kms_intrinsic': float(fwhm_kms_intrinsic),
        'flux_total': flux_total,
        'wing_delta_lambda': float(wing_delta_lambda),
        'blue_wing': float(blue_wing),
        'red_wing': float(red_wing)
    }
