import numpy as np
from typing import Dict, Any, Tuple
from config import SPEED_OF_LIGHT_KMS, SIGMA_CLIP_ITERS, SIGMA_CLIP_THRESHOLD, LOW_RES_FACTOR

def estimate_noise(residuals: np.ndarray, n_iters: int = SIGMA_CLIP_ITERS, threshold: float = SIGMA_CLIP_THRESHOLD) -> float:
    """
    Estimates noise std via iterative sigma clipping on residual points within fit window,
    matching the JS bundle `Sp(va, mo, Xl)` implementation.
    """
    clipped = residuals.copy()
    for _ in range(n_iters):
        if len(clipped) < 6:
            break
        mean = np.mean(clipped)
        std = np.std(clipped, ddof=1)
        mask = np.abs(clipped - mean) <= threshold * std
        if np.sum(mask) == len(clipped):
            break
        clipped = clipped[mask]
        
    if len(clipped) < 2:
        return 1.0e-20
    return float(np.std(clipped, ddof=1))

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
    rest_wavelength: float = 1216.0
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
    if manual_min_wl is None:
        manual_min_wl = center - wing_window
    if manual_max_wl is None:
        manual_max_wl = center + wing_window
        
    # Fit window mask
    mask = (wavelength >= manual_min_wl) & (wavelength <= manual_max_wl)
    wl_win = wavelength[mask]
    sub_win = subtracted_y[mask]
    cont_win = continuum_fit[mask]
    
    # Model on fit window
    g_model = amplitude * np.exp(-((wl_win - center) ** 2) / (2.0 * (sigma ** 2)))
    res_win = sub_win - g_model
    
    # 1. Noise estimate
    mean_cont = np.mean(continuum_fit) if len(continuum_fit) > 0 else 1.0e-13
    norm_res = sub_win / (mean_cont if mean_cont != 0 else 1.0e-30)
    std_norm = estimate_noise(norm_res)
    
    noise_sigma = std_norm * abs(mean_cont)
    if noise_sigma <= 1.0e-30:
        noise_sigma = 1.0e-20
        
    # 2. Reduced Chi-Squared
    chi2 = np.sum((res_win / noise_sigma) ** 2)
    dof = max(len(wl_win) - 3, 1)
    reduced_chi2 = chi2 / dof
    
    # 3. Flux via Trapezoidal integration
    flux = float(np.trapezoid(sub_win, wl_win)) if len(wl_win) > 1 else 0.0
    
    # Flux Error estimation (matching JS bundle: sqrt(N) * noise_sigma * factor * mean_cont)
    n_pts = len(wl_win)
    flux_err = float(np.sqrt(max(n_pts, 1)) * std_norm * LOW_RES_FACTOR * abs(mean_cont))
    
    # 4. SNR
    snr = abs(flux) / flux_err if flux_err > 0 else 0.0
    
    # 5. Equivalent Width (EW) via Trapezoidal integration of (F_sub / F_cont)
    ratio = np.where(cont_win > 0, sub_win / cont_win, 0.0)
    ew = float(np.trapezoid(ratio, wl_win)) if len(wl_win) > 1 else 0.0
    ew_err = abs(ew) * (flux_err / abs(flux)) if abs(flux) > 0 else 0.0
    
    # 6. FWHM
    fwhm_ang = 2.3548 * sigma
    fwhm_kms = (fwhm_ang / center) * SPEED_OF_LIGHT_KMS if center > 0 else 0.0
    
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
        'flux': flux,
        'flux_err': flux_err,
        'snr': float(snr),
        'ew': ew,
        'ew_err': ew_err,
        'fwhm_ang': float(fwhm_ang),
        'fwhm_kms': float(fwhm_kms),
        'wing_delta_lambda': float(wing_delta_lambda),
        'blue_wing': float(blue_wing),
        'red_wing': float(red_wing)
    }
