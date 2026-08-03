import numpy as np
from scipy.optimize import least_squares
from typing import Dict, Any, Tuple, List
from src.statistics import compute_statistics, estimate_noise

SCALE_FACTOR = 1.0e13  # Normalizes amplitudes of order ~1e-13 to ~1.0

def fit_emission_line_two_stage(
    wavelength: np.ndarray,
    subtracted_y: np.ndarray,
    continuum_fit: np.ndarray,
    rest_wl: float = 1216.0,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Two-stage fitting pipeline with numerically stable parameter scaling:
    Stage 1: Coarse grid search over prior grid ranges
    Stage 2: Continuous bounded minimization using scipy.optimize.least_squares
             with amplitude scaled by 1e13 for optimal gradient conditioning.
    """
    if config is None:
        config = {}
        
    amp_grid = config.get('amplitude', {}).get('grid', [5.0e-13, 6.0e-13, 7.0e-13, 8.0e-13, 9.0e-13, 10.0e-13])
    center_grid = config.get('center', {}).get('grid', [1213.0, 1214.0, 1215.0, 1215.5, 1216.0])
    sigma_grid = config.get('sigma', {}).get('grid', [4.0, 5.0, 6.0, 7.0, 7.5, 8.0])
    wing_grid = config.get('wing_window', {}).get('grid', [5.0, 8.0, 10.0, 13.0, 15.0])
    
    # Identify approximate observed peak around rest wavelength
    peak_region_mask = (wavelength >= rest_wl - 15) & (wavelength <= rest_wl + 15)
    if np.any(peak_region_mask):
        wl_reg = wavelength[peak_region_mask]
        fl_reg = subtracted_y[peak_region_mask]
        obs_peak_idx = np.argmax(fl_reg)
        obs_peak_wl = float(wl_reg[obs_peak_idx])
        obs_peak_amp = max(float(fl_reg[obs_peak_idx]), 1.0e-13)
    else:
        obs_peak_wl = rest_wl
        obs_peak_amp = 8.0e-13
        
    # STAGE 1: COARSE GRID SEARCH
    best_chi2 = 1.0e15
    best_grid_params = (obs_peak_amp, obs_peak_wl, 6.0, 10.0)
    
    mean_cont = np.mean(continuum_fit) if len(continuum_fit) > 0 else 1.0e-13
    
    for wing in wing_grid:
        win_mask = (wavelength >= obs_peak_wl - wing) & (wavelength <= obs_peak_wl + wing)
        wl_win = wavelength[win_mask]
        sub_win = subtracted_y[win_mask]
        if len(wl_win) < 4:
            continue
            
        std_norm = estimate_noise(sub_win / (mean_cont if mean_cont != 0 else 1.0e-13))
        noise = std_norm * abs(mean_cont)
        if noise <= 1.0e-30:
            noise = 1.0e-20
            
        for c_val in center_grid:
            for s_val in sigma_grid:
                for a_val in amp_grid:
                    g_model = a_val * np.exp(-((wl_win - c_val) ** 2) / (2.0 * (s_val ** 2)))
                    chi2 = np.sum(((sub_win - g_model) / noise) ** 2) / max(len(wl_win) - 3, 1)
                    
                    if chi2 < best_chi2:
                        best_chi2 = chi2
                        best_grid_params = (a_val, c_val, s_val, wing)
                        
    init_amp, init_center, init_sigma, best_wing = best_grid_params
    
    # STAGE 2: CONTINUOUS NUMERICAL OPTIMIZATION IN SCALED SPACE
    win_mask = (wavelength >= init_center - best_wing) & (wavelength <= init_center + best_wing)
    wl_win = wavelength[win_mask]
    sub_win = subtracted_y[win_mask]
    
    # Scale sub_win by SCALE_FACTOR for numerically balanced residuals
    sub_win_scaled = sub_win * SCALE_FACTOR
    
    def residuals_scaled(params):
        a_scaled, b, c = params
        model_scaled = a_scaled * np.exp(-((wl_win - b) ** 2) / (2.0 * (c ** 2)))
        return sub_win_scaled - model_scaled
        
    amp_min_scaled = config.get('amplitude', {}).get('min', 1.0e-15) * SCALE_FACTOR
    amp_max_scaled = config.get('amplitude', {}).get('max', 1.0e-10) * SCALE_FACTOR
    center_min = config.get('center', {}).get('min', rest_wl - 15.0)
    center_max = config.get('center', {}).get('max', rest_wl + 15.0)
    sigma_min = config.get('sigma', {}).get('min', 1.0)
    sigma_max = config.get('sigma', {}).get('max', 20.0)
    
    lower_bounds = [amp_min_scaled, center_min, sigma_min]
    upper_bounds = [amp_max_scaled, center_max, sigma_max]
    
    x0 = [init_amp * SCALE_FACTOR, init_center, init_sigma]
    
    res = least_squares(
        residuals_scaled,
        x0=x0,
        bounds=(lower_bounds, upper_bounds),
        method='trf',
        ftol=1e-8,
        xtol=1e-8
    )
    
    opt_amp_scaled, opt_center, opt_sigma = res.x
    opt_amp = opt_amp_scaled / SCALE_FACTOR
    
    # Compute full statistical output
    stats = compute_statistics(
        wavelength=wavelength,
        subtracted_y=subtracted_y,
        continuum_fit=continuum_fit,
        amplitude=opt_amp,
        center=opt_center,
        sigma=opt_sigma,
        wing_window=best_wing,
        rest_wavelength=rest_wl
    )
    
    stats['observed_peak_wl'] = obs_peak_wl
    stats['optimizer_success'] = bool(res.success)
    stats['optimizer_message'] = str(res.message)
    
    return stats
