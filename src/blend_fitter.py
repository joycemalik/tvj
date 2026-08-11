import numpy as np
from scipy.optimize import least_squares
from typing import List, Dict, Any, Tuple
from config import SPEED_OF_LIGHT_KMS
from src.statistics import compute_bic

SCALE_FACTOR = 1.0e13

def _tied_components_residuals(params, wl, flux_sub, components_rest_wl):
    """
    params: [A1, A2, ..., An, v, sigma]
    mu_i is computed from v: mu_i = rest_wl_i * (1 + v / c)
    """
    n_comp = len(components_rest_wl)
    amps = params[:n_comp]
    v = params[n_comp]
    sigma = params[n_comp + 1]
    
    model = np.zeros_like(wl)
    for i, rest_wl in enumerate(components_rest_wl):
        # Apply doppler shift
        mu_i = rest_wl * (1.0 + v / SPEED_OF_LIGHT_KMS)
        model += amps[i] * np.exp(-((wl - mu_i) ** 2) / (2.0 * sigma ** 2))
        
    return flux_sub - model


def fit_tied_components(
    wavelength: np.ndarray,
    subtracted_y: np.ndarray,
    noise: float,
    center_guess: float,  # observed center of the primary feature
    sigma_guess: float,
    amp_guess: float,
    components: List[Dict[str, Any]],
    primary_rest_wl: float,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Fit a tied doublet/multiplet model.
    Returns: (list of component parameter dicts, bic)
    """
    comp_rest_wls = [c['rest_wavelength'] for c in components]
    n_comp = len(comp_rest_wls)
    
    # Estimate velocity guess from the primary center
    v_guess = ((center_guess / primary_rest_wl) - 1.0) * SPEED_OF_LIGHT_KMS
    
    # initial guess:
    p0 = []
    # distribute amplitude guess roughly
    for i in range(n_comp):
        p0.append((amp_guess / (i + 1)) * SCALE_FACTOR)
    p0.extend([v_guess, sigma_guess])
    
    bounds_lower = [0.0] * n_comp + [v_guess - 5000.0, 0.5]
    bounds_upper = [np.inf] * n_comp + [v_guess + 5000.0, 20.0]
    
    res = least_squares(
        _tied_components_residuals,
        p0,
        bounds=(bounds_lower, bounds_upper),
        args=(wavelength, subtracted_y * SCALE_FACTOR, comp_rest_wls),
        loss='soft_l1'
    )
    
    p_opt = res.x
    amps_opt = p_opt[:n_comp]
    v_opt = p_opt[n_comp]
    sigma_opt = p_opt[n_comp + 1]
    
    # Calculate BIC
    residuals = _tied_components_residuals(p_opt, wavelength, subtracted_y * SCALE_FACTOR, comp_rest_wls) / SCALE_FACTOR
    chi2_sum = np.sum((residuals / noise) ** 2)
    n_points = len(wavelength)
    k_params = n_comp + 2  # amps + v + sigma
    
    bic = compute_bic(chi2_sum, k_params, n_points)
    
    # Build result
    result_components = []
    for i, c in enumerate(components):
        mu_i = comp_rest_wls[i] * (1.0 + v_opt / SPEED_OF_LIGHT_KMS)
        result_components.append({
            'name': c['name'],
            'rest_wavelength': comp_rest_wls[i],
            'amplitude': amps_opt[i] / SCALE_FACTOR,
            'center': mu_i,
            'sigma': sigma_opt,
            'velocity': v_opt,
        })
    
    return result_components, bic
