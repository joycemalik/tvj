import numpy as np
import os
import json
from typing import Dict, Any

from src.pipeline import fit_continuum
from src.pipeline import load_spectrum


def main():
    spectrum_path = '15dcdr2dswp19731mxlo.txt'
    if not os.path.exists(spectrum_path):
        print(f"File {spectrum_path} not found.")
        return

    wavelength, flux = load_spectrum(spectrum_path)
    
    # 1. Continuum
    continuum_windows = [(1185, 1195), (1235, 1245)]
    amp_cont, spec_idx_cont, continuum_fit, subtracted_y = fit_continuum(
        wavelength, flux, continuum_windows
    )
    
    # 2. Extract window for Lya
    center = 1215.0
    wing = 20.0
    wl_min = center - wing/2.0
    wl_max = center + wing/2.0
    
    mask = (wavelength >= wl_min) & (wavelength <= wl_max)
    wl_w = wavelength[mask]
    sy_w = subtracted_y[mask]
    cont_w = continuum_fit[mask]
    
    # Find observed peak flux
    peak_idx = np.argmax(sy_w)
    obs_peak_wl = wl_w[peak_idx]
    
    # Set amplitude reference for the scan (using OLS-like amplitude for shape)
    # The shape function normalises internally, so amplitude magnitude just needs to be reasonable
    # Let's fix amplitude to some reasonable peak-based value
    peak_flux = float(sy_w[peak_idx])
    
    sigmas = [5.0, 5.5, 6.0, 6.5, 7.0, 7.2, 7.5, 8.0, 8.5, 9.0]
    
    print("|   sigma | chi2core | chi2wing | Rprofile | Jshape |")
    print("| --: | -----: | -----: | -------: | -----: |")
    
    w_c = 1.0
    w_w = 1.0
    w_p = 1.0
    
    for sigma in sigmas:
        # Build standard Gaussian
        amp = peak_flux
        model = amp * np.exp(-((wl_w - center) ** 2) / (2.0 * sigma ** 2))
        
        # Calculate chi2 core and wing
        mean_cont = max(float(np.mean(cont_w)), 1.0e-15)
        core_mask = np.abs(wl_w - center) <= (1.0 * sigma)
        wing_mask = ~core_mask
        
        core_chi2 = 0.0
        wing_chi2 = 0.0
        r_profile = 0.0
        
        if np.any(core_mask):
            diff = (sy_w[core_mask] - model[core_mask]) / mean_cont
            core_chi2 = float(np.sum(diff**2) / np.sum(core_mask))
            
        if np.any(wing_mask):
            diff = (sy_w[wing_mask] - model[wing_mask]) / mean_cont
            wing_chi2 = float(np.sum(diff**2) / np.sum(wing_mask))
            
        # calculate R_profile
        obs_norm = sy_w / peak_flux
        model_norm = model / peak_flux
        r_profile = float(np.mean(np.abs(obs_norm - model_norm)))
        
        # J_shape
        j_shape = w_c * core_chi2 + w_w * wing_chi2 + w_p * r_profile
        
        print(f"| {sigma:3.1f} | {core_chi2:6.3f} | {wing_chi2:6.3f} | {r_profile:8.3f} | {j_shape:6.3f} |")

if __name__ == '__main__':
    main()
