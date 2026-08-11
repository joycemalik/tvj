import numpy as np
import os
import yaml
from src.reader import load_spectrum
from src.continuum import fit_continuum
from src.line_fitter import fit_single_line, load_line_config
from src.statistics import estimate_noise, compute_bic
from src.blend_fitter import fit_tied_components

def test_models():
    spec_path = "15dcdr2dswp19731mxlo.txt"
    wavelength, flux = load_spectrum(spec_path)
    
    # Continuum subtraction
    amp_cont, spec_idx_cont, continuum_fit, subtracted_y = fit_continuum(
        wavelength, flux, [[1150, 1200], [1250, 1300]]
    )
    
    noise = estimate_noise(subtracted_y)
    
    print("--- MODEL A: Lya Only ---")
    lya_cfg = load_line_config("config/Lya.yaml")
    lya_res = fit_single_line(wavelength, subtracted_y, continuum_fit, lya_cfg, top_n=5)
    chi2_A = lya_res['reduced_chi2'] * (len(wavelength) - 3)
    bic_A = compute_bic(chi2_A, 3, len(wavelength))
    print(f"Lya Only BIC: {bic_A:.2f}")

    print("--- MODEL B: Lya + Single NV ---")
    nv_cfg = load_line_config("config/NV.yaml")
    # Fake a single NV model for comparison
    nv_single_cfg = dict(nv_cfg)
    nv_single_cfg['model'] = 'single_gaussian'
    nv_res = fit_single_line(wavelength, subtracted_y, continuum_fit, nv_single_cfg, top_n=5)
    # This is slightly inaccurate because it doesn't fit them jointly, but it's an isolated approximation.
    chi2_B = (lya_res['reduced_chi2'] * (len(wavelength) - 3)) + (nv_res['reduced_chi2'] * (len(wavelength) - 3))
    bic_B = compute_bic(chi2_B, 6, len(wavelength))
    print(f"Lya + Single NV BIC: {bic_B:.2f}")

    print("--- MODEL C: Lya + Tied NV Doublet ---")
    with open("config/NV.yaml", 'r') as f:
        nv_full = yaml.safe_load(f)
        
    components, bic_C_nv = fit_tied_components(
        wavelength=wavelength,
        subtracted_y=subtracted_y,
        noise=noise,
        center_guess=1240.0,
        sigma_guess=5.0,
        amp_guess=nv_res.get('amplitude', 5e-13),
        components=nv_full['components'],
        primary_rest_wl=1240.0
    )
    
    chi2_C = (lya_res['reduced_chi2'] * (len(wavelength) - 3))
    bic_C = compute_bic(chi2_C, 3, len(wavelength)) + bic_C_nv
    print(f"Lya + Tied NV BIC: {bic_C:.2f}")
    
    for c in components:
        print(f"Component: {c['name']} at {c['center']:.2f} A, sigma={c['sigma']:.2f}, A={c['amplitude']:.2e}")

if __name__ == "__main__":
    test_models()
