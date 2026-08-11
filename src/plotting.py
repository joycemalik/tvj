import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Any

def save_publication_plots(record: Dict[str, Any], output_dir: str = "outputs/plots"):
    """
    Generates two publication-quality PNG plots for each spectrum:
    1. Emission line fit plot (Observed spectrum, continuum model, subtracted flux, Gaussian fit)
    2. Residual plot (Subtracted flux - Gaussian fit)
    """
    os.makedirs(output_dir, exist_ok=True)
    spec_name = os.path.splitext(record['spectrum_name'])[0]
    
    wl = record['wavelength']
    flux = record['observed_flux']
    cont = record['continuum_fit']
    sub = record['subtracted_y']
    
    lines = record.get('lines', [])
    best_line = None
    for l in lines:
        if l.get('detected'):
            if l.get('rest_wavelength', 0) == 1216.0 or best_line is None:
                best_line = l
    if not best_line and lines:
        best_line = lines[0]

    if best_line:
        amp = best_line.get('amplitude', 0)
        center = best_line.get('center', 0)
        sigma = best_line.get('sigma', 5.0)
        min_wl = best_line.get('min_wavelength', 0)
        max_wl = best_line.get('max_wavelength', 0)
    else:
        amp, center, sigma, min_wl, max_wl = 0, 0, 5.0, 0, 0
    
    # 1. Emission Fit Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Restrict plot zoom around emission line region
    zoom_mask = (wl >= center - 40) & (wl <= center + 40)
    if not np.any(zoom_mask):
        zoom_mask = np.ones_like(wl, dtype=bool)
        
    wl_z = wl[zoom_mask]
    flux_z = flux[zoom_mask]
    cont_z = cont[zoom_mask]
    sub_z = sub[zoom_mask]
    
    # Gaussian fit evaluation
    g_fit_z = amp * np.exp(-((wl_z - center) ** 2) / (2.0 * (sigma ** 2)))
    
    # Top Panel: Spectrum, Continuum, Subtracted & Gaussian Fit
    ax1.plot(wl_z, flux_z, label='Observed Spectrum', color='blue', alpha=0.6, lw=1.2)
    ax1.plot(wl_z, cont_z, label='Continuum Fit (Power-Law)', color='green', linestyle='--', lw=1.5)
    ax1.plot(wl_z, sub_z, label='Continuum-Subtracted Flux', color='orange', alpha=0.7, lw=1.2)
    ax1.plot(wl_z, g_fit_z, label=f'Gaussian Fit (b={center:.2f}Å, σ={sigma:.2f}Å)', color='red', lw=2.0)
    
    ax1.axvline(min_wl, color='gray', linestyle=':', label='Fit Window Boundaries')
    ax1.axvline(max_wl, color='gray', linestyle=':')
    
    ax1.set_ylabel('Flux (erg s⁻¹ cm⁻² Å⁻¹)', fontsize=12)
    ax1.set_title(f"AGN Emission Line Fit: {spec_name} (Lyα)", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    if best_line:
        info_str = f"χ²_red: {float(best_line.get('reduced_chi2', 0)):.2f}\nSNR: {float(best_line.get('snr', 0)):.2f}\nFlux: {float(best_line.get('flux', 0)):.3e}\nFWHM: {float(best_line.get('fwhm_kms', 0)):.1f} km/s"
    else:
        info_str = "No detected lines"
    ax1.text(0.02, 0.95, info_str, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
             
    # Bottom Panel: Residuals
    res_z = sub_z - g_fit_z
    ax2.plot(wl_z, res_z, color='purple', lw=1.2, label='Residuals')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.7)
    ax2.axvline(min_wl, color='gray', linestyle=':')
    ax2.axvline(max_wl, color='gray', linestyle=':')
    
    ax2.set_xlabel('Wavelength (Å)', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{spec_name}_fit.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    return plot_path
