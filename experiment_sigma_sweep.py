import sys
import numpy as np
from scipy.optimize import curve_fit
from src.reader import load_spectrum
from src.continuum import fit_continuum
from src.statistics import estimate_noise
from config import DEFAULT_CONTINUUM_WINDOWS

def main():
    spec = '15dcdr2dswp19731mxlo.txt'
    wl, fl = load_spectrum(spec)
    amp_cont, idx, cont, sub = fit_continuum(wl, fl, DEFAULT_CONTINUUM_WINDOWS)

    center = 1215.0
    wing_window = 20.0
    half_wing = wing_window / 2.0
    manual_min_wl = center - half_wing
    manual_max_wl = center + half_wing

    mask = (wl >= manual_min_wl) & (wl <= manual_max_wl)
    wl_win = wl[mask]
    sub_win = sub[mask]
    cont_win = cont[mask]
    
    # Calculate noise using our new MAD estimator
    mean_cont = np.mean(cont_win)
    norm_res = sub_win / mean_cont
    noise_sigma_norm = estimate_noise(norm_res)
    noise_sigma = noise_sigma_norm * mean_cont
    
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

    sigmas = [5.0, 5.5, 6.0, 6.5, 7.0, 7.198, 7.5, 8.0]
    
    print(f"{'Sigma':>7} | {'Total chi2':>10} | {'Core chi2':>9} | {'Wing chi2':>9} | {'R_profile':>9} | {'Flux':>12} | {'Gauss EW':>9}")
    print("-" * 80)
    
    for sig in sigmas:
        # Fit amplitude for this specific sigma
        def gaussian(x, a):
            return a * np.exp(-((x - center)**2) / (2 * sig**2))
            
        popt, _ = curve_fit(gaussian, wl_win, sub_win, p0=[5e-13])
        amp = popt[0]
        
        g_model = gaussian(wl_win, amp)
        
        # Total chi2
        res = sub_win - g_model
        total_chi2 = np.sum((res / noise_sigma)**2)
        
        # Regions
        dist = np.abs(wl_win - center)
        core_mask = dist <= sig
        wing_mask = dist > sig
        
        core_chi2 = np.sum((res[core_mask] / noise_sigma)**2) if np.sum(core_mask) > 0 else 0
        wing_chi2 = np.sum((res[wing_mask] / noise_sigma)**2) if np.sum(wing_mask) > 0 else 0
        
        # Profile Error
        y_base = np.min(sub_win)
        m_base = np.min(g_model)
        y_norm = (sub_win - y_base) / np.max(sub_win - y_base)
        m_norm = (g_model - m_base) / np.max(g_model - m_base)
        
        r_profile = np.mean((y_norm - m_norm)**2)
        
        # Flux and EW
        flux = float(_trapz(g_model, wl_win))
        gauss_ew = float(_trapz(g_model / cont_win, wl_win))
        
        sig_str = f"{sig:.3f}" if sig == 7.198 else f"{sig:.1f}"
        print(f"{sig_str:>7} | {total_chi2:>9.1f} | {core_chi2:>9.1f} | {wing_chi2:>9.1f} | {r_profile:>9.4f} | {flux:>12.3e} | {gauss_ew:>9.2f}")
        
if __name__ == '__main__':
    main()
