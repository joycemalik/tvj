import numpy as np
import yaml
from src.reader import load_spectrum
from src.continuum import fit_continuum
from src.candidate_engine import _gaussian
from src.statistics import estimate_noise

def test_core_chi2():
    wl, flux = load_spectrum('15dcdr2dswp19731mxlo.txt')
    _, _, cont, sub_y = fit_continuum(wl, flux, '1150:1175,1300:1350')
    mean_cont = float(np.mean(cont))
    norm_res = sub_y / abs(mean_cont)
    noise = estimate_noise(norm_res) * abs(mean_cont)

    center = 1215.0
    wing = 20.0
    half_wing = wing / 2.0
    mask = (wl >= center - half_wing) & (wl <= center + half_wing)
    w = wl[mask]
    y = sub_y[mask]

    print("Sigma | Amp | Center | chi2_all | chi2_core | core_score")
    for sigma in np.arange(4.0, 6.6, 0.1):
        best_chi2_core = 1e9
        best_chi2_all = 1e9
        best_amp = 0
        best_c = center
        
        for c in np.arange(1214.5, 1215.6, 0.1):
            # Analytic amp
            G = np.exp(-((w - c) ** 2) / (2.0 * sigma ** 2))
            sum_GG = np.sum(G * G)
            if sum_GG == 0: continue
            amp = np.sum(y * G) / sum_GG
            
            model = _gaussian(w, amp, c, sigma)
            chi2_all = np.sum(((y - model)/noise)**2) / max(len(w)-3, 1)
            
            # Compute core chi2 (fixed physical width, e.g. +- 5 A)
            core_mask = np.abs(w - c) <= 5.0
            if np.sum(core_mask) > 3:
                chi2_core = np.sum(((y[core_mask] - model[core_mask])/noise)**2) / (np.sum(core_mask)-3)
            else:
                chi2_core = 1e9
                
            if chi2_core < best_chi2_core:
                best_chi2_core = chi2_core
                best_chi2_all = chi2_all
                best_amp = amp
                best_c = c
                    
        core_score = float(np.exp(-0.5 * ((best_chi2_core - 1.0) / 2.0) ** 2))
        print(f"{sigma:.2f} | {best_amp:.2e} | {best_c:.2f} | {best_chi2_all:.3f} | {best_chi2_core:.3f} | {core_score:.3f}")

if __name__ == '__main__':
    test_core_chi2()
