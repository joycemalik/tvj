import numpy as np
from src.reader import load_spectrum
from src.continuum import fit_continuum
from src.candidate_engine import _gaussian
from src.statistics import estimate_noise

def test_metrics_no_prior():
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

    print("Sigma | Amp | chi2_all | chi2_core | sum | peak_diff | combined_score")
    for sigma in np.arange(4.0, 6.6, 0.1):
        best_chi2_all = 1e9
        best_chi2_core = 1e9
        best_peak = 1e9
        best_amp = 0
        
        for c in np.arange(1214.5, 1215.6, 0.1):
            G = np.exp(-((w - c) ** 2) / (2.0 * sigma ** 2))
            sum_GG = np.sum(G * G)
            if sum_GG == 0: continue
            amp = np.sum(y * G) / sum_GG
            
            model = _gaussian(w, amp, c, sigma)
            chi2_all = np.sum(((y - model)/noise)**2) / max(len(w)-3, 1)
            
            # core chi2 (+- sigma of model)
            core_mask = np.abs(w - c) <= sigma
            if np.sum(core_mask) > 3:
                chi2_core = np.sum(((y[core_mask] - model[core_mask])/noise)**2) / (np.sum(core_mask)-3)
            else:
                chi2_core = chi2_all
                
            # peak difference (absolute difference at the very peak)
            peak_mask = np.abs(w - c) <= 1.0
            peak_diff = np.mean(np.abs(y[peak_mask] - model[peak_mask])) / noise
            
            if chi2_all < best_chi2_all:
                best_chi2_all = chi2_all
                best_chi2_core = chi2_core
                best_peak = peak_diff
                best_amp = amp
                
        # Combine metrics into a score (smaller is better for metrics, so we invert)
        # Score = exp(-chi2_all/2) * exp(-chi2_core/2)
        score_all = np.exp(-0.5 * ((best_chi2_all - 1.0)/2.0)**2)
        score_core = np.exp(-0.5 * ((best_chi2_core - 1.0)/2.0)**2)
        combined = 0.5 * score_all + 0.5 * score_core
        
        print(f"{sigma:.2f} | {best_amp:.2e} | {best_chi2_all:.3f} | {best_chi2_core:.3f} | {best_chi2_all+best_chi2_core:.3f} | {best_peak:.3f} | {combined:.3f}")

if __name__ == '__main__':
    test_metrics_no_prior()
