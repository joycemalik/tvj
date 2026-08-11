import numpy as np
from src.reader import load_spectrum
from src.continuum import fit_continuum
from src.candidate_engine import _gaussian
from src.statistics import estimate_noise

W_CHI2 = 0.28
W_SHAPE = 0.20
W_PEAK = 0.25
W_LEFT = 0.09
W_RIGHT = 0.08

def test_metrics_no_prior(filename, expected_sigma):
    wl, flux = load_spectrum(filename)
    _, _, cont, sub_y = fit_continuum(wl, flux, '1150:1175,1300:1350')
    mean_cont = float(np.mean(cont))
    norm_res = sub_y / abs(mean_cont)
    noise = estimate_noise(norm_res) * abs(mean_cont)

    obs_center = 1216.0 if '16' in filename else 1217.0
    if '15' in filename: obs_center = 1215.0
    wing = 20.0
    half_wing = wing / 2.0

    best_combined = -1
    best_sig = -1
    
    for sigma in np.arange(2.5, 7.0, 0.1):
        best_chi2_all = 1e9
        best_chi2_core = 1e9
        best_c = obs_center
        best_left = 0
        best_right = 0
        
        for c in np.arange(obs_center-1.0, obs_center+1.0, 0.1):
            mask = (wl >= c - half_wing) & (wl <= c + half_wing)
            w = wl[mask]
            y = sub_y[mask]
            
            G = np.exp(-((w - c) ** 2) / (2.0 * sigma ** 2))
            sum_GG = np.sum(G * G)
            if sum_GG == 0: continue
            amp = np.sum(y * G) / sum_GG
            
            model = _gaussian(w, amp, c, sigma)
            res = y - model
            chi2_all = np.sum((res/noise)**2) / max(len(w)-3, 1)
            
            core_mask = np.abs(w - c) <= sigma
            if np.sum(core_mask) > 3:
                chi2_core = np.sum((res[core_mask]/noise)**2) / (np.sum(core_mask)-3)
            else:
                chi2_core = chi2_all
                
            left_mask = (w < c - 1.5 * sigma)
            right_mask = (w > c + 1.5 * sigma)
            left_score = np.exp(-np.sqrt(np.mean(res[left_mask]**2)) / noise) if np.any(left_mask) else 0.5
            right_score = np.exp(-np.sqrt(np.mean(res[right_mask]**2)) / noise) if np.any(right_mask) else 0.5
            
            score_all = np.exp(-0.5 * ((chi2_all - 1.0)/2.0)**2)
            score_core = np.exp(-0.5 * ((chi2_core - 1.0)/2.0)**2)
            
            peak_score_obs  = np.exp(-0.5 * ((c - obs_center) / 3.0) ** 2)
            peak_score_rest = np.exp(-0.5 * ((c - 1216.0)     / 5.0) ** 2)
            peak_score = 0.6 * peak_score_obs + 0.4 * peak_score_rest
            
            combined = W_CHI2 * score_all + W_SHAPE * score_core + W_PEAK * peak_score + W_LEFT * left_score + W_RIGHT * right_score
            
            if combined > best_combined:
                best_combined = combined
                best_sig = sigma
            
    print(f"{filename}: Expected={expected_sigma}, Found={best_sig:.2f} (score={best_combined:.3f})")

if __name__ == '__main__':
    test_metrics_no_prior('15dcdr2dswp19731mxlo.txt', 4.73)
    test_metrics_no_prior('16dcdr2dswp23063mxlo.txt', 4.86)
    test_metrics_no_prior('17dcdr2dswp23064mxlo.txt', 4.49)
