import numpy as np
import yaml
from src.reader import load_spectrum
from src.continuum import fit_continuum
from src.candidate_engine import _continuous_fit, _gaussian, SCALE_FACTOR, _score_candidate
from src.statistics import estimate_noise

W_CHI2  = 0.28
W_PEAK  = 0.25
W_LEFT  = 0.09
W_RIGHT = 0.08
W_SIGMA = 0.20
W_WING  = 0.10

def test_engine_on_sp17():
    wl, flux = load_spectrum('17dcdr2dswp23064mxlo.txt')
    with open('config/Lya.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    _, _, cont, sub_y = fit_continuum(wl, flux, '1150:1175,1300:1350')
    
    rest_wl = 1216.0
    mean_cont = float(np.mean(cont))
    norm_res = sub_y / abs(mean_cont)
    noise = estimate_noise(norm_res) * abs(mean_cont)

    sigma_prior_center = 5.0
    sigma_prior_std = 1.0

    wing = 21.0
    half_wing = wing / 2.0
    
    for sigma in [3, 4, 5, 6, 7]:
        best_proxy = -1.0
        best_chi2 = -1.0
        best_c = -1
        for center in np.arange(1215.0, 1218.0, 0.5):
            mask_w = (wl >= center - half_wing) & (wl <= center + half_wing)
            wl_w = wl[mask_w]
            sy_w = sub_y[mask_w]
            if len(wl_w) < 5: continue

            for amp in np.linspace(5e-13, 2e-12, 6):
                g_model = amp * np.exp(-((wl_w - center) ** 2) / (2.0 * sigma ** 2))
                dof = max(len(wl_w) - 3, 1)
                chi2_red = float(np.sum(((sy_w - g_model) / noise) ** 2) / dof)
                
                chi2_score = float(np.exp(-0.5 * ((chi2_red - 1.0) / 2.0) ** 2))
                sigma_score = float(np.exp(-0.5 * ((sigma - sigma_prior_center) / sigma_prior_std) ** 2))
                peak_score_obs  = float(np.exp(-0.5 * ((center - 1216.5) / 3.0) ** 2))
                peak_score_rest = float(np.exp(-0.5 * ((center - rest_wl)     / 5.0) ** 2))
                peak_score = 0.6 * peak_score_obs + 0.4 * peak_score_rest
                
                proxy_score = W_CHI2 * chi2_score + W_SIGMA * sigma_score + W_PEAK * peak_score
                
                if proxy_score > best_proxy:
                    best_proxy = proxy_score
                    best_chi2 = chi2_red
                    best_c = center
                    
        print(f"Sigma={sigma}: Best Proxy={best_proxy:.4f}  (chi2_red={best_chi2:.4f}, center={best_c:.2f})")

if __name__ == '__main__':
    test_engine_on_sp17()
