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

def scan_sigma():
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

    print("Sigma | Amp | Center | chi2_red | chi2_score | composite")
    for sigma in np.arange(4.0, 6.6, 0.1):
        # optimize amp and center with fixed sigma
        # We can just do a simple grid search for amp and center for now
        best_chi2 = 1e9
        best_amp = 0
        best_c = center
        for c in np.arange(1214.5, 1215.6, 0.1):
            for amp in np.linspace(5e-13, 10e-13, 20):
                model = _gaussian(w, amp, c, sigma)
                chi2_red = np.sum(((y - model)/noise)**2) / max(len(w)-3, 1)
                if chi2_red < best_chi2:
                    best_chi2 = chi2_red
                    best_amp = amp
                    best_c = c
                    
        # Score it
        chi2_score = float(np.exp(-0.5 * ((best_chi2 - 1.0) / 2.0) ** 2))
        sigma_score = float(np.exp(-0.5 * ((sigma - 5.0) / 1.0) ** 2))
        peak_score = 1.0 # assume 1.0 for simplicity
        wing_score = 1.0
        left_score = 0.5
        right_score = 0.5
        
        comp = (W_CHI2*chi2_score + W_PEAK*peak_score + W_LEFT*left_score + W_RIGHT*right_score + W_SIGMA*sigma_score + W_WING*wing_score)
        
        print(f"{sigma:.2f} | {best_amp:.2e} | {best_c:.2f} | {best_chi2:.3f} | {chi2_score:.3f} | {comp:.3f}")

if __name__ == '__main__':
    scan_sigma()
