import numpy as np
from src.reader import load_spectrum
from src.continuum import fit_continuum
from src.candidate_engine import _score_candidate, estimate_noise

wl, flux = load_spectrum('15dcdr2dswp19731mxlo.txt')
_, _, cont, sub_y = fit_continuum(wl, flux, '1150:1175,1300:1350')
mean_cont = float(np.mean(cont)) if len(cont) > 0 else 1.0e-13
norm_res = sub_y / abs(mean_cont)
noise = estimate_noise(norm_res) * abs(mean_cont)

amp = 8.0e-13
center = 1216.43
sigma = 6.50
wing = 20.0

comp_score, comps = _score_candidate(wl, sub_y, 1214.74, amp, center, sigma, wing, noise)
print(f"score: {comp_score}")
print(f"components: {comps}")
