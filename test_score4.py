import numpy as np
from src.reader import load_spectrum
from src.continuum import fit_continuum
from src.candidate_engine import _score_candidate, estimate_noise

wl, flux = load_spectrum('15dcdr2dswp19731mxlo.txt')
_, _, cont, sub_y = fit_continuum(wl, flux, '1150:1175,1300:1350')
mean_cont = float(np.mean(cont)) if len(cont) > 0 else 1.0e-13
norm_res = sub_y / abs(mean_cont)
noise = estimate_noise(norm_res) * abs(mean_cont)

comp_score, comps = _score_candidate(wl, sub_y, 1214.74, 8.0e-13, 1215.13, 4.50, 20.0, noise)
print(f"score 4.50: {comp_score}")
print(f"components 4.50: {comps}")

comp_score, comps = _score_candidate(wl, sub_y, 1214.74, 8.0e-13, 1215.00, 4.73, 20.0, noise)
print(f"score 4.73: {comp_score}")
print(f"components 4.73: {comps}")

comp_score, comps = _score_candidate(wl, sub_y, 1214.74, 8.0e-13, 1215.00, 7.19, 20.0, noise)
print(f"score 7.19: {comp_score}")
print(f"components 7.19: {comps}")
