import numpy as np
from src.reader import load_spectrum
from src.continuum import fit_continuum
from src.candidate_engine import _gaussian, estimate_noise

wl, flux = load_spectrum('15dcdr2dswp19731mxlo.txt')
_, _, cont, sub_y = fit_continuum(wl, flux, '1150:1175,1300:1350')
mean_cont = float(np.mean(cont)) if len(cont) > 0 else 1.0e-13
norm_res = sub_y / abs(mean_cont)
noise = estimate_noise(norm_res) * abs(mean_cont)

center = 1216.73
sigma = 5.90
wing = 20.0
half_wing = wing / 2.0
mask = (wl >= center - half_wing) & (wl <= center + half_wing)
wl_w  = wl[mask]
sy_w  = sub_y[mask]

amp_grid = [5.0e-13, 6.0e-13, 7.0e-13, 8.0e-13, 9.0e-13, 10.0e-13]
for amp in amp_grid:
    model     = _gaussian(wl_w, amp, center, sigma)
    residuals = sy_w - model

    dof = max(len(wl_w) - 3, 1)
    chi2_red = float(np.sum((residuals / noise) ** 2) / dof)
    chi2_score = float(np.exp(-0.5 * ((chi2_red - 1.0) / 2.0) ** 2))

    core_mask = np.abs(wl_w - center) <= sigma
    chi2_core = float(np.sum(((sy_w[core_mask] - model[core_mask]) / noise) ** 2) / (np.sum(core_mask) - 3))
    shape_score = float(np.exp(-0.5 * ((chi2_core - 1.0) / 2.0) ** 2))

    print(f"amp={amp:.1e} chi2_red={chi2_red:.3f} chi2_score={chi2_score:.3f} shape_score={shape_score:.3f}")
