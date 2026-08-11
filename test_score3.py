import numpy as np
from src.reader import load_spectrum
from src.continuum import fit_continuum
from src.candidate_engine import _gaussian, estimate_noise

wl, flux = load_spectrum('15dcdr2dswp19731mxlo.txt')
_, _, cont, sub_y = fit_continuum(wl, flux, '1150:1175,1300:1350')
mean_cont = float(np.mean(cont)) if len(cont) > 0 else 1.0e-13
norm_res = sub_y / abs(mean_cont)
noise = estimate_noise(norm_res) * abs(mean_cont)

amp = 8.0e-13
center = 1216.43
sigma = 6.50
wing = 20.0
half_wing = wing / 2.0
mask = (wl >= center - half_wing) & (wl <= center + half_wing)
wl_w  = wl[mask]
sy_w  = sub_y[mask]

model     = _gaussian(wl_w, amp, center, sigma)
residuals = sy_w - model

right_mask   = wl_w > center
right_res = residuals[right_mask]
rmse = np.sqrt(np.mean(right_res ** 2))
print("right wl:", wl_w[right_mask])
print("right sy:", sy_w[right_mask])
print("right model:", model[right_mask])
print("right res:", right_res)
print("RMSE:", rmse)
print("noise:", noise)
print("RMSE / noise:", rmse / noise)
print("exp(-RMSE/noise):", np.exp(-rmse/noise))
