import numpy as np
from src.reader import load_spectrum
from src.continuum import fit_continuum
from src.statistics import estimate_noise
from src.candidate_engine import _gaussian

wl, flux = load_spectrum('15dcdr2dswp19731mxlo.txt')
_, _, cont, sub_y = fit_continuum(wl, flux, '1150:1175,1300:1350')

mean_cont = float(np.mean(cont))
norm_res = sub_y / abs(mean_cont)
noise = estimate_noise(norm_res) * abs(mean_cont)

center = 1215.0
half_wing = 10.0
mask = (wl >= center - half_wing) & (wl <= center + half_wing)
w = wl[mask]
y = sub_y[mask]

def evaluate_metrics(amp, sigma, name):
    model = _gaussian(w, amp, center, sigma)
    res = y - model
    
    chi2_all = np.sum((res/noise)**2) / max(len(w)-3, 1)
    
    # Core (within +- sigma)
    core_mask = np.abs(w - center) <= sigma
    chi2_core = np.sum((res[core_mask]/noise)**2) / max(np.sum(core_mask)-1, 1)
    
    # Peak (within +- 1 A)
    peak_mask = np.abs(w - center) <= 1.0
    chi2_peak = np.sum((res[peak_mask]/noise)**2) / max(np.sum(peak_mask)-1, 1)
    
    # Signal-weighted chi2
    weights = model / np.max(model)
    chi2_weighted = np.sum(weights * (res/noise)**2) / np.sum(weights)
    
    # Absolute flux difference at peak
    peak_flux_diff = np.mean(np.abs(res[peak_mask])) / noise
    
    print(f"--- {name} (amp={amp:.2e}, sigma={sigma:.2f}) ---")
    print(f"  chi2_all     = {chi2_all:.3f}")
    print(f"  chi2_core    = {chi2_core:.3f}")
    print(f"  chi2_peak    = {chi2_peak:.3f}")
    print(f"  chi2_weight  = {chi2_weighted:.3f}")
    print(f"  peak_diff    = {peak_flux_diff:.3f}")

evaluate_metrics(6.97e-13, 4.73, "Human Approved")
evaluate_metrics(8.00e-13, 6.14, "Auto Broad")
