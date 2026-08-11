import sys
import numpy as np
from scipy.optimize import least_squares
from src.reader import load_spectrum
from src.continuum import fit_continuum
from src.statistics import estimate_noise
from config import DEFAULT_CONTINUUM_WINDOWS

def main():
    spec = '15dcdr2dswp19731mxlo.txt'
    wl, fl = load_spectrum(spec)
    amp_cont, idx, cont, sub = fit_continuum(wl, fl, DEFAULT_CONTINUUM_WINDOWS)

    center = 1215.0
    sigma = 7.198
    wing_window = 20.0
    half_wing = wing_window / 2.0
    manual_min_wl = center - half_wing
    manual_max_wl = center + half_wing

    mask = (wl >= manual_min_wl) & (wl <= manual_max_wl)
    wl_win = wl[mask]
    sub_win = sub[mask]
    cont_win = cont[mask]
    
    mean_cont = np.mean(cont_win)
    norm_res = sub_win / mean_cont
    noise_sigma = estimate_noise(norm_res) * mean_cont
    
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

    def base_g(x):
        return np.exp(-((x - center)**2) / (2 * sigma**2))

    g_shape = base_g(wl_win)

    methods = []

    # 1. Ordinary LS
    def res_ols(params):
        return (sub_win - params[0] * g_shape) / noise_sigma
    ols_res = least_squares(res_ols, x0=[5e-13], loss='linear')
    methods.append(('Ordinary LS', ols_res.x[0]))

    # 2. Soft-L1
    soft_res = least_squares(res_ols, x0=[5e-13], loss='soft_l1', f_scale=1.0)
    methods.append(('Soft-L1', soft_res.x[0]))

    # 3. Huber
    huber_res = least_squares(res_ols, x0=[5e-13], loss='huber', f_scale=1.0)
    methods.append(('Huber', huber_res.x[0]))

    # 4. Profile-weighted
    # Equal weight to core and wings total
    dist = np.abs(wl_win - center)
    core_mask = dist <= sigma
    wing_mask = dist > sigma
    w = np.ones_like(wl_win)
    if np.sum(core_mask) > 0 and np.sum(wing_mask) > 0:
        core_weight = 1.0 / np.sum(core_mask)
        wing_weight = 1.0 / np.sum(wing_mask)
        w[core_mask] = core_weight
        w[wing_mask] = wing_weight
    
    def res_weighted(params):
        return (sub_win - params[0] * g_shape) / noise_sigma * w
    w_res = least_squares(res_weighted, x0=[5e-13], loss='linear')
    methods.append(('Profile-weighted', w_res.x[0]))

    # 5. Reference
    methods.append(('Reference', 5.029e-13))

    print(f"{'Method':<18} | {'Amplitude':>10} | {'Flux':>10} | {'EW':>7} | {'Peak res':>10} | {'Wing res':>10}")
    print("-" * 75)
    
    for name, amp in methods:
        g_model = amp * g_shape
        flux = float(_trapz(g_model, wl_win))
        gauss_ew = float(_trapz(g_model / cont_win, wl_win))
        
        res = sub_win - g_model
        peak_res = np.mean(np.abs(res[core_mask]))
        wing_res = np.mean(np.abs(res[wing_mask]))
        
        print(f"{name:<18} | {amp:>10.3e} | {flux:>10.3e} | {gauss_ew:>7.2f} | {peak_res:>10.3e} | {wing_res:>10.3e}")

if __name__ == '__main__':
    main()
