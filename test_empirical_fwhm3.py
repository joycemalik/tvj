import numpy as np
from src.reader import load_spectrum
from src.continuum import fit_continuum

def get_empirical_fwhm_amp(wl, flux, center, wing, amp):
    mask = (wl >= center - wing/2) & (wl <= center + wing/2)
    wl_w = wl[mask]
    y_w = flux[mask]
    
    max_idx = np.argmin(np.abs(wl_w - center)) # peak at center
    half_max = amp / 2.0
    
    # Left crossing
    left_wl = wl_w[0]
    for i in range(max_idx, -1, -1):
        if y_w[i] < half_max:
            if i < len(y_w)-1:
                f1 = y_w[i]
                f2 = y_w[i+1]
                w1 = wl_w[i]
                w2 = wl_w[i+1]
                left_wl = w1 + (half_max - f1) * (w2 - w1) / (f2 - f1)
            else:
                left_wl = wl_w[i]
            break
            
    # Right crossing
    right_wl = wl_w[-1]
    for i in range(max_idx, len(y_w)):
        if y_w[i] < half_max:
            if i > 0:
                f1 = y_w[i-1]
                f2 = y_w[i]
                w1 = wl_w[i-1]
                w2 = wl_w[i]
                right_wl = w1 + (half_max - f1) * (w2 - w1) / (f2 - f1)
            else:
                right_wl = wl_w[i]
            break
            
    return right_wl - left_wl

wl, flux = load_spectrum('15dcdr2dswp19731mxlo.txt')
_, _, _, sub_y = fit_continuum(wl, flux, '1150:1175,1300:1350')
print("Spec 15 (wing 20):", get_empirical_fwhm_amp(wl, sub_y, 1215.0, 20.0, 7.00e-12))

wl, flux = load_spectrum('16dcdr2dswp23063mxlo.txt')
_, _, _, sub_y = fit_continuum(wl, flux, '1150:1175,1300:1350')
print("Spec 16 (wing 21):", get_empirical_fwhm_amp(wl, sub_y, 1216.0, 21.0, 6.69e-12))
