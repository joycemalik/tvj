import numpy as np
from src.reader import load_spectrum
from src.continuum import fit_continuum

wl, flux = load_spectrum('15dcdr2dswp19731mxlo.txt')
_, _, cont, sub_y = fit_continuum(wl, flux, '1150:1175,1300:1350')

# Empirical FWHM around 1215
mask = (wl >= 1205) & (wl <= 1225)
wl_w = wl[mask]
sy_w = sub_y[mask]
max_flux = np.max(sy_w)
half_max = max_flux / 2.0

# Find indices where flux crosses half_max
above = sy_w >= half_max
first_above = np.argmax(above)
last_above = len(above) - 1 - np.argmax(above[::-1])

fwhm_ang = wl_w[last_above] - wl_w[first_above]
print("Empirical FWHM_ang:", fwhm_ang)
print("Empirical FWHM_kms:", fwhm_ang / 1215.0 * 299792)

print("Formula FWHM_kms for sigma=4.73:", 4.73 * 2.3548 / 1215.0 * 299792)
