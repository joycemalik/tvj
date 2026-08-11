import numpy as np
from src.reader import load_spectrum
from src.continuum import fit_continuum

wl, flux = load_spectrum('15dcdr2dswp19731mxlo.txt')
_, _, _, sub_y = fit_continuum(wl, flux, '1150:1175,1300:1350')

center = 1215.0
wing = 20.0
mask = (wl >= center - wing/2) & (wl <= center + wing/2)
wl_w = wl[mask]
y_w = sub_y[mask]

# Moment 0, 1, 2
m0 = np.sum(y_w)
m1 = np.sum(y_w * wl_w) / m0
m2 = np.sum(y_w * (wl_w - m1)**2) / m0
sigma_mom = np.sqrt(m2)

fwhm_ang = 2.3548 * sigma_mom
fwhm_kms = fwhm_ang / center * 299792
print("Moments FWHM kms:", fwhm_kms)

wl, flux = load_spectrum('16dcdr2dswp23063mxlo.txt')
_, _, _, sub_y = fit_continuum(wl, flux, '1150:1175,1300:1350')
center = 1216.0
wing = 21.0
mask = (wl >= center - wing/2) & (wl <= center + wing/2)
wl_w = wl[mask]
y_w = sub_y[mask]
m0 = np.sum(y_w)
m1 = np.sum(y_w * wl_w) / m0
m2 = np.sum(y_w * (wl_w - m1)**2) / m0
sigma_mom = np.sqrt(m2)
fwhm_ang = 2.3548 * sigma_mom
print("Moments 16 kms:", fwhm_ang / center * 299792)
