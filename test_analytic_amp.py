import numpy as np
from src.reader import load_spectrum
from src.continuum import fit_continuum

wl, flux = load_spectrum('15dcdr2dswp19731mxlo.txt')
_, _, cont, sub_y = fit_continuum(wl, flux, '1150:1175,1300:1350')

center = 1215.0
sigma = 4.73
wing = 20.0
half_wing = wing / 2.0
mask = (wl >= center - half_wing) & (wl <= center + half_wing)
w = wl[mask]
y = sub_y[mask]

# Analytical amp
G = np.exp(-((w - center) ** 2) / (2.0 * sigma ** 2))
amp_analytic = np.sum(y * G) / np.sum(G * G)

print(f"Analytic Amp: {amp_analytic:.3e}")

# Compare to least_squares
from scipy.optimize import least_squares
def residuals(params):
    return y - params[0] * G

res = least_squares(residuals, x0=[7e-13])
print(f"Least Squares Amp: {res.x[0]:.3e}")
