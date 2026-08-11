import numpy as np
import matplotlib.pyplot as plt
from src.reader import load_spectrum
from src.continuum import fit_continuum
from src.candidate_engine import _gaussian

wl, flux = load_spectrum('15dcdr2dswp19731mxlo.txt')
_, _, cont, sub_y = fit_continuum(wl, flux, '1150:1175,1300:1350')

center = 1215.0
half_wing = 15.0
mask = (wl >= center - half_wing) & (wl <= center + half_wing)
w = wl[mask]
y = sub_y[mask]

human_model = _gaussian(w, 6.97e-13, 1215.0, 4.73)
auto_model = _gaussian(w, 8.00e-13, 1214.97, 6.14)

plt.figure(figsize=(10, 6))
plt.step(w, y, where='mid', label='Continuum-subtracted data', color='black')
plt.plot(w, human_model, label='Human Approved (amp=7e-13, sig=4.73)', color='green', linewidth=2)
plt.plot(w, auto_model, label='Auto Broad (amp=8e-13, sig=6.14)', color='red', linestyle='--', linewidth=2)

plt.axvline(1215.0, color='gray', linestyle=':', alpha=0.5)
plt.legend()
plt.title('Spectrum 15: Human vs Auto Fits')
plt.xlabel('Wavelength (A)')
plt.ylabel('Flux')
plt.grid(True, alpha=0.3)
plt.savefig('test_plot_sp15.png')
