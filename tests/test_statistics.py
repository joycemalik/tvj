import pytest
import numpy as np
from src.statistics import compute_statistics

def test_compute_statistics_ideal_gaussian():
    wl = np.linspace(1200, 1230, 301)
    cont = np.full_like(wl, 1.0e-13)
    a, b, c = 8.0e-13, 1215.67, 6.0
    sub_y = a * np.exp(-((wl - b) ** 2) / (2.0 * (c ** 2)))
    
    stats = compute_statistics(
        wavelength=wl,
        subtracted_y=sub_y,
        continuum_fit=cont,
        amplitude=a,
        center=b,
        sigma=c,
        wing_window=30.0,
        rest_wavelength=1216.0
    )
    
    # Analytical integral of Gaussian: a * c * sqrt(2*pi)
    expected_flux = a * c * np.sqrt(2 * np.pi)
    assert pytest.approx(stats['flux'], rel=0.01) == expected_flux
    assert pytest.approx(stats['fwhm_ang'], rel=1e-3) == 2.3548 * c
