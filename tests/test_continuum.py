import pytest
import numpy as np
from src.continuum import fit_continuum, parse_window_ranges

def test_parse_window_ranges():
    res = parse_window_ranges("1150:1175, 1300:1350")
    assert res == [(1150.0, 1175.0), (1300.0, 1350.0)]

def test_fit_continuum_exact_powerlaw():
    # Construct exact power-law data: F = 2.0 * λ^(-1.5)
    wl = np.linspace(1000, 2000, 500)
    true_A = 2.0
    true_alpha = -1.5
    flux = true_A * (wl ** true_alpha)
    
    A_fit, alpha_fit, cont_fit, sub_y = fit_continuum(wl, flux, "1150:1175, 1300:1350")
    
    assert pytest.approx(A_fit, rel=1e-3) == true_A
    assert pytest.approx(alpha_fit, rel=1e-3) == true_alpha
    assert np.allclose(sub_y, 0.0, atol=1e-10)
