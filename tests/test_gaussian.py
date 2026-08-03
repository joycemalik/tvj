import pytest
import numpy as np
from src.gaussian import single_gaussian, multi_gaussian

def test_single_gaussian_peak():
    wl = np.linspace(1200, 1230, 301)
    a, b, c = 8.0e-13, 1215.5, 6.0
    g = single_gaussian(wl, a, b, c)
    
    # Peak value at λ=b should equal a
    peak_idx = np.argmin(np.abs(wl - b))
    assert pytest.approx(g[peak_idx], rel=1e-3) == a

def test_multi_gaussian_sum():
    wl = np.linspace(1200, 1230, 301)
    p1 = {'a': 5.0e-13, 'b': 1214.0, 'c': 4.0}
    p2 = {'a': 3.0e-13, 'b': 1218.0, 'c': 5.0}
    
    g_multi = multi_gaussian(wl, [p1, p2])
    g1 = single_gaussian(wl, 5.0e-13, 1214.0, 4.0)
    g2 = single_gaussian(wl, 3.0e-13, 1218.0, 5.0)
    
    assert np.allclose(g_multi, g1 + g2)
