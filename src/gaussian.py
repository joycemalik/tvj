# pyrefly: ignore [missing-import]
import numpy as np
from typing import List, Dict

def single_gaussian(wavelength: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    """
    Evaluates G(λ) = a * exp( - (λ - b)^2 / (2 * c^2) )
    """
    return amplitude * np.exp(-((wavelength - center) ** 2) / (2.0 * (sigma ** 2)))

def multi_gaussian(wavelength: np.ndarray, params_list: List[Dict[str, float]]) -> np.ndarray:
    """
    Evaluates sum of multiple Gaussians.
    params_list: list of dicts with keys 'a', 'b', 'c' (or 'amplitude', 'center', 'sigma')
    """
    total = np.zeros_like(wavelength, dtype=float)
    for p in params_list:
        a = p.get('a', p.get('amplitude', 0.0))
        b = p.get('b', p.get('center', 0.0))
        c = p.get('c', p.get('sigma', 1.0))
        total += single_gaussian(wavelength, a, b, c)
    return total
