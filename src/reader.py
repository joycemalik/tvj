import numpy as np
from typing import Tuple

def load_spectrum(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a two-column whitespace or tab separated spectrum file (wavelength, flux).
    Filters out non-positive or empty entries if necessary, returning clean numpy arrays.
    """
    data = np.loadtxt(filepath)
    wavelength = data[:, 0]
    flux = data[:, 1]
    
    # Ensure sorted by wavelength
    sort_idx = np.argsort(wavelength)
    wavelength = wavelength[sort_idx]
    flux = flux[sort_idx]
    
    return wavelength, flux
