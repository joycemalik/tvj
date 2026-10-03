
# pyrefly: ignore [missing-import]
import numpy as np
from typing import List, Tuple, Union

def parse_window_ranges(ranges_str: str) -> List[Tuple[float, float]]:
    """
    Parses range string like '1150:1175, 1300:1350' into a list of (min, max) tuples.
    """
    ranges = []
    for item in ranges_str.split(','):
        item = item.strip()
        if not item:
            continue
        parts = item.split(':')
        if len(parts) == 2:
            try:
                r_min, r_max = float(parts[0]), float(parts[1])
                ranges.append((r_min, r_max))
            except ValueError:
                pass
    return ranges

def fit_continuum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    window_ranges: Union[str, List[Tuple[float, float]]]
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Fits continuum using power-law model F = A * λ^α in log-log space via linear regression,
    reproducing the JS bundle logic:
      ln(F) = α * ln(λ) + ln(A)
    
    Returns:
      amplitude (A), spectral_index (α), continuum_fit, subtracted_y
    """
    if isinstance(window_ranges, str):
        parsed_ranges = parse_window_ranges(window_ranges)
    else:
        parsed_ranges = window_ranges

    win_wl = []
    win_fl = []
    
    for r_min, r_max in parsed_ranges:
        mask = (wavelength >= r_min) & (wavelength <= r_max) & (flux > 0)
        win_wl.extend(wavelength[mask])
        win_fl.extend(flux[mask])
        
    win_wl = np.array(win_wl)
    win_fl = np.array(win_fl)
    
    if len(win_wl) < 2:
        # Fallback if window points are sparse or non-positive
        mask = (flux > 0)
        win_wl = wavelength[mask]
        win_fl = flux[mask]
        
    ln_x = np.log(win_wl)
    ln_y = np.log(win_fl)
    
    n = len(ln_x)
    sum_x = np.sum(ln_x)
    sum_y = np.sum(ln_y)
    sum_xy = np.sum(ln_x * ln_y)
    sum_xx = np.sum(ln_x ** 2)
    
    spectral_index = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
    ln_A = (sum_y - spectral_index * sum_x) / n
    amplitude = np.exp(ln_A)
    
    continuum_fit = amplitude * (wavelength ** spectral_index)
    subtracted_y = flux - continuum_fit

    return amplitude, spectral_index, continuum_fit, subtracted_y


def continuum_fit_errors(
    wavelength: np.ndarray,
    flux: np.ndarray,
    window_ranges: Union[str, List[Tuple[float, float]]],
) -> Tuple[float, float]:
    """
    Standard errors of the log-log OLS power-law fit ln F = α ln λ + ln A:
      σ_α   = √(s² / Sxx)
      σ_lnA = √(s² (1/n + x̄²/Sxx)),  σ_A = A σ_lnA
    with s² = Σ residual² / (n − 2), Sxx = Σ (x − x̄)², x = ln λ.
    Returns (σ_α, σ_A).
    """
    ranges = parse_window_ranges(window_ranges) if isinstance(window_ranges, str) else window_ranges
    m = np.zeros_like(wavelength, dtype=bool)
    for r_min, r_max in ranges:
        m |= (wavelength >= r_min) & (wavelength <= r_max) & (flux > 0)
    if m.sum() < 3:
        return float('nan'), float('nan')
    x, y = np.log(wavelength[m]), np.log(flux[m])
    n = len(x)
    sxx = float(np.sum((x - x.mean()) ** 2))
    alpha, ln_a = np.polyfit(x, y, 1)
    s2 = float(np.sum((y - (alpha * x + ln_a)) ** 2) / (n - 2))
    se_alpha = float(np.sqrt(s2 / sxx))
    se_ln_a = float(np.sqrt(s2 * (1.0 / n + x.mean() ** 2 / sxx)))
    return se_alpha, float(np.exp(ln_a) * se_ln_a)
