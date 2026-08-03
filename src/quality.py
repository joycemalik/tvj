from typing import Dict, Any, Tuple

def evaluate_fit_quality(
    stats: Dict[str, Any],
    observed_peak_wl: float,
    snr_min: float = 5.0,
    snr_max: float = 12.0,
    chi2_max: float = 5.0
) -> Tuple[float, bool, str]:
    """
    Evaluates fit using multi-criteria quality scoring:
    - Reduced χ² close to 1.0 (or below chi2_max)
    - SNR in desired range [snr_min, snr_max] (e.g. 5 to 12)
    - Center alignment with observed emission peak
    
    Returns:
      quality_score (0.0 to 1.0), is_acceptable (bool), status_message (str)
    """
    reasons = []
    penalty = 0.0
    
    chi2 = stats.get('reduced_chi2', 999.0)
    snr = stats.get('snr', 0.0)
    center = stats.get('center', 0.0)
    
    # 1. Chi-squared score component
    if chi2 > chi2_max:
        penalty += 0.3
        reasons.append(f"Reduced χ² high ({chi2:.2f} > {chi2_max})")
    elif chi2 < 0.2:
        penalty += 0.1
        reasons.append(f"Reduced χ² abnormally low ({chi2:.2f})")
        
    # 2. SNR check
    if snr < snr_min:
        penalty += 0.4
        reasons.append(f"SNR too low ({snr:.2f} < {snr_min})")
    elif snr > snr_max:
        penalty += 0.2
        reasons.append(f"SNR out of expected range ({snr:.2f} > {snr_max})")
        
    # 3. Peak alignment check
    peak_diff = abs(center - observed_peak_wl)
    if peak_diff > 2.0:
        penalty += 0.3
        reasons.append(f"Peak misalignment ({peak_diff:.2f} Å off)")
        
    score = max(0.0, 1.0 - penalty)
    is_acceptable = (score >= 0.6) and (snr_min <= snr <= snr_max)
    status = "ACCEPTED" if is_acceptable else f"REJECTED: {', '.join(reasons)}"
    
    return score, is_acceptable, status
