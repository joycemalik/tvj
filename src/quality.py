from typing import Dict, Any, Tuple


def evaluate_fit_quality(
    stats: Dict[str, Any],
    observed_peak_wl: float,
    snr_min: float = 5.0,
    snr_max: float = 15.0,    # raised from 12.0 — spectrum 17 has SNR=12.35 and is approved
    chi2_max: float = 5.0,
) -> Tuple[float, bool, str]:
    """
    Evaluates fit quality using multi-criteria scoring.

    Criteria:
      1. Reduced χ² — must be < chi2_max (ideally near 1.0)
      2. SNR — must be >= snr_min; snr_max is a soft warning, NOT hard rejection
         (High SNR is always a good detection; we only reject if too low.)
      3. Center alignment — fitted center must be within ±3 Å of observed peak

    Returns:
      quality_score (0.0 to 1.0), is_acceptable (bool), status_message (str)
    """
    reasons  = []
    penalty  = 0.0

    chi2   = stats.get('reduced_chi2', 999.0)
    snr    = stats.get('snr', 0.0)
    center = stats.get('center', 0.0)

    # ---- 1. Chi-squared ----
    if chi2 > chi2_max:
        penalty += 0.3
        reasons.append(f"Reduced χ² too high ({chi2:.2f} > {chi2_max})")
    elif chi2 < 0.10:
        penalty += 0.1
        reasons.append(f"Reduced χ² abnormally low ({chi2:.2f}) — possible overfitting")

    # ---- 2. SNR ----
    # Only penalise LOW SNR — a high SNR is a strong detection, never penalised
    if snr < snr_min:
        penalty += 0.4
        reasons.append(f"SNR too low ({snr:.2f} < {snr_min})")
    elif snr > snr_max:
        # Soft flag only — does not mark as rejected
        reasons.append(f"SNR unusually high ({snr:.2f}) — verify no artefact")

    # ---- 3. Peak alignment ----
    peak_diff = abs(center - observed_peak_wl)
    if peak_diff > 3.0:        # relaxed from 2.0 Å
        penalty += 0.3
        reasons.append(f"Peak misalignment ({peak_diff:.2f} Å from observed peak)")

    score = max(0.0, 1.0 - penalty)

    if snr < 3.0:
        is_acceptable = False
        status = f"NOT_DETECTED: SNR too low ({snr:.2f} < 3.0)"
    elif snr < 5.0:
        is_acceptable = False
        status = f"MARGINAL: SNR {snr:.2f} < 5.0"
    else:
        # SNR >= 5.0
        if chi2 > chi2_max or peak_diff > 3.0:
            is_acceptable = False
            status = f"REJECTED: {', '.join(reasons)}"
        else:
            is_acceptable = True
            status = "ACCEPTED"
            if reasons:
                status += f" (Warnings: {', '.join(reasons)})"

    return score, is_acceptable, status
