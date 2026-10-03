from typing import Dict, Any, Tuple


def evaluate_fit_quality(
    stats: Dict[str, Any],
    observed_peak_wl: float,
    snr_min: float = 3.0,
    snr_max: float = 15.0,
    chi2_max: float = 5.0,
) -> Tuple[float, bool, str]:
    """
    Detection and quality flags for a fitted line.

    Detection: significance = flux / flux_err >= snr_min (3σ by default).
    Warnings (do not reject): reduced χ² above chi2_max, a width or centre at
    its configured bound, centre more than 3 Å from the observed peak.
    A low reduced χ² is not penalised.

    Returns quality_score (0..1), detected (bool), status message.
    """
    snr    = float(stats.get('snr', 0.0))
    chi2   = float(stats.get('reduced_chi2', 999.0))
    center = float(stats.get('center', 0.0))

    warnings = []
    penalty = 0.0
    if chi2 > chi2_max:
        penalty += 0.3
        warnings.append(f"reduced χ² {chi2:.2f} > {chi2_max}")
    if stats.get('at_bound'):
        penalty += 0.2
        warnings.append("parameter at its allowed bound")
    # The least-squares centre carries its own error; the grid peak is only a seed.
    peak_diff = abs(center - observed_peak_wl)
    if peak_diff > 3.0 and not stats.get('refined'):
        penalty += 0.2
        warnings.append(f"centre {peak_diff:.2f} Å from observed peak")

    if snr < snr_min:
        return 0.0, False, f"NOT_DETECTED: significance {snr:.2f} < {snr_min}"

    status = "ACCEPTED"
    if warnings:
        status += f" (Warnings: {', '.join(warnings)})"
    return max(0.0, 1.0 - penalty), True, status
