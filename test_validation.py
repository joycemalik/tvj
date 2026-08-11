import pytest
from src.quality import evaluate_fit_quality

def test_validation_tiers():
    """
    Explicitly test the scientific validation tiers:
    - SNR = 1.6 -> NOT DETECTED
    - SNR = 4.8 -> MARGINAL
    - SNR = 12.0, chi2 = 1.0 -> ACCEPTED
    - SNR = 20.0, chi2 = 20.0 -> REJECTED
    """
    obs_peak = 1216.0
    
    # 1. NOT DETECTED (SNR < 3)
    stats_not_detected = {'snr': 1.6, 'reduced_chi2': 1.21, 'center': 1216.0}
    score, is_acc, status = evaluate_fit_quality(stats_not_detected, obs_peak)
    assert not is_acc
    assert 'NOT_DETECTED' in status

    # 2. MARGINAL (3 <= SNR < 5)
    stats_marginal = {'snr': 4.8, 'reduced_chi2': 0.632, 'center': 1216.0}
    score, is_acc, status = evaluate_fit_quality(stats_marginal, obs_peak)
    assert not is_acc
    assert 'MARGINAL' in status

    # 3. ACCEPTED (SNR >= 5, chi2 <= 5, peak ok)
    stats_accepted = {'snr': 12.18, 'reduced_chi2': 0.974, 'center': 1216.0}
    score, is_acc, status = evaluate_fit_quality(stats_accepted, obs_peak)
    assert is_acc
    assert 'ACCEPTED' in status
    
    # 4. REJECTED (SNR >= 5, but chi2 > 5)
    stats_rejected = {'snr': 19.9, 'reduced_chi2': 19.37, 'center': 1216.0}
    score, is_acc, status = evaluate_fit_quality(stats_rejected, obs_peak)
    assert not is_acc
    assert 'REJECTED' in status

    # 5. REJECTED due to peak misalignment
    stats_misaligned = {'snr': 12.0, 'reduced_chi2': 1.0, 'center': 1220.0} # diff > 3.0
    score, is_acc, status = evaluate_fit_quality(stats_misaligned, obs_peak)
    assert not is_acc
    assert 'REJECTED' in status
