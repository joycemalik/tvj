"""
test_multi_line.py
------------------
Tests for:
  - joint_fitter.detect_overlapping_groups()
  - joint_fitter.joint_gaussian_fit()
  - line_fitter.fit_single_line()
  - pipeline.run_single_spectrum_pipeline() multi-line output structure
"""

import pytest
import numpy as np
from src.joint_fitter import detect_overlapping_groups, joint_gaussian_fit, apply_joint_fitting
from src.line_fitter import _not_detected_record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_line_result(center, sigma, rest_wl=None, detected=True, wing=15.0):
    return {
        'line_name':       f'TestLine_{center:.0f}',
        'rest_wavelength': rest_wl if rest_wl else center,
        'center':          center,
        'sigma':           sigma,
        'wing_window':     wing,
        'min_wavelength':  center - wing,
        'max_wavelength':  center + wing,
        'amplitude':       5.0e-13,
        'detected':        detected,
    }


def _make_spectrum_with_two_gaussians(
    c1=1215.0, s1=5.0, a1=8e-13,
    c2=1240.0, s2=4.0, a2=4e-13,
    cont_val=2e-13, n=500,
):
    """Synthetic spectrum with two Gaussian emission lines on flat continuum."""
    lo = min(c1, c2) - 40
    hi = max(c1, c2) + 40
    wl = np.linspace(lo, hi, n)
    cont = cont_val * np.ones(n)
    line1 = a1 * np.exp(-((wl - c1) ** 2) / (2 * s1 ** 2))
    line2 = a2 * np.exp(-((wl - c2) ** 2) / (2 * s2 ** 2))
    sub_y = line1 + line2
    return wl, sub_y, cont


# ---------------------------------------------------------------------------
# detect_overlapping_groups tests
# ---------------------------------------------------------------------------

class TestDetectOverlapping:
    def test_two_well_separated_lines(self):
        """Lines 30 Å apart with σ=5 should NOT overlap."""
        results = [
            _make_line_result(1215.0, 5.0),
            _make_line_result(1549.0, 6.0),
        ]
        groups = detect_overlapping_groups(results)
        # Each line should be its own group
        assert len(groups) == 2
        assert all(len(g) == 1 for g in groups)

    def test_two_overlapping_lines(self):
        """Lines 6 Å apart with σ=5 and σ=4 → |μ₁-μ₂|=6 < σ₁+σ₂=9 → overlap."""
        results = [
            _make_line_result(1215.0, 5.0),
            _make_line_result(1221.0, 4.0),
        ]
        groups = detect_overlapping_groups(results)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_three_lines_chain_overlap(self):
        """A-B overlap, B-C overlap → all three in one group."""
        results = [
            _make_line_result(1215.0, 5.0),
            _make_line_result(1222.0, 4.0),   # overlaps with 1215 (|Δ|=7 < 9)
            _make_line_result(1228.0, 4.0),   # overlaps with 1222 (|Δ|=6 < 8)
        ]
        groups = detect_overlapping_groups(results)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_singleton_list(self):
        results = [_make_line_result(1215.0, 5.0)]
        groups = detect_overlapping_groups(results)
        assert len(groups) == 1
        assert groups[0] == [0]

    def test_empty_list(self):
        groups = detect_overlapping_groups([])
        assert groups == []


# ---------------------------------------------------------------------------
# apply_joint_fitting tests
# ---------------------------------------------------------------------------

class TestApplyJointFitting:
    def test_no_overlap_returns_unchanged(self):
        """Non-overlapping lines should pass through unchanged."""
        wl, sub_y, cont = _make_spectrum_with_two_gaussians()
        results = [
            {**_make_line_result(1215.0, 5.0), 'amplitude': 8e-13, 'flux': 1e-12,
             'snr': 8.0, 'ew': 40.0, 'fwhm_kms': 4000.0, 'fwhm_ang': 11.0,
             'wing_delta_lambda': 16.0, 'blue_wing': 1200.0, 'red_wing': 1230.0,
             'reduced_chi2': 0.5, 'flux_err': 1e-13, 'ew_err': 4.0},
            {**_make_line_result(1549.0, 6.0), 'amplitude': 4e-13, 'flux': 5e-13,
             'snr': 5.0, 'ew': 20.0, 'fwhm_kms': 3000.0, 'fwhm_ang': 14.0,
             'wing_delta_lambda': 15.0, 'blue_wing': 1534.0, 'red_wing': 1564.0,
             'reduced_chi2': 0.8, 'flux_err': 1e-13, 'ew_err': 2.0},
        ]
        updated, joint_applied = apply_joint_fitting(wl, sub_y, cont, results)
        assert not joint_applied
        assert len(updated) == 2

    def test_single_line_no_joint(self):
        """A single detected line should never trigger joint fitting."""
        wl, sub_y, cont = _make_spectrum_with_two_gaussians()
        results = [
            {**_make_line_result(1215.0, 5.0), 'amplitude': 8e-13, 'flux': 1e-12,
             'snr': 8.0, 'ew': 40.0, 'fwhm_kms': 4000.0, 'fwhm_ang': 11.0,
             'wing_delta_lambda': 16.0, 'blue_wing': 1200.0, 'red_wing': 1230.0,
             'reduced_chi2': 0.5, 'flux_err': 1e-13, 'ew_err': 4.0},
        ]
        updated, joint_applied = apply_joint_fitting(wl, sub_y, cont, results)
        assert not joint_applied


# ---------------------------------------------------------------------------
# _not_detected_record tests
# ---------------------------------------------------------------------------

class TestNotDetectedRecord:
    def test_structure(self):
        rec = _not_detected_record('Lyman Alpha', 1216.0, 'Outside range')
        assert rec['detected'] is False
        assert rec['rest_wavelength'] == 1216.0
        assert 'NOT_DETECTED' in rec['fit_status']
        assert rec['amplitude'] == 0.0
        assert rec['snr'] == 0.0
        assert rec['reduced_chi2'] == 999.0
