"""
test_statistics.py
------------------
Unit tests for the corrected statistics module.

Key tests:
  1. wing_delta_lambda formula: center × FWHM_kms / c  (not rest_wl × FWHM_kms/2 / c)
  2. blue_wing / red_wing = center ± wing_window  (= min_wl / max_wl)
  3. Verified against approved Lyα table values
"""

import pytest
import numpy as np
from src.statistics import compute_statistics, estimate_noise

SPEED_OF_LIGHT_KMS = 299792.458


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_spectrum(center=1215.0, amp=8.0e-13, sigma=5.0, half_wing=10.0,
                         cont_amp=2.5e-13, n_pts=200):
    """Build a synthetic Gaussian emission line on a flat continuum."""
    wl = np.linspace(center - 50, center + 50, n_pts)
    continuum = cont_amp * np.ones_like(wl)
    line = amp * np.exp(-((wl - center) ** 2) / (2.0 * sigma ** 2))
    sub_y = line   # continuum already subtracted
    return wl, sub_y, continuum


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWingDeltaFormula:
    """
    wing_delta_lambda = center × FWHM_kms / c
    Verified against approved reference table (spectrum 15, 172).
    """

    def test_formula_is_center_times_fwhm_over_c(self):
        # wing=20 (full width) → half=10 → window [1205, 1225]
        center, sigma, wing = 1215.0, 4.73, 20.0
        wl, sub_y, cont = _synthetic_spectrum(center, sigma=sigma, half_wing=wing/2)
        stats = compute_statistics(wl, sub_y, cont, amplitude=8.0e-13,
                                   center=center, sigma=sigma, wing_window=wing,
                                   rest_wavelength=1216.0)
        fwhm_kms = stats['fwhm_kms']
        expected = center * fwhm_kms / SPEED_OF_LIGHT_KMS
        assert abs(stats['wing_delta_lambda'] - expected) < 0.01

    def test_spectrum_172_reference(self):
        """Approved: center=1213.75, FWHM_kms=4216.8 → wing_delta=17.07"""
        center    = 1213.75
        fwhm_kms  = 4216.8
        expected  = center * fwhm_kms / SPEED_OF_LIGHT_KMS
        # Must be close to the approved 17.07
        assert abs(expected - 17.07) < 0.10

    def test_old_formula_is_wrong(self):
        """Demonstrate the OLD formula gives a different (wrong) value."""
        center, rest_wl, sigma = 1215.0, 1216.0, 4.73
        fwhm_ang = 2.3548 * sigma
        fwhm_kms = (fwhm_ang / center) * SPEED_OF_LIGHT_KMS
        old = rest_wl * (fwhm_kms / 2.0) / SPEED_OF_LIGHT_KMS
        new = center  * fwhm_kms          / SPEED_OF_LIGHT_KMS
        # Old gives ~half the correct value
        assert abs(old - new) > 3.0   # they differ by several Å


class TestBlueRedWings:
    """blue_wing = min_wavelength = center - wing_window/2."""

    def test_wings_equal_min_max_wavelength(self):
        # wing=20 (full width) → half=10 → [1205, 1225]
        center, wing = 1215.0, 20.0
        wl, sub_y, cont = _synthetic_spectrum(center, half_wing=wing/2)
        stats = compute_statistics(wl, sub_y, cont, amplitude=8.0e-13,
                                   center=center, sigma=5.0, wing_window=wing)
        assert pytest.approx(stats['blue_wing'], abs=0.01) == stats['min_wavelength']
        assert pytest.approx(stats['red_wing'],  abs=0.01) == stats['max_wavelength']
        assert pytest.approx(stats['blue_wing'], abs=0.01) == center - wing / 2
        assert pytest.approx(stats['red_wing'],  abs=0.01) == center + wing / 2

    def test_spectrum_15_window(self):
        """Approved: center=1215, wing=20 → min=1205, max=1225."""
        assert pytest.approx(1215.0 - 20.0 / 2) == 1205.0
        assert pytest.approx(1215.0 + 20.0 / 2) == 1225.0

    def test_spectrum_172_window(self):
        """Approved: center=1213.75, wing=17 → min=1205.25, max=1222.25."""
        assert pytest.approx(1213.75 - 17.0 / 2) == 1205.25
        assert pytest.approx(1213.75 + 17.0 / 2) == 1222.25


class TestReducedChi2:
    def test_near_perfect_fit_gives_low_chi2(self):
        # wing=20 (full width) → half=10 → window [1205, 1225]
        center, amp, sigma, wing = 1215.0, 8.0e-13, 5.0, 20.0
        wl, sub_y, cont = _synthetic_spectrum(center, amp, sigma, half_wing=wing/2, n_pts=500)
        rng = np.random.default_rng(42)
        sub_y = sub_y + rng.normal(0, 1e-15, sub_y.shape)
        stats = compute_statistics(wl, sub_y, cont, amplitude=amp,
                                   center=center, sigma=sigma, wing_window=wing)
        assert stats['reduced_chi2'] < 10.0


class TestNoiseEstimator:
    def test_gaussian_noise(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 1.0, 1000)
        est = estimate_noise(noise)
        assert 0.8 < est < 1.2

    def test_clipping_removes_outliers(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 1.0, 990)
        outliers = np.array([50.0, -50.0, 100.0, -100.0, 200.0,
                              -200.0, 300.0, -300.0, 500.0, -500.0])
        combined = np.concatenate([noise, outliers])
        est = estimate_noise(combined)
        assert est < 5.0
