import math

import numpy as np
import pandas as pd
import pytest

from src.variability import fvar, fvar_by_year
from src.refine import refine_line


def _by_hand(F, E):
    n = len(F)
    m = sum(F) / n
    s2 = sum((f - m) ** 2 for f in F) / (n - 1)
    mse = sum(e * e for e in E) / n
    return n, m, s2, mse, math.sqrt(s2 - mse) / m


class TestFvar:
    def test_matches_hand_calculation_high_ratio_case(self):
        F = [10.0, 14.0, 7.0, 12.0, 9.0]
        E = [0.5, 0.6, 0.4, 0.5, 0.5]
        n, m, s2, mse, fv = _by_hand(F, E)
        r = fvar(F, E)
        assert s2 / mse >= 10
        assert r['fvar'] == pytest.approx(fv)
        assert r['fvar_err'] == pytest.approx(math.sqrt(mse / n) / m)
        assert r['regime'] == 'S² ≫ σ²err'

    def test_low_ratio_case_uses_first_formula(self):
        F = [10.0, 11.0, 9.0, 10.5, 9.5]
        E = [0.6, 0.6, 0.6, 0.6, 0.6]
        n, m, s2, mse, fv = _by_hand(F, E)
        r = fvar(F, E)
        assert 1 < s2 / mse < 10
        assert r['fvar_err'] == pytest.approx(math.sqrt(1 / (2 * n)) * mse / (m ** 2 * fv))
        assert r['regime'] == 'S² ≈ σ²err'

    def test_cutoff_is_inclusive_at_ten(self):
        F = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # sample variance = 2.5
        E = np.full(5, 0.5)                      # mean square error = 0.25 -> ratio exactly 10
        assert fvar(F, E)['regime'] == 'S² ≫ σ²err'

    def test_noise_dominated_is_not_variable(self):
        r = fvar([10.0, 10.1, 9.9], [1.0, 1.0, 1.0])
        assert math.isnan(r['fvar']) and r['regime'] == 'not variable'

    def test_single_point_is_skipped(self):
        assert math.isnan(fvar([10.0], [1.0])['fvar'])

    def test_grouped_by_line_and_year(self):
        df = pd.DataFrame({
            'line': ['A'] * 4 + ['B'] * 2,
            'year': [1980, 1980, 1981, 1981, 1980, 1980],
            'jd': [1.0, 3.0, 400.0, 410.0, 2.0, 5.0],
            'flux': [10, 14, 7, 12, 5, 6],
            'flux_err': [0.5] * 6,
        })
        t = fvar_by_year(df)
        assert list(zip(t.line, t.year)) == [('A', 1980), ('A', 1981), ('B', 1980)]
        assert t.loc[(t.line == 'A') & (t.year == 1981), 'jd_span'].item() == 10.0


def _wl():
    return np.arange(990.0, 1300.0, 1.448)


class TestRefinement:
    def test_recovers_noiseless_gaussian(self):
        wl = _wl()
        A, mu, s = 6e-13, 1214.3, 6.4
        rng = np.random.default_rng(1)
        flux = 3e-13 + 1e-16 * (wl - 1216) + A * np.exp(-0.5 * ((wl - mu) / s) ** 2)
        flux = flux + rng.normal(0, 1e-16, wl.shape)       # tiny noise so DER_SNR is defined
        cfg = {'sigma': {'min': 4.25, 'max': 10.19}, 'refine': {'window': [1180, 1265]}}
        r = refine_line(wl, flux, 1216.0, cfg, {'center': 1216.0, 'sigma': 5.0, 'amplitude': 4e-13})
        assert r['center'] == pytest.approx(mu, abs=0.01)
        assert r['sigma'] == pytest.approx(s, rel=0.01)
        assert r['flux'] == pytest.approx(math.sqrt(2 * math.pi) * A * s, rel=0.01)

    def test_masked_airglow_spike_is_ignored(self):
        wl = _wl()
        A, mu, s = 2e-13, 1029.0, 5.0
        flux = 3e-13 + A * np.exp(-0.5 * ((wl - mu) / s) ** 2)
        flux = flux + 6e-13 * np.exp(-0.5 * ((wl - 1049.8) / 1.5) ** 2)   # geocoronal Lyα
        flux = flux + np.random.default_rng(2).normal(0, 1e-15, wl.shape)
        cfg = {'sigma': {'min': 2.2, 'max': 14.0}, 'mask_ranges': [[1043.0, 1056.0]],
               'refine': {'window': [998, 1085], 'center_tol': 10.0}}
        r = refine_line(wl, flux, 1025.72, cfg, {'center': 1030.0, 'sigma': 5.0, 'amplitude': 1e-13})
        assert r['center'] == pytest.approx(mu, abs=0.3)
        assert r['flux'] == pytest.approx(math.sqrt(2 * math.pi) * A * s, rel=0.05)
