import math

import numpy as np
import pytest

from src.manuscript_fit import fit_line_manuscript, normalised_continuum_rms
from src.variability import fvar


def _spectrum(seed=3):
    wl = np.arange(990.0, 1832.0, 1.448)
    cont = 3e-13 * (wl / 1300.0) ** -1.0
    rng = np.random.default_rng(seed)
    line = 6e-13 * np.exp(-0.5 * ((wl - 1214.0) / 6.0) ** 2)
    flux = cont + line + rng.normal(0, 0.02, wl.shape) * cont
    return wl, flux, cont


def test_normalised_rms_recovers_injected_noise():
    wl, flux, cont = _spectrum()
    assert normalised_continuum_rms(wl, flux, cont) == pytest.approx(0.02, rel=0.25)


def test_manuscript_fit_recovers_line_and_uses_iraf_error():
    wl, flux, cont = _spectrum()
    sc = normalised_continuum_rms(wl, flux, cont)
    cfg = {'sigma': {'min': 4.25, 'max': 10.19}, 'refine': {'window': [1180, 1250]}}
    r = fit_line_manuscript(wl, flux, cont, 1216.0, cfg, {'center': 1215.0, 'sigma': 6.0, 'amplitude': 5e-13}, sc)
    assert r['center'] == pytest.approx(1214.0, abs=0.3)
    assert r['sigma'] == pytest.approx(6.0, rel=0.05)
    assert r['flux'] == pytest.approx(math.sqrt(2 * math.pi) * 6e-13 * 6.0, rel=0.05)
    # σ_F = √N · σ_c · Δλ · F_λ(b)
    expected = math.sqrt(r['n_pix_line']) * sc * r['dispersion'] * r['continuum_at_center']
    assert r['flux_err'] == pytest.approx(expected, rel=1e-9)
    assert r['fwhm_ang_err'] == pytest.approx(2.3548 * r['sigma_err'], rel=1e-9)
    assert r['ew_err'] == pytest.approx(r['ew'] * r['flux_err'] / r['flux'], rel=1e-9)


def test_rmax_and_error():
    F = [10.0, 14.0, 7.0]
    E = [0.5, 0.7, 0.35]
    r = fvar(F, E)
    assert r['rmax'] == pytest.approx(2.0)
    assert r['rmax_err'] == pytest.approx(2.0 * math.sqrt(0.05 ** 2 + 0.05 ** 2))
