"""
line_fitter.py
--------------
Single-line Gaussian fitting function used by the multi-line pipeline.

Each call fits ONE spectral emission line in a continuum-subtracted spectrum
and returns the best-fit Gaussian parameters plus all derived statistics.

This module delegates the heavy lifting to candidate_engine.generate_candidates()
(two-stage: coarse grid → continuous scipy refinement) and quality.evaluate_fit_quality().
"""

import os
import yaml
import numpy as np
from typing import Dict, Any, Optional

from src.candidate_engine import generate_candidates
from src.quality import evaluate_fit_quality


def fit_single_line(
    wavelength: np.ndarray,
    subtracted_y: np.ndarray,
    continuum_fit: np.ndarray,
    line_config: Dict[str, Any],
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    Fit a single emission line to a continuum-subtracted spectrum.

    Parameters
    ----------
    wavelength      : 1-D array of wavelengths (Å)
    subtracted_y    : 1-D array of continuum-subtracted flux
    continuum_fit   : 1-D array of continuum model flux (same length as wavelength)
    line_config     : dict loaded from the per-line YAML config file. Must contain:
                        - rest_wavelength (float, Å)
                        - optionally: amplitude, center, sigma, wing_window sub-dicts,
                          snr_min, snr_max, chi2_red_max
    top_n           : maximum number of ranked candidates to evaluate

    Returns
    -------
    dict with keys:
      line_name         : str
      rest_wavelength   : float (Å)
      detected          : bool  (False if no candidate passed quality check)
      amplitude         : float
      center            : float (Å)
      sigma             : float (Å)
      wing_window       : float (Å)
      min_wavelength    : float (Å)   = center - wing_window
      max_wavelength    : float (Å)   = center + wing_window
      flux              : float
      flux_err          : float
      snr               : float
      ew                : float (Å)
      fwhm_ang          : float (Å)
      fwhm_kms          : float (km/s)
      wing_delta_lambda : float (Å)
      blue_wing         : float (Å)   = min_wavelength
      red_wing          : float (Å)   = max_wavelength
      reduced_chi2      : float
      quality_score     : float (0..1)
      fit_status        : str  ('ACCEPTED' | 'REJECTED: ...' | 'NOT_DETECTED')
      rank              : int  (1 = best candidate chosen)
      observed_peak_wl  : float (Å)
      composite_score   : float
      score_components  : dict
      shortlist         : list of top-5 candidates (scalars only)
    """
    rest_wl   = float(line_config.get('rest_wavelength', 1216.0))
    line_name = str(line_config.get('line_name', f'Line_{rest_wl:.0f}'))

    # ---- Check that the line falls within the spectrum wavelength range ----
    wl_min, wl_max = float(wavelength[0]), float(wavelength[-1])
    search_radius  = float(line_config.get('peak_search_radius', 15.0))
    line_lo = rest_wl - search_radius
    line_hi = rest_wl + search_radius

    if line_lo > wl_max or line_hi < wl_min:
        return _not_detected_record(line_name, rest_wl, reason="Line outside spectrum range")

    # ---- Check that there is flux in the line region ----
    region_mask = (wavelength >= line_lo) & (wavelength <= line_hi)
    if not np.any(region_mask) or np.all(subtracted_y[region_mask] <= 0):
        return _not_detected_record(line_name, rest_wl, reason="No positive flux in line region")

    # ---- Run two-stage candidate engine ----
    candidates = generate_candidates(
        wavelength=wavelength,
        subtracted_y=subtracted_y,
        continuum_fit=continuum_fit,
        rest_wl=rest_wl,
        config=line_config,
        top_n=top_n,
    )

    if not candidates:
        return _not_detected_record(line_name, rest_wl, reason="No valid fitting candidates found")

    # ---- Select best candidate: first check quality, else take rank-1 ----
    snr_min  = float(line_config.get('snr_min', 5.0))
    snr_max  = float(line_config.get('snr_max', 15.0))
    chi2_max = float(line_config.get('chi2_red_max', 5.0))

    best = None
    for cand in candidates:
        obs_peak = cand.get('observed_peak_wl', rest_wl)
        _, is_ok, _ = evaluate_fit_quality(
            stats=cand,
            observed_peak_wl=obs_peak,
            snr_min=snr_min,
            snr_max=snr_max,
            chi2_max=chi2_max,
        )
        if is_ok:
            best = cand
            break

    # Fall back to rank-1 if none passed quality (we still report with status REJECTED)
    if best is None:
        best = candidates[0]

    # ---- Final quality evaluation on the chosen candidate ----
    obs_peak      = best.get('observed_peak_wl', rest_wl)
    q_score, is_ok, status = evaluate_fit_quality(
        stats=best,
        observed_peak_wl=obs_peak,
        snr_min=snr_min,
        snr_max=snr_max,
        chi2_max=chi2_max,
    )

    # ---- Build result record ----
    shortlist = []
    for cand in candidates[:5]:
        row = {k: v for k, v in cand.items()
               if not isinstance(v, (dict, list, np.ndarray))}
        row['score_components'] = cand.get('score_components', {})
        shortlist.append(row)

    record = {
        'line_name':         line_name,
        'rest_wavelength':   rest_wl,
        'detected':          bool(is_ok),
        'quality_score':     float(q_score),
        'fit_status':        status,
        **best,
    }
    record['shortlist'] = shortlist
    return record


def load_line_config(config_path: str) -> Dict[str, Any]:
    """Load a per-line YAML config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _not_detected_record(line_name: str, rest_wl: float, reason: str = "") -> Dict[str, Any]:
    """Return a zeroed record for a line that could not be detected/fitted."""
    return {
        'line_name':         line_name,
        'rest_wavelength':   rest_wl,
        'detected':          False,
        'quality_score':     0.0,
        'fit_status':        f'NOT_DETECTED: {reason}',
        'amplitude':         0.0,
        'center':            rest_wl,
        'sigma':             0.0,
        'wing_window':       0.0,
        'min_wavelength':    rest_wl,
        'max_wavelength':    rest_wl,
        'flux':              0.0,
        'flux_err':          0.0,
        'snr':               0.0,
        'ew':                0.0,
        'ew_err':            0.0,
        'fwhm_ang':          0.0,
        'fwhm_kms':          0.0,
        'wing_delta_lambda': 0.0,
        'blue_wing':         rest_wl,
        'red_wing':          rest_wl,
        'reduced_chi2':      999.0,
        'composite_score':   0.0,
        'score_components':  {},
        'rank':              0,
        'observed_peak_wl':  rest_wl,
        'shortlist':         [],
    }
