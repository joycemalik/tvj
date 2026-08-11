"""
pipeline.py
-----------
End-to-end spectral fitting pipeline for a single spectrum file.

Workflow
--------
1. Read text spectrum (wavelength, flux columns)
2. Power-law continuum fit & subtraction
3. For each enabled line in the line catalog:
   a. Load per-line config YAML
   b. Run two-stage fitting (coarse grid → continuous optimizer)
   c. Evaluate fit quality
4. Detect and apply joint fitting for overlapping lines
5. Compute global reduced χ²
6. Return complete record

Usage
-----
Single spectrum:
    record = run_single_spectrum_pipeline("spectrum.txt")

Returns a dict with:
    spectrum_name, continuum_amplitude, spectral_index,
    lines        : list of per-line result dicts
    global_reduced_chi2 : float
    joint_fit_applied   : bool
    wavelength, observed_flux, continuum_fit, subtracted_y  (arrays)
"""

import os
import yaml
import numpy as np
from typing import Dict, Any, Optional, List

from src.reader import load_spectrum
from src.continuum import fit_continuum
from src.line_fitter import fit_single_line, load_line_config
from src.joint_fitter import apply_joint_fitting
from src.statistics import estimate_noise
from src.quality import evaluate_fit_quality
from config import DEFAULT_CONTINUUM_WINDOWS


# ---------------------------------------------------------------------------
# Default single-line config (fallback if no lines.yaml)
# ---------------------------------------------------------------------------
_FALLBACK_LINE_CATALOG = [
    {'name': 'Lyman Alpha', 'rest_wavelength': 1216.0, 'config': 'config/Lya.yaml', 'enabled': True}
]


def _load_line_catalog(catalog_path: str) -> List[Dict[str, Any]]:
    """Load lines.yaml catalog, returns list of line entries."""
    if not os.path.exists(catalog_path):
        return _FALLBACK_LINE_CATALOG
    with open(catalog_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data.get('lines', _FALLBACK_LINE_CATALOG)


def _compute_global_chi2(
    wavelength: np.ndarray,
    subtracted_y: np.ndarray,
    line_results: List[Dict[str, Any]],
    noise: float,
) -> float:
    """
    Compute global reduced χ² across all detected emission lines.

    The model is the sum of all fitted Gaussians. Residuals are computed
    only over the union of all fitting windows.
    """
    detected = [r for r in line_results if r.get('detected', False)]
    if not detected:
        return 999.0

    # Build combined model over all windows
    total_model = np.zeros_like(subtracted_y)
    combined_mask = np.zeros(len(wavelength), dtype=bool)

    for r in detected:
        amp    = float(r.get('amplitude', 0.0))
        center = float(r.get('center', 0.0))
        sigma  = float(r.get('sigma', 1.0))
        wl_min = float(r.get('min_wavelength', center - 20))
        wl_max = float(r.get('max_wavelength', center + 20))

        mask = (wavelength >= wl_min) & (wavelength <= wl_max)
        combined_mask |= mask
        total_model[mask] += amp * np.exp(-((wavelength[mask] - center) ** 2) / (2.0 * sigma ** 2))

    if not np.any(combined_mask):
        return 999.0

    residuals = (subtracted_y - total_model)[combined_mask]
    n_params  = 3 * len(detected)   # (A, μ, σ) per line
    dof       = max(int(np.sum(combined_mask)) - n_params, 1)
    chi2_red  = float(np.sum((residuals / noise) ** 2) / dof)
    return chi2_red


# ---------------------------------------------------------------------------
# Single-spectrum pipeline
# ---------------------------------------------------------------------------

def run_single_spectrum_pipeline(
    spectrum_path: str,
    lines_config_path: str = "config/lines.yaml",
    # Legacy single-line support (backward compatible):
    line_config_path: Optional[str] = None,
    continuum_windows: Optional[str] = DEFAULT_CONTINUUM_WINDOWS,
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    End-to-end scientific pipeline for a single spectrum file.

    Parameters
    ----------
    spectrum_path     : path to two-column (wavelength, flux) text file
    lines_config_path : path to lines.yaml catalog (multi-line mode)
    line_config_path  : legacy — path to single-line YAML (e.g. config/Lya.yaml).
                        If given, only that line is fitted (backward compatible).
    continuum_windows : wavelength ranges for power-law continuum fit
    top_n             : max candidates per line

    Returns
    -------
    dict with:
      spectrum_name, continuum_amplitude, spectral_index
      lines                : list of N per-line result dicts
      global_reduced_chi2  : float
      joint_fit_applied    : bool
      wavelength           : ndarray
      observed_flux        : ndarray
      continuum_fit        : ndarray
      subtracted_y         : ndarray
      # Legacy flat keys (best Lyα or first detected line) for backward compatibility:
      amplitude, center, sigma, wing_window, min_wavelength, max_wavelength,
      flux, snr, fwhm_kms, fwhm_ang, ew, reduced_chi2, blue_wing, red_wing,
      wing_delta_lambda, is_acceptable, fit_status, quality_score, shortlist
    """
    spectrum_name = os.path.basename(spectrum_path)
    wavelength, flux = load_spectrum(spectrum_path)

    # ---- Continuum subtraction ----
    amp_cont, spec_idx_cont, continuum_fit, subtracted_y = fit_continuum(
        wavelength, flux, continuum_windows
    )

    # ---- Global noise estimate (used for global χ²) ----
    mean_cont = float(np.mean(continuum_fit)) if len(continuum_fit) > 0 else 1.0e-13
    if mean_cont == 0.0:
        mean_cont = 1.0e-13
    norm_res  = subtracted_y / abs(mean_cont)
    std_norm  = estimate_noise(norm_res)
    noise     = std_norm * abs(mean_cont)
    if noise <= 1.0e-30:
        noise = 1.0e-20

    # ---- Load line catalog ----
    if line_config_path and os.path.exists(line_config_path):
        # Legacy single-line mode
        cfg = load_line_config(line_config_path)
        line_entries = [{'name': cfg.get('line_name', 'Lyman Alpha'),
                         'rest_wavelength': cfg.get('rest_wavelength', 1216.0),
                         'config': line_config_path,
                         'enabled': True,
                         '_preloaded_cfg': cfg}]
    else:
        line_entries = _load_line_catalog(lines_config_path)

    # ---- Fit each enabled line ----
    line_results: List[Dict[str, Any]] = []

    for entry in line_entries:
        if not entry.get('enabled', True):
            continue

        # Load per-line config
        if '_preloaded_cfg' in entry:
            cfg = entry['_preloaded_cfg']
        else:
            cfg_path = entry.get('config', '')
            if not os.path.exists(cfg_path):
                # Inline config from catalog entry (minimal fallback)
                cfg = {
                    'line_name': entry.get('name', ''),
                    'rest_wavelength': float(entry.get('rest_wavelength', 1216.0)),
                }
            else:
                cfg = load_line_config(cfg_path)

        # Ensure line_name is set
        if 'line_name' not in cfg:
            cfg['line_name'] = entry.get('name', f"Line_{entry.get('rest_wavelength', 0):.0f}")

        result = fit_single_line(
            wavelength=wavelength,
            subtracted_y=subtracted_y,
            continuum_fit=continuum_fit,
            line_config=cfg,
            top_n=top_n,
        )
        line_results.append(result)

    if not line_results:
        raise RuntimeError(
            "No emission lines fitted. Check lines.yaml catalog and spectrum wavelength range."
        )

    # ---- Joint fitting for overlapping detected lines ----
    line_results, joint_fit_applied = apply_joint_fitting(
        wavelength=wavelength,
        subtracted_y=subtracted_y,
        continuum_fit=continuum_fit,
        line_results=line_results,
    )

    # ---- Re-evaluate quality for any lines that underwent joint fitting ----
    for i, r in enumerate(line_results):
        if r.get('joint_fit', False):
            cfg_path = r.get('config', '')
            cfg = load_line_config(cfg_path) if os.path.exists(cfg_path) else {}
            snr_min  = float(cfg.get('snr_min', 5.0))
            snr_max  = float(cfg.get('snr_max', 15.0))
            chi2_max = float(cfg.get('chi2_red_max', 5.0))
            obs_peak = r.get('observed_peak_wl', r.get('rest_wavelength', 1216.0))
            
            q_score, is_ok, status = evaluate_fit_quality(
                stats=r,
                observed_peak_wl=obs_peak,
                snr_min=snr_min,
                snr_max=snr_max,
                chi2_max=chi2_max,
            )
            r['detected'] = bool(is_ok)
            r['quality_score'] = float(q_score)
            r['fit_status'] = status

    # ---- Global reduced χ² ----
    global_chi2 = _compute_global_chi2(wavelength, subtracted_y, line_results, noise)

    # ---- Assemble complete record ----
    # Find best detected line for legacy flat-key output (prefer Lyα, else first detected)
    best_line = None
    for r in line_results:
        if r.get('detected', False):
            if r.get('rest_wavelength', 0) == 1216.0 or best_line is None:
                best_line = r

    record: Dict[str, Any] = {
        'spectrum_name':       spectrum_name,
        'continuum_amplitude': float(amp_cont),
        'spectral_index':      float(spec_idx_cont),
        'lines':               line_results,
        'global_reduced_chi2': float(global_chi2),
        'joint_fit_applied':   bool(joint_fit_applied),
        # Raw arrays for plotting
        'wavelength':          wavelength,
        'observed_flux':       flux,
        'continuum_fit':       continuum_fit,
        'subtracted_y':        subtracted_y,
    }

    return record
