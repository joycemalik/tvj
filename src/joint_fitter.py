"""
joint_fitter.py
---------------
Joint multi-Gaussian fitting for overlapping emission lines.

When two (or more) lines are too close together to be fit independently
(|μ₁ − μ₂| < σ₁ + σ₂), they must be fit simultaneously:

    F_model(λ) = G₁(λ) + G₂(λ) + ... + Gₙ(λ)

where each Gaussian is:

    Gₖ(λ) = Aₖ · exp[ −(λ − μₖ)² / (2σₖ²) ]

This module:
  1. Detects which lines overlap
  2. Groups overlapping lines into clusters
  3. For each cluster, runs a joint scipy.optimize.least_squares
     over all 3N parameters simultaneously
  4. Returns updated fit records for all lines in the cluster
"""

import numpy as np
from scipy.optimize import least_squares
from typing import List, Dict, Any, Tuple

from src.statistics import compute_statistics, estimate_noise

SCALE_FACTOR = 1.0e13


# ---------------------------------------------------------------------------
# Overlap detection
# ---------------------------------------------------------------------------

def detect_overlapping_groups(line_results: List[Dict[str, Any]]) -> List[List[int]]:
    """
    Given a list of single-line fit results (each with 'center' and 'sigma'),
    returns groups of indices that should be jointly fit.

    Two lines overlap if |μ₁ − μ₂| < (σ₁ + σ₂).

    NOTE: overlap detection compares centers and sigmas only.
    Wing window is not used here because it is the fitting boundary,
    not the physical line width.

    Uses union-find to group transitively overlapping lines.
    Lines that don't overlap with anything are their own singleton group.

    Parameters
    ----------
    line_results : list of dicts, each containing 'center' and 'sigma'

    Returns
    -------
    List of groups, each group is a list of indices into line_results.
    E.g. [[0], [1, 2], [3]] means lines 1 and 2 overlap.
    """
    n = len(line_results)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            r_i = line_results[i]
            r_j = line_results[j]
            mu_i   = float(r_i.get('center', r_i.get('rest_wavelength', 0)))
            mu_j   = float(r_j.get('center', r_j.get('rest_wavelength', 0)))
            sig_i  = float(r_i.get('sigma', 5.0))
            sig_j  = float(r_j.get('sigma', 5.0))
            if abs(mu_i - mu_j) < (sig_i + sig_j):
                union(i, j)

    # Collect groups
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return list(groups.values())


# ---------------------------------------------------------------------------
# Joint fitting
# ---------------------------------------------------------------------------

def joint_gaussian_fit(
    wavelength: np.ndarray,
    subtracted_y: np.ndarray,
    continuum_fit: np.ndarray,
    line_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Fit multiple overlapping Gaussians jointly over a combined wavelength window.

    The combined window spans from the minimum min_wavelength to the maximum
    max_wavelength across all lines in the group.

    Parameters
    ----------
    wavelength      : full spectrum wavelength array
    subtracted_y    : continuum-subtracted flux array
    continuum_fit   : continuum model array
    line_results    : list of single-line fit result dicts (from line_fitter)
                      Each must contain: amplitude, center, sigma, wing_window,
                      min_wavelength, max_wavelength, rest_wavelength, line_name

    Returns
    -------
    Updated list of line result dicts with jointly fitted parameters.
    Adds 'joint_fit': True to each updated record.
    """
    if len(line_results) == 0:
        return line_results
    if len(line_results) == 1:
        return line_results   # Nothing to jointly fit

    # ---- Build combined fit window (wing_window = FULL width, half = wing/2) ----
    wl_min = min(r.get('min_wavelength',
                        r['center'] - r.get('wing_window', 20.0) / 2.0)
                 for r in line_results)
    wl_max = max(r.get('max_wavelength',
                        r['center'] + r.get('wing_window', 20.0) / 2.0)
                 for r in line_results)
    mask   = (wavelength >= wl_min) & (wavelength <= wl_max)
    wl_w   = wavelength[mask]
    sy_w   = subtracted_y[mask]

    if len(wl_w) < 3 * len(line_results):
        # Not enough data points for joint fit
        return line_results

    # ---- Noise estimate ----
    mean_cont = float(np.mean(continuum_fit[mask])) if np.any(mask) else 1.0e-13
    if mean_cont == 0.0:
        mean_cont = 1.0e-13
    norm_res = sy_w / abs(mean_cont)
    std_norm = estimate_noise(norm_res)
    noise    = std_norm * abs(mean_cont)
    if noise <= 1.0e-30:
        noise = 1.0e-20

    # ---- Build initial parameter vector x0 = [a1_sc, μ1, σ1, a2_sc, μ2, σ2, ...] ----
    x0      = []
    bounds_lo = []
    bounds_hi = []

    for r in line_results:
        amp   = float(r.get('amplitude', 1.0e-13))
        mu    = float(r.get('center',    r['rest_wavelength']))
        sigma = float(r.get('sigma',     5.0))
        rest  = float(r.get('rest_wavelength', mu))

        x0.extend([amp * SCALE_FACTOR, mu, sigma])
        bounds_lo.extend([0.0,              max(wl_w[0],  rest - 10.0), 1.0])
        bounds_hi.extend([1.0e-10 * SCALE_FACTOR, min(wl_w[-1], rest + 10.0), 20.0])

    n_gauss   = len(line_results)
    sy_scaled = sy_w * SCALE_FACTOR

    def residuals(params):
        model = np.zeros_like(sy_scaled)
        for k in range(n_gauss):
            a_sc = params[3 * k]
            mu   = params[3 * k + 1]
            sig  = params[3 * k + 2]
            if sig > 0:
                model += a_sc * np.exp(-((wl_w - mu) ** 2) / (2.0 * sig ** 2))
        return sy_scaled - model

    try:
        result = least_squares(
            residuals,
            x0=x0,
            bounds=(bounds_lo, bounds_hi),
            method='trf',
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=5000,
        )
        params_opt = result.x
    except Exception:
        # Joint fit failed — return originals unchanged
        return line_results

    # ---- Update each line result with jointly fitted parameters ----
    updated = []
    for k, r in enumerate(line_results):
        opt_amp   = params_opt[3 * k]     / SCALE_FACTOR
        opt_mu    = params_opt[3 * k + 1]
        opt_sigma = params_opt[3 * k + 2]
        wing      = float(r.get('wing_window', 20.0))
        rest_wl   = float(r.get('rest_wavelength', opt_mu))

        new_stats = compute_statistics(
            wavelength=wavelength,
            subtracted_y=subtracted_y,
            continuum_fit=continuum_fit,
            amplitude=opt_amp,
            center=opt_mu,
            sigma=opt_sigma,
            wing_window=wing,
            rest_wavelength=rest_wl,
        )

        updated_record = {**r, **new_stats, 'joint_fit': True}
        updated.append(updated_record)

    return updated


# ---------------------------------------------------------------------------
# Top-level helper
# ---------------------------------------------------------------------------

def apply_joint_fitting(
    wavelength: np.ndarray,
    subtracted_y: np.ndarray,
    continuum_fit: np.ndarray,
    line_results: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Detects overlapping line groups and applies joint fitting to each cluster.

    Parameters
    ----------
    wavelength, subtracted_y, continuum_fit : spectrum arrays
    line_results : list of single-line fit dicts (detected lines only)

    Returns
    -------
    (updated_line_results, joint_fit_applied)
      where joint_fit_applied is True if any group had > 1 line.
    """
    # Only consider detected lines for overlap analysis
    detected_idx    = [i for i, r in enumerate(line_results) if r.get('detected', False)]
    not_detected_idx = [i for i, r in enumerate(line_results) if not r.get('detected', False)]

    detected_results = [line_results[i] for i in detected_idx]

    if len(detected_results) < 2:
        return line_results, False

    groups = detect_overlapping_groups(detected_results)
    joint_applied = any(len(g) > 1 for g in groups)

    updated_detected = [None] * len(detected_results)
    for group in groups:
        group_results = [detected_results[i] for i in group]
        if len(group) > 1:
            fitted_group = joint_gaussian_fit(wavelength, subtracted_y, continuum_fit, group_results)
        else:
            fitted_group = group_results   # singleton — no joint fit needed

        for local_i, global_i in enumerate(group):
            updated_detected[global_i] = fitted_group[local_i]

    # Re-assemble in original order
    final = [None] * len(line_results)
    for local_i, global_i in enumerate(detected_idx):
        final[global_i] = updated_detected[local_i]
    for global_i in not_detected_idx:
        final[global_i] = line_results[global_i]

    return final, joint_applied
