"""
candidate_engine.py
-------------------
Two-stage fitting engine for emission-line Gaussian fitting.

Philosophy:
  We are solving a SELECTION problem, not a pure optimisation problem.
  The task is: "Find the Gaussian parameters a human expert would have chosen."

  Wing Window Convention (confirmed from original JS source):
    The original calculator uses: I = R.wingWindow / 2
      manualMinWavelength = R.b - I
      manualMaxWavelength = R.b + I
    So wing_window is the FULL window width.
    half_wing = wing_window / 2  →  fitting region = [center−half, center+half]

  Stage 1 — Coarse discrete grid search:
    Enumerate physically plausible (wing_window, center_init, sigma_init) triplets
    to locate the best fitting window and initial parameter basin.
    Wing is treated as FULL width throughout.

  Stage 2 — Continuous scipy optimisation:
    For each wing candidate, run scipy.optimize.least_squares to refine
    (amplitude, center, sigma) to non-integer precision.
    Sigma bounds from config (e.g. 3.0–8.0 Å for Lyα) constrain the solution
    to the physically relevant range.

Wing-window selection:
  After Stage-2, all candidates are ranked purely by composite_score
  (single consistent metric). The composite_score includes:
    - chi2_score    (χ²_red near 1 is best)
    - peak_score    (center near observed/rest peak)
    - left_score    (residual quality left of center)
    - right_score   (residual quality right of center)
    - sigma_penalty (broad σ relative to half-wing is penalised)

  This guarantees the displayed score is always the same as the ranking score.

Default weights:
  chi2  = 0.45
  peak  = 0.30
  left  = 0.12
  right = 0.08
  sigma = 0.05   (penalty for unreasonably broad Gaussians)
"""

import numpy as np
from scipy.optimize import least_squares
from typing import Dict, Any, List, Tuple

from src.statistics import compute_statistics, estimate_noise


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCALE_FACTOR = 1.0e13          # normalise flux ~1e-13 → ~1 for numerical stability

# Scoring weights
W_CHI2  = 0.28
W_PEAK  = 0.25
W_LEFT  = 0.09
W_RIGHT = 0.08
W_SHAPE = 0.20   # Weight for the core shape (core chi2)
W_WING  = 0.10   # prefer the wing window closest to the scientific expectation

# Default discrete grids (wing is FULL width)
DEFAULT_SIGMA_COARSE  = [3, 4, 5, 6, 7, 8]     # Å — coarse basin finding
DEFAULT_WING_VALS     = list(range(14, 26))     # 14..25 Å full-width
DEFAULT_CENTER_STEP   = 0.5
DEFAULT_CENTER_RANGE  = 5.0
DEFAULT_AMP_STEPS     = 6


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gaussian(wl: np.ndarray, amp: float, center: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-((wl - center) ** 2) / (2.0 * sigma ** 2))


def _continuous_fit(
    wl_win: np.ndarray,
    sub_win: np.ndarray,
    init_amp: float,
    init_center: float,
    fixed_sigma: float,  # Sigma is chosen by Stage 1 objective function and fixed here
    amp_min_sc: float,
    amp_max_sc: float,
) -> Tuple[float, float, float, bool]:
    """
    Stage-2: refine ONLY (amplitude, center) with scipy least_squares.
    Sigma is fixed to preserve the physical shape chosen by the priors.
    """
    sub_scaled = sub_win * SCALE_FACTOR

    def residuals(params):
        a_sc, b = params
        return sub_scaled - a_sc * np.exp(-((wl_win - b) ** 2) / (2.0 * fixed_sigma ** 2))

    center_lo = float(wl_win[0])
    center_hi = float(wl_win[-1])

    x0 = [
        max(amp_min_sc, min(amp_max_sc, init_amp * SCALE_FACTOR)),
        max(center_lo,  min(center_hi,  init_center)),
    ]

    try:
        result = least_squares(
            residuals, x0=x0,
            bounds=([amp_min_sc, center_lo],
                    [amp_max_sc, center_hi]),
            method='trf', ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=2000,
        )
        return (float(result.x[0] / SCALE_FACTOR),
                float(result.x[1]),
                float(fixed_sigma),
                bool(result.success))
    except Exception:
        return float(init_amp), float(init_center), float(fixed_sigma), False


def _score_candidate(
    wl: np.ndarray,
    sub_y: np.ndarray,
    obs_peak_wl: float,
    amp: float,
    center: float,
    sigma: float,
    wing: float,              # FULL width
    noise: float,
    rest_wl: float = 1216.0,
    wing_prior_center: float  = 20.0,  # expected wing (full Å) — from config
    wing_prior_std: float     = 3.0,   # prior width (Å) — how strongly to enforce
) -> Tuple[float, Dict[str, float]]:
    """
    Score a single (amp, center, sigma, wing) candidate after Stage-2 refinement.

    wing is the FULL window width; half_wing = wing/2.

    Returns (composite_score 0..1, component_dict).
    """
    half_wing = wing / 2.0
    mask = (wl >= center - half_wing) & (wl <= center + half_wing)
    wl_w  = wl[mask]
    sy_w  = sub_y[mask]
    if len(wl_w) < 4:
        return 0.0, {}

    model     = _gaussian(wl_w, amp, center, sigma)
    residuals = sy_w - model

    # 1. Chi-squared: reward χ²_red ≈ 1
    dof       = max(len(wl_w) - 3, 1)
    chi2_red  = float(np.sum((residuals / noise) ** 2) / dof)
    chi2_score = float(np.exp(-0.5 * ((chi2_red - 1.0) / 2.0) ** 2))

    # 2. Peak alignment
    peak_score_obs  = float(np.exp(-0.5 * ((center - obs_peak_wl) / 3.0) ** 2))
    peak_score_rest = float(np.exp(-0.5 * ((center - rest_wl)     / 5.0) ** 2))
    peak_score = 0.6 * peak_score_obs + 0.4 * peak_score_rest

    # 3 & 4. Left/right residual quality
    left_mask   = wl_w < center
    right_mask  = wl_w > center
    left_score  = float(np.exp(-np.sqrt(np.mean(residuals[left_mask]  ** 2)) / noise)) if np.any(left_mask)  else 0.5
    right_score = float(np.exp(-np.sqrt(np.mean(residuals[right_mask] ** 2)) / noise)) if np.any(right_mask) else 0.5

    # 5. Shape Score (Core chi2 within +- 1 sigma)
    core_mask = np.abs(wl_w - center) <= sigma
    if np.sum(core_mask) > 3:
        chi2_core = float(np.sum(((sy_w[core_mask] - model[core_mask]) / noise) ** 2) / (np.sum(core_mask) - 3))
    else:
        chi2_core = chi2_red
    shape_score = float(np.exp(-0.5 * ((chi2_core - 1.0) / 2.0) ** 2))

    # 6. Wing prior score
    wing_score = float(np.exp(-0.5 * ((wing - wing_prior_center) / wing_prior_std) ** 2))

    composite = (
        W_CHI2  * chi2_score  +
        W_PEAK  * peak_score  +
        W_LEFT  * left_score  +
        W_RIGHT * right_score +
        W_SHAPE * shape_score +
        W_WING  * wing_score
    )

    components = {
        'chi2_red':           round(chi2_red,    4),
        'chi2_score':         round(chi2_score,  4),
        'peak_score':         round(peak_score,  4),
        'left_score':         round(left_score,  4),
        'right_score':        round(right_score, 4),
        'shape_score':        round(shape_score, 4),
        'wing_score':         round(wing_score,  4),
        'wing_prior_center':  round(wing_prior_center,  4),
        'wing_prior_std':     round(wing_prior_std,     4),
    }
    return float(composite), components


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_candidates(
    wavelength: np.ndarray,
    subtracted_y: np.ndarray,
    continuum_fit: np.ndarray,
    rest_wl: float = 1216.0,
    config: Dict[str, Any] = None,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    if config is None:
        config = {}

    search_radius = float(config.get('peak_search_radius', 15.0))
    peak_region   = (wavelength >= rest_wl - search_radius) & (wavelength <= rest_wl + search_radius)
    if np.any(peak_region):
        wl_reg       = wavelength[peak_region]
        fl_reg       = subtracted_y[peak_region]
        idx          = int(np.argmax(fl_reg))
        obs_peak_wl  = float(wl_reg[idx])
        obs_peak_amp = max(float(fl_reg[idx]), 1.0e-14)
    else:
        obs_peak_wl  = rest_wl
        obs_peak_amp = 8.0e-13

    mean_cont = float(np.mean(continuum_fit)) if len(continuum_fit) > 0 else 1.0e-13
    if mean_cont == 0.0:
        mean_cont = 1.0e-13
    norm_res   = subtracted_y / abs(mean_cont)
    std_norm   = estimate_noise(norm_res)
    noise      = std_norm * abs(mean_cont)
    if noise <= 1.0e-30:
        noise = 1.0e-20

    amp_cfg    = config.get('amplitude') or {}
    center_cfg = config.get('center') or {}
    sigma_cfg  = config.get('sigma') or {}
    wing_cfg   = config.get('wing_window') or {}

    c_range     = float(center_cfg.get('range', DEFAULT_CENTER_RANGE))
    c_step      = float(center_cfg.get('step',  DEFAULT_CENTER_STEP))
    center_grid = np.arange(obs_peak_wl - c_range, obs_peak_wl + c_range + c_step * 0.5, c_step)

    sigma_min    = float(sigma_cfg.get('min', 2.5))
    sigma_max    = float(sigma_cfg.get('max', 6.5))
    sigma_step   = float(sigma_cfg.get('step', 0.1))
    sigma_grid   = np.arange(sigma_min, sigma_max + sigma_step * 0.5, sigma_step)

    wing_vals = wing_cfg.get('grid', DEFAULT_WING_VALS)
    wing_grid = [float(w) for w in wing_vals]
    
    _wing_vals_f = [float(w) for w in wing_vals]
    _wing_mid    = float(np.median(_wing_vals_f)) if _wing_vals_f else 20.0
    wing_prior_center = float(wing_cfg.get('prior_center', _wing_mid))
    wing_prior_std    = float(wing_cfg.get('prior_std',    3.0))

    amp_min_sc = float(amp_cfg.get('min', 1.0e-15)) * SCALE_FACTOR
    amp_max_sc = float(amp_cfg.get('max', 1.0e-10)) * SCALE_FACTOR

    best_per_wing: Dict[float, Tuple[float, float, float, float]] = {}

    for wing in wing_grid:
        half_wing     = wing / 2.0
        best_proxy_w  = -1.0
        best_params_w = (obs_peak_amp, obs_peak_wl, float(np.median(sigma_grid)))

        wing_score = float(np.exp(-0.5 * ((wing - wing_prior_center) / wing_prior_std) ** 2))

        for center in center_grid:
            mask_w = (wavelength >= center - half_wing) & (wavelength <= center + half_wing)
            if np.sum(mask_w) < 5:
                continue
            wl_w = wavelength[mask_w]
            sy_w = subtracted_y[mask_w]

            peak_score_obs  = float(np.exp(-0.5 * ((center - obs_peak_wl) / 3.0) ** 2))
            peak_score_rest = float(np.exp(-0.5 * ((center - rest_wl)     / 5.0) ** 2))
            peak_score = 0.6 * peak_score_obs + 0.4 * peak_score_rest

            for sigma in sigma_grid:
                G = np.exp(-((wl_w - center) ** 2) / (2.0 * sigma ** 2))
                sum_GG = np.sum(G * G)
                if sum_GG == 0: continue
                
                amp = np.sum(sy_w * G) / sum_GG
                if amp < 1e-15: continue
                
                g_model  = amp * G
                dof      = max(len(wl_w) - 3, 1)
                chi2_red = float(np.sum(((sy_w - g_model) / noise) ** 2) / dof)
                chi2_score = float(np.exp(-0.5 * ((chi2_red - 1.0) / 2.0) ** 2))
                
                core_mask = np.abs(wl_w - center) <= sigma
                if np.sum(core_mask) > 3:
                    chi2_core = float(np.sum(((sy_w[core_mask] - g_model[core_mask]) / noise) ** 2) / (np.sum(core_mask) - 3))
                else:
                    chi2_core = chi2_red
                
                shape_score = float(np.exp(-0.5 * ((chi2_core - 1.0) / 2.0) ** 2))
                
                left_mask   = wl_w < center
                right_mask  = wl_w > center
                res_temp = sy_w - g_model
                left_score  = float(np.exp(-np.sqrt(np.mean(res_temp[left_mask]  ** 2)) / noise)) if np.any(left_mask)  else 0.5
                right_score = float(np.exp(-np.sqrt(np.mean(res_temp[right_mask] ** 2)) / noise)) if np.any(right_mask) else 0.5
                
                proxy_score = (
                    W_CHI2  * chi2_score +
                    W_PEAK  * peak_score +
                    W_LEFT  * left_score +
                    W_RIGHT * right_score +
                    W_SHAPE * shape_score +
                    W_WING  * wing_score
                )
                
                if wing == 20.0:
                    print(f"center={center:.2f} sigma={sigma:.2f} p={proxy_score:.3f} ch={chi2_score:.3f} sh={shape_score:.3f}")
                if proxy_score > best_proxy_w:
                    best_proxy_w  = proxy_score
                    best_params_w = (float(amp), float(center), float(sigma))

        best_per_wing[wing] = (best_proxy_w, *best_params_w)

    refined: List[Tuple[float, float, float, float, float, Dict[str, float]]] = []

    for wing, (_, init_amp, init_center, init_sigma) in best_per_wing.items():
        half_wing = wing / 2.0
        mask_w    = (wavelength >= init_center - half_wing) & (wavelength <= init_center + half_wing)
        if np.sum(mask_w) < 5:
            continue
        wl_w = wavelength[mask_w]
        sy_w = subtracted_y[mask_w]

        opt_amp, opt_center, opt_sigma, _ = _continuous_fit(
            wl_w, sy_w,
            init_amp, init_center, init_sigma,
            amp_min_sc, amp_max_sc,
        )

        composite, comps = _score_candidate(
            wl=wavelength, sub_y=subtracted_y,
            obs_peak_wl=obs_peak_wl,
            amp=opt_amp, center=opt_center, sigma=opt_sigma, wing=wing,
            noise=noise, rest_wl=rest_wl,
            wing_prior_center=wing_prior_center,
            wing_prior_std=wing_prior_std,
        )

        refined.append((composite, opt_amp, opt_center, opt_sigma, wing, comps))

    if not refined:
        return []

    # ------------------------------------------------------------------
    # 6. Rank by composite_score (DESCENDING) — single consistent metric.
    #    The displayed score for rank-1 will always be >= rank-2's score.
    # ------------------------------------------------------------------
    refined.sort(key=lambda x: x[0], reverse=True)

    # ------------------------------------------------------------------
    # 7. Compute full statistics for each top candidate (de-dup by wing)
    # ------------------------------------------------------------------
    ranked   = []
    seen     = set()

    for entry in refined:
        composite, amp, center, sigma, wing, comps = entry
        if wing in seen:
            continue
        seen.add(wing)

        stats = compute_statistics(
            wavelength=wavelength,
            subtracted_y=subtracted_y,
            continuum_fit=continuum_fit,
            amplitude=amp,
            center=center,
            sigma=sigma,
            wing_window=wing,
            rest_wavelength=rest_wl,
        )
        stats['composite_score']  = round(float(composite), 4)
        stats['score_components'] = comps
        stats['rank']             = len(ranked) + 1
        stats['observed_peak_wl'] = obs_peak_wl
        ranked.append(stats)

        if len(ranked) >= top_n:
            break

    # Attach optimization landscape grid data to the candidates for UI visualization
    landscape_matrix = []
    # Build grid of wing vs sigma composite scores from refined results
    wings_sorted  = sorted(list(set(r[4] for r in refined)))
    sigmas_sorted = sorted(list(set(r[3] for r in refined)))
    score_lookup  = {(r[4], r[3]): r[0] for r in refined}
    
    for s in sigmas_sorted:
        row = []
        for w in wings_sorted:
            row.append(round(score_lookup.get((w, s), 0.0), 4))
        landscape_matrix.append(row)

    landscape_data = {
        'wings': wings_sorted,
        'sigmas': sigmas_sorted,
        'grid': landscape_matrix
    }

    for r in ranked:
        r['landscape'] = landscape_data

    return ranked
