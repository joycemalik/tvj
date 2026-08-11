"""
candidate_engine.py
-------------------
Discrete grid-search fitting engine.

Philosophy (per user insight):
  "We are solving a DECISION-MAKING problem, not an optimisation problem."

Instead of a continuous solver that finds a mathematically optimal minimum,
this engine enumerates ~8 000 physically-plausible discrete candidates and
scores each one with a multi-criteria metric that mirrors how a human expert
astronomer selects a fit.

Scoring formula (weighted sum, 0..1, higher is better):
  Score = w_chi2  * chi2_component
        + w_peak  * peak_alignment_component
        + w_left  * left_wing_component
        + w_right * right_wing_component
        + w_snr   * snr_component

Default weights:
  chi2  = 0.35
  peak  = 0.25
  left  = 0.20
  right = 0.10
  snr   = 0.10
"""

import numpy as np
from typing import Dict, Any, List, Tuple

from src.statistics import compute_statistics, estimate_noise


# ---------------------------------------------------------------------------
# Default discrete parameter grids
# ---------------------------------------------------------------------------
DEFAULT_AMP_STEPS    = 20
DEFAULT_CENTER_STEP  = 0.5        # Angstrom
DEFAULT_CENTER_RANGE = 5.0        # +/- Angstrom around observed peak
DEFAULT_SIGMA_VALS   = [4, 5, 6, 7, 8, 9, 10]   # expert-preferred integer Angstrom values
DEFAULT_WING_VALS    = list(range(10, 26))        # 10..25 Angstrom

# Scoring weights
W_CHI2  = 0.35
W_PEAK  = 0.25
W_LEFT  = 0.20
W_RIGHT = 0.10
W_SNR   = 0.10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gaussian(wl: np.ndarray, amp: float, center: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-((wl - center) ** 2) / (2.0 * sigma ** 2))


def _score_candidate(
    wl: np.ndarray,
    sub_y: np.ndarray,
    obs_peak_wl: float,
    amp: float,
    center: float,
    sigma: float,
    wing: float,
    noise: float,
    snr_target: float = 7.0,
    rest_wl: float = 1216.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate a single (amp, center, sigma, wing) candidate.
    Returns (composite_score, component_dict).
    """
    mask = (wl >= center - wing) & (wl <= center + wing)
    wl_w = wl[mask]
    sy_w = sub_y[mask]
    if len(wl_w) < 4:
        return 0.0, {}

    model     = _gaussian(wl_w, amp, center, sigma)
    residuals = sy_w - model

    # 1. chi2 component
    # Prefer chi2_red near 1.0.
    # Penalise overfitting (chi2_red < 0.15) almost as strongly as underfitting.
    dof       = max(len(wl_w) - 3, 1)
    chi2_red  = float(np.sum((residuals / noise) ** 2) / dof)
    if chi2_red < 0.15:
        # Overfitting penalty: mirror the chi2 > 1 penalty around 0.15
        effective_chi2 = 1.0 + (0.15 - chi2_red) * 4.0  # inflate toward bad territory
    else:
        effective_chi2 = chi2_red
    chi2_score = float(np.exp(-0.5 * ((effective_chi2 - 1.0) / 2.0) ** 2))

    # 2. Peak alignment component
    # Use obs_peak_wl (the auto-detected peak) with a WIDE tolerance (sigma=3 A)
    # so it provides a weak, soft prior rather than a hard anchor.
    # Additionally reward being near the rest wavelength (wide sigma=5 A)
    # to avoid locking onto noise spikes.
    peak_offset_obs  = abs(center - obs_peak_wl)
    peak_offset_rest = abs(center - rest_wl)
    # Blend: 60% toward observed peak (wide, forgiving), 40% toward rest wavelength
    peak_score_obs  = float(np.exp(-0.5 * (peak_offset_obs  / 3.0) ** 2))
    peak_score_rest = float(np.exp(-0.5 * (peak_offset_rest / 5.0) ** 2))
    peak_score = 0.6 * peak_score_obs + 0.4 * peak_score_rest

    # 3 & 4. Left/right wing residual quality
    left_mask  = wl_w < center
    right_mask = wl_w > center
    if np.any(left_mask):
        left_score = float(np.exp(-np.sqrt(np.mean(residuals[left_mask] ** 2)) / noise))
    else:
        left_score = 0.5

    if np.any(right_mask):
        right_score = float(np.exp(-np.sqrt(np.mean(residuals[right_mask] ** 2)) / noise))
    else:
        right_score = 0.5

    # 5. SNR component (prefer SNR near expert-validated target)
    raw_snr   = amp / noise if noise > 0 else 0.0
    snr_score = float(np.exp(-0.5 * ((raw_snr - snr_target) / (snr_target * 0.5)) ** 2))

    # Diagnostic Shape Components (Step 4A)
    dist = np.abs(wl_w - center)
    core_mask = dist <= sigma
    wing_mask = dist > sigma
    
    res_w = sy_w - model
    chi2_core = float(np.sum((res_w[core_mask] / noise) ** 2) / max(1, np.sum(core_mask)))
    chi2_wing = float(np.sum((res_w[wing_mask] / noise) ** 2) / max(1, np.sum(wing_mask)))
    
    y_base = np.min(sy_w)
    m_base = np.min(model)
    y_norm = (sy_w - y_base) / (np.max(sy_w - y_base) + 1e-30)
    m_norm = (model - m_base) / (np.max(model - m_base) + 1e-30)
    R_profile = float(np.mean((y_norm - m_norm) ** 2))
    
    J_shape = float(1.0 * chi2_core + 1.5 * chi2_wing + 1.0 * R_profile)

    composite = (
        W_CHI2  * chi2_score  +
        W_PEAK  * peak_score  +
        W_LEFT  * left_score  +
        W_RIGHT * right_score +
        W_SNR   * snr_score
    )

    components = {
        'chi2_red':    round(chi2_red,    4),
        'chi2_score':  round(chi2_score,  4),
        'peak_score':  round(peak_score,  4),
        'left_score':  round(left_score,  4),
        'right_score': round(right_score, 4),
        'snr_score':   round(snr_score,   4),
        'chi2_core':   round(chi2_core,   4),
        'chi2_wing':   round(chi2_wing,   4),
        'R_profile':   round(R_profile,   6),
        'J_shape':     round(J_shape,     4),
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
    """
    Discrete grid-search over (amplitude, center, sigma, wing_window).
    Returns the top_n candidates ranked by composite score (descending).

    Each candidate dict contains all fields from compute_statistics() plus:
      - composite_score  : float  (0..1, higher = better)
      - score_components : dict   (per-criterion breakdown)
      - rank             : int    (1 = best)
      - observed_peak_wl : float
    """
    if config is None:
        config = {}

    # ------------------------------------------------------------------
    # 1. Locate observed emission peak near rest wavelength
    # ------------------------------------------------------------------
    search_radius = float(config.get('peak_search_radius', 15.0))
    peak_region   = (wavelength >= rest_wl - search_radius) & (wavelength <= rest_wl + search_radius)
    if np.any(peak_region):
        wl_reg       = wavelength[peak_region]
        fl_reg       = subtracted_y[peak_region]
        idx          = np.argmax(fl_reg)
        obs_peak_wl  = float(wl_reg[idx])
        obs_peak_amp = max(float(fl_reg[idx]), 1.0e-14)
    else:
        obs_peak_wl  = rest_wl
        obs_peak_amp = 8.0e-13

    # ------------------------------------------------------------------
    # 2. Noise estimate
    # ------------------------------------------------------------------
    mean_cont = float(np.mean(continuum_fit)) if len(continuum_fit) > 0 else 1.0e-13
    if mean_cont == 0:
        mean_cont = 1.0e-13
    broad_mask = (wavelength >= rest_wl - 20) & (wavelength <= rest_wl + 20)
    norm_res   = subtracted_y[broad_mask] / abs(mean_cont) if np.any(broad_mask) else subtracted_y / abs(mean_cont)
    std_norm   = estimate_noise(norm_res)
    noise      = std_norm * abs(mean_cont)
    if noise <= 1.0e-30:
        noise = 1.0e-20

    # ------------------------------------------------------------------
    # 3. Parameter grids
    # ------------------------------------------------------------------
    amp_cfg    = config.get('amplitude') or {}
    center_cfg = config.get('center') or {}
    sigma_cfg  = config.get('sigma') or {}
    wing_cfg   = config.get('wing_window') or {}
    wing_val   = float(wing_cfg.get('value', wing_cfg.get('grid', [20.0])[0]))

    amp_lo     = float(amp_cfg.get('min', obs_peak_amp * 0.4))
    amp_hi     = float(amp_cfg.get('max', obs_peak_amp * 2.0))
    amp_steps  = int(amp_cfg.get('steps', DEFAULT_AMP_STEPS))
    amp_grid   = np.linspace(amp_lo, amp_hi, amp_steps)

    c_range     = float(center_cfg.get('range', DEFAULT_CENTER_RANGE))
    c_step      = float(center_cfg.get('step',  DEFAULT_CENTER_STEP))
    center_grid = np.arange(obs_peak_wl - c_range, obs_peak_wl + c_range + c_step * 0.5, c_step)

    sigma_vals = sigma_cfg.get('grid', DEFAULT_SIGMA_VALS)
    sigma_grid = [float(s) for s in sigma_vals]

    wing_vals = wing_cfg.get('grid', DEFAULT_WING_VALS)
    wing_grid = [float(w) for w in wing_vals]

    snr_target = float(config.get('snr_target', 7.0))

    # ------------------------------------------------------------------
    # 4. Generate all combinations
    # ------------------------------------------------------------------
    candidates_params = []
    print(f'         => Grid ranges: {len(amp_grid)} amplitudes × {len(center_grid)} centers × {len(sigma_grid)} sigmas × 1 wing')
    
    for s in sigma_grid:
        for c in center_grid:
            for a in amp_grid:
                candidates_params.append((c, s, wing_val, a))

    n_candidates = len(candidates_params)
    print(f'[ENGINE] Evaluating {n_candidates} candidate models via coarse discrete grid search...')
    
    results_raw: List[Tuple[float, float, float, float, float, Dict[str, float]]] = []
    # (score, amp, center, sigma, wing, components)

    half_wing = wing_val / 2.0
    for center in center_grid:
        mask_w = (wavelength >= center - half_wing) & (wavelength <= center + half_wing)
        if np.sum(mask_w) < 5:
            continue
        for sigma in sigma_grid:
            for amp in amp_grid:
                composite, comps = _score_candidate(
                    wl=wavelength,
                    sub_y=subtracted_y,
                    obs_peak_wl=obs_peak_wl,
                    amp=float(amp),
                    center=float(center),
                    sigma=sigma,
                    wing=half_wing,
                    noise=noise,
                    snr_target=snr_target,
                    rest_wl=rest_wl,
                )
                if composite > 0.0:
                    results_raw.append((composite, float(amp), float(center), sigma, wing_val, comps))

    # Sort descending
    print(f'[OPTIMIZER] Found {len(results_raw)} valid candidates. Sorting by composite objective...')
    results_raw.sort(key=lambda x: x[0], reverse=True)
    top_raw = results_raw[:top_n]
    if top_raw:
        print(f'[STATS] Top candidate initial guess: sigma={top_raw[0][3]:.2f}, A={top_raw[0][1]:.2e}')

    # ------------------------------------------------------------------
    # 5. Compute full statistics for each top candidate (Stage B Amplitude)
    # ------------------------------------------------------------------
    ranked = []
    for rank, (score, grid_amp, center, sigma, wing, comps) in enumerate(top_raw, start=1):
        # Stage B: Profile-weighted amplitude estimator
        # To avoid peak domination, we compute A = sum(w * g * y) / sum(w * g^2)
        # using a weight that suppresses the core: w_i = 1 - g_i
        half_w = wing / 2.0
        mask = (wavelength >= center - half_w) & (wavelength <= center + half_w)
        wl_win = wavelength[mask]
        sub_y_win = subtracted_y[mask]
        
        g_model = np.exp(-((wl_win - center) ** 2) / (2.0 * (sigma ** 2)))
        w_i = 1.0 - g_model  # Zero weight at the exact peak, approaching 1 in the wings
        
        num = np.sum(w_i * g_model * sub_y_win)
        den = np.sum(w_i * (g_model ** 2))
        
        if den > 0:
            stage_b_amp = float(num / den)
        else:
            stage_b_amp = grid_amp

        # Fallback if unphysical negative amplitude due to noise
        if stage_b_amp <= 0:
            stage_b_amp = grid_amp

        stats = compute_statistics(
            wavelength=wavelength,
            subtracted_y=subtracted_y,
            continuum_fit=continuum_fit,
            amplitude=stage_b_amp,
            center=center,
            sigma=sigma,
            wing_window=wing,
            rest_wavelength=rest_wl,
        )
        stats['composite_score']  = round(float(score), 4)
        stats['score_components'] = comps
        stats['rank']             = rank
        stats['observed_peak_wl'] = obs_peak_wl
        ranked.append(stats)

    return ranked
