"""
candidate_engine.py
-------------------
Discrete grid-search fitting engine.

Composite score (0..1, higher is better):
  S = 0.35*s_chi2 + 0.20*s_peak + 0.15*s_left + 0.10*s_right
    + 0.00*s_snr  + 0.12*s_wing + 0.08*s_sigma

  chi2 is computed on power-law-subtracted data (sub_y) so the broad BLR wing
  correctly discriminates sigma. Flux amplitude uses plain OLS on local-baseline-
  corrected net_win to match the reference tool's approach.

  s_wing  = exp(-0.5 * ((W  - W_prior )  / W_std )^2)   (from YAML wing_window.prior_*)
  s_sigma = exp(-0.5 * ((sig- sig_prior)  / sig_std)^2)  (from YAML sigma.prior_*)

For each (sigma, wing_window, center) triplet the flux amplitude is computed
via plain profile-weighted OLS on the local-baseline-corrected signal (net_win),
and a separate scoring amplitude via plain OLS on sub_y for chi2 computation.
"""

import numpy as np
from typing import Dict, Any, List, Tuple

try:
    from scipy.ndimage import gaussian_filter1d as _gf1d
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from src.statistics import compute_statistics, estimate_noise, local_baseline


# ---------------------------------------------------------------------------
# Fallback grids (used only when YAML config provides no values)
# ---------------------------------------------------------------------------
DEFAULT_CENTER_STEP  = 0.5
DEFAULT_CENTER_RANGE = 5.0
DEFAULT_SIGMA_VALS   = list(np.arange(2.5, 10.5, 0.5))
DEFAULT_WING_VALS    = list(range(14, 26))

# Scoring weights (must sum to 1.0)
W_CHI2   = 0.35
W_PEAK   = 0.20
W_LEFT   = 0.15
W_RIGHT  = 0.10
W_SNR    = 0.00
W_WING   = 0.12
W_SIGMA  = 0.08

# Wing prior coupled to sigma (campaign mean: wing_mean/sigma_mean = 16.8/6.32 ≈ 2.66)
WING_SIGMA_RATIO = 2.65



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian(wl: np.ndarray, amp: float, center: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-((wl - center) ** 2) / (2.0 * sigma ** 2))


def _plain_amplitude(
    wl_win: np.ndarray,
    sub_y_win: np.ndarray,
    center: float,
    sigma: float,
    fallback: float = 0.0,
) -> float:
    """Plain profile-weighted OLS on power-law-subtracted data for chi2 scoring."""
    n = len(wl_win)
    if n < 3:
        return fallback
    g = np.exp(-((wl_win - center) ** 2) / (2.0 * sigma ** 2))
    denom = float(np.sum(g ** 2))
    if denom <= 0:
        return fallback
    A = float(np.sum(g * sub_y_win) / denom)
    return A if A > 0 else fallback


def _score_candidate(
    wl_w: np.ndarray,
    sub_y_w: np.ndarray,
    obs_peak_wl: float,
    amp: float,
    center: float,
    sigma: float,
    half_wing: float,
    noise: float,
    wing_prior_center: float = 19.0,
    wing_prior_std: float    = 2.0,
    sigma_prior_center: float = 5.0,
    sigma_prior_std: float    = 1.0,
    snr_target: float = 7.0,
    rest_wl: float = 1216.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Score a single (amp, center, sigma, half_wing) candidate.

    wl_w / sub_y_w: pre-sliced power-law-subtracted flux (not baseline-corrected).
    amp: plain OLS amplitude on sub_y (used for chi2 scoring only — the unbiased
         narrow-core amplitude for flux is stored separately in results_raw).

    Using sub_y for scoring correctly discriminates sigma: a small sigma that
    misses the broad BLR wing gives large chi2 residuals at the wings of sub_y.
    """
    if len(wl_w) < 4:
        return 0.0, {}

    # Plain Gaussian model against power-law-subtracted data
    g_raw     = np.exp(-((wl_w - center) ** 2) / (2.0 * sigma ** 2))
    model     = amp * g_raw
    residuals = sub_y_w - model

    # 1. chi2 score
    # Sharper Gaussian (sigma=0.5 vs old 1.5) so chi2 discriminates sigma/wing strongly.
    # Overfitting penalty: chi2 << 0.15 is treated as badly as chi2=1.6 (not free lunch).
    dof      = max(len(wl_w) - 3, 1)
    chi2_red = float(np.sum((residuals / noise) ** 2) / dof)
    if chi2_red < 0.15:
        eff_chi2 = 1.0 + (0.15 - chi2_red) * 10.0   # steep linear rise
    else:
        eff_chi2 = chi2_red
    chi2_score = float(np.exp(-0.5 * ((eff_chi2 - 1.0) / 1.5) ** 2))

    # 2. Peak alignment
    d_obs  = abs(center - obs_peak_wl)
    d_rest = abs(center - rest_wl)
    peak_score = (0.6 * float(np.exp(-0.5 * (d_obs  / 3.0) ** 2)) +
                  0.4 * float(np.exp(-0.5 * (d_rest / 5.0) ** 2)))

    # 3 & 4. Wing residual quality
    left_mask  = wl_w < center
    right_mask = wl_w > center
    left_score  = (float(np.exp(-np.sqrt(np.mean(residuals[left_mask] ** 2)) / noise))
                   if np.any(left_mask) else 0.5)
    right_score = (float(np.exp(-np.sqrt(np.mean(residuals[right_mask] ** 2)) / noise))
                   if np.any(right_mask) else 0.5)

    # 5. SNR score
    raw_snr   = amp / noise if noise > 0 else 0.0
    snr_score = float(np.exp(-0.5 * ((raw_snr - snr_target) / (snr_target * 0.5)) ** 2))

    # 6. Wing prior — Gaussian prior on the full wing width (W = 2*half_wing)
    full_wing  = 2.0 * half_wing
    wing_score = float(np.exp(-0.5 * ((full_wing - wing_prior_center) / wing_prior_std) ** 2))

    # 7. Sigma prior — Gaussian prior on sigma
    sigma_score = float(np.exp(-0.5 * ((sigma - sigma_prior_center) / sigma_prior_std) ** 2))

    composite = (
        W_CHI2  * chi2_score  +
        W_PEAK  * peak_score  +
        W_LEFT  * left_score  +
        W_RIGHT * right_score +
        W_SNR   * snr_score   +
        W_WING  * wing_score  +
        W_SIGMA * sigma_score
    )

    components = {
        'chi2_red':     round(chi2_red,     4),
        'chi2_score':   round(chi2_score,   4),
        'peak_score':   round(peak_score,   4),
        'left_score':   round(left_score,   4),
        'right_score':  round(right_score,  4),
        'snr_score':    round(snr_score,    4),
        'wing_score':   round(wing_score,   4),
        'sigma_score':  round(sigma_score,  4),
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
    Discrete grid-search over (sigma, wing_window, center); amplitude is
    computed analytically (Stage-B) for every triplet.

    Returns top_n candidates ranked by composite score (descending).
    Each dict contains all fields from compute_statistics() plus:
      composite_score, score_components, rank, observed_peak_wl.
    """
    if config is None:
        config = {}

    # ------------------------------------------------------------------
    # 1. Locate observed emission peak (using smoothed sub_y to suppress
    #    noise spikes and asymmetric absorption near the line core).
    # ------------------------------------------------------------------
    search_radius = float(config.get('peak_search_radius', 15.0))
    peak_region   = ((wavelength >= rest_wl - search_radius) &
                     (wavelength <= rest_wl + search_radius))

    # Smooth sub_y with a ~3-pixel Gaussian kernel before peak detection.
    # This prevents noise spikes / power-law slope from shifting the apparent
    # peak away from the true emission centroid (critical for asymmetric profiles).
    if _HAS_SCIPY and len(subtracted_y) > 10:
        sub_smooth = _gf1d(subtracted_y, sigma=3.0)
    else:
        # Fallback: 7-pixel boxcar via np.convolve
        k = np.ones(7) / 7.0
        sub_smooth = np.convolve(subtracted_y, k, mode='same')

    if np.any(peak_region):
        wl_reg       = wavelength[peak_region]
        fl_reg       = sub_smooth[peak_region]   # smoothed for robust peak
        idx          = int(np.argmax(fl_reg))
        obs_peak_wl  = float(wl_reg[idx])
        obs_peak_amp = max(float(subtracted_y[peak_region][idx]), 1.0e-14)
    else:
        obs_peak_wl  = rest_wl
        obs_peak_amp = 8.0e-13

    # Pre-compute raw observed flux (needed for local baseline estimation)
    obs_flux = subtracted_y + continuum_fit

    # ------------------------------------------------------------------
    # 2. Global noise from a line-free region on each side of the peak.
    #    Using ±20 Å around rest_wl would include the emission line and
    #    inflate the MAD; instead use further-out continuum bands.
    # ------------------------------------------------------------------
    mean_cont = float(np.mean(continuum_fit)) if len(continuum_fit) > 0 else 1.0e-13
    if mean_cont == 0:
        mean_cont = 1.0e-13

    # Two 20 Å bands well outside the line (skip ±40 Å around rest_wl)
    blue_mask = (wavelength >= rest_wl - 60) & (wavelength <= rest_wl - 40)
    red_mask  = (wavelength >= rest_wl + 40) & (wavelength <= rest_wl + 60)
    cont_region = np.concatenate([
        subtracted_y[blue_mask] if np.any(blue_mask) else np.array([]),
        subtracted_y[red_mask]  if np.any(red_mask)  else np.array([]),
    ])
    if len(cont_region) >= 8:
        noise = estimate_noise(cont_region)
    else:
        # fallback: use broad region normalised by continuum
        broad_mask = (wavelength >= rest_wl - 20) & (wavelength <= rest_wl + 20)
        region     = subtracted_y[broad_mask] if np.any(broad_mask) else subtracted_y
        noise      = estimate_noise(region / abs(mean_cont)) * abs(mean_cont)
    if noise <= 1.0e-30:
        noise = 1.0e-20

    # ------------------------------------------------------------------
    # 3. Build parameter grids from config
    # ------------------------------------------------------------------
    center_cfg = config.get('center')       or {}
    sigma_cfg  = config.get('sigma')        or {}
    wing_cfg   = config.get('wing_window')  or {}

    # Center grid
    c_range     = float(center_cfg.get('range', DEFAULT_CENTER_RANGE))
    c_step      = float(center_cfg.get('step',  DEFAULT_CENTER_STEP))
    center_grid = np.arange(
        obs_peak_wl - c_range,
        obs_peak_wl + c_range + c_step * 0.5,
        c_step,
    )

    # Sigma grid: from min/max/step or explicit grid
    if 'grid' in sigma_cfg:
        sigma_grid = [float(s) for s in sigma_cfg['grid']]
    elif 'min' in sigma_cfg or 'max' in sigma_cfg:
        s_min  = float(sigma_cfg.get('min',  2.5))
        s_max  = float(sigma_cfg.get('max',  8.0))
        s_step = float(sigma_cfg.get('step', 0.5))
        sigma_grid = [round(float(s), 6) for s in
                      np.arange(s_min, s_max + s_step * 0.5, s_step)]
    else:
        sigma_grid = [float(s) for s in DEFAULT_SIGMA_VALS]

    # Wing grid: full list
    wing_grid = ([float(w) for w in wing_cfg['grid']]
                 if 'grid' in wing_cfg else [float(w) for w in DEFAULT_WING_VALS])

    # Priors (read from config for scorer)
    wing_prior_center  = float(wing_cfg.get('prior_center',  19.0))
    wing_prior_std     = float(wing_cfg.get('prior_std',      2.0))
    sigma_prior_center = float(sigma_cfg.get('prior_center',  5.0))
    sigma_prior_std    = float(sigma_cfg.get('prior_std',     1.0))
    snr_target         = float(config.get('snr_target', 7.0))

    total = len(wing_grid) * len(sigma_grid) * len(center_grid)
    print(f'         => Grid: {len(wing_grid)} wings x {len(sigma_grid)} sigmas x '
          f'{len(center_grid)} centers = {total} candidates (amplitude analytic)')
    print(f'[ENGINE] Evaluating {total} candidate models via discrete grid search...')

    # ------------------------------------------------------------------
    # 4. Evaluate all (wing, sigma, center) triplets
    # ------------------------------------------------------------------
    results_raw: List[Tuple[float, float, float, float, float, Dict]] = []

    for wing in wing_grid:
        half_wing = wing / 2.0
        for sigma in sigma_grid:
            for center in center_grid:
                wl_mask = ((wavelength >= center - half_wing) &
                           (wavelength <= center + half_wing))
                if np.sum(wl_mask) < 4:
                    continue

                wl_win  = wavelength[wl_mask]
                obs_win = obs_flux[wl_mask]

                bl_win = local_baseline(wavelength, obs_flux, float(center), wl_win)
                if bl_win is None:
                    bl_win = continuum_fit[wl_mask]

                net_win = obs_win - bl_win   # narrow-core + noise only
                sub_win = subtracted_y[wl_mask]  # power-law-subtracted (for scoring)

                # Flux amplitude: plain OLS on local-baseline-corrected signal.
                # G_eff OLS is NOT used — it amplifies broad-wing curvature by 2-3x
                # for the low-pixel IUE spectra, giving +50-80% errors.
                # Plain OLS on net_win matches the reference tool's approach
                # (RMSE=12.8% at exact reference params across 6 campaign spectra).
                amp_flux = _plain_amplitude(
                    wl_win, net_win, float(center), sigma,
                    fallback=obs_peak_amp,
                )
                if amp_flux <= 0:
                    continue

                # Scoring amplitude: plain OLS on sub_y → captures total emission shape;
                # chi2 against sub_y correctly penalises sigma that misses BLR wings.
                amp_score = _plain_amplitude(
                    wl_win, sub_win, float(center), sigma,
                    fallback=obs_peak_amp,
                )

                composite, comps = _score_candidate(
                    wl_w=wl_win,
                    sub_y_w=sub_win,
                    obs_peak_wl=obs_peak_wl,
                    amp=amp_score,
                    center=float(center),
                    sigma=sigma,
                    half_wing=half_wing,
                    noise=noise,
                    wing_prior_center=wing_prior_center,
                    wing_prior_std=wing_prior_std,
                    sigma_prior_center=sigma_prior_center,
                    sigma_prior_std=sigma_prior_std,
                    snr_target=snr_target,
                    rest_wl=rest_wl,
                )
                if composite > 0.0:
                    results_raw.append(
                        (composite, amp_flux, float(center), sigma, wing, comps)
                    )

    print(f'[OPTIMIZER] Found {len(results_raw)} valid candidates. '
          f'Sorting by composite objective...')
    results_raw.sort(key=lambda x: x[0], reverse=True)
    top_raw = results_raw[:top_n]

    if top_raw:
        b = top_raw[0]
        print(f'[STATS] Top candidate: sigma={b[3]:.2f} A, wing={b[4]:.0f} A, '
              f'center={b[2]:.2f} A, A={b[1]:.3e}')

    # ------------------------------------------------------------------
    # 5. Compute full statistics for each top candidate
    # ------------------------------------------------------------------
    ranked: List[Dict[str, Any]] = []
    for rank, (score, amp, center, sigma, wing, comps) in enumerate(top_raw, start=1):
        stats = compute_statistics(
            wavelength=wavelength,
            subtracted_y=subtracted_y,
            continuum_fit=continuum_fit,
            amplitude=amp,
            center=center,
            sigma=sigma,
            wing_window=wing,
            rest_wavelength=rest_wl,
            noise=noise,
        )
        stats['composite_score']  = round(float(score), 4)
        stats['score_components'] = comps
        stats['rank']             = rank
        stats['observed_peak_wl'] = obs_peak_wl
        ranked.append(stats)

    return ranked
