import os
import yaml
from typing import Dict, Any, Optional

from src.reader import load_spectrum
from src.continuum import fit_continuum
from src.candidate_engine import generate_candidates
from src.quality import evaluate_fit_quality
from config import DEFAULT_CONTINUUM_WINDOWS


def run_single_spectrum_pipeline(
    spectrum_path: str,
    line_config_path: str = "config/Lya.yaml",
    continuum_windows: Optional[str] = DEFAULT_CONTINUUM_WINDOWS,
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    End-to-end scientific pipeline for a single spectrum file:
    1. Read text spectrum
    2. Power-law continuum fit & subtraction
    3. Discrete candidate grid-search (replaces continuous optimizer)
    4. Multi-criteria quality evaluation on best candidate
    5. Return best-fit record + full candidate shortlist
    """
    spectrum_name = os.path.basename(spectrum_path)
    wavelength, flux = load_spectrum(spectrum_path)

    # Load configuration
    if os.path.exists(line_config_path):
        with open(line_config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            'rest_wavelength': 1216.0,
            'snr_min': 5.0,
            'snr_max': 12.0,
            'chi2_red_max': 5.0
        }

    rest_wl = cfg.get('rest_wavelength', 1216.0)

    # Continuum subtraction
    amp_cont, spec_idx_cont, continuum_fit, subtracted_y = fit_continuum(
        wavelength, flux, continuum_windows
    )

    # Candidate grid search – returns ranked list, best is index 0
    candidates = generate_candidates(
        wavelength=wavelength,
        subtracted_y=subtracted_y,
        continuum_fit=continuum_fit,
        rest_wl=rest_wl,
        config=cfg,
        top_n=top_n,
    )

    if not candidates:
        raise RuntimeError("No valid candidates found. Check spectrum data and config.")

    # Best candidate is rank-1
    best = candidates[0]

    # Quality evaluation on best candidate
    obs_peak = best.get('observed_peak_wl', rest_wl)
    score, is_acceptable, status = evaluate_fit_quality(
        stats=best,
        observed_peak_wl=obs_peak,
        snr_min=cfg.get('snr_min', 5.0),
        snr_max=cfg.get('snr_max', 12.0),
        chi2_max=cfg.get('chi2_red_max', 5.0),
    )

    # Assemble complete record
    record = {
        'spectrum_name':     spectrum_name,
        'continuum_amplitude': float(amp_cont),
        'spectral_index':    float(spec_idx_cont),
        'quality_score':     float(best.get('composite_score', score)),
        'is_acceptable':     bool(is_acceptable),
        'fit_status':        status,
        **best,
    }

    # Raw arrays for plotting
    record['wavelength']    = wavelength
    record['observed_flux'] = flux
    record['continuum_fit'] = continuum_fit
    record['subtracted_y']  = subtracted_y

    # Attach top-5 shortlist (strip numpy arrays, keep only scalars)
    shortlist = []
    for cand in candidates[:5]:
        shortlist.append({k: v for k, v in cand.items()
                          if not isinstance(v, (dict, list))})
        # also include score_components dict
        shortlist[-1]['score_components'] = cand.get('score_components', {})
    record['shortlist'] = shortlist

    return record
