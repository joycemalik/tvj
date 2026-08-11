"""
report.py
---------
Saves pipeline results across multiple spectra into structured outputs:
  1. CSV  — one row per (spectrum, emission line)   outputs/csv/emission_line_fits.csv
  2. JSON — one file per spectrum                   outputs/json/<spectrum_name>.json
  3. Markdown summary report                        outputs/reports/batch_fit_summary.md
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_value(v: Any) -> Any:
    """Convert numpy scalars to Python natives; drop arrays."""
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
    except ImportError:
        pass
    return v


def _flatten_record_for_json(record: Dict[str, Any]) -> Dict[str, Any]:
    """Strip numpy arrays from a record for JSON serialisation."""
    out = {}
    for k, v in record.items():
        if k in ('wavelength', 'observed_flux', 'continuum_fit', 'subtracted_y'):
            continue   # too large for JSON
        if isinstance(v, list) and k == 'lines':
            # Recursively clean each line result
            out[k] = [_flatten_record_for_json(line) for line in v]
        else:
            cleaned = _clean_value(v)
            if cleaned is not None:
                out[k] = cleaned
    return out


def _build_line_rows(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a pipeline record into one flat dict per emission line.
    Each row contains spectrum-level metadata + per-line fit parameters.
    """
    spectrum_name   = record.get('spectrum_name', '')
    cont_amp        = record.get('continuum_amplitude', 0.0)
    spec_idx        = record.get('spectral_index', 0.0)
    global_chi2     = record.get('global_reduced_chi2', 999.0)
    joint_fit       = record.get('joint_fit_applied', False)

    lines = record.get('lines', [])
    if not lines:
        # Legacy: single-line record — wrap in a list
        lines = [{
            'line_name':       record.get('fit_status', 'Unknown'),
            'rest_wavelength': 1216.0,
            'detected':        record.get('is_acceptable', False),
            'amplitude':       record.get('amplitude', 0.0),
            'center':          record.get('center', 0.0),
            'sigma':           record.get('sigma', 0.0),
            'wing_window':     record.get('wing_window', 0.0),
            'min_wavelength':  record.get('min_wavelength', 0.0),
            'max_wavelength':  record.get('max_wavelength', 0.0),
            'flux':            record.get('flux', 0.0),
            'flux_err':        record.get('flux_err', 0.0),
            'snr':             record.get('snr', 0.0),
            'ew':              record.get('ew', 0.0),
            'fwhm_ang':        record.get('fwhm_ang', 0.0),
            'fwhm_kms':        record.get('fwhm_kms', 0.0),
            'wing_delta_lambda': record.get('wing_delta_lambda', 0.0),
            'blue_wing':       record.get('blue_wing', 0.0),
            'red_wing':        record.get('red_wing', 0.0),
            'reduced_chi2':    record.get('reduced_chi2', 999.0),
            'fit_status':      record.get('fit_status', ''),
        }]

    rows = []
    for line in lines:
        row = {
            # Spectrum-level
            'spectrum_name':       spectrum_name,
            'continuum_amplitude': float(_clean_value(cont_amp) or 0.0),
            'spectral_index':      float(_clean_value(spec_idx) or 0.0),
            'global_reduced_chi2': float(_clean_value(global_chi2) or 999.0),
            'joint_fit_applied':   bool(joint_fit),
            # Line-level
            'line_name':           line.get('line_name', ''),
            'rest_wavelength':     float(line.get('rest_wavelength', 0.0)),
            'detected':            bool(line.get('detected', False)),
            'fit_status':          str(line.get('fit_status', '')),
            'amplitude':           float(_clean_value(line.get('amplitude', 0.0)) or 0.0),
            'center':              float(_clean_value(line.get('center', 0.0)) or 0.0),
            'sigma':               float(_clean_value(line.get('sigma', 0.0)) or 0.0),
            'wing_window':         float(_clean_value(line.get('wing_window', 0.0)) or 0.0),
            'min_wavelength':      float(_clean_value(line.get('min_wavelength', 0.0)) or 0.0),
            'max_wavelength':      float(_clean_value(line.get('max_wavelength', 0.0)) or 0.0),
            'flux':                float(_clean_value(line.get('flux', 0.0)) or 0.0),
            'flux_err':            float(_clean_value(line.get('flux_err', 0.0)) or 0.0),
            'snr':                 float(_clean_value(line.get('snr', 0.0)) or 0.0),
            'ew':                  float(_clean_value(line.get('ew', 0.0)) or 0.0),
            'fwhm_ang':            float(_clean_value(line.get('fwhm_ang', 0.0)) or 0.0),
            'fwhm_kms':            float(_clean_value(line.get('fwhm_kms', 0.0)) or 0.0),
            'wing_delta_lambda':   float(_clean_value(line.get('wing_delta_lambda', 0.0)) or 0.0),
            'blue_wing':           float(_clean_value(line.get('blue_wing', 0.0)) or 0.0),
            'red_wing':            float(_clean_value(line.get('red_wing', 0.0)) or 0.0),
            'reduced_chi2':        float(_clean_value(line.get('reduced_chi2', 999.0)) or 999.0),
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_batch_results(
    results: List[Dict[str, Any]],
    output_dir: str = "outputs",
):
    """
    Save pipeline results across multiple spectra.

    Parameters
    ----------
    results    : list of pipeline records (one per spectrum)
    output_dir : root output directory

    Returns
    -------
    (csv_path, md_path)
    """
    csv_dir    = os.path.join(output_dir, "csv")
    json_dir   = os.path.join(output_dir, "json")
    report_dir = os.path.join(output_dir, "reports")

    os.makedirs(csv_dir,    exist_ok=True)
    os.makedirs(json_dir,   exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    all_line_rows: List[Dict[str, Any]] = []

    for record in results:
        # Per-spectrum JSON
        spec_name = os.path.splitext(record.get('spectrum_name', 'spectrum'))[0]
        json_path = os.path.join(json_dir, f"{spec_name}.json")
        clean_rec = _flatten_record_for_json(record)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_rec, f, indent=2)

        # Collect per-line rows for CSV
        all_line_rows.extend(_build_line_rows(record))

    # ---- CSV (one row per spectrum × line) ----
    df = pd.DataFrame(all_line_rows)
    csv_path = os.path.join(csv_dir, "emission_line_fits.csv")
    df.to_csv(csv_path, index=False)

    # ---- Markdown summary ----
    md_path = os.path.join(report_dir, "batch_fit_summary.md")
    n_spectra  = len(results)
    n_detected = sum(1 for row in all_line_rows if row.get('detected', False))
    n_lines    = len(all_line_rows)

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Batch Emission Line Fitting Summary Report\n\n")
        f.write(f"Total spectra processed: **{n_spectra}**\n\n")
        f.write(f"- Total (spectrum, line) pairs attempted: **{n_lines}**\n")
        f.write(f"- Detected / accepted: **{n_detected}** ({n_detected/n_lines*100:.1f}%)\n\n")

        f.write("## Detailed Results Table\n\n")
        headers = [
            "Spectrum", "Line", "Status",
            "Center (Å)", "Amplitude", "Sigma (Å)",
            "Wing (Å)", "Min wl (Å)", "Max wl (Å)",
            "Flux", "SNR", "EW (Å)", "FWHM (km/s)", "χ²_red",
        ]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")

        for row in all_line_rows:
            fit_stat = str(row.get('fit_status', '')).upper()
            if 'MARGINAL' in fit_stat:
                status_short = '⚠ MARGINAL'
            elif 'NOT_DETECTED' in fit_stat or 'NOT DETECTED' in fit_stat:
                status_short = '— NOT DETECTED'
            elif 'REJECTED' in fit_stat:
                status_short = '✕ REJECTED'
            elif 'ACCEPTED' in fit_stat:
                status_short = '✓ ACCEPTED'
            else:
                status_short = '— NOT DETECTED'
                
            cols = [
                row['spectrum_name'],
                row['line_name'],
                status_short,
                f"{row['center']:.2f}",
                f"{row['amplitude']:.3e}",
                f"{row['sigma']:.2f}",
                f"{row['wing_window']:.1f}",
                f"{row['min_wavelength']:.2f}",
                f"{row['max_wavelength']:.2f}",
                f"{row['flux']:.3e}",
                f"{row['snr']:.2f}",
                f"{row['ew']:.2f}",
                f"{row['fwhm_kms']:.1f}",
                f"{row['reduced_chi2']:.3f}",
            ]
            f.write("| " + " | ".join(cols) + " |\n")

        # Global chi2 per spectrum
        f.write("\n## Global Reduced χ² per Spectrum\n\n")
        f.write("| Spectrum | Global χ²_red | Joint Fit |\n")
        f.write("| --- | --- | --- |\n")
        for record in results:
            name  = record.get('spectrum_name', '')
            gchi2 = record.get('global_reduced_chi2', 999.0)
            joint = "Yes" if record.get('joint_fit_applied', False) else "No"
            f.write(f"| {name} | {gchi2:.3f} | {joint} |\n")

    return csv_path, md_path
