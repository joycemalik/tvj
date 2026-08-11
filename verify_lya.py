"""
verify_lya.py
-------------
Verification script: compares pipeline output against the 6 approved Lyα fits.

Run from the project root:
    python verify_lya.py

The approved reference values are hard-coded from the user's Excel table:
  Spectrum | center | flux      | SNR   | EW    | sigma | wing_delta | FWHM_kms | wing | min_wl | max_wl | chi2
  15       | 1215   | 7.00e-12  | 10.04 | 47.44 | 4.73  | 16.95      | 4183.4   | 20   | 1205   | 1225   | 0.44
  16       | 1216   | 6.69e-12  | 11.86 | 57.6  | 4.86  | 14.12      | 3483.3   | 21   | 1205.5 | 1226.5 | 0.38
  17       | 1217   | 7.20e-12  | 12.35 | 55.47 | 4.49  | 15.31      | 3770.5   | 21   | 1206.5 | 1227.5 | 1.23
  18       | 1214.5 | 5.29e-12  | 8.44  | 45.51 | 5.39  | 18.84      | 4650.2   | 19   | 1205   | 1224   | 0.25
  153      | 1213   | 5.21e-12  | 5.93  | 33.44 | 5.64  | 17.66      | 4364.9   | 18   | 1204   | 1222   | 0.29
  172      | 1213.75| 5.64e-12  | 6.95  | 36.62 | 5.27  | 17.07      | 4216.8   | 17   | 1205.25| 1222.25| 0.26

NOTE: spectrum files must be present in the project root or a 'spectra/' subfolder.
"""

import os
import sys
import glob

# ------ approved reference table ------
APPROVED = {
    '15': {'center': 1215.0,   'sigma': 4.73, 'wing': 20, 'snr': 10.04, 'fwhm_kms': 4183.4, 'wing_delta': 16.95, 'chi2': 0.44, 'ew': 47.44, 'flux': 7.00e-12},
    '16': {'center': 1216.0,   'sigma': 4.86, 'wing': 21, 'snr': 11.86, 'fwhm_kms': 3483.3, 'wing_delta': 14.12, 'chi2': 0.38, 'ew': 57.6,  'flux': 6.69e-12},
    '17': {'center': 1217.0,   'sigma': 4.49, 'wing': 21, 'snr': 12.35, 'fwhm_kms': 3770.5, 'wing_delta': 15.31, 'chi2': 1.23, 'ew': 55.47, 'flux': 7.20e-12},
    '18': {'center': 1214.5,   'sigma': 5.39, 'wing': 19, 'snr':  8.44, 'fwhm_kms': 4650.2, 'wing_delta': 18.84, 'chi2': 0.25, 'ew': 45.51, 'flux': 5.29e-12},
    '153':{'center': 1213.0,   'sigma': 5.64, 'wing': 18, 'snr':  5.93, 'fwhm_kms': 4364.9, 'wing_delta': 17.66, 'chi2': 0.29, 'ew': 33.44, 'flux': 5.21e-12},
    '172':{'center': 1213.75,  'sigma': 5.27, 'wing': 17, 'snr':  6.95, 'fwhm_kms': 4216.8, 'wing_delta': 17.07, 'chi2': 0.26, 'ew': 36.62, 'flux': 5.64e-12},
}

PASS_THRESHOLDS = {
    'center':     1.0,    # ±1.0 Å
    'sigma':      1.5,    # ±1.5 Å
    'wing':       2.0,    # ±2.0 Å
    'snr':        0.25,   # ±25% relative
    'fwhm_kms':   0.25,   # ±25% relative
    'wing_delta': 0.25,   # ±25% relative
    'chi2':       1.0,    # |Δχ²| absolute (chi2 can vary more)
    'ew':         0.30,   # ±30% relative
}


def _find_spectrum(spec_id: str) -> str:
    """Search for a spectrum file by its numeric ID in common locations."""
    patterns = [
        f"*{spec_id}dcdr2d*mxlo.txt",
        f"*{spec_id}*.txt",
        f"spectra/*{spec_id}*.txt",
        f"uploads/*{spec_id}*.txt",
    ]
    for pat in patterns:
        found = glob.glob(pat)
        if found:
            return found[0]
    return None


def run_verification():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    from src.pipeline import run_single_spectrum_pipeline

    print("=" * 70)
    print("Lyα VERIFICATION — comparing pipeline output vs. approved table")
    print("=" * 70)

    overall_pass = True
    results_table = []

    for spec_id, ref in APPROVED.items():
        filepath = _find_spectrum(spec_id)
        if filepath is None:
            print(f"\n[MISSING] Spectrum {spec_id} — file not found. Skipping.")
            continue

        print(f"\nProcessing spectrum {spec_id}: {filepath}")
        try:
            record = run_single_spectrum_pipeline(filepath, line_config_path="config/Lya.yaml")
        except Exception as e:
            print(f"  ERROR: {e}")
            overall_pass = False
            continue

        # Extract Lyα result from 'lines' array or legacy flat keys
        lya = None
        for line in record.get('lines', []):
            if abs(line.get('rest_wavelength', 0) - 1216.0) < 1.0:
                lya = line
                break
        if lya is None:
            lya = record   # legacy fallback

        computed = {
            'center':     lya.get('center', 0),
            'sigma':      lya.get('sigma', 0),
            'wing':       lya.get('wing_window', 0),
            'snr':        lya.get('snr', 0),
            'fwhm_kms':   lya.get('fwhm_kms', 0),
            'wing_delta': lya.get('wing_delta_lambda', 0),
            'chi2':       lya.get('reduced_chi2', 999),
            'ew':         lya.get('ew', 0),
            'flux':       lya.get('flux', 0),
        }

        spec_pass = True
        rows = []
        for param, ref_val in ref.items():
            if param == 'flux':
                continue   # skip flux for now (depends heavily on continuum)
            comp_val = computed.get(param, 0)
            thr = PASS_THRESHOLDS.get(param, 0.2)

            # Determine pass/fail
            if param in ('center', 'sigma', 'wing', 'chi2'):
                diff = abs(comp_val - ref_val)
                ok = diff <= thr
                diff_str = f"Δ={diff:.3f}"
            else:
                # relative
                rel = abs(comp_val - ref_val) / max(abs(ref_val), 1e-30)
                ok = rel <= thr
                diff_str = f"Δ={rel*100:.1f}%"

            status = "✓ PASS" if ok else "✗ FAIL"
            if not ok:
                spec_pass = False
                overall_pass = False
            rows.append(f"  {param:12s}  ref={ref_val:10.3f}  got={comp_val:10.3f}  {diff_str:12s}  {status}")

        print(f"  Spectrum {spec_id} — {'ALL PASS' if spec_pass else 'SOME FAIL'}")
        for row in rows:
            print(row)

    print("\n" + "=" * 70)
    print("OVERALL:", "ALL PASS ✓" if overall_pass else "SOME FAILURES ✗")
    print("=" * 70)
    return overall_pass


if __name__ == "__main__":
    ok = run_verification()
    sys.exit(0 if ok else 1)
