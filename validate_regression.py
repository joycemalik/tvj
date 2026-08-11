import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from src.pipeline import run_single_spectrum_pipeline

# ── Reference Lyα parameters from Approved_From_Jasil_ALL_COLUMNS.xlsx ──
REFERENCE = {
    '15dcdr2dswp19731mxlo.txt': dict(
        center=1215.00, flux=6.995e-12, snr=10.04, ew=47.44,
        fwhm_ang=16.95, chi2=0.44, wing=20
    ),
    '16dcdr2dswp23063mxlo.txt': dict(
        center=1216.00, flux=6.689e-12, snr=11.86, ew=57.60,
        fwhm_ang=14.13, chi2=0.38, wing=21
    ),
    '17dcdr2dswp23064mxlo.txt': dict(
        center=1217.00, flux=7.203e-12, snr=12.35, ew=55.47,
        fwhm_ang=15.31, chi2=1.23, wing=21
    ),
    '18dcdr2dswp23444mxlo.txt': dict(
        center=1214.50, flux=5.291e-12, snr=8.44, ew=45.51,
        fwhm_ang=18.84, chi2=0.25, wing=19
    ),
    '153dcdr2dswp46607mxlo.txt': dict(
        center=1213.00, flux=5.208e-12, snr=5.93, ew=33.44,
        fwhm_ang=17.66, chi2=0.29, wing=18
    ),
    '172dcdr2dswp49597mxlo.txt': dict(
        center=1213.75, flux=5.642e-12, snr=6.95, ew=36.62,
        fwhm_ang=17.07, chi2=0.26, wing=17
    ),
}

# Derive sigma from FWHM: sigma = FWHM / 2.3548
for v in REFERENCE.values():
    v['sigma'] = v['fwhm_ang'] / 2.3548

print(f"\n{'='*120}")
print(f"{'LYMAN ALPHA REGRESSION VALIDATION':^120}")
print(f"{'='*120}")
header = f"  {'Spec':<7}  {'PARAM':<12}  {'REFERENCE':>15}  {'FITTED':>15}  {'ERROR':>12}  {'%ERR':>8}  {'PASS?':>6}"
print(header)
print('-'*120)

results = {}
for spec, ref in REFERENCE.items():
    if not os.path.exists(spec):
        print(f"  {spec[:3]}  MISSING FILE")
        continue

    try:
        rec = run_single_spectrum_pipeline(spec)
    except Exception as e:
        print(f"  {spec[:3]}  ERROR: {e}")
        continue

    lya = next((l for l in rec['lines'] if 'Lyman' in l.get('line_name', '')), None)
    if not lya:
        print(f"  {spec[:3]}  Lya not found in output")
        continue

    def row(name, ref_val, fit_val, tol_pct):
        if ref_val == 0:
            pct = float('nan')
        else:
            pct = 100.0 * (fit_val - ref_val) / abs(ref_val)
        ok = abs(pct) <= tol_pct
        print(f"  {spec[:3]:<7}  {name:<12}  {ref_val:>15.4g}  {fit_val:>15.4g}  {fit_val-ref_val:>12.4g}  {pct:>7.1f}%  {'OK' if ok else 'FAIL':>6}")
        return ok

    fwhm_fit = 2.3548 * lya.get('sigma', 0)
    ok_center = row('center (A)',  ref['center'],  lya.get('center', 0),      1.0)
    ok_sigma  = row('sigma (A)',   ref['sigma'],   lya.get('sigma',  0),     15.0)
    ok_flux   = row('flux(trapz)', ref['flux'],    lya.get('flux',   0),     20.0)
    gflux = lya.get('gaussian_flux', lya.get('amplitude', 0) * lya.get('sigma', 0) * 3.14159**0.5 * 2**0.5)
    row('gaussian_flux', ref['flux'],  gflux, 20.0)
    ok_snr    = row('SNR',         ref['snr'],     lya.get('snr',    0),     30.0)
    ok_ew     = row('EW (A)',      ref['ew'],      lya.get('ew',     0),     25.0)
    ok_chi2   = row('chi2red',     ref['chi2'],    lya.get('reduced_chi2', 0), 999.0)
    ok_wing   = row('wing (A)',    ref['wing'],    lya.get('wing_window', 0), 10.0)
    print(f"  {spec[:3]:<7}  {'amplitude':<12}  {'ref=A*s*sqrt2pi':>15}  {lya.get('amplitude',0):>15.4g}  {'':>12}  {'':>8}  {'':>6}")
    print(f"  {spec[:3]:<7}  {'min_wl':<12}  {ref['center']-ref['wing']/2:>15.4g}  {lya.get('min_wavelength',0):>15.4g}"
          f"  {lya.get('min_wavelength',0)-(ref['center']-ref['wing']/2):>12.4g}  {'':>8}  {'':>6}")
    print(f"  {spec[:3]:<7}  {'max_wl':<12}  {ref['center']+ref['wing']/2:>15.4g}  {lya.get('max_wavelength',0):>15.4g}"
          f"  {lya.get('max_wavelength',0)-(ref['center']+ref['wing']/2):>12.4g}  {'':>8}  {'':>6}")
    print()
    results[spec] = dict(center=ok_center, sigma=ok_sigma, flux=ok_flux, snr=ok_snr, ew=ok_ew)

print('='*120)
print('\nSUMMARY')
print(f"  {'Spec':<10}  {'Center':>8}  {'Sigma':>8}  {'Flux':>8}  {'SNR':>8}  {'EW':>8}")
for spec, r in results.items():
    def sym(b): return 'OK' if b else 'FAIL'
    print(f"  {spec[:3]:<10}  {sym(r['center']):>8}  {sym(r['sigma']):>8}  {sym(r['flux']):>8}  {sym(r['snr']):>8}  {sym(r['ew']):>8}")

if __name__ == '__main__':
    pass
