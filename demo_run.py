import warnings, sys, io, glob
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from src.pipeline import run_single_spectrum_pipeline

# Approved reference values for spectrum 15
APPROVED = {
    'center':     1215.0,
    'sigma':      4.73,
    'wing':       20.0,
    'snr':        10.04,
    'fwhm_kms':   4183.4,
    'wing_delta': 16.95,
    'chi2':       0.44,
    'ew':         47.44,
}

# Find spectrum 15 file
files = glob.glob('*15dcdr2d*.txt') + glob.glob('spectra/*15dcdr2d*.txt') + glob.glob('uploads/*15dcdr2d*.txt')
filepath = files[0] if files else None

if filepath is None:
    print('Spectrum 15 not found. Looking for any spectrum...')
    files = glob.glob('*.txt')
    if files:
        filepath = files[0]
        print('Using:', filepath)
    else:
        print('No .txt spectra found in project root.')
        sys.exit(1)

print('Processing:', filepath)
rec = run_single_spectrum_pipeline(filepath, line_config_path='config/Lya.yaml')

# Find Lya result
lya = None
for line in rec.get('lines', []):
    if abs(line.get('rest_wavelength', 0) - 1216.0) < 1.0:
        lya = line
        break
if lya is None:
    lya = rec

print()
print('=== SPECTRUM 15 RESULT vs. APPROVED TABLE ===')
print()
print(f'{"Parameter":18s} {"Approved":>12s} {"Got":>12s} {"Verdict":>12s}')
print('-' * 56)

comparisons = [
    ('center',     APPROVED['center'],     lya.get('center', 0),              1.0,   'Å'),
    ('sigma',      APPROVED['sigma'],      lya.get('sigma', 0),               1.5,   'Å'),
    ('wing',       APPROVED['wing'],       lya.get('wing_window', 0),         2.0,   'Å'),
    ('snr',        APPROVED['snr'],        lya.get('snr', 0),                 0.25,  'rel'),
    ('fwhm_kms',   APPROVED['fwhm_kms'],   lya.get('fwhm_kms', 0),           0.25,  'rel'),
    ('wing_delta', APPROVED['wing_delta'], lya.get('wing_delta_lambda', 0),  0.25,  'rel'),
    ('chi2',       APPROVED['chi2'],       lya.get('reduced_chi2', 999),      1.0,   'abs'),
    ('ew',         APPROVED['ew'],         lya.get('ew', 0),                  0.30,  'rel'),
]

all_pass = True
for name, ref, got, thr, thr_type in comparisons:
    if thr_type == 'rel':
        err = abs(got - ref) / max(abs(ref), 1e-30)
        ok  = err <= thr
        diff = f'Δ={err*100:.1f}%'
    else:
        err = abs(got - ref)
        ok  = err <= thr
        diff = f'Δ={err:.3f}'
    symbol = 'PASS' if ok else 'FAIL'
    if not ok:
        all_pass = False
    print(f'{name:18s} {ref:12.3f} {got:12.3f}   {diff:12s}  {symbol}')

print()
print('min_wavelength:', round(lya.get('min_wavelength', 0), 2), '  (approved: 1205.0)')
print('max_wavelength:', round(lya.get('max_wavelength', 0), 2), '  (approved: 1225.0)')
print()
print('OVERALL:', 'ALL PASS' if all_pass else 'SOME FAILURES')

print()
print('--- Top-3 candidates ---')
for i, cand in enumerate(lya.get('shortlist', [])[:3]):
    print(f'  #{i+1}: score={cand.get("composite_score",0):.4f}  '
          f'wing={cand.get("wing_window",0):.0f}  '
          f'sigma={cand.get("sigma",0):.3f}  '
          f'center={cand.get("center",0):.2f}  '
          f'chi2={cand.get("reduced_chi2",0):.3f}')
