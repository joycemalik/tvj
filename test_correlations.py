import numpy as np

APPROVED = {
    '15': {'center': 1215.0,   'sigma': 4.73, 'wing': 20, 'snr': 10.04, 'fwhm_kms': 4183.4, 'wing_delta': 16.95, 'chi2': 0.44, 'ew': 47.44, 'flux': 7.00e-12},
    '16': {'center': 1216.0,   'sigma': 4.86, 'wing': 21, 'snr': 11.86, 'fwhm_kms': 3483.3, 'wing_delta': 14.12, 'chi2': 0.38, 'ew': 57.6,  'flux': 6.69e-12},
    '17': {'center': 1217.0,   'sigma': 4.49, 'wing': 21, 'snr': 12.35, 'fwhm_kms': 3770.5, 'wing_delta': 15.31, 'chi2': 1.23, 'ew': 55.47, 'flux': 7.20e-12},
    '18': {'center': 1214.5,   'sigma': 5.39, 'wing': 19, 'snr':  8.44, 'fwhm_kms': 4650.2, 'wing_delta': 18.84, 'chi2': 0.25, 'ew': 45.51, 'flux': 5.29e-12},
    '153':{'center': 1213.0,   'sigma': 5.64, 'wing': 18, 'snr':  5.93, 'fwhm_kms': 4364.9, 'wing_delta': 17.66, 'chi2': 0.29, 'ew': 33.44, 'flux': 5.21e-12},
    '172':{'center': 1213.75,  'sigma': 5.27, 'wing': 17, 'snr':  6.95, 'fwhm_kms': 4216.8, 'wing_delta': 17.07, 'chi2': 0.26, 'ew': 36.62, 'flux': 5.64e-12},
}

for k, v in APPROVED.items():
    sigma = v['sigma']
    wing_delta = v['wing_delta']
    ew = v['ew']
    snr = v['snr']
    wing = v['wing']
    flux = v['flux']
    
    ratio = wing_delta / sigma
    print(f"Spec {k}: ratio={ratio:.4f} ew={ew} snr={snr} wing={wing}")
