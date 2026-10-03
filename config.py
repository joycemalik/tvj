"""
Global constants and default configuration for spectrum fitting.
"""

# Speed of light in km/s
SPEED_OF_LIGHT_KMS = 299792.458

# Default noise estimation parameters (sigma-clipping)
SIGMA_CLIP_ITERS = 4
SIGMA_CLIP_THRESHOLD = 3.0
LOW_RES_FACTOR = 1.676

# IUE SWP low-dispersion resolution (~6 Å FWHM observed frame); spectra are
# de-redshifted by (1+z) with z(3C 273) = 0.158, so rest-frame FWHM = 6/1.158.
INSTRUMENTAL_FWHM_REST = 6.0 / 1.158

# Continuum fitting defaults: the five rest-frame windows of the campaign tables
# (jd.xlsx); with these the spectral indices reproduce the tabulated values.
DEFAULT_CONTINUUM_WINDOWS = "1100:1150, 1325:1370, 1425:1475, 1590:1620, 1660:1700"
