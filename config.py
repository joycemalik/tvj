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

# Continuum fitting defaults
DEFAULT_CONTINUUM_WINDOWS = "1150:1175, 1300:1350, 1425:1475, 1700:1750, 1800:1850, 1930:1945"
