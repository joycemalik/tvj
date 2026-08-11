import sys
import numpy as np
from src.reader import load_spectrum
from src.continuum import fit_continuum
from src.statistics import compute_statistics
from config import DEFAULT_CONTINUUM_WINDOWS

def main():
    spec = '15dcdr2dswp19731mxlo.txt'
    wl, fl = load_spectrum(spec)
    amp_cont, idx, cont, sub = fit_continuum(wl, fl, DEFAULT_CONTINUUM_WINDOWS)

    # Approved parameters
    center = 1215.0
    sigma = 4.73
    wing_window = 20
    manual_min_wl = 1205.0
    manual_max_wl = 1225.0

    # Since the reference implies flux ~ 6.995e-12, and A ~ flux / (sigma*sqrt(2pi))
    # We'll use scipy.optimize to find the best amplitude for this center and sigma
    # by fitting the Gaussian to the subtracted data.
    from scipy.optimize import curve_fit

    mask = (wl >= manual_min_wl) & (wl <= manual_max_wl)
    wl_win = wl[mask]
    sub_win = sub[mask]

    def gaussian(x, a):
        return a * np.exp(-((x - center)**2) / (2 * sigma**2))

    popt, _ = curve_fit(gaussian, wl_win, sub_win, p0=[5e-13])
    amplitude = popt[0]

    # Now let's calculate what the JS app would do
    g_model = gaussian(wl_win, amplitude)
    cont_win = cont[mask]
    
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    flux_raw = _trapz(sub_win, wl_win)
    flux_gauss = _trapz(g_model, wl_win)

    # Calculate EW from raw and gauss
    ew_raw = _trapz(np.where(cont_win > 0, sub_win / cont_win, 0.0), wl_win)
    ew_gauss = _trapz(np.where(cont_win > 0, g_model / cont_win, 0.0), wl_win)

    print(f"--- Reproduction Experiment for {spec} ---")
    print(f"Inputs: center={center}, sigma={sigma}, min_wl={manual_min_wl}, max_wl={manual_max_wl}")
    print(f"Fitted Amplitude: {amplitude:.4e}")
    print(f"Analytic Gaussian Flux: {amplitude * sigma * np.sqrt(2*np.pi):.4e}")
    print()
    print("Integration Results:")
    print(f"Flux (integrating subtracted raw): {flux_raw:.4e}")
    print(f"Flux (integrating Gaussian model): {flux_gauss:.4e}  <-- Expected ~6.995e-12")
    print(f"EW (integrating subtracted raw):   {ew_raw:.2f}")
    print(f"EW (integrating Gaussian model):   {ew_gauss:.2f}  <-- Expected ~47.44")

    # Run the current compute_statistics
    stats = compute_statistics(
        wavelength=wl,
        subtracted_y=sub,
        continuum_fit=cont,
        amplitude=amplitude,
        center=center,
        sigma=sigma,
        wing_window=wing_window,
        manual_min_wl=manual_min_wl,
        manual_max_wl=manual_max_wl
    )
    print("\n--- Current compute_statistics output ---")
    print(f"Flux:       {stats['flux']:.4e}")
    print(f"SNR:        {stats['snr']:.2f} (Expected 10.04)")
    print(f"EW:         {stats['ew']:.2f}")
    print(f"FWHM (A):   {stats['fwhm_ang']:.2f} (Expected 16.95)")
    print(f"FWHM (km/s):{stats['fwhm_kms']:.1f} (Expected 4183.4)")
    print(f"Chi2 red:   {stats['reduced_chi2']:.2f} (Expected 0.44)")

if __name__ == '__main__':
    main()
