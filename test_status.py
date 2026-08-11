from src.pipeline import run_single_spectrum_pipeline

res = run_single_spectrum_pipeline('18dcdr2dswp23444mxlo.txt')
for l in res.get('lines', []):
    print(f"{l['line_name']}: SNR={l['snr']:.2f}, chi2={l.get('reduced_chi2', 0):.2f}, status='{l.get('fit_status', '')}'")
