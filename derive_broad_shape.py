"""
derive_broad_shape.py
---------------------
Measure the broad-component shape of the narrow + broad lines (Lyα, C IV) from the data.

Every spectrum is fitted with the broad centre and σ free. The campaign values
are the medians over fits where the broad component is well constrained:
broad F/σ_F ≥ 5, no parameter at a bound, and ΔBIC > 10 (two Gaussians
strongly preferred over one). Copy the printed values into the line YAML as
broad.center_fixed / broad.sigma_fixed.

    python derive_broad_shape.py [--workers N]
"""
import argparse
import contextlib
import glob
import io
import os
from multiprocessing import Pool

import numpy as np

import src.pipeline as pipeline
from src.line_fitter import load_line_config

FREE_KEYS = ('center_fixed', 'sigma_fixed')


def _free_config(path):
    cfg = load_line_config(path)
    if isinstance(cfg.get('broad'), dict):
        cfg['broad'] = {k: v for k, v in cfg['broad'].items() if k not in FREE_KEYS}
    return cfg


pipeline.load_line_config = _free_config      # also applied in each worker on import


def fit(path):
    with contextlib.redirect_stdout(io.StringIO()):
        res = pipeline.run_single_spectrum_pipeline(path)
    return [dict(spectrum=os.path.basename(path), line=ln['line_name'], **{k: ln.get(k) for k in (
        'center_broad', 'sigma_broad', 'snr_broad', 'delta_bic', 'at_bound', 'sigma_narrow')})
        for ln in res['lines'] if 'sigma_broad' in ln]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=max(1, min((os.cpu_count() or 2) - 2, 10)))
    args = ap.parse_args()
    files = sorted(glob.glob('spectrum/*.txt'))
    with Pool(args.workers) as pool:
        rows = [r for chunk in pool.imap_unordered(fit, files, chunksize=2) for r in chunk]
    for line in sorted({r['line'] for r in rows}):
        all_ = [r for r in rows if r['line'] == line]
        good = [r for r in all_ if r['snr_broad'] >= 5 and not r['at_bound']
                and np.isfinite(r['delta_bic']) and r['delta_bic'] > 10]
        mu = np.array([r['center_broad'] for r in good])
        sg = np.array([r['sigma_broad'] for r in good])
        print(f"{line}: {len(good)} of {len(all_)} spectra well constrained")
        if len(good) < 3:
            continue
        q = lambda a: np.percentile(a, [25, 50, 75])
        print(f"  center_fixed: {np.median(mu):.2f}   (IQR {q(mu)[0]:.2f}–{q(mu)[2]:.2f} Å)")
        print(f"  sigma_fixed:  {np.median(sg):.2f}   (IQR {q(sg)[0]:.2f}–{q(sg)[2]:.2f} Å, "
              f"FWHM {2.3548 * np.median(sg):.1f} Å)")


if __name__ == '__main__':
    main()
