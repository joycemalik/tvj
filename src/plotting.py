import math
import os
from typing import Any, Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.linewidth': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'legend.frameon': False, 'legend.fontsize': 7,
}


def save_publication_plots(record: Dict[str, Any], output_dir: str = "outputs/plots") -> str:
    """
    One PNG per spectrum: the rest-frame spectrum with its power-law continuum, then,
    for every line with a least-squares fit, the data ± 1σ, local continuum, Gaussian
    component(s) and total model, with the normalised residuals below.
    """
    os.makedirs(output_dir, exist_ok=True)
    spec_name = os.path.splitext(record['spectrum_name'])[0]
    lines = [l for l in record.get('lines', []) if l.get('plot')]

    U = 1e-13
    unit = r'$F_\lambda$ ($10^{-13}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)'
    ncol = 3
    nrow = math.ceil(len(lines) / ncol)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(11, 3.0 + 3.4 * nrow))
        outer = fig.add_gridspec(1 + nrow, ncol, height_ratios=[1.0] + [1.25] * nrow,
                                 hspace=0.55, wspace=0.28)

        ax0 = fig.add_subplot(outer[0, :])
        wl, fl, cont = record['wavelength'], record['observed_flux'], record['continuum_fit']
        good = fl != 0
        ax0.plot(wl[good], fl[good] / U, color='k', lw=0.6, label='Observed')
        ax0.plot(wl, cont / U, color='0.5', lw=0.9, ls='--',
                 label=f"Power law $F = A\\lambda^\\alpha$ (α = {record.get('spectral_index', float('nan')):.3f})")
        for l in lines:
            if l.get('detected'):
                ax0.axvline(l['center'], color='0.6', lw=0.5, ls=':')
        ax0.set_xlabel('Rest wavelength (Å)')
        ax0.set_ylabel(unit)
        ax0.set_ylim(0, np.nanpercentile(fl[good], 99.5) / U * 1.15)
        ax0.set_title(spec_name, fontsize=9, loc='left')
        ax0.legend(loc='upper right')

        for i, l in enumerate(lines):
            r, c = divmod(i, ncol)
            inner = outer[1 + r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            p = l['plot']
            x = np.asarray(p['x'])
            ax = fig.add_subplot(inner[0])
            axr = fig.add_subplot(inner[1], sharex=ax)
            ax.errorbar(x, np.asarray(p['y']) / U, yerr=p['yerr'] / U, fmt='o', ms=2, color='k',
                        ecolor='0.6', elinewidth=0.6, lw=0)
            ax.plot(x, np.asarray(p['continuum']) / U, color='0.5', lw=0.8, ls='--')
            for k, comp in enumerate(p['components']):
                ax.plot(x, (np.asarray(comp['y']) + np.asarray(p['continuum'])) / U,
                        color='#c0504d' if k == 0 else '0.5', lw=0.8, ls='-' if k == 0 else ':')
            ax.plot(x, np.asarray(p['model']) / U, color='#1f4e8c', lw=1.2)
            for lo, hi in p.get('mask_ranges', []):
                ax.axvspan(lo, hi, color='0.92', zorder=0)
            status = l.get('verification', {}).get('status', '')
            det = '' if l.get('detected') else ', not detected'
            ax.set_title(f"{l['line_name']} — {status}{det}\n"
                         f"F = {l['flux']:.2e} ± {l['flux_err']:.1e} erg s$^{{-1}}$ cm$^{{-2}}$, "
                         f"σ = {l['sigma']:.2f} Å, χ²$_\\nu$ = {l['reduced_chi2']:.2f}",
                         fontsize=7, loc='left')
            plt.setp(ax.get_xticklabels(), visible=False)
            axr.axhline(0, color='k', lw=0.6)
            for lev in (-2, -1, 1, 2):
                axr.axhline(lev, color='0.6', lw=0.5, ls='--' if abs(lev) == 1 else ':')
            axr.plot(x, p['norm_resid'], 'o', ms=2, color='k')
            axr.set_ylim(-4.5, 4.5)
            axr.set_xlabel('Rest wavelength (Å)')
            if c == 0:
                ax.set_ylabel(r'$F_\lambda$ ($10^{-13}$)')
                axr.set_ylabel(r'$\Delta/\sigma$')

        plot_path = os.path.join(output_dir, f"{spec_name}_fit.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return plot_path
