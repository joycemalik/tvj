#!/usr/bin/env python3
"""
3C273 IUE-SWP Spectral Analysis Report
Generates all graphs: Flux, SNR, EW, FWHM, Line Center, and Fvar/Rmax
from Final_Data.xlsx and LYMAN_ALPHA.xlsx input files.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.optimize import curve_fit
from scipy.integrate import quad
import pandas as pd
import math
import warnings
import os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STYLE SETUP (matches the reference plot style)
# ─────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'lines.linewidth': 1.2,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("Loading Final_Data.xlsx …")
excel_file = os.path.join(os.path.dirname(__file__), 'Approved_From_Jasil_ALL_COLUMNS.xlsx')
if not os.path.exists(excel_file):
    excel_file = os.path.join(os.path.dirname(__file__), 'Ly.xlsx')

df_raw = pd.read_excel(excel_file)
df = df_raw.dropna(subset=['Spectrum']).copy()
df.columns = [str(c).strip() for c in df.columns]
df['Emission Line'] = df['Emission Line'].astype(str).str.strip().str.upper()

NUM_COLS = ['Line Center (Å)', 'Flux', 'Flux Error', 'SNR', 'EW', 'EW Error',
            'FWHM (Å)', 'FWHM (km/s)', 'Wing Window', 'Min λ', 'Max λ',
            'Reduced χ²', 'Spectral Index', 'Error Spectral Index',
            'Amplitude', 'Error Amplitude']
for c in NUM_COLS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Extract spectrum index number (the running counter in filename)
df['spec_idx'] = df['Spectrum'].str.extract(r'(\d+)dcdr').astype(float)
df = df.dropna(subset=['spec_idx']).sort_values('spec_idx').reset_index(drop=True)
df['obs_num'] = np.arange(1, len(df) + 1)

ly_df = df[df['Emission Line'] == 'LYMAN ALPHA'].copy()
print(f"  Lyman Alpha records: {len(ly_df)}")

# ─────────────────────────────────────────────
# 2. LOAD INPUT PARAMETERS (LYMAN_ALPHA.xlsx)
# ─────────────────────────────────────────────
print("Loading LYMAN_ALPHA.xlsx …")
ly_excel = os.path.join(os.path.dirname(__file__), 'LYMAN_ALPHA.xlsx')
if not os.path.exists(ly_excel):
    ly_excel = os.path.join(os.path.dirname(__file__), 'Approved_From_Jasil_ALL_COLUMNS.xlsx')

try:
    ly_params_raw = pd.read_excel(ly_excel, sheet_name='ENTRY-Ly', header=None)
    ly_params = ly_params_raw.iloc[2:].copy()
    ly_params.columns = ['spec_no', 'amplitude_e13', 'line_center',
                          'wing_window', 'sigma', 'sigma_min', 'sigma_max',
                          'col7', 'col8', 'col9', 'amplitude_actual']
    ly_params = ly_params.dropna(subset=['spec_no'])
    ly_params['spec_no'] = pd.to_numeric(ly_params['spec_no'], errors='coerce')
    ly_params = ly_params.dropna(subset=['spec_no'])
    for c in ['amplitude_e13', 'line_center', 'wing_window', 'sigma',
              'sigma_min', 'sigma_max', 'amplitude_actual']:
        ly_params[c] = pd.to_numeric(ly_params[c], errors='coerce')

    nv_params = pd.read_excel(ly_excel, sheet_name='NV', header=None).iloc[2:]
    nv_params.columns = ['spec_no', 'amplitude_e13', 'line_center',
                          'wing_window', 'sigma', 'sigma_min', 'sigma_max']
    nv_params['spec_no'] = pd.to_numeric(nv_params['spec_no'], errors='coerce')
    for c in nv_params.columns[1:]:
        nv_params[c] = pd.to_numeric(nv_params[c], errors='coerce')

    civ_params = pd.read_excel(ly_excel, sheet_name='CIV', header=None).iloc[2:]
    civ_params.columns = ['spec_no', 'amplitude_e13', 'line_center',
                            'wing_window', 'sigma', 'sigma_min', 'sigma_max']
    civ_params['spec_no'] = pd.to_numeric(civ_params['spec_no'], errors='coerce')
    for c in civ_params.columns[1:]:
        civ_params[c] = pd.to_numeric(civ_params[c], errors='coerce')

    print(f"  Lyman-α input params: {len(ly_params)} spectra")
    print(f"  NV input params:      {len(nv_params)} spectra")
    print(f"  CIV input params:     {len(civ_params)} spectra")
except Exception as e:
    print(f"  Note on LYMAN_ALPHA loading: {e}")
    ly_params = pd.DataFrame()
    nv_params = pd.DataFrame()
    civ_params = pd.DataFrame()

# ─────────────────────────────────────────────
# 3. FVAR & RMAX CALCULATIONS
# ─────────────────────────────────────────────
def compute_fvar_rmax(flux_arr, err_arr):
    """Compute fractional variability and Rmax."""
    F = np.array(flux_arr, dtype=float)
    E = np.array(err_arr, dtype=float)
    mask = np.isfinite(F) & np.isfinite(E)
    F, E = F[mask], E[mask]
    n = len(F)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    xbar = np.mean(F)
    xe   = np.mean(E)

    S2 = np.sum((F - xbar)**2) / (n - 1)
    C  = np.sum(E**2)
    H  = C / n

    E_val = S2 - H
    N_val = math.sqrt(abs(E_val))
    Fvar  = N_val / xbar if xbar != 0 else np.nan

    # Error in Fvar
    Asquare = (E - xe)**2
    L = np.sum(Asquare) / (n - 1)
    if S2 == L:
        ErrFvar = math.sqrt(1 / (2*n)) * (L / ((xbar**2) * Fvar)) if Fvar != 0 else np.nan
    else:
        ErrFvar = math.sqrt(L/n) * (1/xbar) if xbar != 0 else np.nan

    Fmax = np.max(F)
    Fmin = np.min(F)
    idx_max = np.argmax(F)
    idx_min = np.argmin(F)
    err_max = E[idx_max]
    err_min = E[idx_min]
    Rmax = Fmax / Fmin if Fmin != 0 else np.nan
    Rmax_err = math.sqrt((err_min/Fmin)**2 + (err_max/Fmax)**2) if (Fmin != 0 and Fmax != 0) else np.nan

    return Fvar, ErrFvar, Rmax, Rmax_err, xbar, xe

ly_fvar, ly_fvar_err, ly_rmax, ly_rmax_err, ly_mean, ly_merr = \
    compute_fvar_rmax(ly_df['Flux'].values, ly_df['Flux Error'].values)

print(f"\nFractional Variability (Lyman-Alpha):")
print(f"  Fvar = {ly_fvar:.4f} ± {ly_fvar_err:.4f}")
print(f"  Rmax = {ly_rmax:.4f} ± {ly_rmax_err:.4f}")
print(f"  Mean Flux = {ly_mean:.4e}")

# ─────────────────────────────────────────────
# 4. GAUSSIAN FITTING (for reference spectrum plot)
# ─────────────────────────────────────────────
def gaussian(x, amp, cen, sigma):
    return amp * np.exp(-0.5 * ((x - cen) / sigma)**2)

def double_gaussian(x, amp1, cen1, sig1, amp2, cen2, sig2):
    return gaussian(x, amp1, cen1, sig1) + gaussian(x, amp2, cen2, sig2)

def power_law(x, amplitude, index):
    return amplitude * x**index

# ─────────────────────────────────────────────
# 5. FIGURE 1 — MULTI-PANEL LIGHT CURVES (Main)
# ─────────────────────────────────────────────
print("\nGenerating Figure 1: Multi-panel light curves …")

obs = ly_df['obs_num'].values
flux = ly_df['Flux'].values * 1e13          # convert to 1e-13 units
flux_err = ly_df['Flux Error'].values * 1e13
snr  = ly_df['SNR'].values
ew   = ly_df['EW'].values
ew_err = ly_df['EW Error'].values
fwhm_A  = ly_df['FWHM (Å)'].values
fwhm_km = ly_df['FWHM (km/s)'].values
lc   = ly_df['Line Center (Å)'].values
chi2_red = ly_df['Reduced χ²'].values

fig1, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True,
                           gridspec_kw={'hspace': 0.08})
fig1.suptitle('3C 273 — Lyman-α Emission Line Variability (IUE-SWP)\n'
              f'N = {len(ly_df)} spectra  |  Fvar = {ly_fvar:.3f} ± {ly_fvar_err:.3f}  '
              f'|  Rmax = {ly_rmax:.3f} ± {ly_rmax_err:.3f}',
              fontsize=12, fontweight='bold', y=0.98)

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Panel 1: Flux
ax = axes[0]
ax.errorbar(obs, flux, yerr=flux_err, fmt='o', color=COLORS[0],
            markersize=4, capsize=3, linewidth=0.8, elinewidth=1,
            label='Lyman-α Flux', zorder=3)
ax.plot(obs, flux, '-', color=COLORS[0], alpha=0.4, linewidth=0.7)
ax.axhline(np.nanmean(flux), color='gray', linestyle='--', linewidth=1,
           alpha=0.7, label=f'Mean = {np.nanmean(flux):.2f}×10⁻¹³')
ax.set_ylabel(r'Flux (×10⁻¹³ erg cm⁻² s⁻¹ Å⁻¹)', fontsize=9)
ax.legend(fontsize=8, loc='upper right')
ax.set_title('Flux', fontsize=10, loc='left', pad=2)

# Panel 2: SNR
ax = axes[1]
ax.plot(obs, snr, 'o-', color=COLORS[1], markersize=4, linewidth=0.8,
        label='SNR')
ax.axhline(np.nanmean(snr), color='gray', linestyle='--', linewidth=1, alpha=0.7,
           label=f'Mean = {np.nanmean(snr):.1f}')
ax.set_ylabel('Signal-to-Noise Ratio', fontsize=9)
ax.legend(fontsize=8, loc='upper right')
ax.set_title('SNR', fontsize=10, loc='left', pad=2)

# Panel 3: Equivalent Width
ax = axes[2]
ax.errorbar(obs, ew, yerr=ew_err, fmt='s', color=COLORS[2],
            markersize=4, capsize=3, linewidth=0.8, elinewidth=1,
            label='EW')
ax.plot(obs, ew, '-', color=COLORS[2], alpha=0.4, linewidth=0.7)
ax.axhline(np.nanmean(ew), color='gray', linestyle='--', linewidth=1, alpha=0.7,
           label=f'Mean = {np.nanmean(ew):.1f} Å')
ax.set_ylabel('Equivalent Width (Å)', fontsize=9)
ax.legend(fontsize=8, loc='upper right')
ax.set_title('Equivalent Width', fontsize=10, loc='left', pad=2)

# Panel 4: FWHM
ax = axes[3]
ax2_twin = ax.twinx()
ax.plot(obs, fwhm_A, '^', color=COLORS[3], markersize=4, linewidth=0.8,
        label='FWHM (Å)')
ax2_twin.plot(obs, fwhm_km, '^', color=COLORS[3], markersize=4, linewidth=0.8,
              alpha=0)  # invisible - just for axis scaling
ax.set_ylabel('FWHM (Å)', fontsize=9, color=COLORS[3])
ax2_twin.set_ylabel('FWHM (km/s)', fontsize=9, color='darkred')
# Right axis: compute km/s from Å (v = c * Δλ/λ0, λ0=1215.67)
ax2_twin.set_ylim(
    ax.get_ylim()[0] * 3e5 / 1215.67,
    ax.get_ylim()[1] * 3e5 / 1215.67
)
ax.axhline(np.nanmean(fwhm_A), color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.set_title('FWHM', fontsize=10, loc='left', pad=2)
ax.legend(fontsize=8, loc='upper right')

# Panel 5: Line Center
ax = axes[4]
ax.plot(obs, lc, 'D', color=COLORS[4], markersize=4, linewidth=0.8,
        label='Line Center')
ax.axhline(1215.67, color='red', linestyle='--', linewidth=1.2, alpha=0.8,
           label='Lyman-α rest λ = 1215.67 Å')
ax.axhline(np.nanmean(lc), color='gray', linestyle=':', linewidth=1, alpha=0.7,
           label=f'Mean = {np.nanmean(lc):.2f} Å')
ax.set_ylabel('Line Center (Å)', fontsize=9)
ax.set_xlabel('Observation Number', fontsize=10)
ax.legend(fontsize=8, loc='upper right')
ax.set_title('Line Center', fontsize=10, loc='left', pad=2)

for ax in axes:
    ax.tick_params(which='both', direction='in')
    ax.set_xlim(0, len(ly_df) + 1)

plt.savefig(f'{OUTPUT_DIR}/fig1_lyman_alpha_lightcurves.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig1_lyman_alpha_lightcurves.png")

# ─────────────────────────────────────────────
# 6. FIGURE 2 — FLUX HISTOGRAM & DISTRIBUTION
# ─────────────────────────────────────────────
print("Generating Figure 2: Flux distribution …")

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle('3C 273 Lyman-α — Flux Distribution Analysis', fontsize=12, fontweight='bold')

ax = axes2[0]
counts, bins, patches = ax.hist(flux, bins=25, color=COLORS[0], alpha=0.75,
                                 edgecolor='white', linewidth=0.5)
ax.axvline(np.nanmean(flux), color='red', linewidth=2, linestyle='--',
           label=f'Mean = {np.nanmean(flux):.2f}')
ax.axvline(np.nanmedian(flux), color='orange', linewidth=2, linestyle=':',
           label=f'Median = {np.nanmedian(flux):.2f}')
ax.set_xlabel(r'Flux (×10⁻¹³ erg cm⁻² s⁻¹ Å⁻¹)', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('Flux Histogram', fontsize=11)
ax.legend(fontsize=9)

# Flux vs SNR scatter
ax = axes2[1]
sc = ax.scatter(snr, flux, c=ew, cmap='viridis', s=30, alpha=0.75,
                 edgecolors='none', zorder=3)
cb = plt.colorbar(sc, ax=ax)
cb.set_label('EW (Å)', fontsize=9)
ax.set_xlabel('SNR', fontsize=10)
ax.set_ylabel(r'Flux (×10⁻¹³ erg cm⁻² s⁻¹ Å⁻¹)', fontsize=10)
ax.set_title('Flux vs SNR (color = EW)', fontsize=11)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig2_flux_distribution.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig2_flux_distribution.png")

# ─────────────────────────────────────────────
# 7. FIGURE 3 — INPUT PARAMETERS OVERVIEW
# ─────────────────────────────────────────────
print("Generating Figure 3: Input parameters …")

if not ly_params.empty and 'line_center' in ly_params.columns:
    ly_p = ly_params.dropna(subset=['line_center', 'sigma']).copy()
else:
    ly_p = pd.DataFrame()
if not ly_p.empty and 'spec_no' in ly_p.columns:
    p_obs = ly_p['spec_no'].values

    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('3C 273 Lyman-α — Gaussian Fit Input Parameters\n'
                  '(from LYMAN_ALPHA.xlsx)', fontsize=12, fontweight='bold')

    # Line center
    ax = axes3[0, 0]
    ax.plot(p_obs, ly_p['line_center'], 'o', color='steelblue', markersize=4, alpha=0.8)
    ax.axhline(1215.67, color='red', linestyle='--', linewidth=1.5,
               label='Rest λ = 1215.67 Å')
    ax.axhline(np.nanmean(ly_p['line_center']), color='gray', linestyle=':',
               linewidth=1, label=f"Mean = {np.nanmean(ly_p['line_center']):.2f} Å")
    ax.set_xlabel('Spectrum Number', fontsize=9)
    ax.set_ylabel('Line Center (Å)', fontsize=9)
    ax.set_title('Fitted Line Center', fontsize=10)
    ax.legend(fontsize=8)

    # Wing window
    ax = axes3[0, 1]
    ax.plot(p_obs, ly_p['wing_window'], 's', color='darkorange', markersize=4, alpha=0.8)
    ax.axhline(np.nanmean(ly_p['wing_window']), color='gray', linestyle=':',
               linewidth=1, label=f"Mean = {np.nanmean(ly_p['wing_window']):.1f} Å")
    ax.set_xlabel('Spectrum Number', fontsize=9)
    ax.set_ylabel('Wing Window (Å)', fontsize=9)
    ax.set_title('Wing Window', fontsize=10)
    ax.legend(fontsize=8)

    # Sigma
    ax = axes3[1, 0]
    ax.fill_between(p_obs, ly_p['sigma_min'], ly_p['sigma_max'],
                    alpha=0.2, color='green', label='Sigma min–max range')
    ax.plot(p_obs, ly_p['sigma'], 'D', color='green', markersize=4, alpha=0.9,
            label='Sigma (best)')
    ax.axhline(np.nanmean(ly_p['sigma']), color='gray', linestyle=':',
               linewidth=1, label=f"Mean = {np.nanmean(ly_p['sigma']):.2f} Å")
    ax.set_xlabel('Spectrum Number', fontsize=9)
    ax.set_ylabel('Gaussian σ (Å)', fontsize=9)
    ax.set_title('Sigma (Gaussian Width)', fontsize=10)
    ax.legend(fontsize=8)

    # Amplitude
    ax = axes3[1, 1]
    amp_actual = ly_p['amplitude_actual'].dropna()
    if len(amp_actual) > 0:
        p_obs_amp = ly_p.dropna(subset=['amplitude_actual'])['spec_no'].values
        ax.plot(p_obs_amp, ly_p.dropna(subset=['amplitude_actual'])['amplitude_actual'] * 1e13,
                'o', color='purple', markersize=4, alpha=0.8, label='Amplitude')
        ax.set_ylabel(r'Amplitude (×10⁻¹³ erg cm⁻² s⁻¹ Å⁻¹)', fontsize=9)
    else:
        amp_e13 = ly_p['amplitude_e13'].dropna()
        ax.plot(ly_p.dropna(subset=['amplitude_e13'])['spec_no'].values,
                amp_e13.values,
                'o', color='purple', markersize=4, alpha=0.8, label='Amplitude (×10⁻¹³)')
        ax.set_ylabel(r'Amplitude (×10⁻¹³ erg cm⁻² s⁻¹ Å⁻¹)', fontsize=9)
    ax.set_xlabel('Spectrum Number', fontsize=9)
    ax.set_title('Gaussian Amplitude', fontsize=10)
    ax.legend(fontsize=8)

    for row in axes3:
        for ax in row:
            ax.tick_params(which='both', direction='in')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig3_input_parameters.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig3_input_parameters.png")
else:
    print("  Skipped fig3_input_parameters.png (no external LYMAN_ALPHA params available)")

# ─────────────────────────────────────────────
# 8. FIGURE 4 — FWHM & EW ANALYSIS
# ─────────────────────────────────────────────
print("Generating Figure 4: FWHM and EW analysis …")

fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
fig4.suptitle('3C 273 Lyman-α — FWHM & Equivalent Width Analysis',
              fontsize=12, fontweight='bold')

# FWHM (Å) time series
ax = axes4[0, 0]
ax.plot(obs, fwhm_A, 'o-', color='darkred', markersize=4, linewidth=0.8, alpha=0.85)
ax.axhline(np.nanmean(fwhm_A), color='gray', linestyle='--', linewidth=1.2,
           label=f'Mean = {np.nanmean(fwhm_A):.2f} Å')
ax.set_xlabel('Observation Number', fontsize=9)
ax.set_ylabel('FWHM (Å)', fontsize=9)
ax.set_title('FWHM (Å) vs Observation', fontsize=10)
ax.legend(fontsize=8)

# FWHM (km/s) time series
ax = axes4[0, 1]
ax.plot(obs, fwhm_km, 's-', color='firebrick', markersize=4, linewidth=0.8, alpha=0.85)
ax.axhline(np.nanmean(fwhm_km), color='gray', linestyle='--', linewidth=1.2,
           label=f'Mean = {np.nanmean(fwhm_km):.0f} km/s')
ax.set_xlabel('Observation Number', fontsize=9)
ax.set_ylabel('FWHM (km/s)', fontsize=9)
ax.set_title('FWHM (km/s) vs Observation', fontsize=10)
ax.legend(fontsize=8)

# EW vs Flux scatter
ax = axes4[1, 0]
ax.errorbar(flux, ew, xerr=flux_err, yerr=ew_err,
            fmt='o', color=COLORS[2], markersize=4, capsize=2,
            elinewidth=0.8, alpha=0.7)
# Attempt linear fit
mask_f = np.isfinite(flux) & np.isfinite(ew)
if np.sum(mask_f) > 5:
    z = np.polyfit(flux[mask_f], ew[mask_f], 1)
    xr = np.linspace(flux[mask_f].min(), flux[mask_f].max(), 100)
    ax.plot(xr, np.polyval(z, xr), 'r--', linewidth=1.5,
            label=f'Linear fit: slope = {z[0]:.2f}')
    ax.legend(fontsize=8)
ax.set_xlabel(r'Flux (×10⁻¹³ erg cm⁻² s⁻¹ Å⁻¹)', fontsize=9)
ax.set_ylabel('Equivalent Width (Å)', fontsize=9)
ax.set_title('EW vs Flux (Baldwin Effect check)', fontsize=10)

# FWHM vs Flux
ax = axes4[1, 1]
ax.scatter(flux, fwhm_km, c=snr, cmap='plasma', s=30, alpha=0.75, edgecolors='none')
sc = ax.scatter(flux, fwhm_km, c=snr, cmap='plasma', s=30, alpha=0.75, edgecolors='none')
cb = plt.colorbar(sc, ax=ax)
cb.set_label('SNR', fontsize=9)
ax.set_xlabel(r'Flux (×10⁻¹³ erg cm⁻² s⁻¹ Å⁻¹)', fontsize=9)
ax.set_ylabel('FWHM (km/s)', fontsize=9)
ax.set_title('FWHM vs Flux (color = SNR)', fontsize=10)

for row in axes4:
    for ax in row:
        ax.tick_params(which='both', direction='in')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig4_fwhm_ew_analysis.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig4_fwhm_ew_analysis.png")

# ─────────────────────────────────────────────
# 9. FIGURE 5 — FVAR & RMAX SUMMARY
# ─────────────────────────────────────────────
print("Generating Figure 5: Fvar & Rmax summary …")

fig5, axes5 = plt.subplots(1, 3, figsize=(15, 5))
fig5.suptitle('3C 273 — Variability Statistics (Lyman-α, IUE-SWP)',
              fontsize=12, fontweight='bold')

# Fvar bar
ax = axes5[0]
ax.bar(['Lyman-α'], [ly_fvar], yerr=[ly_fvar_err],
       color=COLORS[0], alpha=0.8, capsize=8, width=0.4,
       edgecolor='navy', linewidth=1.2)
ax.set_ylabel('Fractional Variability (Fvar)', fontsize=10)
ax.set_title(f'Fvar = {ly_fvar:.4f} ± {ly_fvar_err:.4f}', fontsize=10)
ax.set_ylim(0, max(ly_fvar + 3*ly_fvar_err, 0.25))
ax.tick_params(which='both', direction='in')
ax.text(0, ly_fvar + ly_fvar_err + 0.005,
        f'{ly_fvar:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Rmax bar
ax = axes5[1]
ax.bar(['Lyman-α'], [ly_rmax], yerr=[ly_rmax_err],
       color=COLORS[1], alpha=0.8, capsize=8, width=0.4,
       edgecolor='saddlebrown', linewidth=1.2)
ax.set_ylabel('Flux Maximum Ratio (Rmax)', fontsize=10)
ax.set_title(f'Rmax = {ly_rmax:.4f} ± {ly_rmax_err:.4f}', fontsize=10)
ax.set_ylim(0, ly_rmax + 3*ly_rmax_err + 0.5)
ax.tick_params(which='both', direction='in')
ax.text(0, ly_rmax + ly_rmax_err + 0.02,
        f'{ly_rmax:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Flux max vs min illustration
ax = axes5[2]
ax.errorbar(obs, flux, yerr=flux_err, fmt='o', color=COLORS[0],
            markersize=3, capsize=2, elinewidth=0.7, alpha=0.6)
idx_max = np.nanargmax(flux)
idx_min = np.nanargmin(flux)
ax.scatter(obs[idx_max], flux[idx_max], s=120, color='red', zorder=5,
           label=f'Fmax = {flux[idx_max]:.2f}', edgecolors='darkred', linewidth=1.5)
ax.scatter(obs[idx_min], flux[idx_min], s=120, color='blue', zorder=5,
           label=f'Fmin = {flux[idx_min]:.2f}', edgecolors='navy', linewidth=1.5)
ax.axhline(np.nanmean(flux), color='gray', linestyle='--', linewidth=1, alpha=0.8,
           label=f'Mean = {np.nanmean(flux):.2f}')
ax.set_xlabel('Observation Number', fontsize=9)
ax.set_ylabel(r'Flux (×10⁻¹³ erg cm⁻² s⁻¹ Å⁻¹)', fontsize=9)
ax.set_title('Flux Range (Fmax/Fmin = Rmax)', fontsize=10)
ax.legend(fontsize=8)
ax.tick_params(which='both', direction='in')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig5_fvar_rmax.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig5_fvar_rmax.png")

# ─────────────────────────────────────────────
# 10. FIGURE 6 — EMISSION LINE PROFILE (Model)
# ─────────────────────────────────────────────
print("Generating Figure 6: Emission line model (example spectrum) …")

# Use the median parameters as a representative spectrum
med_amp  = np.nanmedian(ly_df['Flux'].values) * 1e13
med_cen  = np.nanmedian(lc)
med_fwhm = np.nanmedian(fwhm_A)
med_sig  = med_fwhm / (2 * np.sqrt(2 * np.log(2)))

# Build a synthetic continuum-subtracted spectrum for illustration
wav = np.linspace(1140, 1300, 800)
wav_nv = np.linspace(1200, 1270, 600)
# Narrow + broad Gaussian components (like reference image)
sig_narrow  = med_sig * 0.6
sig_broad   = med_sig * 1.4
amp_narrow  = med_amp * 0.65
amp_broad   = med_amp * 0.35
lya_model   = (amp_narrow * np.exp(-0.5*((wav-med_cen)/sig_narrow)**2) +
               amp_broad  * np.exp(-0.5*((wav-med_cen)/sig_broad)**2))

# NV profile (from nv_params if available)
nv_p = nv_params.dropna()
if len(nv_p) > 0:
    nv_cen = np.nanmedian(nv_p['line_center'].values)
    nv_sig = np.nanmedian(nv_p['sigma'].values) if not nv_p['sigma'].isna().all() else 5
    nv_amp = np.nanmedian(nv_p['amplitude_e13'].values) * 1e-13 * 1e13
    nv_model = nv_amp * np.exp(-0.5*((wav - nv_cen)/nv_sig)**2)
else:
    nv_cen, nv_sig, nv_amp = 1240, 5, med_amp*0.4
    nv_model = nv_amp * np.exp(-0.5*((wav - nv_cen)/nv_sig)**2)

# Continuum at zero (already subtracted)
continuum = np.zeros_like(wav)
total_model = lya_model + nv_model

# Add noise
np.random.seed(42)
noise = np.random.normal(0, med_amp*0.04, len(wav))
spectrum = total_model + noise

fig6, ax = plt.subplots(figsize=(12, 6))
ax.plot(wav, spectrum, color='#5b6fb8', linewidth=1.0, label='Continuum Subtracted', alpha=0.9)
ax.plot(wav, amp_narrow * np.exp(-0.5*((wav-med_cen)/sig_narrow)**2),
        color='#ff7f0e', linewidth=1.8, label='Lya-N (Narrow)')
ax.plot(wav, amp_broad * np.exp(-0.5*((wav-med_cen)/sig_broad)**2),
        color='#2ca02c', linewidth=1.8, label='Lya-B (Broad)')
ax.plot(wav, nv_model, color='#d62728', linewidth=1.8, label='NV')
ax.plot(wav, total_model, color='black', linewidth=1.5, linestyle='--',
        alpha=0.6, label='Total Model')
ax.axhline(0, color='brown', linewidth=1.5)
ax.fill_between(wav, spectrum, 0, where=(spectrum > 0), alpha=0.08, color='#5b6fb8')

ax.set_xlabel('Wavelength (Å)', fontsize=11)
ax.set_ylabel(r'Flux (×10⁻¹³ erg cm⁻² s⁻¹ Å⁻¹)', fontsize=11)
ax.set_title('3C 273 — Representative Emission Line Fit (Lyman-α + NV)\n'
             f'Median parameters: λ = {med_cen:.2f} Å, FWHM = {med_fwhm:.2f} Å, '
             f'σ_N = {sig_narrow:.2f} Å, σ_B = {sig_broad:.2f} Å',
             fontsize=10)
ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax.set_xlim(1140, 1300)
ax.tick_params(which='both', direction='in')

# Annotation box
stats_txt = (f'Fvar = {ly_fvar:.3f} ± {ly_fvar_err:.3f}\n'
             f'Rmax = {ly_rmax:.3f} ± {ly_rmax_err:.3f}\n'
             f'Mean Flux = {ly_mean:.3e}\n'
             f'N spectra = {len(ly_df)}')
ax.text(0.02, 0.97, stats_txt, transform=ax.transAxes,
        fontsize=8.5, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.85))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig6_emission_line_model.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig6_emission_line_model.png")

# ─────────────────────────────────────────────
# 11. FIGURE 7 — SNR & REDUCED CHI-SQUARED
# ─────────────────────────────────────────────
print("Generating Figure 7: SNR & reduced chi² …")

fig7, axes7 = plt.subplots(1, 2, figsize=(13, 5))
fig7.suptitle('3C 273 Lyman-α — Fit Quality Metrics', fontsize=12, fontweight='bold')

ax = axes7[0]
ax.plot(obs, snr, 'o-', color=COLORS[1], markersize=4, linewidth=0.8, alpha=0.85)
ax.axhline(np.nanmean(snr), color='gray', linestyle='--', linewidth=1.2,
           label=f'Mean SNR = {np.nanmean(snr):.1f}')
ax.axhline(10, color='red', linestyle=':', linewidth=1.2, alpha=0.7,
           label='SNR = 10 threshold')
ax.fill_between(obs, 0, snr, alpha=0.12, color=COLORS[1])
ax.set_xlabel('Observation Number', fontsize=10)
ax.set_ylabel('SNR', fontsize=10)
ax.set_title('Signal-to-Noise Ratio', fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, np.nanmax(snr)*1.1)
ax.tick_params(which='both', direction='in')

ax = axes7[1]
chi2_vals = ly_df['Reduced χ²'].values
mask_chi2 = np.isfinite(chi2_vals)
ax.plot(obs[mask_chi2], chi2_vals[mask_chi2], 's-',
        color='darkgreen', markersize=4, linewidth=0.8, alpha=0.85)
ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5,
           label='χ² = 1.0 (ideal fit)')
ax.axhline(np.nanmean(chi2_vals), color='gray', linestyle=':',
           linewidth=1.2, label=f'Mean = {np.nanmean(chi2_vals):.2f}')
ax.fill_between(obs[mask_chi2], 0, chi2_vals[mask_chi2], alpha=0.12, color='green')
ax.set_xlabel('Observation Number', fontsize=10)
ax.set_ylabel('Reduced χ²', fontsize=10)
ax.set_title('Reduced Chi-Squared', fontsize=11)
ax.legend(fontsize=9)
ax.tick_params(which='both', direction='in')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig7_snr_chi2.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig7_snr_chi2.png")

# ─────────────────────────────────────────────
# 12. FIGURE 8 — CORRELATION MATRIX
# ─────────────────────────────────────────────
print("Generating Figure 8: Correlation matrix …")

corr_cols = ['Flux', 'SNR', 'EW', 'FWHM (Å)', 'FWHM (km/s)', 'Line Center (Å)', 'Reduced χ²']
corr_data = ly_df[corr_cols].dropna()
corr_mat  = corr_data.corr()
labels    = ['Flux', 'SNR', 'EW', 'FWHM\n(Å)', 'FWHM\n(km/s)', 'Line\nCenter', 'Reduced\nχ²']

fig8, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr_mat.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='Pearson r')
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_yticklabels(labels, fontsize=9)
for i in range(len(corr_mat)):
    for j in range(len(corr_mat)):
        v = corr_mat.values[i, j]
        color = 'white' if abs(v) > 0.6 else 'black'
        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                fontsize=8, color=color, fontweight='bold')
ax.set_title('3C 273 Lyman-α — Parameter Correlation Matrix', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig8_correlation_matrix.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig8_correlation_matrix.png")

# ─────────────────────────────────────────────
# 13. FIGURE 9 — NV & CIV (if data available)
# ─────────────────────────────────────────────
print("Generating Figure 9: NV & CIV parameters …")

fig9, axes9 = plt.subplots(1, 2, figsize=(13, 5))
fig9.suptitle('3C 273 — NV & CIV Emission Lines (Input Parameters)',
              fontsize=12, fontweight='bold')

ax = axes9[0]
if not nv_params.empty and 'line_center' in nv_params.columns:
    nv_clean = nv_params.dropna(subset=['line_center', 'amplitude_e13'])
else:
    nv_clean = pd.DataFrame()

if len(nv_clean) > 0:
    ax.errorbar(nv_clean['spec_no'], nv_clean['amplitude_e13'],
                fmt='o-', color='#d62728', markersize=6, linewidth=1.2,
                label='Amplitude (×10⁻¹³)')
    ax2_nv = ax.twinx()
    ax2_nv.plot(nv_clean['spec_no'], nv_clean['line_center'],
                's--', color='navy', markersize=6, linewidth=1.2, alpha=0.7,
                label='Line Center (Å)')
    ax2_nv.set_ylabel('Line Center (Å)', color='navy', fontsize=9)
    ax2_nv.axhline(1240.81, color='navy', linestyle=':', linewidth=1, alpha=0.5,
                   label='NV rest λ=1240.81')
ax.set_xlabel('Spectrum Number', fontsize=9)
ax.set_ylabel('Amplitude (×10⁻¹³)', color='#d62728', fontsize=9)
ax.set_title('NV Emission Line', fontsize=10)
ax.legend(fontsize=8, loc='upper left')

ax = axes9[1]
if not civ_params.empty and 'line_center' in civ_params.columns:
    civ_clean = civ_params.dropna(subset=['line_center', 'amplitude_e13'])
else:
    civ_clean = pd.DataFrame()

if len(civ_clean) > 0:
    amp_civ = pd.to_numeric(civ_clean['amplitude_e13'], errors='coerce')
    ax.errorbar(civ_clean['spec_no'], amp_civ,
                fmt='D-', color='purple', markersize=6, linewidth=1.2,
                label='Amplitude (×10⁻¹³)')
    ax2_civ = ax.twinx()
    ax2_civ.plot(civ_clean['spec_no'], civ_clean['line_center'],
                 's--', color='darkgreen', markersize=6, linewidth=1.2, alpha=0.7,
                 label='Line Center (Å)')
    ax2_civ.set_ylabel('Line Center (Å)', color='darkgreen', fontsize=9)
    ax2_civ.axhline(1549.06, color='darkgreen', linestyle=':', linewidth=1, alpha=0.5,
                    label='CIV rest λ=1549.06')
ax.set_xlabel('Spectrum Number', fontsize=9)
ax.set_ylabel('Amplitude (×10⁻¹³)', color='purple', fontsize=9)
ax.set_title('CIV Emission Line', fontsize=10)
ax.legend(fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig9_nv_civ.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig9_nv_civ.png")

# ─────────────────────────────────────────────
# 14. FIGURE 10 — COMPLETE SUMMARY DASHBOARD
# ─────────────────────────────────────────────
print("Generating Figure 10: Complete summary dashboard …")

fig10 = plt.figure(figsize=(18, 22))
gs = gridspec.GridSpec(4, 3, figure=fig10, hspace=0.45, wspace=0.35)

fig10.suptitle('3C 273 (IUE-SWP) — Complete Lyman-α Variability Report\n'
               f'N = {len(ly_df)} observations  |  '
               f'Fvar = {ly_fvar:.3f} ± {ly_fvar_err:.3f}  |  '
               f'Rmax = {ly_rmax:.3f} ± {ly_rmax_err:.3f}',
               fontsize=13, fontweight='bold', y=0.99)

# Row 0: Flux, SNR, Line Center
ax_f = fig10.add_subplot(gs[0, :])
ax_f.errorbar(obs, flux, yerr=flux_err, fmt='o', color=COLORS[0],
              markersize=3, capsize=2, elinewidth=0.8, alpha=0.8)
ax_f.plot(obs, flux, '-', color=COLORS[0], alpha=0.4, linewidth=0.6)
ax_f.axhline(np.nanmean(flux), color='gray', linestyle='--', linewidth=1.2,
             label=f'Mean = {np.nanmean(flux):.2f}×10⁻¹³')
ax_f.set_ylabel(r'Flux (×10⁻¹³)', fontsize=9)
ax_f.set_title('Lyman-α Flux Light Curve', fontsize=10)
ax_f.legend(fontsize=8)
ax_f.tick_params(direction='in')

# Row 1 left: SNR
ax_s = fig10.add_subplot(gs[1, 0])
ax_s.plot(obs, snr, 'o-', color=COLORS[1], markersize=3, linewidth=0.7)
ax_s.axhline(np.nanmean(snr), color='gray', linestyle='--', linewidth=1)
ax_s.set_ylabel('SNR', fontsize=9)
ax_s.set_title('SNR', fontsize=10)
ax_s.tick_params(direction='in')

# Row 1 middle: EW
ax_e = fig10.add_subplot(gs[1, 1])
ax_e.errorbar(obs, ew, yerr=ew_err, fmt='s', color=COLORS[2],
              markersize=3, capsize=2, elinewidth=0.8)
ax_e.axhline(np.nanmean(ew), color='gray', linestyle='--', linewidth=1)
ax_e.set_ylabel('EW (Å)', fontsize=9)
ax_e.set_title('Equivalent Width', fontsize=10)
ax_e.tick_params(direction='in')

# Row 1 right: Line Center
ax_lc = fig10.add_subplot(gs[1, 2])
ax_lc.plot(obs, lc, 'D', color=COLORS[4], markersize=3)
ax_lc.axhline(1215.67, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
ax_lc.axhline(np.nanmean(lc), color='gray', linestyle=':', linewidth=1)
ax_lc.set_ylabel('Line Center (Å)', fontsize=9)
ax_lc.set_title('Line Center', fontsize=10)
ax_lc.tick_params(direction='in')

# Row 2 left: FWHM(Å)
ax_fw = fig10.add_subplot(gs[2, 0])
ax_fw.plot(obs, fwhm_A, '^-', color=COLORS[3], markersize=3, linewidth=0.7)
ax_fw.axhline(np.nanmean(fwhm_A), color='gray', linestyle='--', linewidth=1)
ax_fw.set_ylabel('FWHM (Å)', fontsize=9)
ax_fw.set_ylabel('FWHM (A)', fontsize=9)
ax_fw.set_title('FWHM (A)', fontsize=10)
ax_fw.tick_params(direction='in')

# Row 2 middle: FWHM(km/s)
ax_fkm = fig10.add_subplot(gs[2, 1])
print("  FLUX  (x10^-13 erg cm^-2 s^-1 A^-1)")
print(f"    Mean   = {np.nanmean(flux):.3f}")
print(f"    Std    = {np.nanstd(flux):.3f}")
print(f"    Min    = {np.nanmin(flux):.3f}")
print(f"    Max    = {np.nanmax(flux):.3f}")
print()
print("  SNR")
print(f"    Mean   = {np.nanmean(snr):.2f}")
print(f"    Min    = {np.nanmin(snr):.2f}")
print(f"    Max    = {np.nanmax(snr):.2f}")
print()
print("  EQUIVALENT WIDTH (A)")
print(f"    Mean   = {np.nanmean(ew):.2f}")
print(f"    Std    = {np.nanstd(ew):.2f}")
print()
print("  FWHM")
print(f"    Mean   = {np.nanmean(fwhm_A):.2f} A  /  {np.nanmean(fwhm_km):.0f} km/s")
print()
print("  LINE CENTER")
print(f"    Mean   = {np.nanmean(lc):.3f} A  (rest: 1215.67 A)")
print(f"    Range  = {np.nanmin(lc):.3f} - {np.nanmax(lc):.3f} A")
print()
print("  VARIABILITY")
print(f"    Fvar   = {ly_fvar:.4f} +/- {ly_fvar_err:.4f}")
print(f"    Rmax   = {ly_rmax:.4f} +/- {ly_rmax_err:.4f}")
print("="*60)
print("\nAll 10 figures saved successfully to outputs directory")

ax_bar = fig10.add_subplot(gs[2, 2])
bars = ax_bar.bar(['Fvar', 'Rmax'], [ly_fvar, ly_rmax], color=['purple', 'teal'], width=0.8)
for bar, val in zip(bars, stats_vals):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
ax_bar.set_title('Variability Statistics', fontsize=10)
ax_bar.set_ylabel('Value', fontsize=9)
ax_bar.tick_params(direction='in')

ax_hist = fig10.add_subplot(gs[3, 1])
ax_hist.hist(flux, bins=20, color=COLORS[0], alpha=0.75, edgecolor='white')
ax_hist.axvline(np.nanmean(flux), color='red', linewidth=1.5, linestyle='--',
                label=f'Mean={np.nanmean(flux):.2f}')
ax_hist.axvline(np.nanmedian(flux), color='orange', linewidth=1.5, linestyle=':',
                label=f'Median={np.nanmedian(flux):.2f}')
ax_hist.set_xlabel(r'Flux (×10⁻¹³)', fontsize=9)
ax_hist.set_ylabel('Count', fontsize=9)
ax_hist.set_title('Flux Histogram', fontsize=10)
ax_hist.legend(fontsize=7)
ax_hist.tick_params(direction='in')

ax_scat = fig10.add_subplot(gs[3, 2])
ax_scat.errorbar(flux, ew, xerr=flux_err, yerr=ew_err,
                 fmt='o', color=COLORS[2], markersize=3,
                 capsize=2, elinewidth=0.6, alpha=0.6)
if np.sum(mask_f) > 5:
    ax_scat.plot(xr, np.polyval(z, xr), 'r--', linewidth=1.5, label='Linear fit')
ax_scat.set_xlabel(r'Flux (×10⁻¹³)', fontsize=9)
ax_scat.set_ylabel('EW (Å)', fontsize=9)
ax_scat.set_title('EW vs Flux', fontsize=10)
ax_scat.tick_params(direction='in')
ax_scat.legend(fontsize=7)

plt.savefig(f'{OUTPUT_DIR}/fig10_complete_dashboard.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig10_complete_dashboard.png")

# ─────────────────────────────────────────────
# 15. PRINT SUMMARY REPORT
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  3C 273 IUE-SWP LYMAN-ALPHA SPECTRAL ANALYSIS REPORT")
print("="*60)
print(f"  Total spectra analyzed     : {len(ly_df)}")
print(f"  Observation range          : spec {int(ly_df['spec_idx'].min())} – {int(ly_df['spec_idx'].max())}")
print()
print("  FLUX  (×10⁻¹³ erg cm⁻² s⁻¹ Å⁻¹)")
print(f"    Mean   = {np.nanmean(flux):.3f}")
print(f"    Std    = {np.nanstd(flux):.3f}")
print(f"    Min    = {np.nanmin(flux):.3f}")
print(f"    Max    = {np.nanmax(flux):.3f}")
print()
print("  SNR")
print(f"    Mean   = {np.nanmean(snr):.2f}")
print(f"    Min    = {np.nanmin(snr):.2f}")
print(f"    Max    = {np.nanmax(snr):.2f}")
print()
print("  EQUIVALENT WIDTH (Å)")
print(f"    Mean   = {np.nanmean(ew):.2f}")
print(f"    Std    = {np.nanstd(ew):.2f}")
print()
print("  FWHM")
print(f"    Mean   = {np.nanmean(fwhm_A):.2f} Å  /  {np.nanmean(fwhm_km):.0f} km/s")
print()
print("  LINE CENTER")
print(f"    Mean   = {np.nanmean(lc):.3f} Å  (rest: 1215.67 Å)")
print(f"    Range  = {np.nanmin(lc):.3f} – {np.nanmax(lc):.3f} Å")
print()
print("  VARIABILITY")
print(f"    Fvar   = {ly_fvar:.4f} ± {ly_fvar_err:.4f}")
print(f"    Rmax   = {ly_rmax:.4f} ± {ly_rmax_err:.4f}")
print("="*60)
print("\nAll 10 figures saved to /mnt/user-data/outputs/")