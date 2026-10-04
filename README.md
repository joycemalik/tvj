# 3C 273 IUE UV Emission-Line Pipeline

**Least-squares emission-line fitting, statistical verification and per-year fractional variability for 255 archival IUE SWP spectra of the quasar 3C 273.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![Deploy](https://img.shields.io/badge/Deploy-Vercel-black?style=flat-square&logo=vercel)](https://vercel.com)

---

## Overview

For every spectrum the pipeline fits a power-law continuum, locates each emission line on a coarse grid, and fits it by **weighted least squares** as a Gaussian plus a local linear continuum (overlapping lines jointly). Each fit reports parameters with covariance-based uncertainties and is **verified** with standard statistical tests. Detected line fluxes are joined to observation dates and the **fractional variability amplitude F_var** is computed per line and per calendar year (Vaughan et al. 2003).

Lines fitted: Lyβ + O VI, Lyα, N V, O I + Si II, C II, Si IV + O IV], C IV, He II, O III].

The full method, with every formula and its source, is on the **About** page (`/about`, `templates/about.html`).

---

## Table of Contents

1. [Data](#1-data)
2. [Method](#2-method)
3. [Verification](#3-verification)
4. [Graphs](#4-graphs)
5. [Variability (F_var)](#5-variability-f_var)
6. [Usage](#6-usage)
7. [Configuration](#7-configuration)
8. [Output schema](#8-output-schema)
9. [Repository structure](#9-repository-structure)
10. [Web application routes](#10-web-application-routes)
11. [Deployment](#11-deployment)
12. [References](#12-references)

---

## 1. Data

- **Spectra** — `spectrum/*.txt`, 255 IUE SWP low-dispersion (MXLO) spectra, two columns: wavelength (Å), F_λ (erg s⁻¹ cm⁻² Å⁻¹).
- **Rest frame** — the files are divided by (1+z), z = 0.158. Evidence: pixel spacing 1.448 Å = 1.676/(1+z) Å, and geocoronal Lyα airglow (1215.67 Å observed) sits at 1049.8 Å. Usable range 993.6–1832 Å.
- **Dates** — `jd.xlsx`, sheet `Sheet2` (SWP log): LDATEOBS, LJD-OBS per spectrum. Five spectra (251–255) have no date and are excluded from F_var.
- **Instrument** — resolution ≈ 6 Å FWHM observed (5.18 Å rest frame); neighbouring pixels are correlated (lag-1 ρ ≈ 0.5) because MXLO spectra are resampled.

## 2. Method

| Step | Module | What it does |
|---|---|---|
| Continuum | `src/continuum.py` | OLS of ln F on ln λ in five line-free windows (1100–1150, 1325–1370, 1425–1475, 1590–1620, 1660–1700 Å): F_c = A λ^α, with standard errors σ_α = √(s²/S_xx), σ_lnA = √(s²(1/n + x̄²/S_xx)). Reproduces the tabulated spectral indices of the campaign to < 5×10⁻⁵. |
| Peak | `src/candidate_engine.py` | Brightest smoothed pixel within a per-line radius of the rest wavelength, per spectrum; masked ranges (airglow) skipped. |
| Seed grid | `src/candidate_engine.py` | Coarse (μ, σ, W) grid inside the per-line ranges, ranked by a composite score. The best candidate only seeds the fit. |
| Least-squares fit | `src/refine.py` | M(λ) = Σ A_k exp[−(λ−μ_k)²/2σ_k²] + c₀ + c₁(λ−λ̄), minimised with `scipy.optimize.least_squares` (bounds on A, μ, σ). Lyα and N V are fitted together. Pixel noise: DER_SNR on the residuals (Stoehr et al. 2008). Covariance C = (JᵀJ)⁻¹·max(1, χ²_red). |
| Statistics | `src/refine.py` | F = √(2π) A σ; σ_F² = 2π(σ²C_AA + A²C_σσ + 2AσC_Aσ); EW = F/F_c(μ); FWHM = 2√(2 ln 2) σ; FWHM_int = √(FWHM² − 5.18²); χ²_red with ν = n_pix − n_par. |
| Detection | `src/quality.py` | Detected when F/σ_F ≥ 3. A low χ²_red is not penalised. |

**Width ranges (Lyα).** FWHM 10–24 Å ⇒ σ = 4.25–10.19 Å (continuous); wing window 10–24 Å. The range brackets the approved fits (FWHM 10.6–20.0 Å, 2600–4950 km s⁻¹) and published 3C 273 broad-line widths (Paltani & Türler 2005). For every line the lower limit is the instrumental resolution (σ ≥ 2.2 Å).

**Reference-compatible flux.** `flux_window` integrates the Gaussian only over μ ± W/2, the convention of the reference tables; it captures erf(W/(2√2σ)) of the line (72–92 % for the approved Lyα windows). `flux` is the full Gaussian flux.

## 3. Verification

Every least-squares fit is checked (`src/refine.py::verify_fit`). **VERIFIED** = all required checks pass; otherwise **CHECK** with the failed tests listed.

| Check | Statistic | Pass | Reference |
|---|---|---|---|
| Goodness of fit | p = P(χ²_{ν/c} ≥ χ²/c), c = (1+ρ²)/(1−ρ²) for lag-1 pixel correlation ρ | p ≥ 0.01 | Bevington & Robinson 2003; Satterthwaite 1946 |
| Residual randomness | Wald–Wolfowitz runs test on residual signs | p ≥ 0.01 | Wald & Wolfowitz 1940 |
| Residual normality | Shapiro–Wilk on (F − M)/σ | p ≥ 0.01 | Shapiro & Wilk 1965 |
| Model-independent flux | direct integral of data − continuum − companions vs Gaussian over μ ± 3σ | \|z\| ≤ 2 | — |
| Detection | F/σ_F | ≥ 3 | — |
| Parameter bounds | σ, μ not at configured limits | inside | — |
| Resolved (info) | FWHM vs instrumental FWHM | > 5.18 Å | IUE NEWSIPS manual |

Over all 1,057 detected fits in the 255 spectra: 86 % of Si IV + O IV] fits verify; a single Gaussian fails the goodness-of-fit test for 65 % of Lyα and N V fits, and 37 % of C IV fits show systematic residual structure (runs test) — broad quasar lines are generally not Gaussian. The model-independent flux check passes for 98.8 % of fits, so the fluxes themselves are robust.

## 4. Graphs

Shown for the selected line in `/` and `/v2` (`static/js/fitplots.js`) and written for every line to a PNG per spectrum (`src/plotting.py`, linked as "Publication figure"):

1. **Rest-frame spectrum** — observed flux, power-law continuum, shaded continuum windows, fitted line centres.
2. **Line fit** — data ± 1σ pixel noise, local continuum, each Gaussian component, total model.
3. **Normalised residuals** (F − M)/σ with ±1σ, ±2σ lines.
4. **Residual distribution** — histogram vs. N(0, 1).

### Profile Diagnostic Viewer (`/v2`)

Header (detections, verified count, continuum α ± err, A ± err, Excel / PNG buttons) → Graph 1 and the line summary → Graphs 2–4 and the parameter table for the selected line → collapsible sections that show their key result while closed:

- **Verification** for the selected line (§3).
- **Graph 5 — variability context**: the spectrum is dated from `jd.xlsx` by its SWP number; the plot shows the selected line's detected fluxes from the same calendar year with this spectrum highlighted, and a table compares every line with that year's F̄ and F_var.
- **Graph 6 — full-spectrum decomposition**: the spectrum with each selected line's Gaussian in its own colour (Okabe & Ito palette), plus fit and light-curve panels per line. A line selector (e.g. only Lyα and Si IV) controls what is shown and downloaded.
- **Campaign reference**: F_var per line per year for all dated spectra, with the campaign workbook download.

Every graph downloads as PNG (2×) or SVG.

### Excel outputs

| File | Sheets |
|---|---|
| Per spectrum (viewer → *Export to Excel*) | README · Lines (reference column names) · Continuum · Verification (one row per check) · Variability context |
| `outputs/variability_results.xlsx` (`/download/variability_results.xlsx`) | README (formulas, units, sources) · Fvar by year · Light curves (JD, JD − JD_min, F, F_err) · Line fits · Spectra |

## 5. Variability (F_var)

`src/variability.py`, per emission line and per calendar year, using detected fluxes only:

```
F̄ = (1/N) Σ F_i
σ²_F = 1/(N−1) Σ (F_i − F̄)²
ΔF̄² = (1/N) Σ ΔF_i²
F_var = √(σ²_F − ΔF̄²) / F̄

err(F_var) = √(1/2N) · ΔF̄² / (F̄² F_var)     if σ²_F/ΔF̄² < 10   (S² ≈ σ²_err)
           = √(ΔF̄²/N) / F̄                    if σ²_F/ΔF̄² ≥ 10   (S² ≫ σ²_err)
```

(Vaughan et al. 2003; Sukanya et al. 2018). If σ²_F ≤ ΔF̄² the scatter is consistent with the errors and the year is reported as "not variable"; years with N < 2 are skipped.

## 6. Usage

```bash
pip install -r requirements.txt              # web app
pip install -r requirements-analysis.txt     # batch tools (pandas, openpyxl)
python app.py                                 # http://localhost:5000
```

**Batch (all spectra, all lines):**

```bash
python run_variability.py --workers 6         # fit everything (≈15 min)
python run_variability.py --reuse             # recompute F_var from outputs/line_fluxes.csv
```

Writes `outputs/line_fluxes.csv` (every fit, with verification status), `outputs/light_curves.csv` (JD − JD_min, F, F_err), `outputs/fvar_by_year.csv` and `outputs/variability_results.xlsx`.

**Python:**

```python
from src.pipeline import run_single_spectrum_pipeline
rec = run_single_spectrum_pipeline("spectrum/70dcdr2dswp35476mxlo.txt")
lya = next(l for l in rec["lines"] if l["rest_wavelength"] == 1216.0)
print(lya["flux"], lya["flux_err"], lya["verification"]["status"])
```

## 7. Configuration

`config/lines.yaml` lists the lines; each has a YAML file (`config/Lya.yaml`, `config/LyB.yaml`, …):

```yaml
line_name: "Lyman Alpha (Lya)"
rest_wavelength: 1216.0
center:      {range: 5.0, step: 0.25}            # seed grid around the observed peak
sigma:       {min: 4.25, max: 10.19, step: 0.2,   # FWHM 10–24 Å; σ continuous in the fit
              prior_center: 6.3, prior_std: 1.5}
wing_window: {min: 10, max: 24, step: 1, prior_center: 18.0, prior_std: 2.5}
peak_search_radius: 15.0
refine:
  window: [1180, 1265]                             # least-squares window (Å)
  center_tol: 8.0
  companions:                                      # fitted together with this line
    - {rest: 1240.0, sigma_min: 2.2, sigma_max: 12.0, sigma_seed: 6.0}
mask_ranges: []                                    # e.g. [[1043, 1056]] for airglow (Lyβ)
snr_min: 3.0                                       # detection: F/σ_F ≥ 3
chi2_red_max: 5.0                                  # warning only
```

## 8. Output schema

`run_single_spectrum_pipeline()` returns a dict with `wavelength`, `observed_flux`, `continuum_fit`, `subtracted_y`, `spectral_index` ± `spectral_index_err`, `continuum_amplitude` ± `continuum_amplitude_err`, `continuum_windows`, and `lines`. Each line has:

| Key | Meaning |
|---|---|
| `center`, `center_err`, `sigma`, `sigma_err`, `amplitude`, `amplitude_err` | least-squares parameters ± 1σ |
| `flux`, `flux_err`, `snr` | √(2π)Aσ, propagated error, F/σ_F |
| `flux_window` | Gaussian integrated over μ ± W/2 (reference convention) |
| `ew`, `ew_err` | F/F_c(μ) and error (Å) |
| `fwhm_ang`, `fwhm_kms`, `fwhm_kms_intrinsic` | observed and instrument-corrected widths |
| `reduced_chi2`, `dof`, `pixel_noise` | fit statistics |
| `wing_window`, `min_wavelength`, `max_wavelength` | reference-style window |
| `detected`, `fit_status`, `at_bound` | detection and warnings |
| `verification` | `{status, failed, checks[]}` (§3) |
| `plot` | arrays for the graphs: x, y, yerr, model, continuum, components, norm_resid |

## 9. Repository structure

```
app.py                  Flask app (/, /v2, /v3, /about, /fit, /api/fvar)
run_variability.py      batch fit of spectrum/ + F_var per line per year
analysis.py             campaign figures from the reference tables and F_var output
config.py               constants, continuum windows, instrumental FWHM
config/                 lines.yaml + one YAML per line
src/
  continuum.py          power law + standard errors
  candidate_engine.py   peak search and seed grid
  refine.py             least-squares fit, DER_SNR noise, verification, plot arrays
  line_fitter.py        seed → refine → quality per line
  quality.py            detection and warnings
  statistics.py         reference-style window statistics, local baseline
  variability.py        F_var with piecewise error, JD loader, per-spectrum context
  excel_export.py       organised campaign workbook
  plotting.py           publication PNG per spectrum
  pipeline.py           orchestrator
static/js/fitplots.js   browser graphs and verification table
templates/              index.html (V1), index_v2.html (V2), spectrum-analysis.html (V3), about.html
spectrum/, jd.xlsx      input data
outputs/                results (fvar_by_year.csv etc. are versioned for the web app)
tests/                  pytest suite
```

## 10. Web application routes

| Method | Route | Description |
|---|---|---|
| GET | `/` | V1 single-spectrum fitter |
| GET | `/v2` | V2 Profile Diagnostic Viewer (F_var table, per-spectrum fit, Excel export) |
| GET | `/v3` | V3 multi-epoch viewer (reference tables) |
| GET | `/about` | Method documentation |
| POST | `/fit` | Fit one spectrum, full JSON |
| POST | `/upload_fit`, GET `/stream/<id>` | Same, with streamed log |
| GET | `/plots/<file>` | Publication PNG |
| GET | `/api/fvar` | F_var per line per year |
| GET | `/api/fvar_context?spectrum=<file>` | Date, same-year light curve and F_var for one spectrum |
| GET | `/download/variability_results.xlsx` | Campaign workbook |

## 11. Deployment

`vercel.json` routes all requests to `app.py`. The web app needs only `requirements.txt`; `outputs/fvar_by_year.csv` must be committed for `/api/fvar`.

## 12. References

- Bevington, P. R. & Robinson, D. K. 2003, *Data Reduction and Error Analysis for the Physical Sciences*, 3rd ed., McGraw-Hill.
- Edelson, R. et al. 2002, ApJ, 568, 610.
- Nichols, J. S. & Linsky, J. L. 1996, AJ, 111, 517 (IUE NEWSIPS).
- Paltani, S. & Türler, M. 2005, A&A, 435, 811 — *The mass of the black hole in 3C 273*.
- Peterson, B. M. 1993, PASP, 105, 247 — *Reverberation mapping of active galactic nuclei*.
- Peterson, B. M. et al. 2004, ApJ, 613, 682.
- Press, W. H. et al. 2007, *Numerical Recipes*, 3rd ed., Cambridge University Press.
- Rodríguez-Pascual, P. M. et al. 1997, ApJS, 110, 9 — introduces F_var.
- Satterthwaite, F. E. 1946, Biometrics Bulletin, 2, 110.
- Shapiro, S. S. & Wilk, M. B. 1965, Biometrika, 52, 591.
- Stoehr, F. et al. 2008, ASP Conf. Ser., 394, 505 — DER_SNR.
- Sukanya et al. 2018 — piecewise F_var error (as provided by the project team).
- Türler, M. et al. 1999, A&AS, 134, 89 — *30 years of multi-wavelength observations of 3C 273*.
- Ulrich, M.-H. et al. 1980, MNRAS, 192, 561 — *Detailed ultraviolet observations of the quasar 3C 273 with IUE*.
- Ulrich, M.-H., Maraschi, L. & Urry, C. M. 1997, ARA&A, 35, 445 — *Variability of active galactic nuclei*.
- Vaughan, S., Edelson, R., Warwick, R. S. & Uttley, P. 2003, MNRAS, 345, 1271.
- Wald, A. & Wolfowitz, J. 1940, Ann. Math. Stat., 11, 147.
