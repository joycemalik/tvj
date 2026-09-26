# 3C273 Lyman Alpha Spectral Pipeline

**An automated, physics-grounded spectroscopic fitting pipeline for 215 archival IUE SWP ultraviolet observations of the quasar 3C273, targeting the Lyman Alpha emission line at λ1216 Å.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![Deploy](https://img.shields.io/badge/Deploy-Vercel-black?style=flat-square&logo=vercel)](https://vercel.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Overview

This repository contains a complete end-to-end pipeline for automated emission-line fitting in archival UV spectra. The pipeline ingests raw IUE SWP two-column text files, fits a power-law continuum, searches a discrete grid of 5,600 Gaussian candidate models, refines the best candidates with non-linear least squares, and produces quality-controlled measurements of flux, equivalent width (EW), FWHM, signal-to-noise ratio (SNR), and reduced chi-squared for every observation.

A Flask web application provides three interactive frontends — a real-time fit workbench (V1), an academic diagnostic studio (V2), and a campaign time-series viewer (V3) — alongside a scientific documentation page (About) that explains every formula in the pipeline with full citations.

The target object is **quasar 3C273** (z ≈ 0.158), observed with the International Ultraviolet Explorer (**IUE**) Short Wavelength Prime (SWP) camera across 215 archival epochs. The Lyman Alpha line (rest λ1216 Å, observed ≈1409 Å) is the primary fitting target, though the pipeline supports any UV emission line configurable via YAML.

---

## Table of Contents

1. [Scientific Background](#1-scientific-background)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Installation](#4-installation)
5. [Usage](#5-usage)
   - [Web Application](#web-application)
   - [Python API](#python-api)
   - [Batch Campaign](#batch-campaign)
6. [Configuration](#6-configuration)
7. [Pipeline Stages — Technical Detail](#7-pipeline-stages--technical-detail)
   - [Stage 1 — Data Ingestion](#stage-1--data-ingestion)
   - [Stage 2 — Continuum Modeling](#stage-2--continuum-modeling)
   - [Stage 3 — Peak Detection](#stage-3--peak-detection)
   - [Stage 4A — Grid Search](#stage-4a--grid-search)
   - [Stage 4B — Amplitude Refinement](#stage-4b--amplitude-refinement)
   - [Stage 5 — Physical Statistics](#stage-5--physical-statistics)
   - [Stage 6 — Quality Evaluation](#stage-6--quality-evaluation)
8. [Output Schema](#8-output-schema)
9. [Web Application Routes](#9-web-application-routes)
10. [Deployment (Vercel)](#10-deployment-vercel)
11. [Results — 3C273 Campaign](#11-results--3c273-campaign)
12. [References](#12-references)

---

## 1. Scientific Background

### The Object: Quasar 3C273

3C273 is one of the brightest and closest quasars (z = 0.1583, d_L ≈ 749 Mpc), making it an ideal laboratory for long-baseline UV variability studies. Its broad Lyman Alpha emission line — produced by hydrogen recombination in the broad-line region (BLR) — encodes information about the AGN ionising luminosity, BLR geometry, and variability timescales.

### The Instrument: IUE SWP

The IUE SWP camera observed from 1150–2000 Å (rest frame) with a dispersion of ≈1.67 Å px⁻¹ in low-resolution mode. The 215 observations used here span the full IUE mission lifetime (1978–1996), providing an 18-year baseline for variability analysis.

### The Emission Line: Lyman Alpha λ1216 Å

Lyman Alpha is the strongest UV emission line in AGN spectra, arising from the 2p→1s transition of hydrogen. In 3C273, it is redshifted to ≈1409 Å (observed frame), well within the IUE SWP bandpass. Its width (FWHM ≈ 5,000–15,000 km/s) and flux are sensitive tracers of BLR physics and ionising continuum variability.

---

## 2. Pipeline Architecture

```
IUE SWP .txt spectrum
        │
        ▼
┌─────────────────────┐
│   1. Data Ingestion │  src/reader.py
│   2-col ASCII → np  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. Power-law        │  src/continuum.py
│     Continuum Fit    │  OLS in log-log space
│   F_c = A · λ^α     │  continuum windows: YAML
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. Peak Detection   │  src/candidate_engine.py
│   Locate emission    │
│   peak in ±15 Å      │
│   search window      │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│  4A. Discrete Grid Search   (5,600 models)   │  src/candidate_engine.py
│   σ ∈ [2.5, 6.5] Å  ×  W ∈ [14, 25] Å      │
│   Amplitude: analytic Stage-B weighted LS    │
│   Score = 0.35·s_χ² + 0.25·s_peak +         │
│           0.20·s_left + 0.10·s_right +       │
│           0.10·s_SNR                         │
└──────────┬───────────────────────────────────┘
           │
           ▼
┌─────────────────────┐
│  4B. Scipy Refine    │  src/optimizer.py
│   least_squares on   │
│   top-N candidates   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5. Physical Stats   │  src/statistics.py
│   Flux, EW, FWHM,   │
│   SNR, χ²_red,       │
│   flux_err, ew_err   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  6. Quality Gates    │  src/quality.py
│   ACCEPTED /         │
│   REJECTED           │
└──────────┬──────────┘
           │
           ▼
     Result record dict
     (JSON-serialisable)
```

---

## 3. Repository Structure

```
tvj/
├── app.py                    # Flask web application (routes, SSE streaming)
├── config.py                 # Continuum window definitions
├── requirements.txt          # Python dependencies
├── vercel.json               # Vercel serverless deployment config
│
├── config/                   # Per-line YAML fitting configurations
│   ├── lines.yaml            # Master line catalog (which lines to fit)
│   ├── Lya.yaml              # Lyman Alpha λ1216 Å  ← primary target
│   ├── NV.yaml               # N V λ1240 Å
│   ├── CII.yaml              # C II λ1335 Å
│   ├── SiIV.yaml             # Si IV λ1397 Å
│   ├── CIV.yaml              # C IV λ1549 Å
│   ├── HeII.yaml             # He II λ1640 Å
│   ├── OIII.yaml             # O III] λ1663 Å
│   ├── AlIII.yaml            # Al III λ1857 Å
│   └── CIII.yaml             # C III] λ1909 Å
│
├── src/                      # Core pipeline modules
│   ├── pipeline.py           # End-to-end orchestrator
│   ├── reader.py             # ASCII spectrum loader
│   ├── continuum.py          # Power-law continuum fitter
│   ├── candidate_engine.py   # 5,600-point grid search + Stage-B amplitude
│   ├── optimizer.py          # Scipy non-linear refinement
│   ├── line_fitter.py        # Single-line fit coordinator
│   ├── joint_fitter.py       # Overlapping-line joint fitting
│   ├── statistics.py         # Flux, EW, FWHM, SNR, χ², noise estimation
│   ├── quality.py            # Quality scoring and gates
│   ├── gaussian.py           # Gaussian model functions
│   ├── plotting.py           # Matplotlib publication plots
│   ├── report.py             # Text report generator
│   ├── validation.py         # Input validation utilities
│   └── batch.py              # Campaign batch runner
│
├── templates/                # Jinja2 HTML templates
│   ├── index.html            # V1 — Spectrum Fit Workbench (dark Nolan theme)
│   ├── index_v2.html         # V2 — Diagnostic Studio (academic off-white)
│   ├── spectrum-analysis.html# V3 — Campaign Viewer (time series, Plotly)
│   └── about.html            # About — full scientific documentation
│
├── tests/                    # Unit and regression test suite
│   └── *.py
│
├── outputs/                  # Generated plots (git-ignored)
├── uploads/                  # Temporary upload directory (git-ignored)
└── analysis.py               # Standalone campaign analysis script
```

---

## 4. Installation

### Requirements

- Python 3.10+
- pip

### Local setup

```bash
git clone https://github.com/joycemalik/tvj.git
cd tvj
pip install -r requirements.txt
python app.py
```

The server starts at `http://localhost:5000`.

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| Flask | 3.0.3 | Web framework |
| Werkzeug | 3.0.3 | WSGI utilities |
| numpy | 1.26.4 | Array operations |
| scipy | 1.13.1 | Non-linear least squares |
| PyYAML | 6.0.2 | Line configuration files |
| matplotlib | 3.9.2 | Publication-quality plots |
| requests | 2.32.3 | HTTP utilities |

---

## 5. Usage

### Web Application

Navigate to `http://localhost:5000` after starting the server. Four pages are available:

| Route | Page | Description |
|---|---|---|
| `/` | V1 — Workbench | Upload a spectrum, run pipeline, inspect full diagnostics in real time |
| `/v2` | V2 — Studio | Academic studio with SSE log streaming and campaign summary stats |
| `/v3` | V3 — Campaign | Client-side time-series viewer for the full 215-spectrum campaign |
| `/about` | About | Scientific documentation of every pipeline stage with full formulas |

**To run a fit on V1:**
1. Drag-and-drop any IUE SWP `.txt` file onto the upload zone, or click **LOAD SPECTRUM 70 DEMO** to use a built-in archival observation.
2. The pipeline runs server-side; results render automatically into four diagnostic graphs, a parameter table, and a QC checklist.

### Python API

```python
from src.pipeline import run_single_spectrum_pipeline

record = run_single_spectrum_pipeline("path/to/spectrum.txt")

# Access results
print(record['lines'][0]['flux'])       # Integrated Lyα flux
print(record['lines'][0]['fwhm_kms'])   # FWHM in km/s
print(record['lines'][0]['snr'])        # Detection SNR
print(record['global_reduced_chi2'])    # Global fit quality
```

The returned `record` is a plain Python dict — fully JSON-serialisable. See [Output Schema](#8-output-schema) for all keys.

### Batch Campaign

```python
from src.batch import run_campaign

results = run_campaign(
    spectrum_dir="path/to/spectra/",
    output_csv="campaign_results.csv"
)
```

Or from the command line:

```bash
python run_pipeline.py path/to/spectrum.txt
```

---

## 6. Configuration

Each emission line is configured via a YAML file in `config/`. The Lyman Alpha configuration (`config/Lya.yaml`) is the primary example:

```yaml
line_name: "Lyman Alpha (Lya)"
rest_wavelength: 1216.0       # Å

sigma:
  min: 2.5                    # Hard lower bound (Å)
  max: 6.5                    # Hard upper bound (Å)
  step: 0.1                   # Grid step (Å)
  prior_center: 5.0           # Gaussian prior mean (Å)
  prior_std: 1.0              # Gaussian prior width (Å)

wing_window:
  grid: [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]  # Full widths (Å)
  prior_center: 19.0          # Å
  prior_std: 2.0              # Å

center:
  range: 5.0                  # Search ±5 Å around peak
  step: 0.1                   # Grid step (Å)

peak_search_radius: 15.0      # Å around rest wavelength
snr_min: 5.0
snr_max: 15.0
chi2_red_max: 5.0
```

**Wing window convention:** `wing_window` is the **full** integration width *W*. The integration limits are:

```
λ_min = μ − W/2
λ_max = μ + W/2
```

To add a new emission line, create a YAML file in `config/` following this template and register it in `config/lines.yaml`.

---

## 7. Pipeline Stages — Technical Detail

### Stage 1 — Data Ingestion

`src/reader.py` reads any two-column whitespace-separated ASCII file:

```
wavelength_1   flux_1
wavelength_2   flux_2
...
```

Comment lines (`#`) and blank lines are skipped. The spectrum is returned as two `numpy` arrays. No units are assumed — the pipeline operates in whatever units the input file carries (IUE flux is in erg s⁻¹ cm⁻² Å⁻¹).

### Stage 2 — Continuum Modeling

`src/continuum.py` fits a **power-law continuum**:

```
F_c(λ) = A · λ^α
```

by ordinary least squares in log–log space:

```
ln F = ln A + α · ln λ
```

Fitting is restricted to **line-free continuum windows** defined in `config.py`, avoiding regions contaminated by emission lines. The power-law index α and amplitude A are the fit outputs. The continuum is then evaluated at every pixel and subtracted to produce the continuum-subtracted spectrum used for all subsequent fitting.

### Stage 3 — Peak Detection

Within a configurable search radius (default ±15 Å around the rest wavelength), the pipeline locates the wavelength of peak positive flux in the continuum-subtracted spectrum. This observed peak wavelength anchors the subsequent center-grid search.

### Stage 4A — Grid Search

`src/candidate_engine.py` evaluates a discrete grid of Gaussian models parameterised by (σ, W):

| Parameter | Range | Step | Grid points |
|---|---|---|---|
| σ (Gaussian width) | 2.5 – 6.5 Å | 0.1 Å | 40 |
| W (wing window, full) | 14 – 25 Å | 1 Å | 12 |
| μ (line center) | ±5 Å around peak | 0.1 Å | ~100 |

The amplitude at each (σ, μ, W) is computed analytically via **Stage-B profile-weighted amplitude refinement**:

```
Â = Σᵢ wᵢ · gᵢ · Fₛᵤb,ᵢ  /  Σᵢ wᵢ · gᵢ²
```

where `gᵢ = G(λᵢ; μ, σ, 1)` is the unit-amplitude Gaussian, `Fₛᵤb,ᵢ` is the continuum-subtracted flux, and the profile-weights `wᵢ = 1 − gᵢ` downweight the line core to reduce sensitivity to noise at the peak.

For each model, a **composite score** combines five sub-scores:

```
S = 0.35 · s_χ² + 0.25 · s_peak + 0.20 · s_left + 0.10 · s_right + 0.10 · s_SNR
```

where:
- **s_χ²** — goodness of fit within the wing window
- **s_peak** — proximity of model peak to observed emission peak
- **s_left / s_right** — flux coverage in the blue / red wing halves
- **s_SNR** — detection significance

The top-N candidates (default N=20) by composite score are passed to Stage 4B.

### Stage 4B — Amplitude Refinement

`src/optimizer.py` runs `scipy.optimize.least_squares` on each top-N candidate, refining the center and amplitude while holding σ and W at the Stage-4A values. The candidate with the lowest final χ² is selected as the best fit.

### Stage 5 — Physical Statistics

`src/statistics.py` computes all reported quantities from the accepted fit:

**Noise estimation (MAD):**
```
σ_noise = 1.4826 × median(|rᵢ − median(r)|)
```
where `rᵢ = Fᵢ − F_model,ᵢ` are residuals in the continuum region.

**Integrated line flux:**
```
F = Â · σ · √(2π)
```

**Flux uncertainty:**
```
σ_F = √n · σ_noise · δ_res · F̄_c
```
where `n` is the number of pixels in the wing window, `δ_res = 1.676 Å px⁻¹` is the IUE SWP dispersion, and `F̄_c` is the mean continuum in the window.

**Signal-to-noise ratio:**
```
SNR = F / σ_F
```

**Equivalent width:**
```
EW = F / F̄_c
```

**FWHM:**
```
FWHM_Å  = 2.355 · σ
FWHM_kms = FWHM_Å · c / μ     (c = 2.998 × 10⁵ km/s)
```

**Reduced chi-squared:**
```
χ²_red = (1/ν) · Σ [(Fᵢ − F_model,ᵢ)² / σ_noise²]
```
where `ν = n_pixels − n_free_params` (degrees of freedom).

**EW uncertainty:**
```
σ_EW = σ_F / F̄_c
```

### Stage 6 — Quality Evaluation

`src/quality.py` applies threshold gates and assigns a quality score:

| Gate | Condition | Action |
|---|---|---|
| SNR minimum | SNR ≥ `snr_min` (5.0) | FAIL → REJECTED |
| SNR maximum | SNR ≤ `snr_max` (15.0) | WARNING if exceeded |
| χ²_red | χ²_red ≤ `chi2_red_max` (5.0) | FAIL → REJECTED |
| Composite score | S ≥ 0.3 | FAIL → REJECTED |

A spectrum passes all gates → status `ACCEPTED`. Any failed gate → status `REJECTED: <reason>`.

---

## 8. Output Schema

`run_single_spectrum_pipeline()` returns a dict with the following keys:

```python
{
    # Spectrum metadata
    "spectrum_name":        str,     # filename
    "global_reduced_chi2":  float,   # χ²_red across all fitted lines
    "joint_fit_applied":    bool,    # True if overlapping-line joint fit ran

    # Arrays (list[float] when JSON-serialised)
    "wavelength":           list,    # Å
    "observed_flux":        list,    # input flux
    "continuum_fit":        list,    # power-law continuum
    "subtracted_y":         list,    # continuum-subtracted flux

    # Continuum parameters
    "continuum_amplitude":  float,   # A in F_c = A·λ^α
    "spectral_index":       float,   # α

    # Per-line results (one dict per fitted line)
    "lines": [{
        "line_name":         str,
        "rest_wavelength":   float,  # Å
        "detected":          bool,
        "fit_status":        str,    # "ACCEPTED" | "REJECTED: ..."
        "amplitude":         float,  # Gaussian amplitude
        "center":            float,  # μ (Å)
        "sigma":             float,  # σ (Å)
        "wing_window":       float,  # W full width (Å)
        "min_wavelength":    float,  # μ − W/2 (Å)
        "max_wavelength":    float,  # μ + W/2 (Å)
        "flux":              float,  # integrated flux
        "flux_err":          float,  # 1σ flux uncertainty
        "snr":               float,  # detection SNR
        "ew":                float,  # equivalent width (Å)
        "ew_err":            float,  # EW uncertainty (Å)
        "fwhm_ang":          float,  # FWHM (Å)
        "fwhm_kms":          float,  # FWHM (km/s)
        "reduced_chi2":      float,  # χ²_red for this line
        "composite_score":   float,  # Stage-4A composite score
        "score_components":  dict,   # s_χ², s_peak, s_left, s_right, s_SNR
        "quality_score":     float,  # quality gate aggregate score
        "rank":              int,    # candidate rank (1 = best)
        "observed_peak_wl":  float,  # observed peak wavelength (Å)
        "shortlist":         list,   # top-5 candidates
    }]
}
```

---

## 9. Web Application Routes

| Method | Route | Description |
|---|---|---|
| GET | `/` | V1 Workbench |
| GET | `/v2` | V2 Diagnostic Studio |
| GET | `/v3` | V3 Campaign Viewer |
| GET | `/about` | Scientific documentation |
| POST | `/upload_fit` | Upload file, returns `{task_id}` |
| GET | `/stream/<task_id>` | SSE stream: pipeline logs + result JSON |
| POST | `/fit` | Synchronous fit endpoint (returns full JSON) |
| GET | `/plots/<filename>` | Serve generated plot images |
| GET | `/api/analysis_summary` | Campaign-level statistics (Fvar, Rmax, means) |

---

## 10. Deployment (Vercel)

The app is configured for Vercel serverless deployment via `vercel.json`. All file I/O (uploads, plots) is redirected to `/tmp` on serverless environments.

```bash
npm i -g vercel
vercel --prod
```

On serverless, the `/stream/<task_id>` SSE endpoint degrades gracefully; the synchronous `/fit` endpoint remains fully functional.

---

## 11. Results — 3C273 Campaign

Across the full 215-spectrum IUE SWP campaign, the pipeline measures the following Lyman Alpha variability statistics:

| Statistic | Value |
|---|---|
| Spectra processed | 215 |
| Mean flux F̄ | ~ 8 × 10⁻¹³ erg s⁻¹ cm⁻² Å⁻¹ |
| Mean SNR | > 10 |
| Fractional variability Fvar | Computed per campaign run |
| Max-to-min flux ratio Rmax | Computed per campaign run |

**Variability statistics:**

```
Fvar = √(S² − ⟨σ²_err⟩) / F̄

Rmax = F_max / F_min
```

where `S²` is the sample variance of the flux measurements and `⟨σ²_err⟩` is the mean squared measurement uncertainty (Vaughan et al. 2003).

---

## 12. References

1. Kinney, A. L. et al. (1991). *An atlas of ultraviolet spectra of star-forming galaxies.* ApJS, 75, 645.
2. Bahcall, J. N. et al. (1991). *Hubble Space Telescope spectroscopy of the broad emission lines of 3C 273.* ApJ, 377, L5.
3. Ulrich, M.-H., Maraschi, L., & Urry, C. M. (1997). *Variability of active galactic nuclei.* ARA&A, 35, 445.
4. Cristiani, S., & Vio, R. (1990). *The emission-line variability of 3C 273.* A&A, 227, 385.
5. Vaughan, S. et al. (2003). *On characterizing the variability properties of X-ray light curves from active galactic nuclei.* MNRAS, 345, 1271.
6. Bevington, P. R., & Robinson, D. K. (2003). *Data Reduction and Error Analysis for the Physical Sciences.* McGraw-Hill. (χ² statistics)
7. Press, W. H. et al. (2007). *Numerical Recipes: The Art of Scientific Computing.* Cambridge University Press. (OLS, least squares)
8. Pei, Y. C. (1992). *Interstellar dust from the Milky Way.* ApJ, 395, 130.
9. Peterson, B. M. (1993). *Reverberation mapping of active galactic nuclei.* PASP, 105, 247.
10. Bowen, D. V. et al. (1994). *IUE observations of 3C 273.* ApJ, 421, 87.
11. Wamsteker, W. et al. (1990). *Variability in the UV continuum of 3C 273.* ApJ, 354, 446.
12. Turler, M. et al. (1999). *Modeling the long-term radio-to-gamma-ray light curves of 3C 273.* A&AS, 134, 89.
13. Maoz, D. et al. (1994). *A Hubble Space Telescope UV survey of broad-line AGN.* ApJ Suppl, 91, 1.
14. Rodriguez-Pascual, P. M. et al. (1997). *IUE Atlas of Cataclysmic Variables.* A&AS, 121, 495.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

IUE archive data courtesy of **NASA/ESA MAST** (Mikulski Archive for Space Telescopes).

---

*Developed as part of a research project on long-baseline UV variability of the quasar 3C273 using archival IUE spectroscopy.*
