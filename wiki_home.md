# 3C273 Lyman Alpha Spectral Pipeline — Wiki

Welcome to the project wiki. This page provides a navigational index to all documentation for the pipeline.

---

## Quick Start

```bash
git clone https://github.com/joycemalik/tvj.git
cd tvj
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

Upload any IUE SWP `.txt` spectrum (two columns: wavelength Å, flux) or click **LOAD SPECTRUM 70 DEMO** to see a real archival observation processed live.

---

## Pages

| Page | Contents |
|---|---|
| [[Pipeline Stages]] | Full technical detail of every stage: continuum, grid search, scoring, statistics |
| [[Configuration Reference]] | All YAML config keys for emission-line fitting |
| [[Output Schema]] | Every key returned by `run_single_spectrum_pipeline()` |
| [[Web Application]] | Route reference, SSE streaming, API endpoints |
| [[Deployment]] | Vercel serverless setup, environment variables |
| [[Campaign Results]] | 3C273 variability statistics: Fvar, Rmax |
| [[Scientific Background]] | IUE SWP instrument, 3C273, Lyman Alpha physics |
| [[References]] | Full bibliography |

---

## Architecture at a Glance

```
IUE SWP spectrum (.txt)
   → Power-law continuum (OLS log-log)
   → 5,600-candidate Gaussian grid (σ × wing window)
   → Stage-B amplitude refinement (analytic weighted LS)
   → Composite scoring (χ², peak, wings, SNR)
   → Scipy least-squares refinement
   → Physical statistics (flux, EW, FWHM, SNR, χ²_red)
   → Quality gates → ACCEPTED / REJECTED
```

Composite score formula:

```
S = 0.35·s_χ² + 0.25·s_peak + 0.20·s_left + 0.10·s_right + 0.10·s_SNR
```

---

## Emission Lines Supported

| Line | Rest λ (Å) | Config |
|---|---|---|
| Lyman Alpha | 1216.0 | `config/Lya.yaml` |
| N V | 1240.0 | `config/NV.yaml` |
| C II | 1335.0 | `config/CII.yaml` |
| Si IV | 1397.0 | `config/SiIV.yaml` |
| C IV | 1549.0 | `config/CIV.yaml` |
| He II | 1640.0 | `config/HeII.yaml` |
| O III] | 1663.0 | `config/OIII.yaml` |
| Al III | 1857.0 | `config/AlIII.yaml` |
| C III] | 1909.0 | `config/CIII.yaml` |

---

## Web Application

| Route | Description |
|---|---|
| `/` | V1 — Spectrum Fit Workbench |
| `/v2` | V2 — Diagnostic Studio |
| `/v3` | V3 — Campaign Viewer |
| `/about` | Scientific documentation |
| `/api/analysis_summary` | Campaign Fvar, Rmax, mean flux/SNR/EW/FWHM |

---

## Data

IUE SWP archival observations of quasar **3C273** (z = 0.1583).  
215 spectra · 900–1830 Å · primary target: Lyα λ1216 Å  
Courtesy of **NASA/ESA MAST**.

---

*See the [README](https://github.com/joycemalik/tvj#readme) for full technical documentation.*
