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
IUE SWP rest-frame spectrum (.txt)
   → Power-law continuum (log-log OLS, 5 line-free windows, with standard errors)
   → Per-spectrum peak + coarse (μ, σ, W) seed grid
   → Weighted least-squares fit: Gaussian(s) + local linear continuum (Lyα + N V jointly)
   → Flux √(2π)Aσ ± covariance error, EW, FWHM (observed / intrinsic), χ²_red
   → Detection F/σ_F ≥ 3
   → Verification: χ² p-value (pixel-correlation corrected), runs test, Shapiro–Wilk,
     model-independent flux check
   → F_var per line per calendar year (Vaughan et al. 2003)
```

---

## Emission Lines Supported

| Line | Rest λ (Å) | Config |
|---|---|---|
| Lyman Beta + O VI | 1025.7 | `config/LyB.yaml` (airglow 1043–1056 Å masked) |
| Lyman Alpha | 1216.0 | `config/Lya.yaml` |
| N V | 1240.0 | `config/NV.yaml` |
| O I + Si II | 1305.0 | `config/OI.yaml` |
| C II | 1335.0 | `config/CII.yaml` |
| Si IV + O IV] | 1400.0 | `config/SiIV.yaml` |
| C IV | 1549.0 | `config/CIV.yaml` |
| He II | 1640.0 | `config/HeII.yaml` |
| O III] | 1665.0 | `config/OIII.yaml` |

---

## Web Application

| Route | Description |
|---|---|
| `/` | V1 — Spectrum Fit Workbench |
| `/v2` | V2 — Diagnostic Studio |
| `/v3` | V3 — Campaign Viewer |
| `/about` | Scientific documentation |
| `/api/fvar` | F_var per emission line per calendar year (`python run_variability.py`) |

---

## Data

IUE SWP archival observations of quasar **3C273** (z = 0.1583).  
255 rest-frame spectra · 994–1832 Å · dates from `jd.xlsx` (SWP log)  
Courtesy of **NASA/ESA MAST**.

---

*See the [README](https://github.com/joycemalik/tvj#readme) for full technical documentation.*
