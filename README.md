# pak_quality_estimation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20824536.svg)](https://doi.org/10.5281/zenodo.20824536)

**Daily gridded PM2.5 estimation for Pakistan at 0.1° resolution using a LightGBM model trained on satellite, meteorological, and ground-station data.**

This repository is the public code release accompanying the manuscript:

> **Recent monitor history improves daily PM2.5 estimation but weakens national
> transfer under sparse monitoring: a support-aware model for Pakistan.**
> Rehan Ahmad, Mahad Naveed, Abid Omar (Pakistan Air Quality Initiative, PAQI).
> Submitted to *Atmospheric Chemistry and Physics* (ACP).

It is also the estimation engine behind [Hawanama](https://hawanama.com), PAQI's air
quality intelligence platform. The cloud-deployment/orchestration code has been
omitted from this public release; what remains is the scientific pipeline — data
extraction, feature engineering, model training, and inference.

> **Note on scope.** The performance figures below describe the **operational
> production model** (a 305-feature LightGBM). The ACP paper studies a **support-aware
> temporal-dropout** variant and reports its own held-out-2025 numbers (backbone MAE
> 22.9; support-aware 14.3 with local history / 23.2 without; F1@150 0.729→0.838).
> See `MODEL_CARD.md` for the relationship between the two.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Data Sources](#data-sources)
- [Feature Engineering](#feature-engineering)
- [Model](#model)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)
- [Data availability](#data-availability)

---

## How It Works

The system ingests satellite imagery, meteorological reanalysis, and ground-station observations, engineers 305 features on a regular grid covering Pakistan, and runs a LightGBM model to predict PM2.5 concentrations for every 0.1° cell (~10 km).

1. **Ingest** raw data from TROPOMI (Sentinel-5P), GEOS-CF, MODIS AOD, ERA5/ERA5-Land, and the ground-station network
2. **Build** per-date Parquet partitions in a grid feature store (MET, AOD, TROPOMI stages)
3. **Merge** stages into a master feature table — one row per grid cell, 305 columns
4. **Predict** PM2.5 with a tuned LightGBM model, masked to Pakistan's boundary
5. **Publish** outputs as Cloud Optimized GeoTIFF (COG), JSON, and CSV

---

## Architecture

```
    Sentinel-5P        MODIS MCD19A2       ERA5 / ERA5-Land       Ground Stations
    (TROPOMI)          (AOD @ 550nm)       (hourly reanalysis)    (PM2.5/PM10)
        │                   │                     │                      │
        ▼                   ▼                     ▼                      ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  TIER 1 — Raw Ingestion  (daily/monthly schedule)                       │
  └────────────────────────────────┬─────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  TIER 2 — Feature Store  (grid partitions per date per stage)           │
  └────────────────────────────────┬─────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  TIER 3 — Inference  (LightGBM prediction on merged feature table)      │
  │  141×175 grid (24,675 cells) → predict → GeoTIFF + Gaussian smooth      │
  └────────────────────────────────┬─────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  TIER 4 — Publishing  (COG / JSON / CSV per date)                       │
  └──────────────────────────────────────────────────────────────────────────┘
```

---

## Data Sources

| Source | Resolution | Cadence | Variables |
|--------|-----------|---------|-----------|
| **TROPOMI** (Sentinel-5P) | ~5.5 km | Daily | NO2, SO2, CO, HCHO, O3, AAI, ALH, CH4, Cloud |
| **GEOS-CF** (NASA) | ~25 km | Daily | NH3 |
| **MODIS AOD** (MCD19A2.061) | 1 km | Daily | Optical depth @ 470 nm, 550 nm |
| **ERA5 / ERA5-Land** | ~31 km / ~9 km | Monthly | T2m, d2m, wind, pressure, BLH, cloud cover, humidity |
| **Ground stations** | Point | Hourly | PM2.5, PM10 |

See `DATA.md` for access terms (public inputs vs. the restricted ground-station stream).

---

## Feature Engineering

**305 input features** across four categories:

### Meteorological (~100 features)

Core variables from ERA5/ERA5-Land (T2m, dewpoint, u/v wind at 10 m and 100 m, surface pressure, boundary layer height, cloud cover, relative humidity, specific humidity, VPD) plus extensive temporal engineering:

- **Lags:** 1-day and 3-day for wind speed, BLH
- **Rolling statistics:** 3-day and 7-day mean, std, min, max for wind, BLH, RH, VPD, ventilation coefficient
- **Derived stability:** stagnation flags, calm-wind counts (< 3 m/s, < 7 m/s), ventilation coefficient anomalies
- **Tendencies:** BLH, MSLP, surface pressure tendencies with 3d/7d rolling mean and std
- **Wind direction:** sin/cos encoding with 7d and 14d rolling means, directional variance
- **Calendar:** day-of-year sin/cos (3 harmonics), heating season and burning season flags

### TROPOMI Gas Columns (~160 features)

For each of 10 products (NO2, SO2, CO, HCHO, O3, AAI, ALH, CH4, Cloud, NH3):
- **Robust statistics:** median, mean, std, min, max, percentiles (p10/p25/p75/p90), IQR, MAD, range, CV
- **Spatial texture:** skewness, kurtosis
- **Multiscale gradients:** delta median and delta std between 0.5°, 1°, and 2° neighborhoods

### Aerosol Optical Depth (~6 features)

- Optical depth at 470 nm and 550 nm (QA-filtered)
- AOD uncertainty, QA cloud mask, QA adjacency, QA AOD flags

### Calendar (~8 features)

- Day-of-year (3 harmonics: sin/cos pairs)
- Heating season flag, burning season flag

---

## Model

### LightGBM — Production

A gradient-boosted decision tree trained on station-level observations (2020-2023), validated on 2024, tested on 2025.

| Detail | Value |
|--------|-------|
| Model type | LightGBM |
| Features | 305 |
| Trees (n_estimators) | 12,000 |
| Max depth | 12 |
| Num leaves | 255 |
| Learning rate | 0.01 |
| Subsample | 1.0 |
| Column sample | 0.5 |
| Regularization | L2=10.0, min split gain=0.05 |
| Tuning | RandomSearch with rolling year-wise CV |

**Top 10 features by importance (split count):**

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `blh_lag1d` | 4,608 |
| 2 | `sp` | 4,561 |
| 3 | `blh_lag3d` | 4,509 |
| 4 | `WS10_lag3d` | 4,106 |
| 5 | `WS10_lag1d` | 3,878 |
| 6 | `aai_skew` | 3,474 |
| 7 | `aai_delta_median_r1_r05` | 3,413 |
| 8 | `WD10_cos_rm_14d` | 3,205 |
| 9 | `WS10_rollmin_7d` | 3,180 |
| 10 | `aai_delta_std_r2_r05` | 3,164 |

---

## Performance

> These are **operational production-model** metrics. The ACP paper reports separate
> numbers for the support-aware temporal-dropout experiments (see `MODEL_CARD.md`).

### Test Set (2025, Jan–Jul, n = 15,261)

| Metric | Value |
|--------|-------|
| **MAE** | 18.6 µg/m³ |
| **RMSE** | 28.4 µg/m³ |
| **R²** | 0.623 |
| **Bias** | +1.6 µg/m³ |
| **F1 @ 150 µg/m³** | 0.528 |

### Cross-Validation (rolling year-wise, 2021–2024)

| Metric | Mean | Std |
|--------|------|-----|
| MAE | 27.7 µg/m³ | 2.4 |
| RMSE | 44.3 µg/m³ | 2.9 |
| R² | 0.658 | 0.023 |
| F1 @ 150 | 0.691 | 0.008 |

### Seasonal Breakdown (Test)

| Season | MAE | RMSE | R² |
|--------|-----|------|-----|
| Winter (Nov–Feb) | 30.6 | 44.1 | 0.446 |
| Pre-monsoon (Mar–May) | 14.5 | 19.3 | 0.417 |
| Monsoon (Jun–Sep) | 11.6 | 14.7 | 0.249 |

---

## Project Structure

```
├── feature_engineering/            Feature computation & data lake
│   ├── feature_family/             Per-modality feature modules (MET, AOD, TROPOMI)
│   ├── feature_store/              Parquet store builders (station, grid, master)
│   └── main_feature_pipeline.py    Orchestrator
│
├── inference/                      Inference engine
│   └── feature_store/              Grid reader & main inference script
│       └── run_grid_inference_from_store.py
│
├── training/                       Model development
│   ├── main.py                     Benchmark trainer (tuning + rolling CV)
│   ├── baselines/                  Climatology, persistence, GEOS-CF baselines
│   ├── results/                    Tuning & evaluation metrics (JSON)
│   └── utils/                      Metrics, evaluation, dataset prep, trainer
│
├── extraction_and_preprocessing/   Raw data scripts & station labels
├── shared_features/                Shared feature definitions
├── scripts/                        Data-lake / sync helper scripts
├── requirements.txt                Dependencies
├── MODEL_CARD.md                   Model card (+ relationship to the ACP paper)
├── DATA.md                         Data sources and availability
├── CITATION.cff                    How to cite
└── LICENSE                         MIT
```

---

## Usage

Trained model weights are **not** committed (see `.gitignore`); they are distributed
with the archived release (Zenodo DOI [10.5281/zenodo.20824536](https://doi.org/10.5281/zenodo.20824536); see `ZENODO.md`) or available on request.
The daily gridded 2024–2025 PM2.5 product is included under
`data/gridded_predictions_2024_2025/`, and the held-out 2025 validation table under
`validation/`.

### Run inference for one date

```bash
python -m inference.feature_store.run_grid_inference_from_store \
  --date 2026-01-15 \
  --model <path/to/model.joblib> \
  --format all \
  --output_dir output/
```

### Train

```bash
# With hyperparameter tuning
python -m training.main --do_tuning --tuning_trials 25

# Without tuning
python -m training.main --no_tuning --dev_year 2024 --test_year 2025
```

Ingestion of satellite/reanalysis inputs requires NASA Earthdata, Google Earth Engine,
and Copernicus CDS credentials, supplied through environment variables (no secrets are
stored in this repository).

### Configuration

Object-store locations and credentials are read from environment variables; code
defaults are neutral placeholders, so set these to your own infrastructure. Copy
`.env.example` to `.env` and fill in the values:

| Variable | Purpose |
|---|---|
| `GCP_PROJECT` | Google Cloud project id |
| `PAQI_RAW_BUCKET` | raw satellite/station data bucket (e.g. `gs://your-raw-bucket`) |
| `PAQI_DERIVED_BUCKET` | feature-store bucket |
| `PAQI_LAKE_PATH` | training data-lake path |
| `PAQI_GEOSCF_PATTERN` | GEOS-CF baseline raster path pattern |
| `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` | NASA Earthdata (MODIS AOD) |
| `CDS_API_KEY` | Copernicus CDS (ERA5) |

---

## Citation

If you use this software, please cite the article (see `CITATION.cff`).

## License

Code is released under the MIT License (`LICENSE`). Derived data products are released
under CC-BY-4.0 (see `DATA.md`).

## Data availability

See `DATA.md`. Public inputs are available from their providers (NASA Earthdata,
Copernicus C3S, NASA GMAO). The ground-station PM2.5 stream is available from PAQI on
reasonable request; derived, quality-controlled tables and gridded predictions are
released as a versioned data package.
