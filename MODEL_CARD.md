# Model Card — pak_quality_estimation

## Overview

Daily surface PM2.5 estimation over Pakistan on a fixed 0.1° (~10 km) grid, produced
by a gradient-boosted decision tree (LightGBM) that fuses satellite aerosol
retrievals, atmospheric composition, chemistry, meteorology, and ground-station
observations.

- **Task:** regression of daily mean surface PM2.5 (µg m⁻³) per grid cell.
- **Domain:** Pakistan, 141 × 175 grid (24,675 cells).
- **Estimator:** LightGBM (gradient-boosted trees).
- **Intended use:** daily national air-quality estimation, episode analysis, and
  public communication. **Not** for neighbourhood-scale/roadside exposure or
  sub-daily peaks.

## Training data

- **Inputs:** TROPOMI (NO2, SO2, CO, HCHO, O3, AAI, ALH, CH4, cloud), GEOS-CF NH3,
  MODIS MAIAC AOD (470/550 nm), ERA5/ERA5-Land meteorology, and ground-station PM2.5
  history. 305 engineered features (see `README.md` → Feature Engineering).
- **Target:** quality-controlled daily mean station PM2.5.
- **Split (temporal):** train 2020–2023, development/validation 2024, test 2025.
  The split is strictly temporal to mirror deployment (fit on the past, predict the
  future).

## Production model (this repository)

| Item | Value |
|---|---|
| Trees | 12,000 |
| Max depth | 12 |
| Num leaves | 255 |
| Learning rate | 0.01 |
| Column sample | 0.5 |
| Regularization | L2 = 10.0, min split gain = 0.05 |
| Tuning | RandomSearch with rolling year-wise CV |

**Held-out 2025 (Jan–Jul, n = 15,261):** MAE 18.6, RMSE 28.4, R² 0.623, bias +1.6,
F1@150 0.528 µg m⁻³. Rolling-CV (2021–2024): MAE 27.7 ± 2.4. Strongest errors in
winter (MAE 30.6) when severe smog episodes occur.

## Relationship to the ACP paper

The manuscript *"Daily PM2.5 Estimation under Sparse Monitoring: A Support-Aware
Framework for Pakistan"*
studies a **support-aware temporal-dropout** training strategy: the full recent
monitor-history feature block is withheld for a random subset of training rows so a
single model operates both **with** and **without** local PM2.5 history. The paper's
reported held-out-2025 numbers are:

| Setting | MAE (µg m⁻³) |
|---|---|
| National-predictor backbone (no history) | 22.9 |
| Support-aware, local history available | 14.3 |
| Support-aware, no local history | 23.2 |
| Standard temporal, no local history (failure mode) | 32.4 |

F1@150 rises 0.729 → 0.838 with local support. These differ from the operational
production-model metrics above because they evaluate a different model configuration
(support-regime experiments) on the paper's evaluation protocol. The dropout rate is
`p = 0.2`, selected on the 2024 development split.

## Limitations

- Ground archive includes low-cost sensors → reported errors reflect both model error
  and observation uncertainty; best read at grid-cell/neighbourhood scale.
- Validation is constrained by monitoring-network geography; error in unmonitored
  regions cannot be measured directly.
- Satellite/reanalysis predictors do not observe surface PM2.5 directly; column-to-
  surface mapping depends on boundary-layer depth, humidity, vertical aerosol
  structure, and aerosol type.
- 0.1° daily means only — no roadside gradients, point sources, or diurnal cycles.

## Reproducibility

Configuration (split years, dropout rate, feature groups), training, and inference
code are in this repository. Trained weights are distributed with the archived release
(Zenodo DOI: TBD) or available on request. Ingestion requires NASA Earthdata, Google
Earth Engine, and Copernicus CDS credentials via environment variables.
