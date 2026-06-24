# Model artifacts

This file documents the fitted model artifacts behind the accompanying manuscript — what is
released, what is withheld and why, and the identifiers needed to verify the exact models used.

## What is released here

- The full **model code** and training/inference pipeline.
- The **train/development/test configuration** and **hyperparameters** (`BEST_PARAMS` in
  `training/utils/spatial_blocks.py`; split: train 2020–2023, dev 2024, test 2025; temporal
  dropout rate 0.2; Huber loss on `log1p` targets).
- The **feature registry** (`training/data/features_registry.csv`).
- The **held-out 2025 validation predictions** (`validation/pm25_2025_station_day_validation.csv`):
  observed daily PM2.5 with the support-aware predictions made **with** and **without** the local
  monitor-history block — sufficient to reproduce the paper's headline metrics.
- The **daily gridded PM2.5 outputs** for 2024–2025 (`data/gridded_predictions_2024_2025/`).
- The **model card** (`MODEL_CARD.md`) and figure-generation scripts (`paper_figures/`).

## What is withheld, and why

The **fitted model artifacts** (the serialized LightGBM models) are **not** included in the
public archive. They are trained on a partly **non-public contributor sensor stream** and are
also used in an **operational PAQI system**, so releasing the fitted artifacts would expose
non-public contributor data dependencies and operational integrity. This is a deliberate,
controlled-reproducibility decision, not an omission: the released code, configuration,
hyperparameters, feature registry, validation predictions, and gridded outputs are sufficient to
reproduce the reported results, and the models can be retrained from the documented configuration.

The fitted artifacts are **available from the corresponding author on reasonable request** for
research verification — including to the editor and reviewers during peer review — subject to
PAQI data-use and contributor-licensing constraints.

## Fitted artifacts used for the paper (identifiers for verification)

| Model | Role in paper | Version ID | SHA-256 |
|---|---|---|---|
| Support-aware (archive-quality) | Headline results: with-history MAE 14.3 / no-history MAE 23.2; F1@150 0.838 | `support_aware_20260325_132406` | `f7552326958ac00b4a3a49c96d1f5933e112d3f1bf55cda8feb23f3dd67cce9f` |
| Standard temporal (no dropout) | Comparison: with-history MAE 14.2 / no-history MAE 32.4 | `support_aware_20260513_121120` | `8bc72a3c258dc78878aec4d3f2321631a10809b5e7fea98f2c96961e3d64c709` |

The national-predictor backbone (no monitor history; MAE 22.9) shares the same configuration
with the monitor-history block removed. The SHA-256 values above identify the exact serialized
artifacts; a copy obtained on request can be checked with `shasum -a 256 <file>`.
