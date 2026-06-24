# Figure-generation scripts

Scripts that produce the manuscript figures. They are provided for transparency and
reproduction; some require inputs that are restricted (see notes).

| Script | Figure | Inputs | Reproducible from this archive? |
|---|---|---|---|
| `make_fig_monthly.py` | Seasonal daily PM2.5 maps | gridded fields (`data/gridded_predictions_2024_2025/`) + admin boundaries (geoBoundaries, external) | Yes — fields are included; boundaries are public |
| `make_fig_model_safety.py` | MAE with/without history (3 models) | values inline in the script (from the 2025 evaluation) | Yes — self-contained |
| `make_fig_obs_scale_diagnostics.py` | Low-cost-sensor pairwise disagreement by distance/season | values inline in the script | Yes — self-contained |
| `make_fig1_network.py` | Monitoring-network growth maps | admin boundaries + WorldPop population (external) + monitor metadata (**restricted**: exact low-cost-sensor coordinates are not public) | Partially — code provided; monitor-location input is restricted (see `DATA.md`) |

Notes:
- The bar/line figures (`make_fig_model_safety.py`, `make_fig_obs_scale_diagnostics.py`) carry
  their plotted values inline, so they reproduce without any external data.
- Map figures rely on the gridded product (included) plus public boundary/population layers.
- Table values in the manuscript are reproducible from `validation/pm25_2025_station_day_validation.csv`
  and the evaluation result JSONs under `training/results/`.
- These scripts depend on `matplotlib`, `numpy`, `rasterio`, `cartopy`, `geopandas` (see
  `environment.yml` / `requirements.txt`).
