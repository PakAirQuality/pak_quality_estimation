# 2025 held-out validation table

`pm25_2025_station_day_validation.csv` — the per-row, station-day predictions on the
held-out **2025** test year, sufficient to reproduce the manuscript's headline numbers.

Each row is one quality-controlled station-day. Both predictions come from the **same**
support-aware model (`support_aware_20260325_132406`); they differ only in whether the
recent monitor-history feature block is supplied (`pred_with_history`) or fully masked
(`pred_no_history`) — the two deployment regimes evaluated in the paper.

## Columns

| Column | Description |
|---|---|
| `date` | Observation day (YYYY-MM-DD), 2025 |
| `station_id` | Anonymised, non-reversible station id (salted SHA-1; `PK…`) |
| `city` | City the station is in |
| `province` | Province (blank where unavailable) |
| `pm25_observed` | Quality-controlled daily mean PM2.5 (µg m⁻³) |
| `pred_with_history` | Support-aware prediction **with** local monitor history (µg m⁻³) |
| `pred_no_history` | Support-aware prediction with the monitor-history block **masked** (µg m⁻³) |

Rows: 44,492 station-days · 398 stations.

## Reproduce the headline metrics

```python
import pandas as pd, numpy as np
d = pd.read_csv("pm25_2025_station_day_validation.csv")
y = d.pm25_observed.values
mae = lambda p: np.mean(np.abs(y - p))
f1  = lambda p, t=150: (lambda tp,fp,fn: 2*tp/max(2*tp+fp+fn,1))(
        int(((p>=t)&(y>=t)).sum()), int(((p>=t)&(y<t)).sum()), int(((p<t)&(y>=t)).sum()))
print("with-history  MAE %.2f  F1@150 %.3f" % (mae(d.pred_with_history), f1(d.pred_with_history)))
print("no-history    MAE %.2f  F1@150 %.3f" % (mae(d.pred_no_history),  f1(d.pred_no_history)))
# -> with-history MAE 14.31, F1@150 0.838 ;  no-history MAE 23.22
```

These match the manuscript's support-aware results (14.3 / 23.2 µg m⁻³; F1@150 0.838).

## Privacy

The network is predominantly low-cost sensors at private/residential sites. Exact
coordinates, device ids, and site names are **not** released; `station_id` is a salted,
non-reversible hash. This is the derived, publication-safe table referenced in the paper's
Code and data availability statement; raw individual-sensor records are available from PAQI
on reasonable request.

## Provenance

`build_validation_table.py` documents how this file was produced (two-pass inference with
the saved production model). It requires the restricted master feature lake and the model
weights, so it is provided for transparency rather than for out-of-the-box execution.
