# Data sources and availability

## 1. Public input products (obtain from the provider)

| Stream | Product | Source | Terms |
|---|---|---|---|
| Aerosol optical depth | MODIS MAIAC (MCD19A2.061) | NASA Earthdata | open |
| Atmospheric composition | Sentinel-5P/TROPOMI (NO2, SO2, CO, HCHO, O3, CH4, AAI, ALH, cloud) | NASA Earthdata / Copernicus | open |
| Chemistry context | GEOS-CF (NH3) | NASA GMAO | open |
| Meteorology | ERA5 / ERA5-Land | Copernicus C3S | open (C3S licence) |
| Administrative boundaries | geoBoundaries (Runfola et al., 2020) | geoBoundaries | CC-BY-4.0 |

These products are not redistributed here. Ingestion requires NASA Earthdata, Google
Earth Engine, and Copernicus CDS credentials, supplied via environment variables.

## 2. Derived products (CC-BY-4.0)

- Daily gridded PM2.5 prediction fields (0.1° national grid), as Cloud Optimized
  GeoTIFFs (raw + Gaussian-smoothed) with per-day metadata.
- A derived station-day validation table (observed vs. estimated daily PM2.5 with
  support-regime flags) sufficient to reproduce the paper's headline metrics.

These are distributed with the archived release (Zenodo DOI: TBD).

## 3. Restricted stream

The ground-station PM2.5 archive (public-monitor + low-cost-sensor streams) is
contributed by the Pakistan Air Quality Initiative (PAQI). Raw individual-sensor
records are **not fully public** owing to contributor and device-licensing
constraints; they are available from PAQI on reasonable request. The derived,
quality-controlled station-day table needed to reproduce the reported results is
released as in §2.

## Quality control

Before daily aggregation, hourly observations pass physical-bound, sentinel-value,
spike, and flatline filters; a station-day is retained with at least 18 valid hourly
readings. Temporal monitor-history features are shifted to previous days (no same-day
leakage); nearby-station summaries are computed leave-one-station-out for
station-level evaluation.
