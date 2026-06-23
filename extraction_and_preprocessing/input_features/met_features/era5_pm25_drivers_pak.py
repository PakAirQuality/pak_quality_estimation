# save as download_era5_pm25_drivers.py
# pip install cdsapi
import time
from pathlib import Path
import cdsapi

YEARS  = list(range(2025, 2026))                 # 2018–2025 inclusive
MONTHS = [f"{m:02d}" for m in range(10, 13)]
DAYS   = [f"{d:02d}" for d in range(1, 32)]
HOURS  = [f"{h:02d}:00" for h in range(24)]

# [North, West, South, East]
AREA = [37.3, 60.5, 23.3, 77.9]

OUTDIR = Path("era5_singlelevels_pk_2018_2025_grib")
OUTDIR.mkdir(parents=True, exist_ok=True)

VARS = [
    # Wind
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "100m_u_component_of_wind","100m_v_component_of_wind",
    "instantaneous_10m_wind_gust",  # if this fails, swap to "10m_wind_gust_since_previous_post_processing"
    # Temperature & humidity
    "2m_temperature","2m_dewpoint_temperature",
    # Pressure
    "mean_sea_level_pressure","surface_pressure",
    # Mixing / dispersion
    "boundary_layer_height",
    # Radiation & cloud
    "surface_solar_radiation_downwards","total_cloud_cover",
    # Precipitation
    "total_precipitation",
    # Tier B context
    "surface_sensible_heat_flux","surface_latent_heat_flux",
    "surface_thermal_radiation_downwards","surface_thermal_radiation_downwards_clear_sky",
]

def retrieve_month(c, y, m, retries=3):
    target = OUTDIR / f"era5_pk_{y}_{m}.grib"
    if target.exists():
        print(f"✔ Skipping existing {target.name}")
        return

    req = {
        "product_type": "reanalysis",
        "variable": VARS,
        "year": str(y),
        "month": m,
        "day": DAYS,
        "time": HOURS,
        "area": AREA,         # [N, W, S, E]
        "format": "grib",
    }

    for attempt in range(1, retries + 1):
        try:
            print(f"▼ Requesting {target.name} (attempt {attempt}/{retries})")
            c.retrieve("reanalysis-era5-single-levels", req, str(target))
            print(f"✔ Done {target.name}")
            return
        except Exception as e:
            print(f"… error: {e}")
            if attempt == retries:
                print(f"✖ Failed {target.name} after {retries} attempts")
                return
            sleep = 30 * attempt
            print(f"↻ Retrying in {sleep}s …"); time.sleep(sleep)

if __name__ == "__main__":
    c = cdsapi.Client()
    for y in YEARS:
        for m in MONTHS:
            retrieve_month(c, y, m)
    print("All requests submitted.")