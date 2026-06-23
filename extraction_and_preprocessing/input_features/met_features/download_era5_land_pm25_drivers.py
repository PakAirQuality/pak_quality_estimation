# save as download_era5land_pk_continue.py
# pip install cdsapi
import time
import os
from pathlib import Path
import cdsapi

# ------------ Config ------------
START_YEAR = 2025
START_MONTH = 9      # Continue from May 2024
END_YEAR   = 2025

DAYS   = [f"{d:02d}" for d in range(1, 32)]
HOURS  = [f"{h:02d}:00" for h in range(24)]

AREA = [37.3, 60.5, 23.3, 77.9]  # [North, West, South, East]

# ✅ Folder where your .grib files are stored
OUTDIR = Path(os.environ.get("ERA5LAND_OUTDIR", "data/era5_land"))
OUTDIR.mkdir(parents=True, exist_ok=True)

ERA5LAND_VARS = [
    "2m_temperature", "2m_dewpoint_temperature",
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "surface_pressure",
]

def retrieve_month(c, y, m, retries=3):
    target = OUTDIR / f"era5land_pk_{y}_{m:02d}.grib"
    if target.exists():
        print(f"⏩ Skipping {target.name} (already exists)")
        return

    req = {
        "product_type": "reanalysis",
        "variable": ERA5LAND_VARS,
        "year": str(y),
        "month": f"{m:02d}",
        "day": DAYS,
        "time": HOURS,
        "area": AREA,
        "grid": "0.1/0.1",
        "format": "grib",
    }

    for attempt in range(1, retries + 1):
        try:
            print(f"▼ Downloading {target.name} (attempt {attempt}/{retries})")
            c.retrieve("reanalysis-era5-land", req, str(target))
            print(f"✔ Finished {target.name}")
            return
        except Exception as e:
            print(f"⚠ Error downloading {target.name}: {e}")
            if attempt == retries:
                print(f"✖ Giving up on {target.name}")
                return
            sleep = 30 * attempt
            print(f"↻ Retrying in {sleep}s …")
            time.sleep(sleep)

if __name__ == "__main__":
    c = cdsapi.Client()

    for y in range(START_YEAR, END_YEAR + 1):
        start_m = START_MONTH if y == START_YEAR else 1
        for m in range(start_m, 13):
            retrieve_month(c, y, m)

    print("✅ All months from May 2024 through 2025 requested.")
