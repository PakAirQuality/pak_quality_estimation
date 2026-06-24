#!/usr/bin/env python3
"""
Figure (paper_v2): four characteristic daily PM2.5 fields across the
Pakistani seasonal cycle in 2025.

For each season we pick a single representative day rather than the seasonal
mean. Selection is percentile-based on the in-country mean of the smoothed
daily TIFFs:

  - Winter (Jan, Feb, Dec): 80th percentile (high pollution typical).
  - Pre-monsoon (Mar, Apr, May): 50th percentile (median).
  - Monsoon (Jun, Jul, Aug, Sep): 20th percentile (clean monsoon air).
  - Post-monsoon (Oct, Nov): 60th percentile (winter buildup begins).

This gives a 4-panel figure that reads as a seasonal walk-through.

Visualization mirrors the hawanama estimation app: continuous turbo
colormap stretched 20-180 ug/m^3, muted light-gray background, soft admin
boundaries, 0.85 raster alpha. The figure carries no suptitle or colorbar
title; the LaTeX caption is the canonical descriptive text.

Output: paper_v2/figures/fig8_maps.pdf
"""

from __future__ import annotations

import glob
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import colormaps
from matplotlib.colors import Normalize
from rasterio.features import geometry_mask

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent  # Desktop
EST_ROOT = REPO / "Estimation"
PAPER_FIG_DIR = HERE.parent / "figures"

TIFF_DIR = EST_ROOT / "inference" / "geotiffs_2024_2025"
ADM1_PATH = EST_ROOT / "training" / "data" / "PAK_ADM1.geojson"
OUT_PDF = PAPER_FIG_DIR / "fig8_maps.pdf"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PAK_EXTENT = [60.5, 77.7, 23.0, 37.3]
PM25_VMIN = 20.0
PM25_VMAX = 180.0
BG_LAND = "#F0F0F0"   # near-white — neighbouring-country fill
BG_WATER = "#E6ECEF"  # very light cool grey — sea/ocean
ADMIN_GRAY = "#8B95A0"  # thin grey for international borders
PROVINCE_WHITE = "#FFFFFF"
COUNTRY_BLACK = "#000000"
TEXT_DARK = "#37414b"
RASTER_ALPHA = 0.92

# Pakistani seasonal grouping. (name, months, target percentile)
SEASONS = [
    ("Winter",       [1, 2, 12],   80),
    ("Pre-monsoon",  [3, 4, 5],    50),
    ("Monsoon",      [6, 7, 8, 9], 20),
    ("Post-monsoon", [10, 11],     60),
]

YEAR = 2025

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "savefig.dpi": 400,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def in_country_mean(path: str, country_geom, cached_mask=None) -> tuple[float, np.ndarray]:
    """Return (in-country mean PM2.5, masked array) for a TIFF.

    cached_mask avoids recomputing the geometry mask for every file. If
    provided, it must be aligned to the same raster shape/transform.
    """
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0
        if cached_mask is None:
            cached_mask = geometry_mask(
                [country_geom], out_shape=arr.shape,
                transform=src.transform, invert=True,
            )
    valid = (arr != nodata) & np.isfinite(arr) & (arr >= 0) & cached_mask
    arr = np.where(valid, arr, np.nan).astype(np.float32)
    return float(np.nanmean(arr)), arr


def get_country_mask(country_geom):
    """Compute raster mask once from the first available TIFF."""
    files = sorted(glob.glob(str(TIFF_DIR / f"pm25_{YEAR}-01-*_gaussian_smoothed.tif")))
    if not files:
        # Fall back to any month if January is unavailable for the chosen year.
        files = sorted(glob.glob(str(TIFF_DIR / f"pm25_{YEAR}-*_gaussian_smoothed.tif")))
    with rasterio.open(files[0]) as src:
        mask = geometry_mask(
            [country_geom], out_shape=(src.height, src.width),
            transform=src.transform, invert=True,
        )
        bounds = src.bounds
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    return mask, extent


def pick_day_for_months(months: list[int], percentile: int,
                        country_geom, cached_mask) -> tuple[str, float, np.ndarray]:
    """Return (date_str, mean, masked field) for the representative day across
    the given month list, picked at the requested percentile of in-country
    daily means."""
    files: list[str] = []
    for month in months:
        pattern = str(TIFF_DIR / f"pm25_{YEAR}-{month:02d}-*_gaussian_smoothed.tif")
        files.extend(sorted(glob.glob(pattern)))
    if not files:
        raise FileNotFoundError(f"No TIFFs for {YEAR} months {months}")

    means: list[tuple[float, str]] = []
    for fp in files:
        m, _ = in_country_mean(fp, country_geom, cached_mask)
        means.append((m, fp))

    means.sort(key=lambda x: x[0])
    idx = int(round((percentile / 100.0) * (len(means) - 1)))
    chosen_mean, chosen_path = means[idx]
    chosen_date = Path(chosen_path).stem.replace("pm25_", "").replace("_gaussian_smoothed", "")

    _, arr = in_country_mean(chosen_path, country_geom, cached_mask)
    return chosen_date, chosen_mean, arr


def main():
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading boundaries...")
    provinces = gpd.read_file(ADM1_PATH)
    country_geom = provinces.union_all()

    print("Computing country mask...")
    cached_mask, extent = get_country_mask(country_geom)

    cmap = colormaps["magma"].copy()
    cmap.set_bad(color="#FFFFFF", alpha=0)
    norm = Normalize(vmin=PM25_VMIN, vmax=PM25_VMAX)

    fig, axes = plt.subplots(
        2, 2, figsize=(5.6, 5.0),
        subplot_kw={"projection": ccrs.PlateCarree()},
        gridspec_kw={"wspace": 0.02, "hspace": 0.20},
    )

    im = None
    for ax, (season_name, months, percentile) in zip(axes.flat, SEASONS):
        print(f"  {season_name} (months {months}, p{percentile}) ...",
              end="", flush=True)

        date_str, day_mean, field = pick_day_for_months(
            months, percentile, country_geom, cached_mask,
        )
        print(f" {date_str}  (in-country mean {day_mean:.1f} ug/m^3)")

        ax.set_extent(PAK_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor=BG_LAND, zorder=0, edgecolor="none")
        ax.add_feature(cfeature.OCEAN, facecolor=BG_WATER, zorder=0, edgecolor="none")

        im = ax.imshow(
            field, extent=extent, origin="upper",
            cmap=cmap, norm=norm, transform=ccrs.PlateCarree(),
            zorder=2, interpolation="bilinear", alpha=RASTER_ALPHA,
        )

        # Surrounding-country borders (thin grey)
        ax.add_feature(cfeature.COASTLINE, edgecolor=ADMIN_GRAY,
                       linewidth=0.5, alpha=0.6, zorder=3)
        ax.add_feature(cfeature.BORDERS, linestyle="-",
                       edgecolor=ADMIN_GRAY, alpha=0.6,
                       linewidth=0.5, zorder=4)

        # Province boundaries inside Pakistan (white)
        for geom in provinces.geometry:
            ax.add_geometries([geom], ccrs.PlateCarree(),
                              facecolor="none", edgecolor=PROVINCE_WHITE,
                              linewidth=0.55, alpha=0.95, zorder=5)

        # Pakistan ADM0 outline (bold black) on top
        ax.add_geometries([country_geom], ccrs.PlateCarree(),
                          facecolor="none", edgecolor=COUNTRY_BLACK,
                          linewidth=1.2, zorder=6)

        # Panel title: season name only.
        ax.set_title(season_name,
                     fontsize=11, fontweight="bold",
                     pad=4, color=TEXT_DARK)

        gl = ax.gridlines(
            draw_labels=True,
            xlocs=[60, 65, 70, 75, 80],
            ylocs=[24, 27, 30, 33, 36],
            linewidth=0.3, color="#A6AEB6", alpha=0.5, linestyle=":",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8, "color": TEXT_DARK}
        gl.ylabel_style = {"size": 8, "color": TEXT_DARK}

        ax.spines["geo"].set_edgecolor("black")
        ax.spines["geo"].set_linewidth(0.7)

    fig.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.16)
    cbar_ax = fig.add_axes([0.22, 0.07, 0.56, 0.025])
    cbar = fig.colorbar(
        im, cax=cbar_ax, orientation="horizontal", extend="both",
        ticks=[20, 40, 60, 80, 100, 120, 140, 160, 180],
    )
    cbar.ax.tick_params(labelsize=9.5, colors=TEXT_DARK)
    cbar.outline.set_edgecolor(ADMIN_GRAY)
    cbar.outline.set_linewidth(0.5)
    cbar_ax.text(
        0.5, -2.6,
        r"PM$_{2.5}$ ($\mu$g m$^{-3}$)",
        transform=cbar_ax.transAxes, ha="center", va="top",
        fontsize=11, color=TEXT_DARK,
    )

    # Data/boundary attribution (ACP requirement for maps)
    fig.text(
        0.985, 0.012, "Boundaries: geoBoundaries",
        ha="right", va="bottom", fontsize=6, color=ADMIN_GRAY,
    )

    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> {OUT_PDF}")


if __name__ == "__main__":
    main()
