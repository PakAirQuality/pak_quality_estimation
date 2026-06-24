#!/usr/bin/env python3
"""
Figure 1 (paper_v2): Pakistan low-cost AQ monitoring network expansion across
three eras.

Three panels, one per era (2020-21 sparse, 2022-23 expansion, 2024-25
operational), all sharing:
  - Warm tan/sienna population-density backdrop (shows urban concentration)
  - ADM0 (country) outline with white core + dark halo
  - ADM1 (provincial) boundaries with province name labels
  - Royal-blue circles for QC-passed monitors active in that era

Output: paper_v2/figures/fig1_network.pdf
        paper_v2/figures/fig1_network.png
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PAPER_FIG_DIR = HERE.parent / "figures"
DESKTOP = HERE.parent.parent.parent

POP_TIF = DESKTOP / "Geo-analysis" / "world_pop" / "pak_pop_2025_CN_100m_R2024B_v1.tif"
ADM0_PATH_CANDIDATES = [
    DESKTOP / "pop_opt_ipynb" / "data" / "boundaries"
        / "PAK_ADM0_geoboundaries_gbOpen_ADM0.geojson",
    DESKTOP / "pop_opt_ipynb_v2" / "data" / "boundaries"
        / "PAK_ADM0_geoboundaries_gbOpen_ADM0.geojson",
]
ADM0_PATH = next((p for p in ADM0_PATH_CANDIDATES if p.exists()), None)

ADM1_PATH_CANDIDATES = [
    DESKTOP / "Estimation" / "training" / "data" / "PAK_ADM1.geojson",
    DESKTOP / "Estimation" / "inference" / "output" / "PAK_ADM1.geojson",
    DESKTOP / "Geo-analysis" / "population-optimized-monitors-master"
        / "Boundaries" / "PAK_ADM1.geojson",
]
ADM1_PATH = next((p for p in ADM1_PATH_CANDIDATES if p.exists()), None)
META_PATH = DESKTOP / "Estimation" / "training" / "data" / "metadata" / "aod_metadata.csv"

OUT_PDF = PAPER_FIG_DIR / "fig1_network.pdf"
OUT_PNG = PAPER_FIG_DIR / "fig1_network.png"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
EXTENT_WGS84 = (60.0, 22.5, 78.5, 37.8)  # (xmin, ymin, xmax, ymax)

ERA_LABELS = [
    ([2020, 2021], "2020–2021", "Limited coverage"),
    ([2022, 2023], "2022–2023", "Network growth"),
    ([2024, 2025], "2024–2025", "Expanded coverage"),
]

# Warm tan–sienna basemap, matched to the "Sparse Daily Monitoring" panel
# of Fig. 1 (graphical overview).
POP_VMIN = 0
POP_VMAX = 120
POP_CMAP = LinearSegmentedColormap.from_list(
    "pop_tan",
    [
        (0.00, "#FBF1DC"),  # cream (low pop)
        (0.30, "#F1D9A4"),  # straw
        (0.55, "#D9A864"),  # tan
        (0.80, "#B07A3A"),  # sienna
        (1.00, "#7A4720"),  # umber (very dense)
    ],
    N=256,
)

STATION_BLUE = "#2742E2"
BOUNDARY_WHITE = "#FFFFFF"
LEGEND_BORDER = "#888888"
TEXT_DARK = "#1C1C1C"

DECIMATE = 4

# Short labels for ADM1 polygons; key is the geoBoundaries `shapeName`.
ADM1_SHORT = {
    "Punjab": "Punjab",
    "Sindh": "Sindh",
    "Balochistan": "Balochistan",
    "Khyber Pakhtunkhwa": "KPK",
    "Islamabad Capital Territory": "",
    "Gilgit-Baltistan": "GB",
    "Azad Kashmir": "",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.titlesize": 11,
    "savefig.dpi": 400,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Population raster: decimated read, then reproject to Web Mercator
# ---------------------------------------------------------------------------
def load_population_3857():
    print(f"Reading WorldPop (decimate=1/{DECIMATE})...")
    with rasterio.open(POP_TIF) as src:
        h_out = src.height // DECIMATE
        w_out = src.width // DECIMATE
        pop = src.read(
            1, out_shape=(h_out, w_out),
            resampling=Resampling.average,
        ).astype(np.float32)

        scale_x = src.width / w_out
        scale_y = src.height / h_out
        src_transform = src.transform * src.transform.scale(scale_x, scale_y)
        src_crs = src.crs

        pop = np.where(pop <= -1000, np.nan, pop)
        print(f"  decimated shape: {pop.shape}, valid range: "
              f"{np.nanmin(pop):.2f} – {np.nanmax(pop):.1f}")

        dst_crs = "EPSG:3857"
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src_crs, dst_crs, w_out, h_out,
            *rasterio.transform.array_bounds(h_out, w_out, src_transform),
        )
        dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
        reproject(
            source=pop, destination=dst,
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=Resampling.average,
            src_nodata=np.nan, dst_nodata=np.nan,
        )
        bounds = rasterio.transform.array_bounds(dst_h, dst_w, dst_transform)
        extent_3857 = (bounds[0], bounds[2], bounds[1], bounds[3])
    return dst, extent_3857


def wgs84_extent_to_3857(extent):
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin, ymin = t.transform(extent[0], extent[1])
    xmax, ymax = t.transform(extent[2], extent[3])
    return xmin, xmax, ymin, ymax


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def main():
    if ADM0_PATH is None:
        raise FileNotFoundError("PAK_ADM0_geoboundaries_gbOpen_ADM0.geojson not found")
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Pakistan ADM0...")
    adm0 = gpd.read_file(ADM0_PATH).to_crs("EPSG:3857")

    if ADM1_PATH is not None:
        print(f"Loading Pakistan ADM1 from {ADM1_PATH.name}...")
        adm1 = gpd.read_file(ADM1_PATH).to_crs("EPSG:3857")
    else:
        print("  ADM1 boundaries not found; provincial layer skipped.")
        adm1 = None

    pop, pop_extent = load_population_3857()

    print("Loading AQ monitor metadata (with active year)...")
    meta = pd.read_csv(META_PATH,
                       usecols=["sensor_id", "obs_lat", "obs_lon", "date_utc"])
    meta["year"] = pd.to_datetime(meta["date_utc"]).dt.year
    coords = (meta.dropna(subset=["obs_lat", "obs_lon"])
                  [["sensor_id", "obs_lat", "obs_lon", "year"]]
                  .drop_duplicates(["sensor_id", "year"]))

    cmap = POP_CMAP.copy()
    cmap.set_bad((0, 0, 0, 0))
    norm = Normalize(vmin=POP_VMIN, vmax=POP_VMAX)

    fig, axes = plt.subplots(
        1, 3, figsize=(15.6, 6.5),
        gridspec_kw={"wspace": 0.04},
    )

    xmin, xmax, ymin, ymax = wgs84_extent_to_3857(EXTENT_WGS84)

    counts_per_era = []
    for col_idx, (ax, (years, era, era_desc)) in enumerate(zip(axes, ERA_LABELS)):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")

        # Layer 1: population density (warm tan/sienna palette)
        masked = np.where(pop > 0.5, pop, np.nan)
        ax.imshow(
            masked, extent=pop_extent, origin="upper",
            cmap=cmap, norm=norm, alpha=0.95,
            interpolation="bilinear", zorder=2,
        )

        # Layer 2: Pakistan ADM1 (provincial) boundaries — dark halo + white core
        if adm1 is not None:
            adm1.boundary.plot(ax=ax, color="#1A1A1A", linewidth=1.6,
                               zorder=2.55, alpha=0.55)
            adm1.boundary.plot(ax=ax, color=BOUNDARY_WHITE, linewidth=0.9,
                               zorder=2.6, alpha=1.0)

            # Province labels at representative_point (guaranteed inside polygon).
            for _, row in adm1.iterrows():
                name = row.get("shapeName") or row.get("ADM1_EN") or ""
                short = ADM1_SHORT.get(name, name)
                if not short:
                    continue
                pt = row.geometry.representative_point()
                ax.text(
                    pt.x, pt.y, short,
                    ha="center", va="center",
                    fontsize=7.5, fontweight="bold",
                    color=TEXT_DARK, zorder=6,
                    path_effects=[
                        pe.withStroke(linewidth=2.6, foreground="white"),
                    ],
                )

        # Layer 3: Pakistan ADM0 outline — white core with thin dark halo
        adm0.boundary.plot(ax=ax, color="#3A3A3A", linewidth=2.4,
                           zorder=2.9, alpha=0.55)
        adm0.boundary.plot(ax=ax, color=BOUNDARY_WHITE, linewidth=1.4,
                           zorder=3, alpha=0.95)

        # Layer 4: Active monitors in this era
        mask = coords["year"].isin(years)
        pts = coords[mask].drop_duplicates("sensor_id")
        counts_per_era.append(len(pts))
        if len(pts):
            pts_gdf = gpd.GeoDataFrame(
                pts,
                geometry=gpd.points_from_xy(pts["obs_lon"], pts["obs_lat"]),
                crs="EPSG:4326",
            ).to_crs("EPSG:3857")
            pts_gdf.plot(ax=ax, color=STATION_BLUE, markersize=44,
                         edgecolor="white", linewidth=0.7, zorder=5)

        # Strip ticks/labels but keep a thin gray bounding box around the panel
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ("top", "right", "left", "bottom"):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color(LEGEND_BORDER)
            ax.spines[spine].set_linewidth(0.7)

        # Two-line title block, sitting tight above the panel
        ax.set_title("")
        ax.text(
            0.5, 1.045, era,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=TEXT_DARK,
        )
        ax.text(
            0.5, 1.005, f"{era_desc}, $n = {len(pts)}$ stations",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, color=TEXT_DARK,
        )

    # ---- Shared legend below all three panels (single compact row) ----
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=STATION_BLUE,
               markeredgecolor="white", markeredgewidth=0.7,
               markersize=10, linestyle="None"),
        Line2D([0], [0], color=BOUNDARY_WHITE, linewidth=1.4,
               solid_capstyle="butt"),
        Line2D([0], [0], color=BOUNDARY_WHITE, linewidth=0.9,
               solid_capstyle="butt"),
    ]
    labels = ["Active AQ monitor", "Country boundary", "Province boundary"]

    leg = fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=len(handles), frameon=False,
        fontsize=9.5, handletextpad=0.45,
        columnspacing=1.6, borderpad=0.4,
    )
    for txt in leg.get_texts():
        txt.set_color(TEXT_DARK)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.06)
    print(f"Saving fig1_network.{{pdf,png}}...")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f"  -> {OUT_PDF}")
    print(f"  -> {OUT_PNG}")


if __name__ == "__main__":
    main()
