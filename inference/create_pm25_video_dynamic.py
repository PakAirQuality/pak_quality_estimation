#!/usr/bin/env python3
"""
create_pm25_video_dynamic.py

Same pipeline as create_pm25_video.py, but uses a continuous "turbo" colormap
with Normalize(vmin, vmax). Default range is 0–300 µg/m³.

Notes:
- Keep "fixed" scaling (default) for stable colors across time.
- Use --autoscale period to fit vmin/vmax to the whole period (no flicker).
- Use --autoscale frame only if you really want per-frame scaling (can flicker).
"""
import os
import sys
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.path import Path as MplPath
import matplotlib.patches as mpatches

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

from pakistan_daily_inference import PakistanPM25Predictor

# Feature store imports
from feature_store.paths import GridStoreConfig, VALID_STAGES
from feature_store.reader import GridPartitionReader, join_grid_stages
from feature_store.build_grid_store import (
    build_static_grid,
    build_met_store,
    build_aod_store,
    build_tropomi_store,
)

warnings.filterwarnings("ignore")

# cartopy geoms -> matplotlib paths (version-safe import)
try:
    from cartopy.mpl.patch import geos_to_path
except Exception:
    geos_to_path = None


class PM25VideoGenerator:
    """Generate animated videos of PM2.5 predictions over Pakistan (continuous turbo cmap)."""

    def __init__(
        self,
        predictor: PakistanPM25Predictor,
        vmin: float = 0.0,
        vmax: float = 300.0,
        autoscale: str = "fixed",  # fixed | period | frame
    ):
        self.predictor = predictor
        self.predictions_cache: Dict[str, np.ndarray] = {}

        self.user_vmin = float(vmin)
        self.user_vmax = float(vmax)
        self.autoscale = str(autoscale).lower().strip()

        # Continuous colormap
        self.cmap = plt.get_cmap("turbo").copy()
        self.cmap.set_bad((0, 0, 0, 0))  # transparent NaNs
        self.norm = Normalize(vmin=self.user_vmin, vmax=self.user_vmax, clip=True)

        # derived once from lat/lon grids
        self._grid_tf: Optional[Dict[str, object]] = None

        print(f"[video] PM2.5 Video Generator initialized (turbo, vmin={self.user_vmin}, vmax={self.user_vmax}, autoscale={self.autoscale})")

    # ----------------------------
    # Grid helpers (robust)
    # ----------------------------
    def _grid_coords_flat(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback lat/lon if predictor day-1 fails.
        Tries multiple attribute names to match your generator versions.
        """
        fg = self.predictor.feature_generator

        if hasattr(fg, "grid_lats") and hasattr(fg, "grid_lons"):
            lats = np.asarray(fg.grid_lats, dtype="float32").reshape(-1)
            lons = np.asarray(fg.grid_lons, dtype="float32").reshape(-1)
            return lats, lons

        if hasattr(fg, "lat_grid") and hasattr(fg, "lon_grid"):
            lats = np.asarray(fg.lat_grid, dtype="float32").reshape(-1)
            lons = np.asarray(fg.lon_grid, dtype="float32").reshape(-1)
            return lats, lons

        raise AttributeError(
            "Could not find grid coordinates on feature_generator. "
            "Expected grid_lats/grid_lons or lat_grid/lon_grid."
        )

    def _infer_grid_transform(self, lat_grid: np.ndarray, lon_grid: np.ndarray) -> Dict[str, object]:
        """
        Infers a consistent transform so imshow is not inverted.
        Detects:
          - transpose needed (flatten order mismatch)
          - flipud needed (lat decreasing)
          - fliplr needed (lon decreasing)

        Returns dict with flags + extent.
        """
        eps = 1e-6

        latg = np.asarray(lat_grid)
        long = np.asarray(lon_grid)

        # In a proper (lat,lon) grid:
        # - long[:,0] should be ~constant
        # - latg[0,:] should be ~constant
        transpose = (np.nanstd(long[:, 0]) > eps) or (np.nanstd(latg[0, :]) > eps)
        if transpose:
            latg = latg.T
            long = long.T

        lat_axis = latg[:, 0].copy()
        lon_axis = long[0, :].copy()

        flipud = lat_axis[0] > lat_axis[-1]
        fliplr = lon_axis[0] > lon_axis[-1]

        # axes increasing for extent calculation
        if flipud:
            lat_axis = lat_axis[::-1]
        if fliplr:
            lon_axis = lon_axis[::-1]

        extent = [float(lon_axis.min()), float(lon_axis.max()), float(lat_axis.min()), float(lat_axis.max())]

        return {
            "transpose": transpose,
            "flipud": flipud,
            "fliplr": fliplr,
            "extent": extent,
        }

    def _apply_transform(self, z2d: np.ndarray, tf: Dict[str, object]) -> np.ndarray:
        """Applies inferred transpose/flip operations to a 2D grid."""
        z = z2d
        if tf["transpose"]:
            z = z.T
        if tf["flipud"]:
            z = np.flipud(z)
        if tf["fliplr"]:
            z = np.fliplr(z)
        return z

    # ----------------------------
    # Clipping (fix coastal/ocean "leaks")
    # ----------------------------
    def _pakistan_clip_patch(self, ax):
        """
        Build a clip patch for Pakistan using Natural Earth admin_0 geometry.
        Clips ONLY the raster so coarse coastal pixels cannot spill into the sea.
        """
        if geos_to_path is None:
            return None

        try:
            shp = shpreader.natural_earth(
                resolution="10m",
                category="cultural",
                name="admin_0_countries",
            )
            reader = shpreader.Reader(shp)

            pak_geom = None
            for rec in reader.records():
                attrs = rec.attributes
                if attrs.get("ADMIN") == "Pakistan" or attrs.get("NAME") == "Pakistan":
                    pak_geom = rec.geometry
                    break

            if pak_geom is None:
                return None

            paths = geos_to_path(pak_geom)
            compound = MplPath.make_compound_path(*paths)

            patch = mpatches.PathPatch(
                compound,
                transform=ccrs.PlateCarree(),
                facecolor="none",
                edgecolor="none",
            )
            ax.add_patch(patch)
            return patch

        except Exception as e:
            print(f"[video] clip patch failed (ok to ignore): {e}")
            return None

    # ----------------------------
    # Scaling / normalization
    # ----------------------------
    def _valid_stats(self, vec: np.ndarray) -> Optional[Dict[str, float]]:
        v = vec[np.isfinite(vec)]
        if v.size == 0:
            return None
        return {
            "min": float(v.min()),
            "max": float(v.max()),
            "p01": float(np.percentile(v, 1)),
            "p99": float(np.percentile(v, 99)),
        }

    def _period_scale(self, predictions_dict: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """Compute a stable vmin/vmax over the whole period (min/max across frames)."""
        mn = np.inf
        mx = -np.inf
        for _, vec in predictions_dict.items():
            v = vec[np.isfinite(vec)]
            if v.size == 0:
                continue
            mn = min(mn, float(v.min()))
            mx = max(mx, float(v.max()))

        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return self.user_vmin, self.user_vmax

        # clamp to user range (default 0–300) unless user gave very wide values
        vmin = min(self.user_vmin, mn) if self.user_vmin < mn else self.user_vmin
        vmax = max(self.user_vmax, mx) if self.user_vmax > mx else self.user_vmax

        # If user set strict bounds (like 0–300), keep them:
        vmin = self.user_vmin
        vmax = self.user_vmax
        return float(vmin), float(vmax)

    def _frame_scale(self, vec: np.ndarray) -> Tuple[float, float]:
        """Compute per-frame vmin/vmax (percentile-based), clamped to user bounds."""
        st = self._valid_stats(vec)
        if st is None:
            return self.user_vmin, self.user_vmax

        lo = st["p01"]
        hi = st["p99"]
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return self.user_vmin, self.user_vmax

        vmin = max(self.user_vmin, lo)
        vmax = min(self.user_vmax, hi)
        if vmax <= vmin:
            return self.user_vmin, self.user_vmax
        return float(vmin), float(vmax)

    def _set_norm(self, vmin: float, vmax: float):
        self.norm = Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)

    # ----------------------------
    # Stats text
    # ----------------------------
    def _stats_box(self, vec: np.ndarray) -> str:
        v = vec[np.isfinite(vec)]
        if v.size == 0:
            return "No valid predictions"

        mu = float(v.mean())
        mx = float(v.max())
        mn = float(v.min())
        med = float(np.median(v))
        p95 = float(np.percentile(v, 95))
        std = float(v.std())

        lines = [
            f"PM2.5 µg/m³  Mean {mu:.1f} | Median {med:.1f} | P95 {p95:.1f}",
            f"Min {mn:.1f} | Max {mx:.1f} | Std {std:.1f}",
            f"Color scale: [{self.norm.vmin:.0f}, {self.norm.vmax:.0f}]",
        ]
        return "\n".join(lines)

    # ----------------------------
    # Prediction loop
    # ----------------------------
    def generate_predictions_for_period(
        self, start_date: datetime, end_date: datetime, verbose: bool = False
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Generate predictions for all dates in the period (inclusive)."""

        print(f"[video] Generating predictions: {start_date:%Y-%m-%d} → {end_date:%Y-%m-%d}")

        current_date = start_date
        predictions: Dict[str, np.ndarray] = {}

        # always know grid size even if day-1 fails
        lats, lons = self._grid_coords_flat()
        n = lats.size

        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"[video] Processing {date_str}...")

            try:
                pred, lat_vals, lon_vals, _ = self.predictor.predict(current_date, verbose=verbose)

                pred = np.asarray(pred, dtype="float32").reshape(-1)
                if pred.size != n:
                    raise ValueError(f"Prediction size mismatch: got {pred.size}, expected {n}")
                predictions[date_str] = pred

                # keep coords if predictor returns them
                if lat_vals is not None and lon_vals is not None:
                    lats = np.asarray(lat_vals, dtype="float32").reshape(-1)
                    lons = np.asarray(lon_vals, dtype="float32").reshape(-1)

            except Exception as e:
                print(f"[video] ERROR {date_str}: {e}")
                predictions[date_str] = np.full(n, np.nan, dtype="float32")

            current_date += timedelta(days=1)

        return predictions, lats, lons

    # ----------------------------
    # Static map (imshow + correct orientation + clip)
    # ----------------------------
    def create_static_map(
        self,
        predictions: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        date: str,
        output_file: Optional[Path] = None,
    ) -> plt.Figure:
        """Create a static map for a single day with turbo colormap (not inverted, no ocean leaks)."""

        # For static map, honor autoscale frame if requested, else fixed.
        if self.autoscale == "frame":
            vmin, vmax = self._frame_scale(predictions)
            self._set_norm(vmin, vmax)
        else:
            self._set_norm(self.user_vmin, self.user_vmax)

        grid_shape = self.predictor.feature_generator.lat_grid.shape
        pred_grid = predictions.reshape(grid_shape)
        lat_grid = lats.reshape(grid_shape)
        lon_grid = lons.reshape(grid_shape)

        tf = self._infer_grid_transform(lat_grid, lon_grid)
        z = self._apply_transform(pred_grid, tf)
        extent = tf["extent"]

        z = np.ma.masked_invalid(z)

        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([60.5, 77.9, 23.3, 37.3], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.6)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.25)
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.25)

        im = ax.imshow(
            z,
            origin="lower",
            extent=extent,
            transform=ccrs.PlateCarree(),
            cmap=self.cmap,
            norm=self.norm,
            interpolation="nearest",
        )

        # clip raster to Pakistan to avoid coastal pixel spill
        clip_patch = self._pakistan_clip_patch(ax)
        if clip_patch is not None:
            im.set_clip_path(clip_patch)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
        cbar.set_label("PM2.5 (µg/m³)", fontsize=11)

        ax.set_title(f"Pakistan Daily PM2.5 Predictions\n{date}", fontsize=14, fontweight="bold")
        ax.text(
            0.02,
            0.98,
            self._stats_box(predictions),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth=0.3, alpha=0.7)
        plt.tight_layout()

        if output_file is not None:
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"[video] Saved still: {output_file}")

        return fig

    # ----------------------------
    # Animation (imshow + correct orientation + clip)
    # ----------------------------
    def create_animation(
        self,
        predictions_dict: Dict[str, np.ndarray],
        lats: np.ndarray,
        lons: np.ndarray,
        output_file: str = "pm25_pakistan_animation.mp4",
        fps: int = 8,
    ) -> str:
        """Create animated video of PM2.5 predictions with turbo colormap (clipped; stable transform)."""

        print(f"[video] Creating animation: {len(predictions_dict)} frames → {output_file}")

        dates = sorted(predictions_dict.keys())
        if not dates:
            raise ValueError("No dates found in predictions_dict")

        # Choose scaling mode
        if self.autoscale == "period":
            # You asked for 0–300, so default behavior keeps user bounds.
            # If you want truly fitted bounds, set --vmin/--vmax wide and autoscale=period.
            vmin, vmax = self._period_scale(predictions_dict)
            self._set_norm(vmin, vmax)
        elif self.autoscale == "fixed":
            self._set_norm(self.user_vmin, self.user_vmax)
        else:
            # frame scaling will happen inside animate_frame
            self._set_norm(self.user_vmin, self.user_vmax)

        grid_shape = self.predictor.feature_generator.lat_grid.shape
        lat_grid = lats.reshape(grid_shape)
        lon_grid = lons.reshape(grid_shape)

        # infer transform once and reuse
        tf = self._infer_grid_transform(lat_grid, lon_grid)
        self._grid_tf = tf
        extent = tf["extent"]

        fig = plt.figure(figsize=(14, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([60.5, 77.9, 23.3, 37.3], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="black")
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.2)
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.25)

        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth=0.3, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        vec0 = predictions_dict[dates[0]]
        z0 = self._apply_transform(vec0.reshape(grid_shape), tf)
        z0 = np.ma.masked_invalid(z0)

        im = ax.imshow(
            z0,
            origin="lower",
            extent=extent,
            transform=ccrs.PlateCarree(),
            cmap=self.cmap,
            norm=self.norm,
            interpolation="nearest",
            animated=False,
        )

        # clip once (works for all frames because same artist)
        clip_patch = self._pakistan_clip_patch(ax)
        if clip_patch is not None:
            im.set_clip_path(clip_patch)

        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("PM2.5 (µg/m³)", fontsize=12, fontweight="bold")

        title = ax.set_title("", fontsize=16, fontweight="bold", pad=20)
        stats_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        def animate_frame(frame_idx: int):
            date = dates[frame_idx]
            vec = predictions_dict[date]

            if self.autoscale == "frame":
                vmin, vmax = self._frame_scale(vec)
                im.set_clim(vmin, vmax)
                cbar.update_normal(im)

            z = self._apply_transform(vec.reshape(grid_shape), tf)
            im.set_data(np.ma.masked_invalid(z))

            title.set_text(f"Pakistan Daily PM2.5 Predictions\n{date}")
            stats_text.set_text(self._stats_box(vec))

            return (im, title, stats_text)

        anim = animation.FuncAnimation(
            fig,
            animate_frame,
            frames=len(dates),
            interval=max(1, int(1000 // max(1, fps))),
            blit=False,
            repeat=True,
        )

        if not animation.writers.is_available("ffmpeg"):
            plt.close(fig)
            raise RuntimeError(
                "Matplotlib ffmpeg writer is not available. "
                "Try: `which ffmpeg` and ensure it is on PATH in the SAME env."
            )

        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        print(f"[video] Saving → {output_file}")
        anim.save(output_file, writer=writer, dpi=150)

        plt.close(fig)
        print(f"[video] ✓ Saved: {output_file}")
        return output_file

    # ----------------------------
    # Summary plots (unchanged; uses thresholds for interpretability)
    # ----------------------------
    def create_summary_plots(self, predictions_dict: Dict[str, np.ndarray], output_dir: str = "summary_plots"):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        dates = sorted(predictions_dict.keys())
        daily_stats = []

        for date in dates:
            preds = predictions_dict[date]
            valid = preds[np.isfinite(preds)]
            if valid.size == 0:
                continue

            daily_stats.append(
                {
                    "date": pd.to_datetime(date),
                    "mean": float(valid.mean()),
                    "median": float(np.median(valid)),
                    "max": float(valid.max()),
                    "min": float(valid.min()),
                    "std": float(valid.std()),
                    "p95": float(np.percentile(valid, 95)),
                    "good_pct": float(np.sum(valid < 9.0) / valid.size * 100),
                    "unhealthy_pct": float(np.sum(valid > 55.4) / valid.size * 100),
                }
            )

        if not daily_stats:
            print("[video] No valid stats to plot (all frames NaN).")
            return

        stats_df = pd.DataFrame(daily_stats)

        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        axes[0].plot(stats_df["date"], stats_df["mean"], linewidth=2, label="Mean")
        axes[0].plot(stats_df["date"], stats_df["median"], linewidth=2, label="Median")
        axes[0].fill_between(
            stats_df["date"],
            stats_df["mean"] - stats_df["std"],
            stats_df["mean"] + stats_df["std"],
            alpha=0.3,
            label="±1 STD",
        )
        axes[0].axhline(y=9.0, linestyle="--", alpha=0.6, label="Good/Moderate (9.0)")
        axes[0].axhline(y=35.4, linestyle="--", alpha=0.6, label="Moderate/USG (35.4)")
        axes[0].axhline(y=55.4, linestyle="--", alpha=0.6, label="USG/Unhealthy (55.4)")
        axes[0].set_ylabel("PM2.5 (µg/m³)")
        axes[0].set_title("Daily Average PM2.5 Across Pakistan")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(stats_df["date"], stats_df["max"], linewidth=2, label="Daily Max")
        axes[1].plot(stats_df["date"], stats_df["p95"], linewidth=2, label="95th Percentile")
        axes[1].axhline(y=125.4, linestyle="--", alpha=0.7, label="Very Unhealthy (125.4)")
        axes[1].axhline(y=225.4, linestyle="--", alpha=0.7, label="Hazardous (225.4)")
        axes[1].set_ylabel("PM2.5 (µg/m³)")
        axes[1].set_title("Daily Maximum PM2.5")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(stats_df["date"], stats_df["good_pct"], linewidth=2, label="Good (<9.0)")
        axes[2].plot(stats_df["date"], stats_df["unhealthy_pct"], linewidth=2, label="Unhealthy+ (>55.4)")
        axes[2].set_ylabel("Percentage (%)")
        axes[2].set_xlabel("Date")
        axes[2].set_title("Air Quality Distribution")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        out_png = output_dir / "pm25_time_series.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()

        out_csv = output_dir / "daily_pm25_statistics.csv"
        stats_df.to_csv(out_csv, index=False)

        print(f"[video] Saved: {out_png}")
        print(f"[video] Saved: {out_csv}")


def ensure_store_exists(
    start_date: datetime,
    end_date: datetime,
    config: GridStoreConfig,
    datasets_dir: Path,
    history_days: int = 21,
    tropomi_window: int = 5,
    verbose: bool = False,
) -> None:
    """
    Ensure the feature store has all required partitions for the date range.
    Build any missing partitions.
    """
    print(f"[store] Checking feature store for {start_date.date()} to {end_date.date()}...")

    # Check which dates are missing for each stage
    missing_dates = {stage: [] for stage in VALID_STAGES}

    current = start_date
    while current <= end_date:
        dt = current.date()
        for stage in VALID_STAGES:
            if not config.partition_exists(stage, dt):
                missing_dates[stage].append(dt)
        current += timedelta(days=1)

    # Report missing dates
    total_missing = sum(len(dates) for dates in missing_dates.values())
    if total_missing == 0:
        print("[store] All partitions exist, using cached store.")
        return

    for stage, dates in missing_dates.items():
        if dates:
            print(f"[store] {stage}: {len(dates)} missing partitions")

    print(f"[store] Building {total_missing} missing partitions...")

    # Build static grid first
    build_static_grid(config, overwrite=False)

    # Setup paths
    met_dir = datasets_dir / "features_met"
    aod_dir = datasets_dir / "MCD19A2.061"
    tropomi_dir = datasets_dir / "tropomi_pakistan_2020_2025"
    geos_cf_dir = datasets_dir / "geos_cf_pakistan_2020_2025"

    # Build missing MET partitions
    if missing_dates["met"]:
        min_date = min(missing_dates["met"])
        max_date = max(missing_dates["met"])
        print(f"[store] Building MET: {min_date} to {max_date}")
        build_met_store(
            min_date, max_date, config, met_dir,
            history_days=history_days,
            overwrite=False,
            verbose=verbose,
        )

    # Build missing AOD partitions
    if missing_dates["aod"]:
        min_date = min(missing_dates["aod"])
        max_date = max(missing_dates["aod"])
        print(f"[store] Building AOD: {min_date} to {max_date}")
        build_aod_store(
            min_date, max_date, config, aod_dir,
            overwrite=False,
            verbose=verbose,
        )

    # Build missing TROPOMI partitions
    if missing_dates["tropomi"]:
        min_date = min(missing_dates["tropomi"])
        max_date = max(missing_dates["tropomi"])
        print(f"[store] Building TROPOMI: {min_date} to {max_date}")
        build_tropomi_store(
            min_date, max_date, config, tropomi_dir, geos_cf_dir,
            tropomi_window=tropomi_window,
            overwrite=False,
            verbose=verbose,
        )

    print("[store] Feature store build complete.")


def generate_predictions_from_store(
    start_date: datetime,
    end_date: datetime,
    model_path: Path,
    config: GridStoreConfig,
    verbose: bool = False,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Generate predictions using the feature store instead of raw data.
    """
    print(f"[store] Generating predictions from store: {start_date.date()} to {end_date.date()}")

    # Load model
    payload = joblib.load(model_path)
    if isinstance(payload, dict):
        model = payload.get("model")
        feature_cols = payload.get("feature_cols", [])
        train_medians = payload.get("train_medians")
    else:
        model = payload
        feature_cols = []
        train_medians = None

    # Load static grid for coordinates
    static_df = pd.read_parquet(config.static_path())
    lats = static_df["lat"].values
    lons = static_df["lon"].values
    n_cells = len(static_df)

    predictions: Dict[str, np.ndarray] = {}

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        dt = current.date()
        print(f"[store] Processing {date_str}...")

        try:
            # Read all stages
            df_met = GridPartitionReader(config, "met").read_single_date(dt)
            df_aod = GridPartitionReader(config, "aod").read_single_date(dt)
            df_tropomi = GridPartitionReader(config, "tropomi").read_single_date(dt)

            # Join stages
            master = join_grid_stages(df_met, df_aod, df_tropomi)

            # Select features
            if feature_cols:
                available = [f for f in feature_cols if f in master.columns]
                X = master[available].copy()
            else:
                # Use all numeric columns except coordinates
                exclude = ["cell_id", "lat", "lon", "date", "row", "col"]
                X = master[[c for c in master.columns if c not in exclude]].copy()

            # Impute missing values
            if train_medians is not None:
                # Handle both dict and Series formats
                if isinstance(train_medians, dict):
                    for col in X.columns:
                        if col in train_medians:
                            X[col] = X[col].fillna(train_medians[col])
                else:
                    for col in X.columns:
                        if col in train_medians.index:
                            X[col] = X[col].fillna(train_medians[col])
            X = X.fillna(0.0)

            # Reorder columns if needed
            if feature_cols:
                X = X.reindex(columns=feature_cols, fill_value=0.0)

            # Predict
            pred = model.predict(X)
            predictions[date_str] = np.asarray(pred, dtype="float32").reshape(-1)

            if verbose:
                valid = pred[np.isfinite(pred)]
                print(f"  -> mean={valid.mean():.1f}, max={valid.max():.1f}")

        except Exception as e:
            print(f"[store] ERROR {date_str}: {e}")
            predictions[date_str] = np.full(n_cells, np.nan, dtype="float32")

        current += timedelta(days=1)

    return predictions, lats, lons


def main():
    parser = argparse.ArgumentParser(description="Create PM2.5 video animation for Pakistan (turbo continuous colormap)")
    parser.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", required=True, help="End date (YYYY-MM-DD)")

    parser.add_argument("--model", default="best_model_weight/production_tuned_model_20260103_033834.joblib")
    parser.add_argument("--master_feature_dir", default="../inference")
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--history_days", type=int, default=21)
    parser.add_argument("--tropomi_window", type=int, default=5, help="Odd window size for TROPOMI (5 or 7 recommended)")

    parser.add_argument("--output_video", default="output/pm25_pakistan.mp4")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--create_stills", action="store_true")
    parser.add_argument("--stills_dir", default="output/still_images")
    parser.add_argument("--summary_dir", default="output/summary_plots")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output for debugging")

    # Turbo scaling controls
    parser.add_argument("--vmin", type=float, default=20.0, help="Color scale min (default 0)")
    parser.add_argument("--vmax", type=float, default=180.0, help="Color scale max (default 300)")
    parser.add_argument(
        "--autoscale",
        choices=["fixed", "period", "frame"],
        default="fixed",
        help="fixed: use vmin/vmax; period: (currently keeps user bounds for stability); frame: per-frame percentile scaling (can flicker).",
    )

    # Feature store options
    parser.add_argument("--store_path", default="derived/feature_store/grid",
                       help="Path to grid feature store (default: derived/feature_store/grid)")
    parser.add_argument("--datasets_dir", default="datasets",
                       help="Path to raw datasets directory (for building store)")
    parser.add_argument("--no_store", action="store_true",
                       help="Disable feature store, use direct inference (slower)")

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    # Decide whether to use feature store
    use_store = not args.no_store
    config = GridStoreConfig(base_path=args.store_path)
    datasets_dir = Path(args.datasets_dir)

    if use_store:
        # Ensure store exists, build missing partitions
        ensure_store_exists(
            start_date, end_date, config, datasets_dir,
            history_days=args.history_days,
            tropomi_window=args.tropomi_window,
            verbose=args.verbose,
        )

        # Generate predictions from store
        predictions_dict, lats, lons = generate_predictions_from_store(
            start_date, end_date,
            Path(args.model),
            config,
            verbose=args.verbose,
        )

        # Create a minimal predictor just for grid shape info
        predictor = PakistanPM25Predictor(
            model_path=args.model,
            master_feature_dir=args.master_feature_dir,
            grid_resolution=args.resolution,
            tropomi_window=args.tropomi_window,
            history_days=args.history_days,
        )
    else:
        # Use direct inference (original behavior)
        predictor = PakistanPM25Predictor(
            model_path=args.model,
            master_feature_dir=args.master_feature_dir,
            grid_resolution=args.resolution,
            tropomi_window=args.tropomi_window,
            history_days=args.history_days,
        )

        video_gen = PM25VideoGenerator(
            predictor,
            vmin=args.vmin,
            vmax=args.vmax,
            autoscale=args.autoscale,
        )
        predictions_dict, lats, lons = video_gen.generate_predictions_for_period(
            start_date, end_date, verbose=args.verbose
        )

    # Create video generator (need predictor for grid_shape)
    video_gen = PM25VideoGenerator(
        predictor,
        vmin=args.vmin,
        vmax=args.vmax,
        autoscale=args.autoscale,
    )

    out_vid = video_gen.create_animation(predictions_dict, lats, lons, args.output_video, fps=args.fps)

    video_gen.create_summary_plots(predictions_dict, output_dir=args.summary_dir)

    if args.create_stills:
        stills_dir = Path(args.stills_dir)
        stills_dir.mkdir(exist_ok=True)
        for date_str, preds in predictions_dict.items():
            fig = video_gen.create_static_map(
                preds,
                lats,
                lons,
                date_str,
                output_file=stills_dir / f"pm25_{date_str}.png",
            )
            plt.close(fig)

    print("\n=== DONE ===")
    print(f"Video  : {out_vid}")
    print(f"Summary: {args.summary_dir}")
    if args.create_stills:
        print(f"Stills : {args.stills_dir}")


if __name__ == "__main__":
    main()