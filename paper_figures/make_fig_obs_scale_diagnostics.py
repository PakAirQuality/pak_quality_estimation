#!/usr/bin/env python3
"""
Observation-scale diagnostics for winter LCS evaluation (paper_v2).

Same-day LCS-LCS pairwise MAE by distance band, winter vs non-winter
(from experiments/05_label_noise_floor).

Output: paper_v2/figures/fig_obs_scale_diagnostics.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PAPER_FIG_DIR = HERE.parent / "figures"
OUT_PDF = PAPER_FIG_DIR / "fig_obs_scale_diagnostics.pdf"

# Shared paper-style palette (matches make_fig_model_safety.py)
NAVY = "#1F3A66"
ORANGE = "#D9480F"
ADMIN_GRAY = "#A6AEB6"
TEXT_DARK = "#37414b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "savefig.dpi": 400,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# ----------------------------------------------------------------------
# Same-day LCS-LCS pairwise MAE by distance band
# (from experiments/05_label_noise_floor.md)
# ----------------------------------------------------------------------
BANDS = ["0–1 km", "1–2 km", "2–5 km", "5–10 km"]
WINTER_MAE = [26.2, 28.9, 41.6, 45.5]
NONWINTER_MAE = [11.4, 16.2, 19.3, 21.4]


def _style_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(ADMIN_GRAY)
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(axis="x", colors=TEXT_DARK, length=0)
    ax.tick_params(axis="y", colors=TEXT_DARK, labelsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=ADMIN_GRAY, alpha=0.30, linewidth=0.4, linestyle=":")


def _value_labels(ax, bars, values, dy=0.6):
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dy,
            f"{v:.1f}",
            ha="center", va="bottom",
            fontsize=7.8, color=TEXT_DARK,
        )


def main():
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.0, 2.8))

    x = np.arange(len(BANDS))
    w = 0.36
    bars_win = ax.bar(x - w / 2, WINTER_MAE, width=w, color=NAVY,
                       label="Winter (DJF)", zorder=3)
    bars_non = ax.bar(x + w / 2, NONWINTER_MAE, width=w, color=ORANGE,
                       label="Non-winter", zorder=3)
    _value_labels(ax, bars_win, WINTER_MAE)
    _value_labels(ax, bars_non, NONWINTER_MAE)
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS, color=TEXT_DARK, fontsize=8.5)
    ax.set_ylabel(r"Pairwise MAE ($\mu$g m$^{-3}$)", color=TEXT_DARK, fontsize=9)
    ax.set_ylim(0, 55)
    ax.set_xlim(-0.5, 3.5)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    _style_axes(ax)
    ax.set_title("Nearby LCS disagreement", color=TEXT_DARK,
                  fontsize=10.5, loc="left", pad=6)
    ax.legend(loc="upper left", frameon=False, fontsize=8,
               handlelength=1.0, handletextpad=0.4,
               labelcolor=TEXT_DARK)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT_PDF}")


if __name__ == "__main__":
    main()
