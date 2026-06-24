#!/usr/bin/env python3
"""
Figure 3 (paper_v2): Local PM history available vs. no local PM history MAE for three model variants.

Backbone / Standard temporal / Support-aware dropout,
evaluated on the 2025 forward-time test year.

Output: paper_v2/figures/fig4_model_safety.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PAPER_FIG_DIR = HERE.parent / "figures"
OUT_PDF = PAPER_FIG_DIR / "fig4_model_safety.pdf"

# ---------------------------------------------------------------------------
# Style (shared with other paper figures)
# ---------------------------------------------------------------------------
OBS_COLOR  = "#1F3A66"   # deep navy  — monitored regime
PRED_COLOR = "#D9480F"   # warm orange — unmonitored regime
ADMIN_GRAY = "#A6AEB6"
TEXT_DARK  = "#37414b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "savefig.dpi": 400,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# ---------------------------------------------------------------------------
# Data (from Table 5, 08_results.tex:96-98)
# ---------------------------------------------------------------------------
GROUPS = ["Backbone", "Standard temporal", "Support-aware dropout"]
MON    = [22.9, 14.2, 14.3]   # monitored-regime MAE
UNMON  = [22.9, 32.4, 23.2]   # unmonitored-regime (PM-history masked) MAE
BACKBONE_REF = 22.9

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def main():
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(GROUPS))
    w = 0.26

    fig, ax = plt.subplots(figsize=(4.0, 2.4))

    b1 = ax.bar(x - w / 2, MON, width=w, color=OBS_COLOR,
                label="With local PM history", zorder=3)
    b2 = ax.bar(x + w / 2, UNMON, width=w, color=PRED_COLOR,
                label="No local PM history", zorder=3)

    # Value labels above each bar
    for bars, vals in [(b1, MON), (b2, UNMON)]:
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.35,
                f"{v:.1f}",
                ha="center", va="bottom",
                fontsize=9.5, color=TEXT_DARK,
            )

    # Backbone reference line (coincides with backbone bars; self-evident)
    ax.axhline(BACKBONE_REF, color=ADMIN_GRAY, linestyle=":",
               linewidth=0.9, alpha=0.9, zorder=2)

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Backbone", "Standard\ntemporal", "Support-aware\ndropout"],
        color=TEXT_DARK, fontsize=9,
    )
    ax.set_ylabel(r"MAE ($\mu$g m$^{-3}$)", color=TEXT_DARK, fontsize=10)
    ax.set_ylim(0, 37)
    ax.set_xlim(-0.55, len(GROUPS) - 0.45)
    ax.tick_params(axis="x", colors=TEXT_DARK, length=0)
    ax.tick_params(axis="y", colors=TEXT_DARK, labelsize=9)

    # Spines
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(ADMIN_GRAY)
        ax.spines[spine].set_linewidth(0.6)

    # Grid (horizontal only, behind bars)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.grid(axis="y", color=ADMIN_GRAY, alpha=0.30,
            linewidth=0.4, linestyle=":")

    # Legend
    ax.legend(loc="upper left", frameon=False, fontsize=8,
              handlelength=1.0, handletextpad=0.4,
              labelcolor=TEXT_DARK)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT_PDF}")


if __name__ == "__main__":
    main()
