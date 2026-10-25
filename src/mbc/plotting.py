"""Shared chart style. Each channel keeps the same colour in every figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CHANNEL_COLORS = {
    "tv": "#2a78d6",
    "search": "#eb6834",
    "social": "#1baf7a",
    "radio": "#eda100",
}
BASE_COLOR = "#b9b8b2"
TRUTH_COLOR = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": "#d6d5d0",
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#ecebe7",
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "font.size": 9,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def euro_k(x, _pos=None) -> str:
    return f"€{x / 1000:,.0f}k"
