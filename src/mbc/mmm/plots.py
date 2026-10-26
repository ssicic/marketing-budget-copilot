"""Figures for a fitted MMM."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.ticker import FuncFormatter

from mbc.mmm.data import CHANNELS
from mbc.plotting import CHANNEL_COLORS, TEXT_SECONDARY, TRUTH_COLOR, euro_k

LABEL = {"tv": "TV", "search": "Search", "social": "Social", "radio": "Radio"}


def plot_predictive(dates_train, y_train, pi_train: pd.DataFrame, dates_hold, y_hold,
                    pi_hold: pd.DataFrame, title: str, path: Path,
                    band_label: str = "fitted (90% band)") -> None:
    """Observed sales with the posterior predictive 90% band, in-sample and holdout."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for d, pi, color in [(dates_train, pi_train, "#2a78d6"), (dates_hold, pi_hold, "#eb6834")]:
        if pi is None:
            continue
        ax.fill_between(d, pi["lower"], pi["upper"], color=color, alpha=0.25, lw=0)
        ax.plot(d, pi["mean"], color=color, lw=1.5)
    ax.plot(dates_train, y_train, color=TRUTH_COLOR, lw=1, label="observed")
    if y_hold is not None:
        ax.plot(dates_hold, y_hold, color=TRUTH_COLOR, lw=1)
        ax.axvline(dates_hold.iloc[0], color=TEXT_SECONDARY, ls="--", lw=0.8)
        ax.text(dates_hold.iloc[0], ax.get_ylim()[1], "  holdout", va="top", color=TEXT_SECONDARY)
    ax.yaxis.set_major_formatter(FuncFormatter(euro_k))
    ax.set_title(title)
    ax.plot([], [], color="#2a78d6", label=band_label)
    if pi_hold is not None:
        ax.plot([], [], color="#eb6834", label="forecast (90% band)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_contributions_vs_truth(contrib: xr.DataArray, components: pd.DataFrame, dates,
                                path: Path) -> None:
    """Estimated weekly contribution (mean + 90% band) against the true contribution."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    flat = contrib.stack(sample=("chain", "draw"))
    for ax, c in zip(axes, CHANNELS, strict=True):
        x = flat.sel(channel=f"spend_{c}").transpose("sample", "date").values
        lo, hi = np.quantile(x, [0.05, 0.95], axis=0)
        col = CHANNEL_COLORS[c]
        ax.fill_between(dates, lo, hi, color=col, alpha=0.3, lw=0)
        ax.plot(dates, x.mean(axis=0), color=col, lw=1.5, label="estimated")
        ax.plot(dates, components.loc[pd.DatetimeIndex(dates), c], color=TRUTH_COLOR, lw=1,
                ls="--", label="true")
        ax.set_title(LABEL[c], fontsize=10)
        ax.yaxis.set_major_formatter(FuncFormatter(euro_k))
    axes[0].legend(loc="upper left", ncols=2)
    fig.suptitle("Weekly channel contribution: estimated (90% band) vs truth", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_recovery(table: pd.DataFrame, path: Path) -> None:
    """Forest plot: 90% HDI and mean per channel, true value as a black tick."""
    params = [
        ("adstock_alpha", "Adstock decay α"),
        ("half_saturation_spend", "Half-saturation spend"),
        ("beta_eur", "Max weekly effect (β)"),
        ("roi", "ROI (training window)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (p, label) in zip(axes, params, strict=True):
        sub = table[table["parameter"] == p].set_index("channel").loc[CHANNELS]
        y = np.arange(len(sub))[::-1]
        for yi, (c, r) in zip(y, sub.iterrows(), strict=True):
            col = CHANNEL_COLORS[c]
            ax.plot([r["hdi_low"], r["hdi_high"]], [yi, yi], color=col, lw=3,
                    solid_capstyle="round")
            ax.scatter(r["mean"], yi, s=36, color=col, edgecolor="white", lw=1.5, zorder=3)
            ax.scatter(r["true"], yi, marker="|", s=220, color=TRUTH_COLOR, lw=2, zorder=4)
        ax.set_yticks(y, [LABEL[c] for c in sub.index])
        ax.set_title(label)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        if p in ("half_saturation_spend", "beta_eur"):
            ax.xaxis.set_major_formatter(FuncFormatter(euro_k))
    axes[-1].scatter([], [], marker="|", s=120, color=TRUTH_COLOR, label="true value")
    axes[-1].plot([], [], color=TEXT_SECONDARY, lw=3, label="90% HDI")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Parameter recovery: does the 90% HDI contain the truth?", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roi(roi_table: pd.DataFrame, truth_roi: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    y = np.arange(len(CHANNELS))[::-1]
    for yi, c in zip(y, CHANNELS, strict=True):
        r = roi_table.loc[c]
        col = CHANNEL_COLORS[c]
        ax.plot([r["roi_hdi_low"], r["roi_hdi_high"]], [yi, yi], color=col, lw=6,
                solid_capstyle="round", alpha=0.8)
        ax.scatter(r["roi_mean"], yi, s=50, color=col, edgecolor="white", lw=1.5, zorder=3)
        ax.scatter(truth_roi[c], yi, marker="|", s=300, color=TRUTH_COLOR, lw=2, zorder=4)
        ax.text(r["roi_hdi_high"], yi, f"  {r['roi_mean']:.2f} [{r['roi_hdi_low']:.2f}, "
                f"{r['roi_hdi_high']:.2f}]", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.axvline(1, color=TEXT_SECONDARY, lw=0.8, ls="--")
    ax.set_yticks(y, [LABEL[c] for c in CHANNELS])
    ax.set_xlabel("ROI (sales per € spent)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_title("Channel ROI: mean and 90% HDI (black tick = truth; dashed = break-even)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_prior_vs_posterior(prior: xr.Dataset, posterior: xr.Dataset, var: str, title: str,
                            path: Path) -> None:
    """Overlaid densities per channel: how much did the data move each prior?"""
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.6), sharey=False)
    for ax, c in zip(axes, CHANNELS, strict=True):
        col = CHANNEL_COLORS[c]
        pr = prior[var].sel(channel=f"spend_{c}").values.ravel()
        po = posterior[var].sel(channel=f"spend_{c}").values.ravel()
        hi = np.quantile(np.r_[pr, po], 0.99)
        bins = np.linspace(0, hi, 50)
        ax.hist(pr, bins=bins, density=True, color="#d6d5d0", label="prior")
        ax.hist(po, bins=bins, density=True, color=col, alpha=0.8, label="posterior")
        ax.set_title(LABEL[c])
        ax.set_yticks([])
    axes[0].legend(fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
