"""Figures that show the synthetic data and its planted truth."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from mbc.plotting import BASE_COLOR, CHANNEL_COLORS, TEXT_SECONDARY, apply_style, euro_k
from mbc.simulate.generate import CHANNEL_COLS, SimResult
from mbc.transforms import geometric_adstock, logistic_saturation

CHANNELS = list(CHANNEL_COLORS)


def plot_overview(sim: SimResult, path: Path) -> None:
    """Sales on top, one spend panel per channel below, sharing the time axis."""
    df = sim.data
    holdout_start = np.datetime64(sim.truth["holdout_start"])
    fig, axes = plt.subplots(5, 1, figsize=(10, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1, 1, 1]})

    ax = axes[0]
    ax.plot(df["date"], df["sales"], color="#0b0b0b", lw=1.6)
    holiday = df.loc[(df["black_friday"] == 1) | (df["christmas"] == 1)]
    ax.scatter(holiday["date"], holiday["sales"], s=24, color="#0b0b0b", zorder=3)
    promo = df.loc[df["promo"] == 1]
    ax.scatter(promo["date"], promo["sales"], s=24, facecolor="white",
               edgecolor="#0b0b0b", zorder=3, label="Promo week")
    ax.set_title("Weekly sales (filled dots: holiday weeks, hollow: promo weeks)")
    ax.yaxis.set_major_formatter(FuncFormatter(euro_k))

    for ax, c in zip(axes[1:], CHANNELS, strict=True):
        ax.fill_between(df["date"], df[CHANNEL_COLS[c]], step="mid", color=CHANNEL_COLORS[c],
                        alpha=0.85, lw=0)
        ax.set_title(f"{c.upper() if c == 'tv' else c.capitalize()} spend", fontsize=10)
        ax.yaxis.set_major_formatter(FuncFormatter(euro_k))

    for ax in axes:
        ax.axvspan(holdout_start, df["date"].iloc[-1] + np.timedelta64(7, "D"),
                   color="#ecebe7", zorder=0, lw=0)
    axes[0].text(holdout_start, axes[0].get_ylim()[1], "  holdout", va="top",
                 color=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_true_contributions(sim: SimResult, path: Path) -> None:
    """Stacked area: base sales, then each channel's true contribution on top."""
    comp = sim.components
    base = comp.drop(columns=CHANNELS).sum(axis=1) - comp["noise"]
    layers = [base] + [comp[c] for c in CHANNELS]
    colors = [BASE_COLOR] + [CHANNEL_COLORS[c] for c in CHANNELS]
    labels = ["Base (trend, season, holidays, promo)"] + [
        f"{'TV' if c == 'tv' else c.capitalize()}  (ROI {sim.truth['channels'][c]['roi']:.1f})"
        for c in CHANNELS
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.stackplot(comp.index, *layers, colors=colors, labels=labels, edgecolor="white", lw=0.6)
    ax.plot(comp.index, comp.sum(axis=1), color="#0b0b0b", lw=1, label="Observed sales")
    ax.set_title(f"True decomposition of sales: media drives "
                 f"{sim.truth['media_share_of_sales']:.0%} of the total")
    ax.yaxis.set_major_formatter(FuncFormatter(euro_k))
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", ncols=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_response_curves(sim: SimResult, path: Path) -> None:
    """Weekly contribution vs steady weekly spend, with the average spend level marked.

    The slope of the chord from 0 to the dot is average ROI; the slope of the curve
    at the dot is marginal ROI. Where they differ, the channel is saturating.
    """
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, c in zip(axes, CHANNELS, strict=True):
        t = sim.truth["channels"][c]
        spend = np.linspace(0, 1.6 * t["spend_scale"], 200)
        # Steady-state: with constant spend, normalised adstock equals the spend itself.
        contrib = t["beta"] * logistic_saturation(spend / t["spend_scale"], t["lam"])
        mean_spend = t["total_spend"] / len(sim.data)
        mean_contrib = t["beta"] * logistic_saturation(mean_spend / t["spend_scale"], t["lam"])
        col = CHANNEL_COLORS[c]
        ax.plot(spend, contrib, color=col)
        ax.plot([0, mean_spend], [0, mean_contrib], color=col, ls=":", lw=1.2)
        ax.scatter([mean_spend], [mean_contrib], s=40, color=col, edgecolor="white", lw=1.5,
                   zorder=3)
        ax.plot([0, spend[-1]], [0, spend[-1]], color=TEXT_SECONDARY, lw=0.8, ls="--")
        ax.set_ylim(0, contrib.max() * 1.15)
        ax.set_title(f"{'TV' if c == 'tv' else c.capitalize()}")
        ax.text(0.03, 0.97, f"avg ROI {t['roi']:.2f}\nmarginal {t['marginal_roi']:.2f}",
                transform=ax.transAxes, va="top", fontsize=8, color=TEXT_SECONDARY)
        ax.xaxis.set_major_formatter(FuncFormatter(euro_k))
        ax.yaxis.set_major_formatter(FuncFormatter(euro_k))
        ax.set_xlabel("weekly spend")
    axes[0].set_ylabel("weekly sales contribution")
    fig.suptitle("True response curves (dot: average weekly spend; dashed: break-even, ROI = 1)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_adstock_decay(sim: SimResult, path: Path) -> None:
    """How a single week of spend is spread over the following weeks, per channel."""
    l_max = sim.truth["config"]["l_max"]
    impulse = np.r_[1.0, np.zeros(l_max + 2)]
    weeks = np.arange(len(impulse))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    width = 0.2
    for i, c in enumerate(CHANNELS):
        t = sim.truth["channels"][c]
        w = geometric_adstock(impulse, t["alpha"], l_max)
        ax.bar(weeks + (i - 1.5) * width, w, width=width * 0.9, color=CHANNEL_COLORS[c],
               label=f"{'TV' if c == 'tv' else c.capitalize()}  α={t['alpha']}, "
                     f"half-life {t['adstock_half_life_weeks']:.1f} wk")
    ax.set_xlabel("weeks after spend")
    ax.set_ylabel("share of the effect")
    ax.set_xticks(weeks)
    ax.set_title("Adstock: how one week's spend is spread over time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_all(sim: SimResult, out_dir: Path) -> list[Path]:
    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sim_overview.png": plot_overview,
        "sim_true_contributions.png": plot_true_contributions,
        "sim_response_curves.png": plot_response_curves,
        "sim_adstock_decay.png": plot_adstock_decay,
    }
    out = []
    for name, fn in paths.items():
        fn(sim, out_dir / name)
        out.append(out_dir / name)
    return out
