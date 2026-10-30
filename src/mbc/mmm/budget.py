"""Recommend a budget split from the fitted MMM and check it against the truth.

    python -m mbc.mmm.budget                 # uses models/informed.nc
"""

from __future__ import annotations

import argparse
import json

import arviz as az  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

import mbc  # noqa: F401
from mbc.mmm.data import CHANNELS, ROOT, SPEND_COLUMNS, load_synthetic, split_holdout  # noqa: E402
from mbc.mmm.optimize import Curves, optimise, relative_bounds, uplift_draws  # noqa: E402
from mbc.plotting import CHANNEL_COLORS, TEXT_SECONDARY, apply_style, euro_k  # noqa: E402

WEEKS_PER_QUARTER = 13
LABEL = {"tv": "TV", "search": "Search", "social": "Social", "radio": "Radio"}


def model_curves(idata: az.InferenceData, n_draws: int = 1000, seed: int = 0) -> Curves:
    post = az.extract(idata, group="posterior", num_samples=n_draws, rng=seed)
    target_scale = float(idata.constant_data["target_scale"])
    beta = post["saturation_beta"].sel(channel=SPEND_COLUMNS).transpose("sample", "channel")
    lam = post["saturation_lam"].sel(channel=SPEND_COLUMNS).transpose("sample", "channel")
    return Curves(channels=CHANNELS, beta=beta.values * target_scale, lam=lam.values,
                  scale=idata.constant_data["channel_scale"].sel(channel=SPEND_COLUMNS).values)


def true_curves(truth: dict) -> Curves:
    ch = truth["channels"]
    return Curves(channels=CHANNELS, beta=[[ch[c]["beta"] for c in CHANNELS]],
                  lam=[[ch[c]["lam"] for c in CHANNELS]],
                  scale=np.array([ch[c]["spend_scale"] for c in CHANNELS]))


def hdi(x: np.ndarray, prob: float = 0.9) -> tuple[float, float]:
    lo, hi = az.hdi(np.asarray(x), hdi_prob=prob)
    return float(lo), float(hi)


def run(model_name: str = "informed", band: float = 0.5) -> dict:
    apply_style()
    out = ROOT / "reports" / "budget" / model_name
    out.mkdir(parents=True, exist_ok=True)
    truth = json.loads((ROOT / "data" / "ground_truth.json").read_text())
    idata = az.from_netcdf(ROOT / "models" / f"{model_name}.nc")
    train = split_holdout(load_synthetic(), truth["config"]["holdout_weeks"]).train

    # Current plan: average weekly spend per channel over the training window.
    current = pd.Series(train[SPEND_COLUMNS].mean().to_numpy(), index=CHANNELS)
    budget = float(current.sum())
    bounds = relative_bounds(current, band)

    model, real = model_curves(idata), true_curves(truth)
    rec = optimise(model, budget, bounds)  # what the model recommends
    best = optimise(real, budget, bounds)  # what we'd recommend if we knew the truth

    cur, new, opt = current.to_numpy(), rec.spend.to_numpy(), best.spend.to_numpy()
    promised = uplift_draws(model, new, cur)  # the model's own forecast of the gain
    delivered = float(uplift_draws(real, new, cur)[0])  # what the truth says it gains
    achievable = float(uplift_draws(real, opt, cur)[0])  # the best possible gain

    alloc = pd.DataFrame({"current": current, "model_recommends": rec.spend,
                          "truth_optimal": best.spend})
    alloc["change_pct"] = alloc["model_recommends"] / alloc["current"] - 1
    mroi_model = model.marginal(cur)
    mroi = pd.DataFrame({
        "model_mean": mroi_model.mean(axis=0),
        "model_hdi_low": [hdi(mroi_model[:, i])[0] for i in range(len(CHANNELS))],
        "model_hdi_high": [hdi(mroi_model[:, i])[1] for i in range(len(CHANNELS))],
        "true": real.marginal(cur)[0],
    }, index=CHANNELS)

    q = WEEKS_PER_QUARTER
    lo, hi = hdi(promised)
    summary = {
        "weekly_budget": budget,
        "quarterly_budget": budget * q,
        "bounds_band": band,
        "model_promised_uplift_q": float(promised.mean() * q),
        "model_promised_uplift_q_hdi90": [lo * q, hi * q],
        "prob_model_uplift_positive": float((promised > 0).mean()),
        "true_uplift_of_model_plan_q": delivered * q,
        "true_best_uplift_q": achievable * q,
        "share_of_best_captured": delivered / achievable if achievable > 0 else None,
        "optimiser_converged": rec.success and best.success,
    }
    alloc.to_csv(out / "allocation.csv")
    mroi.to_csv(out / "marginal_roi.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot_allocation(alloc, summary, out / "allocation.png")
    _plot_marginal(mroi, out / "marginal_roi.png")
    return {"summary": summary, "allocation": alloc, "marginal_roi": mroi}


def _plot_allocation(alloc: pd.DataFrame, s: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(CHANNELS))
    w = 0.26
    series = [("current", "Current", "#d6d5d0"), ("model_recommends", "Model recommends", None),
              ("truth_optimal", "Optimal if we knew the truth", "#0b0b0b")]
    for i, (col, label, color) in enumerate(series):
        colors = [CHANNEL_COLORS[c] for c in CHANNELS] if color is None else color
        ax.bar(x + (i - 1) * w, alloc[col] * WEEKS_PER_QUARTER, width=w * 0.92, color=colors,
               label=label, alpha=1.0 if color != "#0b0b0b" else 0.85)
    ax.set_xticks(x, [LABEL[c] for c in CHANNELS])
    ax.yaxis.set_major_formatter(FuncFormatter(euro_k))
    ax.set_ylabel("spend per quarter")
    lo, hi = s["model_promised_uplift_q_hdi90"]
    ax.set_title(
        f"Same €{s['quarterly_budget'] / 1e3:,.0f}k per quarter, split differently. Model promises "
        f"+€{s['model_promised_uplift_q'] / 1e3:,.0f}k (90%: {lo / 1e3:,.0f} to {hi / 1e3:,.0f}k); "
        f"truth: +€{s['true_uplift_of_model_plan_q'] / 1e3:,.0f}k", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#d6d5d0"),
               plt.Rectangle((0, 0), 1, 1, color=CHANNEL_COLORS["tv"]),
               plt.Rectangle((0, 0), 1, 1, color="#0b0b0b", alpha=0.85)]
    ax.legend(handles, [s_[1] for s_ in series], loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_marginal(m: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    y = np.arange(len(CHANNELS))[::-1]
    for yi, c in zip(y, CHANNELS, strict=True):
        r = m.loc[c]
        ax.plot([r["model_hdi_low"], r["model_hdi_high"]], [yi, yi], color=CHANNEL_COLORS[c],
                lw=6, solid_capstyle="round", alpha=0.8)
        ax.scatter(r["model_mean"], yi, s=50, color=CHANNEL_COLORS[c], edgecolor="white",
                   lw=1.5, zorder=3)
        ax.scatter(r["true"], yi, marker="|", s=300, color="#0b0b0b", lw=2, zorder=4)
    ax.axvline(1, color=TEXT_SECONDARY, lw=0.8, ls="--")
    ax.set_yticks(y, [LABEL[c] for c in CHANNELS])
    ax.set_xlabel("sales from the next € at current spend")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_title("Marginal ROI at current spend: model (90% HDI) vs truth (black tick)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="informed")
    p.add_argument("--band", type=float, default=0.5)
    a = p.parse_args()
    r = run(a.model, a.band)
    pd.set_option("display.precision", 2)
    print(r["allocation"].assign(change_pct=lambda d: (100 * d["change_pct"]).round(0)).round(0))
    print(r["marginal_roi"].round(2))
    print(json.dumps(r["summary"], indent=2))


if __name__ == "__main__":
    main()
