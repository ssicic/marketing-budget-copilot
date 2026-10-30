"""Run the geo-lift experiment and compare truth, DiD and the MMM.

    python -m mbc.experiments
"""

from __future__ import annotations

import json

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mbc.experiments.did import (
    did_estimate,
    event_study,
    lift_test_for_mmm,
    permutation_pvalue,
)
from mbc.experiments.geo import GeoConfig, simulate_geo
from mbc.mmm.data import ROOT
from mbc.plotting import CHANNEL_COLORS, TEXT_SECONDARY, TRUTH_COLOR, apply_style
from mbc.simulate import simulate
from mbc.transforms import geometric_adstock, logistic_saturation

OUT = ROOT / "reports" / "experiments"
TREATED_COLOR, CONTROL_COLOR = "#1baf7a", "#52514e"


def mmm_implied_lift(idata: az.InferenceData, social_spend: np.ndarray, cfg: GeoConfig,
                     treated_weight: float, n_draws: int = 1000, seed: int = 0) -> dict:
    """What the fitted (national) MMM says this test should produce, per posterior draw.

    Same logic as the truth: the treated regions are a share `treated_weight` of the
    national business, so their lift is that share of the national response change.
    """
    post = az.extract(idata, group="posterior", num_samples=n_draws, rng=seed)
    col = "spend_social"
    alpha = post["adstock_alpha"].sel(channel=col).values
    lam = post["saturation_lam"].sel(channel=col).values
    beta = post["saturation_beta"].sel(channel=col).values * float(
        idata.constant_data["target_scale"])
    scale = float(idata.constant_data["channel_scale"].sel(channel=col))
    window = np.zeros(len(social_spend), bool)
    window[cfg.start_week : cfg.start_week + cfg.test_weeks] = True
    boosted = np.where(window, social_spend * cfg.lift_factor, social_spend)

    in_window, total = np.empty(len(alpha)), np.empty(len(alpha))
    for i in range(len(alpha)):
        base = beta[i] * logistic_saturation(geometric_adstock(social_spend / scale, alpha[i]),
                                             lam[i])
        test = beta[i] * logistic_saturation(geometric_adstock(boosted / scale, alpha[i]), lam[i])
        diff = treated_weight * (test - base)
        in_window[i], total[i] = diff[window].sum(), diff.sum()
    return {"in_window": in_window, "total": total}


def _hdi(x, prob=0.9):
    lo, hi = az.hdi(np.asarray(x), hdi_prob=prob)
    return float(lo), float(hi)


def plot_parallel_trends(panel: pd.DataFrame, truth: dict, path) -> None:
    """Average sales of treated vs control regions, each indexed to its pre-test mean."""
    start, end = truth["test_weeks"]
    sub = panel[(panel.week >= start - 26) & (panel.week <= end + 8)].copy()
    pre_mean = sub[sub.week < start].groupby("region")["sales"].mean()
    sub["index"] = sub["sales"] / sub["region"].map(pre_mean) * 100
    g = sub.groupby(["week", "treated"])["index"].mean().unstack()
    dates = sub.groupby("week")["date"].first()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.axvspan(dates[start], dates[end], color="#ecebe7", lw=0)
    ax.plot(dates, g[0], color=CONTROL_COLOR, label="control regions (10)")
    ax.plot(dates, g[1], color=TREATED_COLOR, label="treated regions (10): social +50%")
    ax.text(dates[start], ax.get_ylim()[1], "  test", va="top", color=TEXT_SECONDARY)
    ax.set_ylabel("sales, pre-test average = 100")
    ax.set_title("Parallel trends: before the test the two groups move together")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_event_study(es: pd.DataFrame, true_per_week: float, path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.4))
    pre, post = es[es.rel_week < 0], es[es.rel_week >= 0]
    for d, color in [(pre, CONTROL_COLOR), (post, TREATED_COLOR)]:
        ax.vlines(d.rel_week, d.ci_low, d.ci_high, color=color, lw=2, alpha=0.6)
        ax.scatter(d.rel_week, d.coef, color=color, s=30, zorder=3, edgecolor="white")
    ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
    ax.hlines(true_per_week, -0.5, post.rel_week.max() + 0.5, color=TRUTH_COLOR, ls="--", lw=1.2,
              label=f"true effect ≈ €{true_per_week:,.0f} per region-week")
    ax.axvline(-0.5, color=TEXT_SECONDARY, ls=":", lw=1)
    ax.set_xlabel("weeks relative to test start (week -1 = reference)")
    ax.set_ylabel("treated − control (€)")
    ax.set_title("Event study: no gap before the test (grey), a gap during it (green)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_comparison(rows: list[tuple[str, float, float, float, str]], path) -> None:
    """Three answers to one question: how much extra sales did the test cause?"""
    fig, ax = plt.subplots(figsize=(8, 2.8))
    y = np.arange(len(rows))[::-1]
    for yi, (_label, mid, lo, hi, color) in zip(y, rows, strict=True):
        if lo is not None:
            ax.plot([lo / 1e3, hi / 1e3], [yi, yi], color=color, lw=7, alpha=0.6,
                    solid_capstyle="round")
        ax.scatter(mid / 1e3, yi, s=60, color=color, edgecolor="white", lw=1.5, zorder=3)
        txt = f"  €{mid / 1e3:,.1f}k" + (f"  [{lo / 1e3:,.1f}, {hi / 1e3:,.1f}]" if lo else "")
        ax.text(max(mid, hi or mid) / 1e3, yi, txt, va="center", fontsize=8,
                color=TEXT_SECONDARY)
    ax.axvline(rows[0][1] / 1e3, color=TRUTH_COLOR, ls="--", lw=1)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("extra sales during the 8 test weeks (€k)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_title("Lift from the social test: truth vs experiment vs model")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = GeoConfig()
    geo = simulate_geo(cfg)
    panel, truth = geo.panel, geo.truth

    did = did_estimate(panel, pre_weeks=26)
    es = event_study(panel, pre_weeks=12)
    p_perm, _ = permutation_pvalue(panel, pre_weeks=26, n_perm=1000)
    pre_es = es[es.rel_week < -1]
    pre_trend_ok = bool(((pre_es.ci_low <= 0) & (pre_es.ci_high >= 0)).all())

    idata = az.from_netcdf(ROOT / "models" / "informed.nc")
    social = simulate().data["spend_social"].to_numpy()
    mmm = mmm_implied_lift(idata, social, cfg, truth["treated_weight"])

    did_total, did_lo, did_hi = did.total
    mmm_lo, mmm_hi = _hdi(mmm["in_window"])
    result = {
        "truth": truth,
        "did": {"lift": did_total, "ci95": [did_lo, did_hi], "p_value": did.p_value,
                "permutation_p": p_perm, "iroas": did_total / truth["extra_spend"],
                "iroas_ci95": [did_lo / truth["extra_spend"], did_hi / truth["extra_spend"]],
                "pre_trend_coefs_all_cover_zero": pre_trend_ok},
        "mmm": {"lift": float(mmm["in_window"].mean()), "hdi90": [mmm_lo, mmm_hi],
                "iroas": float(mmm["in_window"].mean() / truth["extra_spend"]),
                "lift_total_incl_carryover": float(mmm["total"].mean())},
    }
    (OUT / "geo_lift.json").write_text(json.dumps(result, indent=2))
    es.to_csv(OUT / "event_study.csv", index=False)
    panel.to_csv(ROOT / "data" / "geo_panel.csv", index=False)
    window = slice(cfg.start_week, cfg.start_week + cfg.test_weeks)
    lift = lift_test_for_mmm(panel, did, pd.Series(social[window]), cfg.lift_factor,
                             "spend_social")
    lift.to_csv(ROOT / "data" / "lift_test.csv", index=False)
    result["lift_test_row"] = lift.iloc[0].to_dict()
    (OUT / "geo_lift.json").write_text(json.dumps(result, indent=2, default=float))

    plot_parallel_trends(panel, truth, OUT / "parallel_trends.png")
    plot_event_study(es, truth["lift_per_region_week"], OUT / "event_study.png")
    plot_comparison([
        ("Truth", truth["lift_in_window"], None, None, TRUTH_COLOR),
        ("Experiment (DiD, 95% CI)", did_total, did_lo, did_hi, TREATED_COLOR),
        ("MMM (90% HDI)", result["mmm"]["lift"], mmm_lo, mmm_hi, CHANNEL_COLORS["tv"]),
    ], OUT / "lift_comparison.png")
    print(json.dumps({k: v for k, v in result.items() if k != "truth"}, indent=2))
    print(f"truth: lift €{truth['lift_in_window']:,.0f}, iROAS {truth['iroas_in_window']:.2f}")


if __name__ == "__main__":
    main()
