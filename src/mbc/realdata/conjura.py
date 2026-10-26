"""Load and screen the Conjura eCommerce MMM dataset.

Anderson, A. (2024). Multi-Region Marketing Mix Modelling (MMM) Dataset for Several
eCommerce Brands. figshare. https://doi.org/10.6084/m9.figshare.25314841.v3 (CC BY 4.0)

Fetch it with scripts/download_conjura.sh.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[3] / "data" / "raw" / "conjura" / "conjura_mmm_data.csv"

SPEND_COLS = [
    "google_paid_search_spend",
    "google_shopping_spend",
    "google_pmax_spend",
    "google_display_spend",
    "google_video_spend",
    "meta_facebook_spend",
    "meta_instagram_spend",
    "meta_other_spend",
    "tiktok_spend",
]
ORGANIC_COLS = [
    "direct_clicks",
    "branded_search_clicks",
    "organic_search_clicks",
    "email_clicks",
    "referral_clicks",
    "all_other_clicks",
]
SERIES_ATTRS = [
    "organisation_id",
    "organisation_vertical",
    "organisation_subvertical",
    "territory_name",
    "currency_code",
]


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Read the daily file with lower-case columns and parsed dates.

    A missing spend value means the brand doesn't use that channel in that series, so
    spend NaNs become 0. Clicks and impressions are left as they are.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/download_conjura.sh first.")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df["date_day"] = pd.to_datetime(df["date_day"])
    df[SPEND_COLS] = df[SPEND_COLS].fillna(0.0)
    return df.sort_values(["mmm_timeseries_id", "date_day"]).reset_index(drop=True)


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Sum days into Monday-starting weeks per series, keeping only complete weeks.

    Partial weeks at the start or end of a series would look like sudden drops in
    both sales and spend, which the model would read as a (fake) media effect.
    """
    df = daily.assign(week=daily["date_day"].dt.to_period("W-SUN").dt.start_time)
    sums = {
        "revenue": ("all_purchases_original_price", "sum"),
        "orders": ("all_purchases", "sum"),
        "new_customer_orders": ("first_purchases", "sum"),
        **{c: (c, "sum") for c in SPEND_COLS + ORGANIC_COLS},
    }
    weekly = (
        df.groupby(["mmm_timeseries_id", "week"])
        .agg(n_days=("date_day", "count"), **sums)
        .reset_index()
    )
    weekly = weekly[weekly["n_days"] == 7].drop(columns="n_days")
    attrs = daily.groupby("mmm_timeseries_id")[SERIES_ATTRS].first()
    return weekly.merge(attrs, on="mmm_timeseries_id").reset_index(drop=True)


def screen_series(
    weekly: pd.DataFrame, *, main_share: float = 0.10
) -> pd.DataFrame:
    """One row per series with the statistics that decide whether an MMM can work.

    - weeks: history length (we want 3+ years, i.e. 156+)
    - main_channels: channels with at least `main_share` of total spend
    - spend_to_revenue: if ads are a tiny share of revenue, their effect drowns in noise
    - min_cv: smallest coefficient of variation among main channels; flat spend
      carries no information about its effect
    - max_trend_corr: largest |corr(spend, time)|; trend-like spend is confounded
      with the base trend
    - max_pair_corr: largest |corr| between two main channels; collinear channels
      can't be told apart
    """
    rows = []
    for sid, g in weekly.groupby("mmm_timeseries_id"):
        totals = g[SPEND_COLS].sum()
        total = totals.sum()
        if total <= 0:
            continue
        main = totals[totals / total >= main_share].index.tolist()
        t = np.arange(len(g))
        cv = [g[c].std() / g[c].mean() for c in main]
        trend = [abs(np.corrcoef(t, g[c])[0, 1]) for c in main]
        if len(main) > 1:
            corr = g[main].corr().abs().to_numpy()
            pair = float(corr[~np.eye(len(main), dtype=bool)].max())
        else:
            pair = np.nan
        rows.append(
            {
                "mmm_timeseries_id": sid,
                "organisation_id": g["organisation_id"].iloc[0],
                "vertical": g["organisation_vertical"].iloc[0],
                "territory": g["territory_name"].iloc[0],
                "currency": g["currency_code"].iloc[0],
                "weeks": len(g),
                "start": g["week"].min().date(),
                "end": g["week"].max().date(),
                "revenue_per_week": g["revenue"].mean(),
                "spend_to_revenue": total / g["revenue"].sum(),
                "n_main": len(main),
                "main_channels": ", ".join(c.removesuffix("_spend") for c in main),
                "min_cv": min(cv),
                "max_trend_corr": max(trend),
                "max_pair_corr": pair,
            }
        )
    return pd.DataFrame(rows)


def mmm_candidates(
    screen: pd.DataFrame,
    *,
    min_weeks: int = 156,
    min_channels: int = 2,
    min_cv: float = 0.3,
    min_spend_to_revenue: float = 0.02,
) -> pd.DataFrame:
    """Filter the screen down to series worth trying an MMM on."""
    mask = (
        (screen["weeks"] >= min_weeks)
        & (screen["n_main"] >= min_channels)
        & (screen["min_cv"] >= min_cv)
        & (screen["spend_to_revenue"] >= min_spend_to_revenue)
    )
    return screen[mask].sort_values("weeks", ascending=False).reset_index(drop=True)
