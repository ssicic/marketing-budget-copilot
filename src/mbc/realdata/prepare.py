"""Turn one Conjura series into the weekly table the MMM expects.

Controls are built from the calendar only, never from the sales data, so they can't
leak the answer into the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mbc.realdata.conjura import load_raw, to_weekly
from mbc.simulate.generate import black_friday, holiday_flags

META_CHANNELS = {"meta_facebook_spend": "spend_facebook", "meta_instagram_spend": "spend_instagram"}
REAL_CONTROLS = ["black_friday", "cyber_week", "december_peak", "christmas", "trend"]


def calendar_controls(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Retail calendar flags for Monday-starting weeks.

    black_friday:  week containing Black Friday
    cyber_week:    week starting on Cyber Monday (3 days after Black Friday)
    december_peak: the two weeks after cyber week (pre-Christmas shipping rush)
    christmas:     week containing 24 December
    trend:         0 -> 1 over the series
    """
    years = sorted(set(dates.year))
    cyber = {black_friday(y) + pd.Timedelta(days=3) for y in years}
    dec = {c + pd.Timedelta(weeks=k) for c in cyber for k in (1, 2)}
    h = holiday_flags(dates)
    return pd.DataFrame({
        "black_friday": h["black_friday"].to_numpy(),
        "cyber_week": dates.isin(cyber).astype(int),
        "december_peak": dates.isin(dec).astype(int),
        "christmas": h["christmas"].to_numpy(),
        "trend": np.arange(len(dates)) / (len(dates) - 1),
    }, index=dates)


def brand_table(weekly: pd.DataFrame, series_id: str,
                channels: dict[str, str] = META_CHANNELS) -> pd.DataFrame:
    """Weekly MMM table for one series: date, spend columns, controls, sales (= revenue)."""
    g = weekly[weekly["mmm_timeseries_id"] == series_id].sort_values("week")
    if g.empty:
        raise KeyError(f"series {series_id} not found")
    dates = pd.DatetimeIndex(g["week"])
    if (dates.to_series().diff().dropna() != pd.Timedelta(weeks=1)).any():
        raise ValueError("weeks are not consecutive")
    out = pd.DataFrame({"date": dates})
    for src, dst in channels.items():
        out[dst] = g[src].to_numpy()
    controls = calendar_controls(dates).reset_index(drop=True)
    out = pd.concat([out, controls], axis=1)
    out["sales"] = g["revenue"].to_numpy()  # revenue before discounts (see data dictionary)
    return out


def load_brand(series_prefix: str = "1f064c83") -> pd.DataFrame:
    weekly = to_weekly(load_raw())
    ids = [s for s in weekly["mmm_timeseries_id"].unique() if s.startswith(series_prefix)]
    if len(ids) != 1:
        raise KeyError(f"prefix {series_prefix!r} matches {len(ids)} series")
    return brand_table(weekly, ids[0])
