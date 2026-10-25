"""Generate weekly sales from known media effects.

sales_t = intercept + trend_t + seasonality_t + holidays_t + promo_t
          + sum_c beta_c * saturation(adstock(spend_c / scale_c))
          + noise_t
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from mbc.simulate.config import SimConfig
from mbc.transforms import adstock_half_life, media_response

CHANNEL_COLS = {c: f"spend_{c}" for c in ("tv", "search", "social", "radio")}
CONTROL_COLS = ["promo", "black_friday", "christmas"]


@dataclass
class SimResult:
    data: pd.DataFrame  # what a business would see: date, spend, controls, sales
    components: pd.DataFrame  # true decomposition of sales, per week
    truth: dict  # planted parameters and derived ROIs, JSON-serialisable


# ---------------------------------------------------------------- calendar


def weekly_dates(start: str, n_weeks: int) -> pd.DatetimeIndex:
    dates = pd.date_range(start, periods=n_weeks, freq="7D")
    if dates[0].weekday() != 0:
        raise ValueError("start_date must be a Monday")
    return dates


def _week_start(day: pd.Timestamp) -> pd.Timestamp:
    return day - pd.Timedelta(days=day.weekday())


def black_friday(year: int) -> pd.Timestamp:
    """The day after the fourth Thursday of November."""
    nov1 = pd.Timestamp(year=year, month=11, day=1)
    first_thu = nov1 + pd.Timedelta(days=(3 - nov1.weekday()) % 7)
    return first_thu + pd.Timedelta(weeks=3, days=1)


def holiday_flags(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Flag the week containing Black Friday and the week containing 24 December."""
    years = sorted(set(dates.year))
    bf_weeks = {_week_start(black_friday(y)) for y in years}
    xmas_weeks = {_week_start(pd.Timestamp(year=y, month=12, day=24)) for y in years}
    return pd.DataFrame(
        {
            "black_friday": dates.isin(bf_weeks).astype(int),
            "christmas": dates.isin(xmas_weeks).astype(int),
        },
        index=dates,
    )


def yearly_fourier(dates: pd.DatetimeIndex, n_order: int = 2) -> np.ndarray:
    """Columns sin(2πk·t), cos(2πk·t) for k = 1..n_order, with t = day of year / 365.25.

    Same convention as PyMC-Marketing's YearlyFourier, so the planted seasonality is
    something the model's Fourier terms can represent exactly.
    """
    t = dates.day_of_year.to_numpy() / 365.25
    cols = []
    for k in range(1, n_order + 1):
        cols += [np.sin(2 * np.pi * k * t), np.cos(2 * np.pi * k * t)]
    return np.column_stack(cols)


# ---------------------------------------------------------------- spend patterns


def _tv_spend(rng: np.random.Generator, dates: pd.DatetimeIndex) -> np.ndarray:
    """Flighted: bursts of 4-6 weeks on, 3-8 weeks off, heavier in Q4."""
    n = len(dates)
    spend = np.zeros(n)
    t, on = 0, bool(rng.integers(2))
    while t < n:
        length = rng.integers(4, 7) if on else rng.integers(3, 9)
        if on:
            level = rng.normal(45_000, 5_000)
            for i in range(t, min(t + length, n)):
                q4 = 1.3 if dates[i].month >= 10 else 1.0
                spend[i] = level * q4 * rng.uniform(0.9, 1.1)
        t += length
        on = not on
    return spend


def _search_spend(rng: np.random.Generator, dates: pd.DatetimeIndex) -> np.ndarray:
    """Always on, with a mild seasonal lift towards the end of the year."""
    t = dates.day_of_year.to_numpy() / 365.25
    seasonal = 1 + 0.15 * np.cos(2 * np.pi * (t - 0.9))
    return np.clip(18_000 * seasonal + rng.normal(0, 1_200, len(dates)), 0, None)


def _social_spend(rng: np.random.Generator, dates: pd.DatetimeIndex) -> np.ndarray:
    """Growing steadily from about 4k to 22k per week."""
    trend = np.linspace(4_000, 22_000, len(dates))
    return np.clip(trend + rng.normal(0, 1_500, len(dates)), 0, None)


def _radio_spend(rng: np.random.Generator, dates: pd.DatetimeIndex) -> np.ndarray:
    """Sparse 2-3 week flights at a low level."""
    n = len(dates)
    spend = np.zeros(n)
    t = 0
    while t < n:
        if rng.uniform() < 0.3:
            length = int(rng.integers(2, 4))
            spend[t : t + length] = rng.normal(9_000, 1_000)
            t += length
        t += 1
    return np.clip(spend, 0, None)


SPEND_GENERATORS = {
    "tv": _tv_spend,
    "search": _search_spend,
    "social": _social_spend,
    "radio": _radio_spend,
}


# ---------------------------------------------------------------- simulation


def simulate(config: SimConfig | None = None) -> SimResult:
    cfg = config or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    dates = weekly_dates(cfg.start_date, cfg.n_weeks)
    n = len(dates)

    spend = {c: SPEND_GENERATORS[c](rng, dates) for c in cfg.channels}

    # Media: solve beta so each channel's realised average ROI equals its target.
    contributions, betas, scales = {}, {}, {}
    for c, p in cfg.channels.items():
        scales[c] = float(spend[c].max())
        unit = media_response(
            spend[c], alpha=p.alpha, lam=p.lam, beta=1.0, scale=scales[c], l_max=cfg.l_max
        )
        betas[c] = p.target_roi * spend[c].sum() / unit.sum()
        contributions[c] = betas[c] * unit

    # Base sales
    trend = cfg.intercept * cfg.trend_total_growth * np.arange(n) / (n - 1)
    seasonality = yearly_fourier(dates) @ np.array(cfg.seasonality)
    holidays = holiday_flags(dates)
    promo = (rng.uniform(size=n) < cfg.promo_rate).astype(int)
    promo[(holidays["black_friday"] + holidays["christmas"]).to_numpy() > 0] = 0
    noise = rng.normal(0, cfg.noise_sd, n)

    components = pd.DataFrame(
        {
            "intercept": cfg.intercept,
            "trend": trend,
            "seasonality": seasonality,
            "black_friday": holidays["black_friday"].to_numpy() * cfg.black_friday_effect,
            "christmas": holidays["christmas"].to_numpy() * cfg.christmas_effect,
            "promo": promo * cfg.promo_effect,
            **{c: contributions[c] for c in cfg.channels},
            "noise": noise,
        },
        index=pd.Index(dates, name="date"),
    )
    sales = components.sum(axis=1).to_numpy()

    data = pd.DataFrame(
        {
            "date": dates,
            **{CHANNEL_COLS[c]: spend[c] for c in cfg.channels},
            "promo": promo,
            "black_friday": holidays["black_friday"].to_numpy(),
            "christmas": holidays["christmas"].to_numpy(),
            "sales": sales,
        }
    )

    truth = _ground_truth(cfg, spend, contributions, betas, scales, dates, sales)
    return SimResult(data=data, components=components, truth=truth)


def marginal_roi(
    spend: np.ndarray, *, alpha: float, lam: float, beta: float, scale: float, l_max: int,
    bump: float = 0.01,
) -> float:
    """Extra sales per extra euro if every week's spend were `bump` (1%) higher."""
    base = media_response(spend, alpha=alpha, lam=lam, beta=beta, scale=scale, l_max=l_max)
    more = media_response(
        spend * (1 + bump), alpha=alpha, lam=lam, beta=beta, scale=scale, l_max=l_max
    )
    return float((more.sum() - base.sum()) / (spend.sum() * bump))


def _ground_truth(cfg, spend, contributions, betas, scales, dates, sales) -> dict:
    channels = {}
    for c, p in cfg.channels.items():
        channels[c] = {
            "alpha": p.alpha,
            "adstock_half_life_weeks": round(adstock_half_life(p.alpha), 3),
            "lam": p.lam,
            "spend_scale": scales[c],
            # Adstocked weekly spend (EUR) at which half the max effect is reached.
            # Independent of scaling, so it compares across models directly.
            "half_saturation_spend": float(np.log(3) / p.lam * scales[c]),
            "beta": float(betas[c]),
            "total_spend": float(spend[c].sum()),
            "total_contribution": float(contributions[c].sum()),
            "roi": float(contributions[c].sum() / spend[c].sum()),
            "marginal_roi": marginal_roi(
                spend[c], alpha=p.alpha, lam=p.lam, beta=betas[c], scale=scales[c],
                l_max=cfg.l_max,
            ),
        }
    total_media = sum(contributions[c].sum() for c in cfg.channels)
    config = asdict(cfg)
    config.pop("channels")
    return {
        "description": "Planted parameters for the synthetic MMM dataset. Contributions "
        "and ROIs cover the full period; carry-over past the last week is ignored.",
        "config": config,
        "channels": channels,
        "media_share_of_sales": float(total_media / sales.sum()),
        "date_range": [str(dates[0].date()), str(dates[-1].date())],
        "holdout_start": str(dates[-cfg.holdout_weeks].date()),
    }


def lam_at_scale(lam: float, true_scale: float, model_scale: float) -> float:
    """Convert a saturation `lam` defined on spend/true_scale to spend/model_scale.

    f(lam * x / s_true) == f(lam' * x / s_model)  =>  lam' = lam * s_model / s_true
    """
    return lam * model_scale / true_scale
