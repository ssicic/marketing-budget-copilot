import numpy as np
import pandas as pd
import pytest

from mbc.realdata.conjura import (
    ORGANIC_COLS,
    SPEND_COLS,
    load_raw,
    mmm_candidates,
    screen_series,
    to_weekly,
)


def _daily(sid: str, start: str, n_days: int, rng, **spend_fns) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days, freq="D")
    df = pd.DataFrame(
        {
            "mmm_timeseries_id": sid,
            "organisation_id": f"org_{sid}",
            "organisation_vertical": "Apparel",
            "organisation_subvertical": "Shoes",
            "territory_name": "All Territories",
            "currency_code": "GBP",
            "date_day": dates,
            "all_purchases_original_price": rng.uniform(900, 1100, n_days),
            "all_purchases": rng.integers(5, 15, n_days),
            "first_purchases": rng.integers(1, 5, n_days),
        }
    )
    for c in SPEND_COLS:
        df[c] = spend_fns[c](n_days) if c in spend_fns else np.nan
    for c in ORGANIC_COLS:
        df[c] = 1
    return df


@pytest.fixture
def daily():
    rng = np.random.default_rng(0)
    on_off = lambda n: np.where((np.arange(n) // 14) % 2 == 0, 100.0, 10.0)  # noqa: E731
    a = _daily(
        "a", "2021-01-06", 7 * 170, rng,  # a Wednesday: first and last weeks are partial
        meta_facebook_spend=on_off,
        google_pmax_spend=lambda n: np.repeat(rng.uniform(10, 90, n // 7 + 1), 7)[:n],
    )
    b = _daily("b", "2023-01-02", 7 * 20, rng, meta_facebook_spend=lambda n: np.full(n, 5.0))
    return pd.concat([a, b], ignore_index=True)


def test_load_raw_lowercases_and_fills_spend(tmp_path, daily):
    raw = daily.copy()
    raw.columns = raw.columns.str.upper()
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)
    df = load_raw(path)
    assert "date_day" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date_day"])
    assert df[SPEND_COLS].isna().sum().sum() == 0


def test_load_raw_missing_file_explains_how_to_fetch(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_conjura"):
        load_raw(tmp_path / "nope.csv")


def test_to_weekly_keeps_only_complete_monday_weeks(daily):
    daily[SPEND_COLS] = daily[SPEND_COLS].fillna(0)
    w = to_weekly(daily)
    a = w[w["mmm_timeseries_id"] == "a"]
    assert (a["week"].dt.weekday == 0).all()
    assert len(a) == 169  # 170 weeks of days starting mid-week -> 169 full weeks
    first_week = daily[(daily["mmm_timeseries_id"] == "a")
                       & (daily["date_day"] >= a["week"].iloc[0])
                       & (daily["date_day"] < a["week"].iloc[0] + pd.Timedelta(days=7))]
    assert a["revenue"].iloc[0] == pytest.approx(first_week["all_purchases_original_price"].sum())
    assert a["currency_code"].iloc[0] == "GBP"


def test_screen_and_candidates(daily):
    daily[SPEND_COLS] = daily[SPEND_COLS].fillna(0)
    screen = screen_series(to_weekly(daily)).set_index("mmm_timeseries_id")
    a, b = screen.loc["a"], screen.loc["b"]
    assert a["n_main"] == 2
    assert set(a["main_channels"].split(", ")) == {"meta_facebook", "google_pmax"}
    assert b["min_cv"] == pytest.approx(0.0)  # constant spend carries no information
    cands = mmm_candidates(screen.reset_index())
    assert cands["mmm_timeseries_id"].tolist() == ["a"]
