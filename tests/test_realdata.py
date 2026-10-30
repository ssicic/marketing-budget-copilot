import numpy as np
import pandas as pd
import pytest

from mbc.realdata.conjura import ORGANIC_COLS, SPEND_COLS
from mbc.realdata.prepare import REAL_CONTROLS, brand_table, calendar_controls


def test_calendar_controls_2021():
    dates = pd.date_range("2021-11-01", "2021-12-27", freq="7D")  # Mondays
    c = calendar_controls(dates)
    flagged = {col: [str(d.date()) for d in c.index[c[col] == 1]] for col in REAL_CONTROLS[:-1]}
    assert flagged == {
        "black_friday": ["2021-11-22"],  # Black Friday 26 Nov
        "cyber_week": ["2021-11-29"],  # Cyber Monday
        "december_peak": ["2021-12-06", "2021-12-13"],
        "christmas": ["2021-12-20"],
    }
    assert c["trend"].iloc[0] == 0 and c["trend"].iloc[-1] == 1


def _weekly(n=60, sid="abc"):
    weeks = pd.date_range("2022-01-03", periods=n, freq="7D")
    df = pd.DataFrame({"mmm_timeseries_id": sid, "week": weeks,
                       "revenue": np.linspace(1000, 2000, n), "orders": 10,
                       "new_customer_orders": 5})
    for c in SPEND_COLS + ORGANIC_COLS:
        df[c] = 0.0
    df["meta_facebook_spend"] = 100.0
    df["meta_instagram_spend"] = 50.0
    return df


def test_brand_table_columns_and_values():
    t = brand_table(_weekly(), "abc")
    assert list(t.columns) == ["date", "spend_facebook", "spend_instagram", *REAL_CONTROLS,
                               "sales"]
    assert t["sales"].iloc[0] == 1000 and t["spend_instagram"].iloc[0] == 50


def test_brand_table_rejects_gaps_and_unknown_series():
    w = _weekly().drop(index=10)
    with pytest.raises(ValueError):
        brand_table(w, "abc")
    with pytest.raises(KeyError):
        brand_table(_weekly(), "nope")


def test_semi_synthetic_plants_exact_roi_and_keeps_spend():
    from mbc.realdata.case_study import PLANTED, semi_synthetic

    rng = np.random.default_rng(0)
    dates = pd.date_range("2021-01-04", periods=150, freq="7D")
    df = pd.DataFrame({"date": dates,
                       "spend_facebook": rng.uniform(1000, 3000, 150),
                       "spend_instagram": np.r_[np.zeros(60), rng.uniform(500, 2000, 90)]})
    df = pd.concat([df, calendar_controls(dates).reset_index(drop=True)], axis=1)
    df["sales"] = 20_000 + 5_000 * df["black_friday"] + rng.normal(0, 1_000, 150)
    out, info = semi_synthetic(df)
    for col, p in PLANTED.items():
        c = info["truth"][col]["contribution"]
        assert c.sum() / df[col].sum() == pytest.approx(p["roi"])
    pd.testing.assert_series_equal(out["spend_facebook"], df["spend_facebook"])
    assert info["noise_sd"] > 0
