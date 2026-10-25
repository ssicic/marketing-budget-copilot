import numpy as np
import pandas as pd
import pytest

from mbc.simulate import CHANNEL_COLS, SimConfig, simulate
from mbc.simulate.generate import black_friday, holiday_flags, lam_at_scale, weekly_dates
from mbc.transforms import logistic_saturation


@pytest.fixture(scope="module")
def sim():
    return simulate(SimConfig())


def test_shape_and_columns(sim):
    df = sim.data
    assert len(df) == 156
    assert list(df.columns) == [
        "date", *CHANNEL_COLS.values(), "promo", "black_friday", "christmas", "sales"
    ]
    assert (df[list(CHANNEL_COLS.values())] >= 0).all().all()


def test_deterministic_for_seed():
    a, b = simulate(SimConfig(seed=7)), simulate(SimConfig(seed=7))
    pd.testing.assert_frame_equal(a.data, b.data)
    assert not simulate(SimConfig(seed=8)).data.equals(a.data)


def test_components_add_up_to_sales(sim):
    np.testing.assert_allclose(sim.components.sum(axis=1).to_numpy(), sim.data["sales"])


def test_realised_roi_matches_planted_target(sim):
    for c, p in SimConfig().channels.items():
        assert sim.truth["channels"][c]["roi"] == pytest.approx(p.target_roi)


def test_marginal_roi_below_average_roi(sim):
    """Saturation means the next euro is always worth less than the average euro."""
    for c, t in sim.truth["channels"].items():
        assert 0 < t["marginal_roi"] < t["roi"], c


def test_search_more_saturated_than_social(sim):
    ch = sim.truth["channels"]
    ratio = {c: ch[c]["marginal_roi"] / ch[c]["roi"] for c in ch}
    assert ratio["search"] < ratio["social"]


def test_spend_patterns(sim):
    df = sim.data
    tv = df["spend_tv"].to_numpy()
    assert (tv == 0).mean() > 0.3  # flighted: many dark weeks
    assert df["spend_search"].std() / df["spend_search"].mean() < 0.2  # steady
    slope = np.polyfit(np.arange(len(df)), df["spend_social"], 1)[0]
    assert slope > 0  # growing
    assert (df["spend_radio"] == 0).mean() > 0.4  # sparse


def test_media_share_is_plausible(sim):
    assert 0.1 < sim.truth["media_share_of_sales"] < 0.4


def test_black_friday_dates():
    assert black_friday(2023) == pd.Timestamp("2023-11-24")
    assert black_friday(2024) == pd.Timestamp("2024-11-29")
    assert black_friday(2025) == pd.Timestamp("2025-11-28")


def test_one_holiday_week_per_year():
    dates = weekly_dates("2023-01-02", 156)
    flags = holiday_flags(dates)
    assert flags["black_friday"].sum() == 3
    assert flags["christmas"].sum() == 3
    assert (flags["black_friday"] & flags["christmas"]).sum() == 0


def test_promo_never_on_holiday_weeks(sim):
    df = sim.data
    assert ((df["promo"] == 1) & ((df["black_friday"] == 1) | (df["christmas"] == 1))).sum() == 0


def test_lam_conversion_preserves_curve():
    x = np.linspace(0, 50_000, 20)
    lam, s_true, s_model = 1.5, 50_000.0, 40_000.0
    lam2 = lam_at_scale(lam, s_true, s_model)
    np.testing.assert_allclose(
        logistic_saturation(x / s_true, lam), logistic_saturation(x / s_model, lam2)
    )
