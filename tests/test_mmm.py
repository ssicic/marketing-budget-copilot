import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mbc.mmm import evaluate as ev
from mbc.mmm.data import CONTROLS, SPEND_COLUMNS, add_trend, spend_share, split_holdout
from mbc.mmm.model import build_mmm, model_config
from mbc.simulate import simulate


@pytest.fixture(scope="module")
def frame():
    return add_trend(simulate().data)


# ---------------------------------------------------------------- data


def test_trend_runs_from_zero_to_one(frame):
    assert frame["trend"].iloc[0] == 0 and frame["trend"].iloc[-1] == 1
    assert np.all(np.diff(frame["trend"]) > 0)


def test_holdout_split_is_time_ordered(frame):
    s = split_holdout(frame, 12)
    assert len(s.holdout) == 12 and len(s.train) == len(frame) - 12
    assert s.train["date"].max() < s.holdout["date"].min()
    X, y = s.xy("train")
    assert "sales" not in X and len(y) == len(X)


def test_holdout_split_rejects_bad_sizes(frame):
    with pytest.raises(ValueError):
        split_holdout(frame, 0)


def test_spend_share_sums_to_one(frame):
    assert spend_share(frame).sum() == pytest.approx(1.0)


# ---------------------------------------------------------------- model


def test_informed_beta_prior_scales_with_spend_share(frame):
    share = spend_share(frame)
    cfg = model_config("informed", share)
    sigma = np.asarray(cfg["saturation_beta"].parameters["sigma"])
    np.testing.assert_allclose(sigma, 2 * share.reindex(SPEND_COLUMNS).to_numpy())


def test_informed_alpha_prior_means_favour_slow_broadcast_decay(frame):
    cfg = model_config("informed", spend_share(frame))
    a = np.asarray(cfg["adstock_alpha"].parameters["alpha"])
    b = np.asarray(cfg["adstock_alpha"].parameters["beta"])
    mean = dict(zip(SPEND_COLUMNS, a / (a + b), strict=True))
    assert mean["spend_tv"] > mean["spend_social"] > mean["spend_search"]


def test_weak_priors_are_flat_on_decay(frame):
    cfg = model_config("weak", spend_share(frame))
    assert cfg["adstock_alpha"].parameters == {"alpha": 1.0, "beta": 1.0}


def test_unknown_prior_set_raises(frame):
    with pytest.raises(ValueError):
        model_config("strong", spend_share(frame))


def test_build_mmm_wires_channels_and_controls(frame):
    mmm = build_mmm("informed", spend_share(frame))
    assert mmm.channel_columns == SPEND_COLUMNS
    assert mmm.control_columns == CONTROLS
    assert mmm.adstock.l_max == 8


# ---------------------------------------------------------------- evaluation


def test_mape():
    assert ev.mape([100, 200], [110, 180]) == pytest.approx(0.10)
    with pytest.raises(ValueError):
        ev.mape([0, 1], [0, 1])


def test_predictive_interval_and_coverage():
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, size=(4000, 500))
    pi = ev.predictive_interval(samples)
    actual = rng.normal(0, 1, size=500)
    assert ev.coverage(actual, pi) == pytest.approx(0.90, abs=0.04)


def _fake_idata(n_chain=2, n_draw=300, n_date=10, seed=0):
    rng = np.random.default_rng(seed)
    coords = {"chain": range(n_chain), "draw": range(n_draw), "channel": SPEND_COLUMNS,
              "date": pd.date_range("2024-01-01", periods=n_date, freq="7D")}
    contrib = rng.uniform(0.01, 0.02, (n_chain, n_draw, n_date, 4))
    post = xr.Dataset({
        "channel_contribution": (("chain", "draw", "date", "channel"), contrib),
        "adstock_alpha": (("chain", "draw", "channel"), rng.beta(5, 5, (n_chain, n_draw, 4))),
        "saturation_lam": (("chain", "draw", "channel"), rng.gamma(20, 0.1, (n_chain, n_draw, 4))),
        "saturation_beta": (("chain", "draw", "channel"),
                            rng.uniform(0.1, 0.2, (n_chain, n_draw, 4))),
    }, coords=coords)
    const = xr.Dataset({"target_scale": 1000.0,
                        "channel_scale": (("channel",), np.full(4, 50.0))},
                       coords={"channel": SPEND_COLUMNS})
    return az.InferenceData(posterior=post, constant_data=const), coords["date"]


def test_roi_is_contribution_over_spend():
    idata, dates = _fake_idata()
    spend = pd.DataFrame({c: np.full(len(dates), 2.0) for c in SPEND_COLUMNS})
    contrib = ev.contributions_original_scale(idata)
    roi = ev.roi_samples(contrib, spend)
    expected = contrib.sum("date") / (2.0 * len(dates))
    xr.testing.assert_allclose(roi, expected)
    table = ev.summarise_draws(roi, "roi")
    assert list(table.index) == ["tv", "search", "social", "radio"]
    assert (table["roi_hdi_low"] < table["roi_mean"]).all()
    assert (table["roi_mean"] < table["roi_hdi_high"]).all()


def test_recovery_flags_coverage_correctly():
    idata, dates = _fake_idata()
    roi = xr.ones_like(idata.posterior["adstock_alpha"]) * 2.0
    truth = pd.DataFrame(
        {"adstock_alpha": 0.5, "saturation_lam": 2.0, "beta_eur": 150.0,
         "half_saturation_spend": np.log(3) / 2.0 * 50.0, "roi": 2.0, "saturation_beta": 0.15},
        index=["tv", "search", "social", "radio"],
    )
    truth.loc["radio", "adstock_alpha"] = 0.99  # far outside Beta(5, 5) draws
    table = ev.recovery_table(idata.posterior, roi, truth,
                              idata.constant_data["channel_scale"], 1000.0)
    alpha = table[table.parameter == "adstock_alpha"].set_index("channel")["covered"]
    assert alpha["tv"] and not alpha["radio"]
    assert set(table.parameter) == {"adstock_alpha", "saturation_lam", "beta_eur",
                                    "half_saturation_spend", "roi"}


def test_true_values_for_window_restricts_roi_to_training_weeks():
    sim = simulate()
    components = sim.components
    train = sim.data.iloc[:-12]
    scales = {c: float(train[c].max()) for c in SPEND_COLUMNS}
    tw = ev.true_values_for_window(sim.truth, components, train, scales, target_scale=1.0)
    expected = components["tv"].iloc[:-12].sum() / train["spend_tv"].sum()
    assert tw.loc["tv", "roi"] == pytest.approx(expected)
    # lam converted to the training-window scale keeps the curve identical
    t = sim.truth["channels"]["social"]
    assert tw.loc["social", "saturation_lam"] == pytest.approx(
        t["lam"] * scales["spend_social"] / t["spend_scale"])


@pytest.mark.slow
def test_mmm_fits_end_to_end(frame):
    """Tiny real fit: checks the model builds and samples, not that it converges."""
    import mbc  # noqa: F401

    s = split_holdout(frame, 12)
    mmm = build_mmm("informed", spend_share(s.train))
    X, y = s.xy("train")
    idata = mmm.fit(X, y, draws=30, tune=30, chains=1, progressbar=False, random_seed=0)
    assert set(idata.posterior["channel"].values) == set(SPEND_COLUMNS)
    diag = ev.diagnostics(idata)
    assert "max_r_hat" in diag
