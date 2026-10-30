import numpy as np
import pandas as pd
import pytest

from mbc.experiments.did import analysis_window, did_estimate, event_study, permutation_pvalue
from mbc.experiments.geo import GeoConfig, simulate_geo


def _panel(effect: float, noise: float, seed: int = 0, n_regions: int = 20,
           n_weeks: int = 40, start: int = 30, test_weeks: int = 8) -> pd.DataFrame:
    """Simple panel: region level + common weekly shock + effect in treated test weeks."""
    rng = np.random.default_rng(seed)
    level = rng.uniform(50, 150, n_regions)
    week_shock = rng.normal(0, 20, n_weeks)
    treated = np.zeros(n_regions, int)
    treated[rng.choice(n_regions, n_regions // 2, replace=False)] = 1
    rows = []
    for r in range(n_regions):
        for t in range(n_weeks):
            post = int(start <= t < start + test_weeks)
            y = level[r] + week_shock[t] + effect * treated[r] * post + rng.normal(0, noise)
            rows.append((f"r{r}", t, treated[r], post, y))
    return pd.DataFrame(rows, columns=["region", "week", "treated", "post", "sales"])


def test_did_recovers_effect_exactly_without_noise():
    r = did_estimate(_panel(effect=25.0, noise=0.0), pre_weeks=20)
    assert r.tau == pytest.approx(25.0, abs=1e-8)
    assert r.total[0] == pytest.approx(25.0 * 10 * 8)


def test_did_finds_nothing_when_there_is_no_effect():
    r = did_estimate(_panel(effect=0.0, noise=5.0), pre_weeks=20)
    assert r.ci_low < 0 < r.ci_high


def test_did_confidence_interval_covers_truth_about_95_percent():
    covered = [
        (lambda r: r.ci_low <= 10.0 <= r.ci_high)(did_estimate(_panel(10.0, 8.0, seed=s), 20))
        for s in range(200)
    ]
    # Cluster-robust intervals with 20 clusters run a little narrow, so allow some slack.
    assert 0.85 <= np.mean(covered) <= 0.99


def test_analysis_window_drops_weeks_after_the_test():
    p = _panel(1.0, 0.0, n_weeks=45, start=30)
    w = analysis_window(p, pre_weeks=10)
    assert w["week"].min() == 20 and w["week"].max() == 37


def test_event_study_pre_period_is_flat_without_pre_trends():
    es = event_study(_panel(effect=30.0, noise=0.0), pre_weeks=8)
    pre = es[es.rel_week < 0]
    post = es[es.rel_week >= 0]
    np.testing.assert_allclose(pre["coef"], 0.0, atol=1e-8)
    np.testing.assert_allclose(post["coef"], 30.0, atol=1e-8)


def test_permutation_pvalue_small_for_real_effect_large_for_none():
    p_real, placebo = permutation_pvalue(_panel(30.0, 5.0), pre_weeks=20, n_perm=300)
    p_null, _ = permutation_pvalue(_panel(0.0, 5.0, seed=3), pre_weeks=20, n_perm=300)
    assert p_real < 0.01 and p_null > 0.05
    assert placebo.shape == (300,)


@pytest.fixture(scope="module")
def geo():
    return simulate_geo()


def test_geo_panel_shape_and_assignment(geo):
    p = geo.panel
    assert p["region"].nunique() == 20 and p["week"].nunique() == 156
    assert p.groupby("region")["treated"].first().sum() == 10
    assert p.loc[p.post == 1, "week"].nunique() == 8


def test_geo_weights_sum_to_one(geo):
    assert geo.panel.groupby("region")["weight"].first().sum() == pytest.approx(1.0)


def test_true_lift_only_in_treated_regions_and_matches_truth(geo):
    p = geo.panel
    assert (p.loc[p.treated == 0, "true_extra_sales"] == 0).all()
    in_window = p.loc[p.post == 1, "true_extra_sales"].sum()
    assert in_window == pytest.approx(geo.truth["lift_in_window"])
    assert geo.truth["lift_total_incl_carryover"] > geo.truth["lift_in_window"]


def test_extra_spend_is_half_of_treated_social_spend(geo):
    cfg = GeoConfig()
    p = geo.panel
    treated_test = p[(p.treated == 1) & (p.post == 1)]
    base_spend = treated_test["spend_test_channel"].sum() / cfg.lift_factor
    assert geo.truth["extra_spend"] == pytest.approx(base_spend * (cfg.lift_factor - 1))


def test_lift_test_row_scales_group_lift_to_national(geo):
    from mbc.experiments.did import lift_test_for_mmm

    r = did_estimate(geo.panel)
    spend = pd.Series([100.0, 120.0])
    row = lift_test_for_mmm(geo.panel, r, spend, lift_factor=1.5,
                            channel_column="spend_social").iloc[0]
    assert row["x"] == pytest.approx(110.0) and row["delta_x"] == pytest.approx(55.0)
    # pre-test sales share is a close, observable proxy for the true region weight
    assert row["treated_share"] == pytest.approx(geo.truth["treated_weight"], abs=0.02)
    assert row["delta_y"] == pytest.approx(r.tau * r.n_treated / row["treated_share"])
    assert row["sigma"] > 0
