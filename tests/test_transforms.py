import numpy as np
import pytest
import xarray as xr
from pymc_marketing.mmm.transformers import geometric_adstock as pm_adstock
from pymc_marketing.mmm.transformers import logistic_saturation as pm_logistic

from mbc.transforms import (
    adstock_half_life,
    geometric_adstock,
    logistic_saturation,
    media_response,
)


def test_adstock_impulse_response_is_normalised_geometric_decay():
    x = np.zeros(10)
    x[0] = 1.0
    out = geometric_adstock(x, alpha=0.5, l_max=4)
    weights = np.array([1, 0.5, 0.25, 0.125]) / 1.875
    np.testing.assert_allclose(out[:4], weights)
    np.testing.assert_allclose(out[4:], 0.0)


def test_adstock_preserves_total_spend_when_series_is_long_enough():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(0, 10, 50), np.zeros(8)])
    assert geometric_adstock(x, alpha=0.7, l_max=8).sum() == pytest.approx(x.sum())


def test_adstock_is_causal():
    x = np.zeros(10)
    x[5] = 1.0
    out = geometric_adstock(x, alpha=0.6, l_max=8)
    np.testing.assert_allclose(out[:5], 0.0)


def test_alpha_zero_means_no_carryover():
    x = np.arange(6, dtype=float)
    np.testing.assert_allclose(geometric_adstock(x, alpha=0.0, l_max=8), x)


def test_adstock_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        geometric_adstock(np.ones(3), alpha=1.0)


@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.7])
def test_adstock_matches_pymc_marketing(alpha):
    x = np.random.default_rng(1).uniform(0, 1, 40)
    ours = geometric_adstock(x, alpha=alpha, l_max=8, normalize=True)
    x_da = xr.DataArray(x, dims=("date",))
    theirs = pm_adstock(x_da, alpha=alpha, l_max=8, normalize=True, dim="date").eval()
    np.testing.assert_allclose(ours, theirs, atol=1e-10)


def test_saturation_limits_and_monotonicity():
    x = np.linspace(0, 20, 200)
    y = logistic_saturation(x, lam=1.0)
    assert y[0] == 0.0
    assert np.all(np.diff(y) > 0)
    assert y[-1] == pytest.approx(1.0, abs=1e-6)


def test_saturation_is_concave():
    y = logistic_saturation(np.linspace(0, 5, 100), lam=1.0)
    assert np.all(np.diff(y, 2) < 0)


def test_saturation_half_point():
    lam = 2.5
    assert logistic_saturation(np.log(3) / lam, lam) == pytest.approx(0.5)


def test_saturation_matches_pymc_marketing():
    x = np.linspace(0, 3, 30)
    np.testing.assert_allclose(
        logistic_saturation(x, 1.7),
        pm_logistic(xr.DataArray(x, dims=("date",)), lam=1.7).eval(),
        atol=1e-12,
    )


def test_media_response_scales_with_beta():
    spend = np.array([0, 10, 20, 0, 0], dtype=float)
    r1 = media_response(spend, alpha=0.5, lam=2.0, beta=1.0, scale=20.0)
    r3 = media_response(spend, alpha=0.5, lam=2.0, beta=3.0, scale=20.0)
    np.testing.assert_allclose(r3, 3 * r1)


def test_half_life():
    assert adstock_half_life(0.5) == pytest.approx(1.0)
    assert adstock_half_life(0.0) == 0.0
