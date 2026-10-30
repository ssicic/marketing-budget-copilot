import numpy as np
import pandas as pd
import pytest

from mbc.mmm.optimize import Curves, optimise, relative_bounds, uplift_draws


def _two_channels(beta=(100.0, 100.0), lam=(1.0, 3.0)):
    return Curves(channels=["a", "b"], beta=[beta], lam=[lam], scale=np.array([100.0, 100.0]))


def test_marginal_matches_numerical_derivative():
    c = _two_channels()
    s = np.array([40.0, 70.0])
    h = 1e-4
    num = (c.response(s + h) - c.response(s - h)) / (2 * h)
    np.testing.assert_allclose(c.marginal(s), num, rtol=1e-6)


def test_budget_is_spent_exactly_and_bounds_respected():
    c = _two_channels()
    bounds = {"a": (10.0, 150.0), "b": (10.0, 150.0)}
    a = optimise(c, 120.0, bounds)
    assert a.success
    assert a.spend.sum() == pytest.approx(120.0)
    for ch, (lo, hi) in bounds.items():
        assert lo - 1e-6 <= a.spend[ch] <= hi + 1e-6


def test_interior_optimum_equalises_marginal_roi():
    """The textbook condition: at the optimum, the next euro earns the same everywhere."""
    c = _two_channels()
    a = optimise(c, 120.0, {"a": (0.0, 200.0), "b": (0.0, 200.0)})
    m = c.marginal(a.spend.to_numpy())[0]
    assert m[0] == pytest.approx(m[1], rel=1e-4)


def test_identical_channels_split_evenly():
    c = _two_channels(lam=(2.0, 2.0))
    a = optimise(c, 100.0, {"a": (0.0, 100.0), "b": (0.0, 100.0)})
    np.testing.assert_allclose(a.spend.to_numpy(), [50.0, 50.0], atol=1e-3)


def test_binding_bound_caps_the_better_channel():
    # 'a' is far better, but can't exceed 60
    c = _two_channels(beta=(1000.0, 10.0), lam=(0.5, 0.5))
    a = optimise(c, 100.0, {"a": (0.0, 60.0), "b": (0.0, 100.0)})
    assert a.spend["a"] == pytest.approx(60.0, abs=1e-3)
    assert a.spend["b"] == pytest.approx(40.0, abs=1e-3)


def test_infeasible_bounds_raise():
    with pytest.raises(ValueError):
        optimise(_two_channels(), 500.0, {"a": (0.0, 100.0), "b": (0.0, 100.0)})


def test_optimum_beats_current_on_average():
    rng = np.random.default_rng(0)
    c = Curves(channels=["a", "b"], beta=rng.uniform(80, 120, (200, 2)),
               lam=rng.uniform(0.5, 3, (200, 2)), scale=np.array([100.0, 100.0]))
    current = pd.Series({"a": 80.0, "b": 40.0})
    a = optimise(c, current.sum(), relative_bounds(current))
    up = uplift_draws(c, a.spend.to_numpy(), current.to_numpy())
    assert up.shape == (200,)
    assert up.mean() >= -1e-9


def test_relative_bounds():
    b = relative_bounds(pd.Series({"x": 100.0}), band=0.5)
    assert b == {"x": (50.0, 150.0)}


def test_curves_shape_validation():
    with pytest.raises(ValueError):
        Curves(channels=["a"], beta=[[1.0, 2.0]], lam=[[1.0, 2.0]], scale=np.array([1.0]))
