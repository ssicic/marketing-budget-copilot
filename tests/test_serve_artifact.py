import json
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mbc.mmm.data import CHANNELS, SPEND_COLUMNS
from mbc.serve.artifact import load_artifact, save_artifact
from mbc.serve.export import build_artifact
from mbc.serve.pyfunc import MMMBudgetModel


def _fake_idata(n_chain=2, n_draw=200, n_date=12, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_date, freq="7D")
    coords = {"chain": range(n_chain), "draw": range(n_draw), "channel": SPEND_COLUMNS,
              "date": dates}
    shape = (n_chain, n_draw, 4)
    post = xr.Dataset({
        "channel_contribution": (("chain", "draw", "date", "channel"),
                                 rng.uniform(0.01, 0.02, (n_chain, n_draw, n_date, 4))),
        "adstock_alpha": (("chain", "draw", "channel"), rng.beta(5, 5, shape)),
        "saturation_lam": (("chain", "draw", "channel"), rng.gamma(20, 0.1, shape)),
        "saturation_beta": (("chain", "draw", "channel"), rng.uniform(0.1, 0.2, shape)),
    }, coords=coords)
    const = xr.Dataset({"target_scale": 1000.0,
                        "channel_scale": (("channel",), np.full(4, 50.0))},
                       coords={"channel": SPEND_COLUMNS})
    stats = xr.Dataset({"diverging": (("chain", "draw"), np.zeros((n_chain, n_draw), bool))},
                       coords={"chain": coords["chain"], "draw": coords["draw"]})
    train = pd.DataFrame({"date": dates, "sales": 100.0,
                          **{c: np.full(n_date, 10.0) for c in SPEND_COLUMNS}})
    return az.InferenceData(posterior=post, constant_data=const, sample_stats=stats), train


@pytest.fixture
def exported(tmp_path):
    idata, train = _fake_idata()
    meta = build_artifact(idata, train, tmp_path, source="fake", n_draws=100)
    return tmp_path, meta, idata


def test_artifact_files_are_small_and_complete(exported):
    path, meta, _ = exported
    assert (path / "model.npz").stat().st_size < 50_000
    m = json.loads((path / "model.json").read_text())
    assert m["channels"] == CHANNELS
    assert set(m["roi"]) == set(CHANNELS)
    assert len(m["contributions"]["tv"]["weekly"]["mean"]) == 12
    assert m["current_weekly_spend"]["tv"] == pytest.approx(10.0)


def test_load_roundtrip_converts_beta_to_sales_units(exported):
    path, _, idata = exported
    a = load_artifact(path)
    assert a.curves.beta.shape == (100, 4)
    beta_range = idata.posterior["saturation_beta"].values * 1000.0
    assert beta_range.min() - 1e-3 <= a.curves.beta.min()
    assert a.curves.beta.max() <= beta_range.max() + 1e-3
    np.testing.assert_allclose(a.curves.scale, 50.0)


def test_roi_summary_matches_definition(exported):
    _, meta, idata = exported
    contrib = idata.posterior["channel_contribution"].sel(channel="spend_tv") * 1000.0
    expected = float((contrib.sum("date") / (10.0 * 12)).mean())
    assert meta["roi"]["tv"]["mean"] == pytest.approx(expected)
    lo, hi = meta["roi"]["tv"]["hdi"]
    assert lo < meta["roi"]["tv"]["mean"] < hi


def test_unsupported_version_is_rejected(tmp_path):
    save_artifact(tmp_path, ["a"], np.zeros((2, 1)), np.ones((2, 1)), np.ones((2, 1)),
                  np.ones(1), {})
    m = json.loads((tmp_path / "model.json").read_text())
    m["artifact_version"] = 99
    (tmp_path / "model.json").write_text(json.dumps(m))
    with pytest.raises(ValueError):
        load_artifact(tmp_path)


def test_pyfunc_predicts_interval_per_scenario(exported):
    path, _, _ = exported
    model = MMMBudgetModel()
    ctx = SimpleNamespace(artifacts={"npz": str(path / "model.npz"),
                                     "json": str(path / "model.json")})
    model.load_context(ctx)
    df = pd.DataFrame([{c: 10.0 for c in CHANNELS}, {c: 40.0 for c in CHANNELS}])
    out = model.predict(None, df)
    assert list(out.columns) == ["media_sales_mean", "media_sales_low", "media_sales_high"]
    assert (out["media_sales_low"] <= out["media_sales_mean"]).all()
    assert out["media_sales_mean"].iloc[1] > out["media_sales_mean"].iloc[0]


def test_register_and_promote_moves_champion(exported, tmp_path):
    from mbc.serve.export import register
    from mbc.serve.promote import promote

    path, meta, _ = exported
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    store = tmp_path / "mlruns"
    v1 = register(path, meta, uri, artifact_location=store)
    idata2, train2 = _fake_idata(seed=1)
    path2 = tmp_path / "v2"
    meta2 = build_artifact(idata2, train2, path2, source="fake-2", n_draws=100)
    v2 = register(path2, meta2, uri, artifact_location=store)
    assert (v1, v2) == ("1", "2")

    out = promote(v1, uri, out_dir=tmp_path / "champion")
    assert (out / "VERSION").read_text().strip() == "mmm-budget v1"
    assert load_artifact(out).meta["source_model"] == "fake"

    promote(v2, uri, out_dir=tmp_path / "champion")  # release a new version
    assert load_artifact(out).meta["source_model"] == "fake-2"
    promote(v1, uri, out_dir=tmp_path / "champion")  # roll back
    assert load_artifact(out).meta["source_model"] == "fake"
