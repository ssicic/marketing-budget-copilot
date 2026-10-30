import pytest
from fastapi.testclient import TestClient

from mbc.api.app import create_app
from mbc.serve.artifact import load_artifact

CHANNELS = ["tv", "search", "social", "radio"]


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:  # serves the committed champion artifact
        yield c


@pytest.fixture(scope="module")
def current(client):
    return {x["channel"]: x["current_weekly_spend"] for x in
            client.get("/channels/roi").json()["channels"]}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["channels"] == CHANNELS
    assert body["model_version"].startswith("mmm-budget v")


def test_roi_has_ordered_intervals(client):
    body = client.get("/channels/roi").json()
    assert body["currency"] == "EUR"
    for ch in body["channels"]:
        for key in ("roi", "marginal_roi"):
            i = ch[key]
            assert i["low"] <= i["mean"] <= i["high"]
        assert ch["current_weekly_spend"] > 0


def test_marginal_roi_below_average_roi_for_saturating_channels(client):
    # Diminishing returns: the next euro earns less than the average euro.
    for ch in client.get("/channels/roi").json()["channels"]:
        assert ch["marginal_roi"]["mean"] < ch["roi"]["mean"] * 1.05


def test_contributions_weekly_is_optional(client):
    short = client.get("/channels/contributions").json()
    assert all(c["weekly"] is None for c in short["channels"])
    full = client.get("/channels/contributions", params={"weekly": True}).json()
    w = full["channels"][0]["weekly"]
    assert len(w["dates"]) == len(w["mean"]) == len(w["low"]) == len(w["high"]) > 100


def test_simulate_current_spend_changes_nothing(client, current):
    body = client.post("/simulate", json={"weekly_spend": current}).json()
    assert body["sales_change_total"]["mean"] == pytest.approx(0.0, abs=1e-6)
    assert body["spend_change_total"] == pytest.approx(0.0, abs=1e-6)
    assert body["warnings"] == []


def test_simulate_fills_missing_channels_and_scales_with_weeks(client, current):
    more_tv = {"tv": current["tv"] * 1.2}
    q = client.post("/simulate", json={"weekly_spend": more_tv, "weeks": 13}).json()
    w = client.post("/simulate", json={"weekly_spend": more_tv, "weeks": 1}).json()
    assert q["proposed_weekly_spend"]["search"] == pytest.approx(current["search"])
    assert q["sales_change_total"]["mean"] == pytest.approx(13 * w["sales_change_total"]["mean"])
    assert q["sales_change_total"]["mean"] > 0  # more TV spend, more sales


def test_simulate_warns_outside_seen_range(client, current):
    body = client.post("/simulate", json={"weekly_spend": {"radio": current["radio"] * 3}}).json()
    assert len(body["warnings"]) == 1 and body["warnings"][0].startswith("radio")


@pytest.mark.parametrize("payload", [
    {"weekly_spend": {"podcast": 100}},
    {"weekly_spend": {"tv": -5}},
    {"weekly_spend": {"tv": 100}, "weeks": 0},
])
def test_simulate_rejects_bad_input(client, payload):
    assert client.post("/simulate", json=payload).status_code == 422


def test_optimize_default_keeps_budget_and_limits(client, current):
    body = client.post("/optimize", json={}).json()
    total_weekly = sum(current.values())
    rec = {a["channel"]: a["recommended_weekly"] for a in body["allocation"]}
    assert sum(rec.values()) == pytest.approx(total_weekly, rel=1e-6)
    assert body["total_budget"] == pytest.approx(total_weekly * 13)
    for c in CHANNELS:
        assert current[c] * 0.5 - 1e-6 <= rec[c] <= current[c] * 1.5 + 1e-6
    assert body["sales_change_total"]["mean"] > 0
    assert body["warnings"] == []


def test_optimize_matches_offline_optimiser(client):
    """The API must give the same answer as the batch report from step 1.3."""
    import numpy as np

    from mbc.api.app import DEFAULT_ARTIFACT
    from mbc.mmm.optimize import optimise, relative_bounds

    a = load_artifact(DEFAULT_ARTIFACT)
    import pandas as pd

    cur = pd.Series(a.current_weekly_spend)
    offline = optimise(a.curves, float(cur.sum()), relative_bounds(cur)).spend
    api = {x["channel"]: x["recommended_weekly"] for x in
           client.post("/optimize", json={}).json()["allocation"]}
    np.testing.assert_allclose([api[c] for c in CHANNELS], offline[CHANNELS].to_numpy(),
                               rtol=1e-4)


def test_optimize_small_budget_warns_and_scales_mix(client):
    body = client.post("/optimize", json={"total_budget": 100_000}).json()
    assert body["total_budget"] == pytest.approx(100_000)
    assert sum(a["recommended_weekly"] for a in body["allocation"]) == pytest.approx(100_000 / 13)
    assert len(body["warnings"]) == 1


def test_optimize_rejects_bad_input(client):
    assert client.post("/optimize", json={"total_budget": -1}).status_code == 422
    assert client.post("/optimize", json={"max_change": 2}).status_code == 422


def test_openapi_docs_available(client):
    spec = client.get("/openapi.json").json()
    assert {"/health", "/channels/roi", "/channels/contributions", "/simulate",
            "/optimize"} <= set(spec["paths"])
