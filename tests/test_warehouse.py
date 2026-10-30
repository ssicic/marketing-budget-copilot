import numpy as np
import pandas as pd
import pytest

from mbc.mmm.data import CONTROLS, SPEND_COLUMNS, add_trend, load_synthetic
from mbc.simulate import simulate
from mbc.warehouse import build, connect, read_sql


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("wh") / "test.duckdb"
    counts = build(path)
    return path, counts


def test_build_creates_expected_tables(db):
    _, counts = db
    assert counts["weekly_sales"] == 156
    assert counts["true_components"] == 156
    assert counts["channel_truth"] == 4
    assert counts["geo_panel"] == 20 * 156
    assert counts["mmm_training_frame"] == 156


def test_training_frame_matches_python_pipeline(db):
    path, _ = db
    sql = load_synthetic(path)
    py = add_trend(simulate().data)
    np.testing.assert_allclose(sql["trend"], py["trend"])
    np.testing.assert_allclose(sql[SPEND_COLUMNS].to_numpy(), py[SPEND_COLUMNS].to_numpy())
    np.testing.assert_allclose(sql["sales"], py["sales"])
    assert list(sql.columns) == ["date", *SPEND_COLUMNS, *CONTROLS, "sales"]
    assert pd.api.types.is_datetime64_any_dtype(sql["date"])


def test_read_sql_accepts_raw_queries(db):
    path, _ = db
    df = read_sql("SELECT channel, roi FROM channel_truth ORDER BY roi DESC", path)
    assert df["channel"].iloc[0] == "search"


def test_build_is_repeatable(db):
    path, counts = db
    assert build(path) == counts


def test_connect_missing_db_explains_how_to_build(tmp_path):
    with pytest.raises(FileNotFoundError, match="mbc.warehouse"):
        connect(tmp_path / "missing.duckdb")


def test_connections_are_read_only_by_default(db):
    path, _ = db
    import duckdb

    with connect(path) as con, pytest.raises(duckdb.Error):
        con.execute("DELETE FROM weekly_sales")
