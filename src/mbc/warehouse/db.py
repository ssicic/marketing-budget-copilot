"""Build and query the DuckDB warehouse.

Tables
------
weekly_sales           one row per week: spend per channel, controls, sales
true_components        the planted decomposition of sales (synthetic data only)
channel_truth          planted parameters and ROI per channel
geo_panel              region x week panel of the geo-lift experiment
lift_tests             experiment results in the form the MMM uses for calibration

Views
-----
mmm_training_frame     weekly_sales plus a 0-1 trend, exactly what the MMM is fitted on
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "data" / "mbc.duckdb"
SQL_DIR = Path(__file__).parent / "sql"


def connect(path: Path | str = DB_PATH, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    path = Path(path)
    if read_only and not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m mbc.warehouse` first.")
    return duckdb.connect(str(path), read_only=read_only)


def read_sql(name_or_query: str, path: Path | str = DB_PATH) -> pd.DataFrame:
    """Run a query, or a named file from sql/ (e.g. 'mmm_training_frame'), into pandas."""
    file = SQL_DIR / f"{name_or_query}.sql"
    query = file.read_text() if file.exists() else name_or_query
    with connect(path) as con:
        return con.execute(query).df()


def _write(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> None:
    con.register("_df", df)
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _df")
    con.unregister("_df")


def build(path: Path | str = DB_PATH) -> dict[str, int]:
    """(Re)create the warehouse from the simulators. Returns row counts per table."""
    from mbc.experiments.geo import simulate_geo
    from mbc.simulate import simulate

    sim = simulate()
    geo = simulate_geo()
    channel_truth = (
        pd.DataFrame(sim.truth["channels"]).T.rename_axis("channel").reset_index()
    )
    lift_path = ROOT / "data" / "lift_test.csv"
    lift = pd.read_csv(lift_path) if lift_path.exists() else None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(path)) as con:
        _write(con, "weekly_sales", sim.data)
        _write(con, "true_components", sim.components.reset_index())
        _write(con, "channel_truth", channel_truth)
        _write(con, "geo_panel", geo.panel)
        if lift is not None:
            _write(con, "lift_tests", lift)
        con.execute((SQL_DIR / "create_views.sql").read_text())
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        return {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}
