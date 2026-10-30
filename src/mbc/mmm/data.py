"""Prepare the weekly table for the MMM: add a trend control and split off the holdout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mbc.simulate.generate import CHANNEL_COLS, CONTROL_COLS

ROOT = Path(__file__).resolve().parents[3]

CHANNELS = list(CHANNEL_COLS)  # tv, search, social, radio
SPEND_COLUMNS = [CHANNEL_COLS[c] for c in CHANNELS]
CONTROLS = [*CONTROL_COLS, "trend"]
TARGET = "sales"
DATE = "date"


@dataclass
class Split:
    train: pd.DataFrame
    holdout: pd.DataFrame

    def xy(self, part: str) -> tuple[pd.DataFrame, pd.Series]:
        df = getattr(self, part)
        return df.drop(columns=TARGET), df[TARGET]


def add_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Linear trend from 0 at the first week to 1 at the last.

    PyMC-Marketing doesn't scale controls, so we keep the trend on [0, 1]. Its
    coefficient is then "growth over the whole period, as a share of max sales".
    """
    return df.assign(trend=np.arange(len(df)) / (len(df) - 1))


def load_synthetic(db_path: Path | None = None) -> pd.DataFrame:
    """The MMM training frame, read with SQL from the DuckDB warehouse.

    The trend column is computed in SQL (see warehouse/sql/create_views.sql).
    """
    from mbc.warehouse import DB_PATH, read_sql

    df = read_sql("mmm_training_frame", db_path or DB_PATH)
    df[DATE] = pd.to_datetime(df[DATE])
    return df[[DATE, *SPEND_COLUMNS, *CONTROLS, TARGET]]


def split_holdout(df: pd.DataFrame, holdout_weeks: int = 12) -> Split:
    """Time-ordered split: the model never sees the last `holdout_weeks`."""
    if not 0 < holdout_weeks < len(df):
        raise ValueError("holdout_weeks must be between 1 and len(df) - 1")
    return Split(train=df.iloc[:-holdout_weeks].copy(), holdout=df.iloc[-holdout_weeks:].copy())


def spend_share(train: pd.DataFrame) -> pd.Series:
    totals = train[SPEND_COLUMNS].sum()
    return totals / totals.sum()
