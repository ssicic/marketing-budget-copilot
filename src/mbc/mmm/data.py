"""Prepare the weekly table for the MMM: add a trend control and split off the holdout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mbc.simulate.generate import CHANNEL_COLS, CONTROL_COLS

ROOT = Path(__file__).resolve().parents[3]
SYNTHETIC_PATH = ROOT / "data" / "mmm_weekly.csv"

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


def load_synthetic(path: Path = SYNTHETIC_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m mbc.simulate` first.")
    df = pd.read_csv(path, parse_dates=[DATE])
    return add_trend(df)[[DATE, *SPEND_COLUMNS, *CONTROLS, TARGET]]


def split_holdout(df: pd.DataFrame, holdout_weeks: int = 12) -> Split:
    """Time-ordered split: the model never sees the last `holdout_weeks`."""
    if not 0 < holdout_weeks < len(df):
        raise ValueError("holdout_weeks must be between 1 and len(df) - 1")
    return Split(train=df.iloc[:-holdout_weeks].copy(), holdout=df.iloc[-holdout_weeks:].copy())


def spend_share(train: pd.DataFrame) -> pd.Series:
    totals = train[SPEND_COLUMNS].sum()
    return totals / totals.sum()
