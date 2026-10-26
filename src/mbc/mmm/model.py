"""Build the Bayesian MMM with documented priors.

All priors live on PyMC-Marketing's *scaled* space: each channel's spend is divided
by its max, and sales by max(sales). So "1.0" for a channel effect means "this
channel alone could deliver the biggest sales week".
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pymc_extras.prior import Prior
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM

from mbc.mmm.data import CHANNELS, CONTROLS, DATE, SPEND_COLUMNS, TARGET

PriorSet = Literal["informed", "weak"]

# Prior mean decay per channel (TV, search, social, radio): broadcast media is
# usually believed to carry over longer than search. Beta(a, b) has mean a / (a+b);
# a + b = 6 keeps these mild (a 90% interval covers most of 0-1).
_ALPHA_PRIOR = {"tv": (3.0, 3.0), "search": (1.5, 4.5), "social": (2.0, 4.0), "radio": (3.0, 3.0)}


def model_config(priors: PriorSet, spend_share: pd.Series) -> dict:
    """Priors for the MMM.

    informed: what we would use in practice, mildly informative.
    weak: near-flat priors for sensitivity analysis. If results change a lot between
          the two, the data alone isn't pinning those parameters down.
    """
    share = spend_share.reindex(SPEND_COLUMNS).to_numpy()
    if priors == "informed":
        alpha = Prior(
            "Beta",
            alpha=[_ALPHA_PRIOR[c][0] for c in CHANNELS],
            beta=[_ALPHA_PRIOR[c][1] for c in CHANNELS],
            dims="channel",
        )
        # Spend-share prior: a channel taking 40% of the budget is a priori allowed a
        # bigger effect than one taking 5%. sigma = 2 * share keeps every channel's
        # plausible max contribution well below total sales.
        beta = Prior("HalfNormal", sigma=list(2.0 * share), dims="channel")
    elif priors == "weak":
        alpha = Prior("Beta", alpha=1.0, beta=1.0, dims="channel")
        beta = Prior("HalfNormal", sigma=1.0, dims="channel")
    else:
        raise ValueError(f"unknown prior set {priors!r}")

    return {
        "adstock_alpha": alpha,
        # Gamma(2, 1): mean 2, 90% of mass in about 0.36-4.7. The library default
        # Gamma(3, 1) puts little mass below 1, which would push unsaturated channels
        # (like our social) towards looking saturated.
        "saturation_lam": Prior("Gamma", alpha=2.0, beta=1.0, dims="channel"),
        "saturation_beta": beta,
        # Scaled sales live in (0, 1]; a typical base is around half the max.
        "intercept": Prior("Normal", mu=0.5, sigma=0.5),
        # Holidays, promo and trend: each shifts sales by a modest share of the max.
        "gamma_control": Prior("Normal", mu=0.0, sigma=0.5, dims="control"),
        # Laplace shrinks small Fourier terms towards 0 (like a lasso).
        "gamma_fourier": Prior("Laplace", mu=0.0, b=0.2, dims="fourier_mode"),
        # Weekly noise of a few percent of max sales; HalfNormal(0.1) allows up to ~20%.
        "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=0.1), dims="date"),
    }


def build_mmm(
    priors: PriorSet,
    spend_share: pd.Series,
    *,
    l_max: int = 8,
    yearly_seasonality: int = 2,
    sampler_config: dict | None = None,
) -> MMM:
    return MMM(
        date_column=DATE,
        channel_columns=SPEND_COLUMNS,
        target_column=TARGET,
        control_columns=CONTROLS,
        adstock=GeometricAdstock(l_max=l_max),
        saturation=LogisticSaturation(),
        yearly_seasonality=yearly_seasonality,
        model_config=model_config(priors, spend_share),
        sampler_config=sampler_config,
    )
