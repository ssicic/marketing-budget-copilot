"""Turn a fitted MMM into numbers: diagnostics, fit, ROI and parameter recovery.

Functions take plain xarray / pandas objects, so they can be tested without
fitting a model.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from mbc.mmm.data import CHANNELS, SPEND_COLUMNS
from mbc.simulate.generate import lam_at_scale

HDI_PROB = 0.90
PARAMS = ["adstock_alpha", "saturation_lam", "saturation_beta", "gamma_control",
          "gamma_fourier", "intercept_contribution", "y_sigma"]


# ---------------------------------------------------------------- sampling health


def diagnostics(idata: az.InferenceData, var_names: list[str] = PARAMS) -> dict:
    """Convergence summary across all model parameters.

    r_hat: compares between-chain and within-chain variance. 1.00 means the chains
           agree; above 1.01 means they found different answers.
    ess:   effective sample size. Autocorrelated draws carry less information than
           independent ones; we want at least 400 (bulk and tail).
    divergences: the sampler hit geometry it couldn't follow. Any > 0 is a warning
           that the posterior may be biased in that region.
    """
    names = [v for v in var_names if v in idata.posterior]
    s = az.summary(idata, var_names=names, kind="diagnostics")
    div = int(idata.sample_stats["diverging"].sum())
    n_draws = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
    out = {
        "max_r_hat": float(s["r_hat"].max()),
        "min_ess_bulk": float(s["ess_bulk"].min()),
        "min_ess_tail": float(s["ess_tail"].min()),
        "divergences": div,
        "divergence_rate": div / n_draws,
        "worst_r_hat_param": str(s["r_hat"].idxmax()),
    }
    out["converged"] = bool(
        out["max_r_hat"] < 1.01 and out["min_ess_bulk"] > 400 and out["min_ess_tail"] > 400
        and div == 0
    )
    return out


# ---------------------------------------------------------------- fit quality


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error."""
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if np.any(actual == 0):
        raise ValueError("MAPE undefined when actual contains zeros")
    return float(np.mean(np.abs((actual - predicted) / actual)))


def predictive_interval(samples: np.ndarray, prob: float = HDI_PROB) -> pd.DataFrame:
    """Mean and central interval per time step from a (n_samples, n_dates) array."""
    lo, hi = np.quantile(samples, [(1 - prob) / 2, (1 + prob) / 2], axis=0)
    return pd.DataFrame({"mean": samples.mean(axis=0), "lower": lo, "upper": hi})


def coverage(actual: np.ndarray, interval: pd.DataFrame) -> float:
    """Share of observations inside the predictive interval. ~90% is well calibrated."""
    a = np.asarray(actual)
    return float(np.mean((a >= interval["lower"].to_numpy()) & (a <= interval["upper"].to_numpy())))


# ---------------------------------------------------------------- contributions and ROI


def _hdi(samples: np.ndarray, prob: float = HDI_PROB) -> tuple[float, float]:
    lo, hi = az.hdi(np.asarray(samples).ravel(), hdi_prob=prob)
    return float(lo), float(hi)


def contributions_original_scale(idata: az.InferenceData) -> xr.DataArray:
    """channel_contribution (chain, draw, date, channel) in sales units."""
    return idata.posterior["channel_contribution"] * idata.constant_data["target_scale"]


def roi_samples(contrib: xr.DataArray, spend: pd.DataFrame,
                columns: list[str] = SPEND_COLUMNS) -> xr.DataArray:
    """Average ROI per posterior draw: total contribution / total spend, per channel."""
    total_spend = xr.DataArray(
        spend[columns].sum().to_numpy(), dims="channel", coords={"channel": columns},
    )
    return contrib.sum("date") / total_spend


def summarise_draws(samples: xr.DataArray, name: str) -> pd.DataFrame:
    """Mean, median and 90% HDI per channel."""
    rows = []
    for ch in samples["channel"].values:
        x = samples.sel(channel=ch).values.ravel()
        lo, hi = _hdi(x)
        rows.append({"channel": _short(ch), f"{name}_mean": x.mean(),
                     f"{name}_median": np.median(x), f"{name}_hdi_low": lo,
                     f"{name}_hdi_high": hi})
    return pd.DataFrame(rows).set_index("channel")


def _short(col: str) -> str:
    return col.removeprefix("spend_")


# ---------------------------------------------------------------- parameter recovery


def true_values_for_window(
    truth: dict, components: pd.DataFrame, spend: pd.DataFrame, model_scales: dict[str, float],
    target_scale: float,
) -> pd.DataFrame:
    """Ground truth expressed the way the model sees it, for the training window.

    - ROI is recomputed over the training weeks only, because that's all the model saw.
    - lam is converted to the model's channel scale (max spend in the *training* window).
    - beta is converted to scaled-sales units (divided by the model's target scale).
    """
    rows = {}
    for c in CHANNELS:
        t = truth["channels"][c]
        col = f"spend_{c}"
        window_contrib = components.loc[pd.DatetimeIndex(spend["date"]), c]
        lam_model = lam_at_scale(t["lam"], t["spend_scale"], model_scales[col])
        rows[c] = {
            "adstock_alpha": t["alpha"],
            "saturation_lam": lam_model,
            "saturation_beta": t["beta"] / target_scale,
            "beta_eur": t["beta"],
            "half_saturation_spend": t["half_saturation_spend"],
            "roi": float(window_contrib.sum() / spend[col].sum()),
        }
    return pd.DataFrame(rows).T


def recovery_table(posterior: xr.Dataset, roi: xr.DataArray, truth_window: pd.DataFrame,
                   channel_scale: xr.DataArray, target_scale: float) -> pd.DataFrame:
    """Estimated vs true value for each channel parameter, with 90% HDI and coverage."""
    est = {
        "adstock_alpha": posterior["adstock_alpha"],
        "saturation_lam": posterior["saturation_lam"],
        "beta_eur": posterior["saturation_beta"] * target_scale,
        "half_saturation_spend": np.log(3) / posterior["saturation_lam"] * channel_scale,
        "roi": roi,
    }
    rows = []
    for param, samples in est.items():
        for col in samples["channel"].values:
            c = _short(str(col))
            x = samples.sel(channel=col).values.ravel()
            lo, hi = _hdi(x)
            true = float(truth_window.loc[c, param])
            rows.append({
                "channel": c, "parameter": param, "true": true, "mean": float(x.mean()),
                "hdi_low": lo, "hdi_high": hi, "covered": lo <= true <= hi,
                "rel_error": float((x.mean() - true) / true),
            })
    return pd.DataFrame(rows)
