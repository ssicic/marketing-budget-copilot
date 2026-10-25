"""Media transformations in plain NumPy.

These mirror PyMC-Marketing's `geometric_adstock(normalize=True)` and
`logistic_saturation` exactly, so the data generator plants effects in the same
functional form the model fits. tests/test_transforms.py checks the match.
"""

from __future__ import annotations

import numpy as np


def geometric_adstock(
    x: np.ndarray, alpha: float, l_max: int = 8, normalize: bool = True
) -> np.ndarray:
    """Carry-over: today's effective spend is a decaying sum of past spend.

    adstock[t] = sum_{l=0}^{l_max-1} w_l * x[t-l],  w_l = alpha**l
    With normalize=True the weights sum to 1, so adstock only *spreads* spend over
    time and never inflates it. Spend before the series starts is treated as 0.
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    x = np.asarray(x, dtype=float)
    weights = alpha ** np.arange(l_max)
    if normalize:
        weights = weights / weights.sum()
    # Full convolution, truncated to the input length = causal filter with zero padding.
    return np.convolve(x, weights)[: len(x)]


def logistic_saturation(x: np.ndarray, lam: float) -> np.ndarray:
    """Diminishing returns: (1 - e^{-lam x}) / (1 + e^{-lam x}), rising from 0 towards 1.

    Larger lam means the channel saturates at lower spend. Half of the maximum
    effect is reached at x = ln(3) / lam.
    """
    if lam <= 0:
        raise ValueError(f"lam must be positive, got {lam}")
    x = np.asarray(x, dtype=float)
    e = np.exp(-lam * x)
    return (1.0 - e) / (1.0 + e)


def media_response(
    spend: np.ndarray,
    *,
    alpha: float,
    lam: float,
    beta: float,
    scale: float,
    l_max: int = 8,
) -> np.ndarray:
    """Sales contribution of one channel: beta * saturation(adstock(spend / scale)).

    `scale` is the channel's max spend, which is how PyMC-Marketing scales channels
    before transforming them. `beta` is the maximum weekly contribution in sales units.
    """
    return beta * logistic_saturation(geometric_adstock(spend / scale, alpha, l_max), lam)


def adstock_half_life(alpha: float) -> float:
    """Weeks until the carry-over of a single burst falls to half its first-week effect."""
    if alpha <= 0:
        return 0.0
    return float(np.log(0.5) / np.log(alpha))
