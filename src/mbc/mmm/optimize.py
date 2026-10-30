"""Budget optimisation on top of response curves.

Steady-state view: if a channel gets the same spend s every week, normalised adstock
equals s, so its weekly sales contribution is

    response(s) = beta * logistic(lam * s / scale)

The optimiser picks weekly spend per channel to maximise total expected response,
keeping the total budget fixed and each channel within bounds. "Expected" means the
average over posterior draws, so the answer accounts for uncertainty. The same code
runs on the true curves, which lets us measure what a wrong model costs.

Simplification: spend is assumed constant across weeks, so flighting (TV bursts) is
ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class Curves:
    """Response curve parameters: one row per posterior draw (or a single row for truth)."""

    channels: list[str]
    beta: np.ndarray  # (n_draws, n_channels), max weekly contribution in EUR
    lam: np.ndarray  # (n_draws, n_channels), on spend / scale
    scale: np.ndarray  # (n_channels,), EUR

    def __post_init__(self):
        self.beta = np.atleast_2d(np.asarray(self.beta, float))
        self.lam = np.atleast_2d(np.asarray(self.lam, float))
        self.scale = np.asarray(self.scale, float)
        if self.beta.shape != self.lam.shape or self.beta.shape[1] != len(self.channels):
            raise ValueError("beta and lam must be (n_draws, n_channels)")

    def response(self, spend: np.ndarray) -> np.ndarray:
        """Weekly contribution per draw and channel, shape (n_draws, n_channels)."""
        e = np.exp(-self.lam * np.asarray(spend, float) / self.scale)
        return self.beta * (1 - e) / (1 + e)

    def marginal(self, spend: np.ndarray) -> np.ndarray:
        """Extra sales from the next euro (marginal ROI), shape (n_draws, n_channels)."""
        e = np.exp(-self.lam * np.asarray(spend, float) / self.scale)
        return self.beta * 2 * self.lam * e / ((1 + e) ** 2 * self.scale)


@dataclass
class Allocation:
    spend: pd.Series  # weekly EUR per channel
    expected_response: float  # weekly EUR, averaged over draws
    success: bool
    message: str


def optimise(curves: Curves, total_budget: float, bounds: dict[str, tuple[float, float]],
             x0: np.ndarray | None = None) -> Allocation:
    """Maximise mean total weekly response subject to sum(spend) == total_budget."""
    lo = np.array([bounds[c][0] for c in curves.channels])
    hi = np.array([bounds[c][1] for c in curves.channels])
    if lo.sum() > total_budget + 1e-6 or hi.sum() < total_budget - 1e-6:
        raise ValueError("bounds make the budget infeasible")
    if x0 is None:
        x0 = lo + (hi - lo) * (total_budget - lo.sum()) / (hi - lo).sum()

    # Work in thousands of euros so the optimiser's step sizes are well scaled.
    k = 1_000.0

    def neg(x):
        return -curves.response(x * k).sum(axis=1).mean() / k

    def neg_grad(x):
        return -curves.marginal(x * k).mean(axis=0)

    res = minimize(
        neg, x0 / k, jac=neg_grad, method="SLSQP",
        bounds=list(zip(lo / k, hi / k, strict=True)),
        constraints=[{"type": "eq", "fun": lambda x: x.sum() - total_budget / k,
                      "jac": lambda x: np.ones_like(x)}],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    x = res.x * k
    return Allocation(
        spend=pd.Series(x, index=curves.channels),
        expected_response=float(curves.response(x).sum(axis=1).mean()),
        success=bool(res.success),
        message=str(res.message),
    )


def relative_bounds(current: pd.Series, band: float = 0.5) -> dict[str, tuple[float, float]]:
    """Each channel may move at most +-band (50%) from its current spend.

    The model has only seen spend in roughly this range; beyond it, the curve is an
    extrapolation driven by the functional form and the priors.
    """
    return {c: (float(v * (1 - band)), float(v * (1 + band))) for c, v in current.items()}


def uplift_draws(curves: Curves, new: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Weekly sales change from moving current -> new, one value per posterior draw."""
    return curves.response(new).sum(axis=1) - curves.response(current).sum(axis=1)
