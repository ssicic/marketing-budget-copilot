"""The planted 'truth' for the synthetic dataset.

Every number the model should later recover lives here, so the data-generating
process can be read in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChannelTruth:
    alpha: float  # adstock decay per week
    lam: float  # logistic saturation steepness, on spend / max(spend)
    target_roi: float  # average ROI over the full period; beta is solved from it


@dataclass(frozen=True)
class SimConfig:
    seed: int = 42
    n_weeks: int = 156
    start_date: str = "2023-01-02"  # a Monday
    l_max: int = 8

    channels: dict[str, ChannelTruth] = field(
        default_factory=lambda: {
            # Slow-decaying brand channel, moderately saturated.
            "tv": ChannelTruth(alpha=0.7, lam=1.5, target_roi=1.4),
            # Captures existing demand: immediate and heavily saturated, so its
            # average ROI is high but the next euro is worth much less.
            "search": ChannelTruth(alpha=0.1, lam=3.0, target_roi=2.8),
            # Growing channel, still far from saturation.
            "social": ChannelTruth(alpha=0.3, lam=0.8, target_roi=2.0),
            # Small, weak channel.
            "radio": ChannelTruth(alpha=0.5, lam=1.2, target_roi=0.7),
        }
    )

    # Base sales (EUR per week)
    intercept: float = 250_000.0
    trend_total_growth: float = 0.20  # base grows 20% over the whole period
    # Yearly Fourier coefficients: (sin1, cos1, sin2, cos2), in EUR
    seasonality: tuple[float, float, float, float] = (-12_000.0, -15_000.0, 4_000.0, 6_000.0)
    black_friday_effect: float = 60_000.0
    christmas_effect: float = 45_000.0
    promo_effect: float = 30_000.0
    promo_rate: float = 0.10
    noise_sd: float = 10_000.0

    holdout_weeks: int = 12
