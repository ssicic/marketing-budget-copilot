"""Split the synthetic company into regions and run a social-spend geo-lift test.

Each region is a scaled copy of the national business: region r gets a share w_r of
national spend and of every media effect. Because adstock is linear and saturation
is applied to spend / max(spend), a region's response is exactly w_r times the
national response to national spend. So the true lift of a test is known exactly.

Regions also get their own base level, a small region-specific trend and their own
noise, so treated and control regions are similar but not identical, as in reality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mbc.simulate import SimConfig, simulate
from mbc.transforms import media_response


@dataclass(frozen=True)
class GeoConfig:
    seed: int = 7
    n_regions: int = 20
    n_treated: int = 10
    channel: str = "social"
    start_week: int = 120  # 0-based week index where the test starts
    test_weeks: int = 8
    lift_factor: float = 1.5  # treated regions spend 50% more on the channel
    region_noise: float = 0.05  # weekly noise sd as a share of the region's mean sales
    trend_drift_sd: float = 0.02  # region-specific extra growth over the period


@dataclass
class GeoResult:
    panel: pd.DataFrame  # one row per region-week
    truth: dict


def simulate_geo(cfg: GeoConfig | None = None, sim_config: SimConfig | None = None) -> GeoResult:
    cfg = cfg or GeoConfig()
    sim = simulate(sim_config)
    scfg = sim_config or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    nat, comp, truth = sim.data, sim.components, sim.truth
    n_weeks = len(nat)
    t = np.arange(n_weeks)

    weights = rng.dirichlet(np.full(cfg.n_regions, 5.0))
    treated = np.zeros(cfg.n_regions, bool)
    treated[rng.choice(cfg.n_regions, cfg.n_treated, replace=False)] = True
    window = np.zeros(n_weeks, bool)
    window[cfg.start_week : cfg.start_week + cfg.test_weeks] = True

    channels = list(scfg.channels)
    base_nat = comp.drop(columns=[*channels, "noise"]).sum(axis=1).to_numpy()
    media_nat = comp[channels].sum(axis=1).to_numpy()

    # National response of the test channel, with and without the extra spend.
    ch = truth["channels"][cfg.channel]
    spend = nat[f"spend_{cfg.channel}"].to_numpy()
    boosted = np.where(window, spend * cfg.lift_factor, spend)
    kw = dict(alpha=ch["alpha"], lam=ch["lam"], beta=ch["beta"], scale=ch["spend_scale"],
              l_max=scfg.l_max)
    resp_base, resp_test = media_response(spend, **kw), media_response(boosted, **kw)
    extra_nat = resp_test - resp_base  # >0 from the first test week, decays after it

    rows = []
    for r in range(cfg.n_regions):
        w = weights[r]
        drift = 1 + rng.normal(0, cfg.trend_drift_sd) * t / (n_weeks - 1)
        base = w * base_nat * drift
        media = w * media_nat + (w * extra_nat if treated[r] else 0.0)
        noise = rng.normal(0, cfg.region_noise * (w * nat["sales"].mean()), n_weeks)
        region_spend = w * (boosted if treated[r] else spend)
        rows.append(pd.DataFrame({
            "region": f"r{r:02d}", "week": t, "date": nat["date"], "treated": int(treated[r]),
            "post": window.astype(int), "spend_test_channel": region_spend,
            "sales": base + media + noise, "true_extra_sales": w * extra_nat * treated[r],
            "weight": w,
        }))
    panel = pd.concat(rows, ignore_index=True)

    w_t = weights[treated].sum()
    in_window = float(w_t * extra_nat[window].sum())
    total = float(w_t * extra_nat.sum())
    extra_spend = float(w_t * (boosted - spend).sum())
    truth_out = {
        "channel": cfg.channel,
        "test_weeks": [int(cfg.start_week), int(cfg.start_week + cfg.test_weeks - 1)],
        "test_dates": [str(nat["date"].iloc[cfg.start_week].date()),
                       str(nat["date"].iloc[cfg.start_week + cfg.test_weeks - 1].date())],
        "treated_regions": sorted(panel.loc[panel.treated == 1, "region"].unique().tolist()),
        "treated_weight": float(w_t),
        "extra_spend": extra_spend,
        "lift_in_window": in_window,
        "lift_total_incl_carryover": total,
        "lift_per_region_week": in_window / (cfg.n_treated * cfg.test_weeks),
        "iroas_in_window": in_window / extra_spend,
        "iroas_total": total / extra_spend,
    }
    return GeoResult(panel=panel, truth=truth_out)
