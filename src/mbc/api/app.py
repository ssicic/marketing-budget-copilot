"""FastAPI service over the champion MMM.

    uvicorn mbc.api.app:app --port 8080         # docs at http://localhost:8080/docs

All numbers come from the model artifact (artifacts/champion by default, or the
MBC_ARTIFACT_DIR environment variable). Intervals are central 90% intervals over
posterior draws.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request

from mbc.api import schemas as s
from mbc.mmm.optimize import optimise, uplift_draws
from mbc.serve.artifact import ModelArtifact, load_artifact

DEFAULT_ARTIFACT = Path(__file__).resolve().parents[3] / "artifacts" / "champion"
EXTRAPOLATION_LIMIT = 0.5  # beyond +-50% of current spend the curves are extrapolated


def interval(draws: np.ndarray) -> s.Interval:
    lo, hi = np.quantile(draws, [0.05, 0.95])
    return s.Interval(mean=float(np.mean(draws)), low=float(lo), high=float(hi))


def get_model(request: Request) -> ModelArtifact:
    return request.app.state.model


Model = Annotated[ModelArtifact, Depends(get_model)]


def create_app(artifact_dir: Path | str | None = None) -> FastAPI:
    path = Path(artifact_dir or os.environ.get("MBC_ARTIFACT_DIR", DEFAULT_ARTIFACT))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = load_artifact(path)
        version_file = path / "VERSION"
        app.state.version = version_file.read_text().strip() if version_file.exists() else "dev"
        yield

    app = FastAPI(
        title="Marketing Budget Copilot API",
        version="1.0",
        description="Channel ROI, contributions, what-if simulation and budget optimisation "
                    "from a Bayesian media mix model. Every number comes with a 90% interval.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=s.Health, tags=["meta"])
    def health(request: Request, m: Model):
        return s.Health(status="ok", model_version=request.app.state.version,
                        source_model=m.meta["source_model"],
                        training_window=m.meta["training_window"], channels=m.channels)

    @app.get("/channels/roi", response_model=s.ROIResponse, tags=["channels"])
    def channel_roi(m: Model):
        current = np.array([m.current_weekly_spend[c] for c in m.channels])
        marginal = m.curves.marginal(current)
        return s.ROIResponse(currency=m.meta["currency"], channels=[
            s.ChannelROI(
                channel=c,
                roi=s.Interval(mean=m.roi[c]["mean"], low=m.roi[c]["hdi"][0],
                               high=m.roi[c]["hdi"][1]),
                marginal_roi=interval(marginal[:, i]),
                current_weekly_spend=m.current_weekly_spend[c],
            )
            for i, c in enumerate(m.channels)
        ])

    @app.get("/channels/contributions", response_model=s.ContributionsResponse,
             tags=["channels"])
    def channel_contributions(
        m: Model,
        weekly: Annotated[bool, Query(description="Include weekly series")] = False,
    ):
        out = []
        for c in m.channels:
            d = m.contributions[c]
            out.append(s.ChannelContribution(
                channel=c,
                total=s.Interval(mean=d["total_mean"], low=d["total_hdi"][0],
                                 high=d["total_hdi"][1]),
                share_of_sales=d["share_of_sales_mean"],
                weekly=s.WeeklySeries(**d["weekly"]) if weekly else None,
            ))
        return s.ContributionsResponse(currency=m.meta["currency"],
                                       training_window=m.meta["training_window"], channels=out)

    @app.post("/simulate", response_model=s.SimulateResponse, tags=["budget"])
    def simulate(req: s.SimulateRequest, m: Model):
        unknown = set(req.weekly_spend) - set(m.channels)
        if unknown:
            raise HTTPException(422, f"unknown channels {sorted(unknown)}; use {m.channels}")
        if any(v < 0 for v in req.weekly_spend.values()):
            raise HTTPException(422, "spend can't be negative")
        cur = m.current_weekly_spend
        proposed = {c: float(req.weekly_spend.get(c, cur[c])) for c in m.channels}
        cur_arr = np.array([cur[c] for c in m.channels])
        new_arr = np.array([proposed[c] for c in m.channels])
        change = uplift_draws(m.curves, new_arr, cur_arr) * req.weeks
        warnings = [
            f"{c}: {proposed[c]:,.0f}/week is {proposed[c] / cur[c] - 1:+.0%} vs current; "
            "the model hasn't seen spend this far from history, so treat it as a rough guess"
            for c in m.channels
            if cur[c] > 0 and abs(proposed[c] / cur[c] - 1) > EXTRAPOLATION_LIMIT
        ]
        return s.SimulateResponse(
            weeks=req.weeks, current_weekly_spend=cur, proposed_weekly_spend=proposed,
            spend_change_total=float((new_arr - cur_arr).sum() * req.weeks),
            sales_change_total=interval(change),
            probability_sales_increase=float((change > 0).mean()), warnings=warnings,
        )

    @app.post("/optimize", response_model=s.OptimizeResponse, tags=["budget"])
    def optimize(req: s.OptimizeRequest, m: Model):
        cur = np.array([m.current_weekly_spend[c] for c in m.channels])
        weekly_budget = (req.total_budget / req.weeks) if req.total_budget else float(cur.sum())
        # Keep the current mix as the anchor, scaled to the requested budget.
        anchor = cur * weekly_budget / cur.sum()
        bounds = {c: (anchor[i] * (1 - req.max_change), anchor[i] * (1 + req.max_change))
                  for i, c in enumerate(m.channels)}
        result = optimise(m.curves, weekly_budget, bounds)
        if not result.success:
            raise HTTPException(500, f"optimiser did not converge: {result.message}")
        rec = result.spend.to_numpy()
        change = uplift_draws(m.curves, rec, anchor) * req.weeks
        alloc = [
            s.ChannelAllocation(
                channel=c, current_weekly=float(cur[i]), recommended_weekly=float(rec[i]),
                change_pct=float(rec[i] / anchor[i] - 1),
                at_limit=bool(np.isclose(rec[i], bounds[c][0], rtol=1e-3)
                              or np.isclose(rec[i], bounds[c][1], rtol=1e-3)),
            )
            for i, c in enumerate(m.channels)
        ]
        mroi = m.curves.marginal(rec).mean(axis=0)
        ratio = weekly_budget / cur.sum()
        warnings = [] if abs(ratio - 1) <= EXTRAPOLATION_LIMIT else [
            f"total budget is {ratio - 1:+.0%} vs current spend; the model hasn't seen spend "
            "levels this far from history, so treat the split as a rough guess"
        ]
        return s.OptimizeResponse(
            weeks=req.weeks, total_budget=weekly_budget * req.weeks, allocation=alloc,
            sales_change_total=interval(change),
            probability_sales_increase=float((change > 0).mean()),
            marginal_roi_at_optimum={c: float(mroi[i]) for i, c in enumerate(m.channels)},
            warnings=warnings,
        )

    return app


app = create_app()
