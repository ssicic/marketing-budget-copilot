"""Request and response models. These also define the OpenAPI docs at /docs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Interval(BaseModel):
    mean: float
    low: float = Field(description="Lower bound of the 90% interval")
    high: float = Field(description="Upper bound of the 90% interval")


class Health(BaseModel):
    status: str
    model_version: str
    source_model: str
    training_window: list[str]
    channels: list[str]


class ChannelROI(BaseModel):
    channel: str
    roi: Interval = Field(description="Average ROI over the training window: sales per € spent")
    marginal_roi: Interval = Field(description="Sales from the next € at current weekly spend")
    current_weekly_spend: float


class ROIResponse(BaseModel):
    currency: str
    channels: list[ChannelROI]


class WeeklySeries(BaseModel):
    dates: list[str]
    mean: list[float]
    low: list[float]
    high: list[float]


class ChannelContribution(BaseModel):
    channel: str
    total: Interval = Field(description="Sales caused by the channel over the training window")
    share_of_sales: float
    weekly: WeeklySeries | None = None


class ContributionsResponse(BaseModel):
    currency: str
    training_window: list[str]
    channels: list[ChannelContribution]


class SimulateRequest(BaseModel):
    weekly_spend: dict[str, float] = Field(
        description="Proposed weekly spend per channel (€). Channels left out keep their "
                    "current spend.",
        examples=[{"tv": 18000, "social": 18000}],
    )
    weeks: int = Field(13, ge=1, le=52, description="Length of the period (13 = a quarter)")


class SimulateResponse(BaseModel):
    weeks: int
    current_weekly_spend: dict[str, float]
    proposed_weekly_spend: dict[str, float]
    spend_change_total: float = Field(description="Extra (or saved) spend over the period")
    sales_change_total: Interval = Field(description="Change in media-driven sales over the "
                                                     "period")
    probability_sales_increase: float
    warnings: list[str]


class OptimizeRequest(BaseModel):
    total_budget: float | None = Field(
        None, gt=0, description="Total spend over the period (€). Default: current spend.")
    weeks: int = Field(13, ge=1, le=52)
    max_change: float = Field(
        0.5, gt=0, le=1.0,
        description="Each channel may move at most this share from its current spend. The "
                    "model hasn't seen spend far outside that range.")


class ChannelAllocation(BaseModel):
    channel: str
    current_weekly: float
    recommended_weekly: float
    change_pct: float
    at_limit: bool


class OptimizeResponse(BaseModel):
    weeks: int
    total_budget: float
    allocation: list[ChannelAllocation]
    sales_change_total: Interval = Field(description="Expected change vs current split")
    probability_sales_increase: float
    marginal_roi_at_optimum: dict[str, float]
    warnings: list[str]
