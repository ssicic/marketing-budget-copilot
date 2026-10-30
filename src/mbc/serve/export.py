"""Export a fitted MMM to the serving artifact and register it in MLflow.

    python -m mbc.serve.export --model informed-calibrated

Each export creates a new version of the registered model "mmm-budget". Nothing is
served until a version is promoted (see mbc.serve.promote).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mbc  # noqa: F401
from mbc.mmm import evaluate as ev  # noqa: E402
from mbc.mmm.data import CHANNELS, ROOT, SPEND_COLUMNS, load_synthetic, split_holdout  # noqa: E402
from mbc.serve.artifact import save_artifact  # noqa: E402

REGISTERED_MODEL = "mmm-budget"
HDI = 0.9


def _hdi(x) -> list[float]:
    lo, hi = az.hdi(np.asarray(x).ravel(), hdi_prob=HDI)
    return [float(lo), float(hi)]


def build_artifact(idata: az.InferenceData, train: pd.DataFrame, out_dir: Path,
                   source: str, n_draws: int = 1000, seed: int = 0) -> dict:
    """Write model.npz + model.json for the API. Returns the metadata."""
    post = az.extract(idata, group="posterior", num_samples=n_draws, rng=seed)
    target_scale = float(idata.constant_data["target_scale"])

    def draws(var):
        return post[var].sel(channel=SPEND_COLUMNS).transpose("sample", "channel").values

    alpha, lam = draws("adstock_alpha"), draws("saturation_lam")
    beta = draws("saturation_beta") * target_scale
    scale = idata.constant_data["channel_scale"].sel(channel=SPEND_COLUMNS).values

    contrib = ev.contributions_original_scale(idata)  # (chain, draw, date, channel)
    roi = ev.roi_samples(contrib, train)
    total = contrib.sum("date")
    weekly = contrib.stack(s=("chain", "draw"))
    sales_total = float(train["sales"].sum())

    meta = {
        "source_model": source,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "training_window": [str(train["date"].min().date()), str(train["date"].max().date())],
        "currency": "EUR",
        "hdi_prob": HDI,
        "n_draws": int(alpha.shape[0]),
        "current_weekly_spend": {c: float(train[f"spend_{c}"].mean()) for c in CHANNELS},
        "roi": {
            c: {"mean": float(roi.sel(channel=f"spend_{c}").mean()),
                "hdi": _hdi(roi.sel(channel=f"spend_{c}"))}
            for c in CHANNELS
        },
        "contributions": {
            c: {
                "total_mean": float(total.sel(channel=f"spend_{c}").mean()),
                "total_hdi": _hdi(total.sel(channel=f"spend_{c}")),
                "share_of_sales_mean": float(total.sel(channel=f"spend_{c}").mean() / sales_total),
                "weekly": {
                    "dates": [str(pd.Timestamp(d).date()) for d in weekly["date"].values],
                    "mean": weekly.sel(channel=f"spend_{c}").mean("s").values.round(2).tolist(),
                    "low": weekly.sel(channel=f"spend_{c}").quantile(0.05, "s").values.round(2)
                    .tolist(),
                    "high": weekly.sel(channel=f"spend_{c}").quantile(0.95, "s").values.round(2)
                    .tolist(),
                },
            }
            for c in CHANNELS
        },
        "diagnostics": {k: v for k, v in ev.diagnostics(idata).items()
                        if isinstance(v, (int, float, bool))},
    }
    save_artifact(out_dir, CHANNELS, alpha, lam, beta, scale, meta)
    return meta


def register(artifact_dir: Path, meta: dict, tracking_uri: str,
             artifact_location: Path | None = None) -> str:
    """Log the artifact as an MLflow pyfunc model and register a new version."""
    import mlflow

    from mbc.mmm.train import MLFLOW_ARTIFACTS
    from mbc.serve.pyfunc import MMMBudgetModel

    mlflow.set_tracking_uri(tracking_uri)
    exp = "mmm-serving"
    if mlflow.get_experiment_by_name(exp) is None:
        location = (artifact_location or MLFLOW_ARTIFACTS).as_uri()
        mlflow.create_experiment(exp, artifact_location=location)
    mlflow.set_experiment(exp)
    with mlflow.start_run(run_name=f"export-{meta['source_model']}"):
        mlflow.log_params({"source_model": meta["source_model"], "n_draws": meta["n_draws"],
                           "training_window": " to ".join(meta["training_window"])})
        mlflow.log_metrics({f"roi_{c}": v["mean"] for c, v in meta["roi"].items()})
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=MMMBudgetModel(),
            artifacts={"npz": str(artifact_dir / "model.npz"),
                       "json": str(artifact_dir / "model.json")},
            code_paths=[str(Path(__file__).resolve().parents[1])],
            registered_model_name=REGISTERED_MODEL,
            input_example=pd.DataFrame([meta["current_weekly_spend"]]),
        )
    return str(info.registered_model_version)


def main() -> None:
    from mbc.mmm.train import MLFLOW_DB

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="informed-calibrated")
    a = p.parse_args()
    idata = az.from_netcdf(ROOT / "models" / f"{a.model}.nc")
    train = split_holdout(load_synthetic(), 12).train
    with tempfile.TemporaryDirectory() as tmp:
        meta = build_artifact(idata, train, Path(tmp), source=a.model)
        version = register(Path(tmp), meta, f"sqlite:///{MLFLOW_DB}")
    print(f"Registered {REGISTERED_MODEL} version {version} from {a.model}")
    print(json.dumps({c: round(v["mean"], 2) for c, v in meta["roi"].items()}))
    print(f"Promote with: python -m mbc.serve.promote --version {version}")


if __name__ == "__main__":
    main()
