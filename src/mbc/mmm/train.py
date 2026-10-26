"""Fit the MMM on the synthetic data, evaluate it against the ground truth, log to MLflow.

    python -m mbc.mmm.train                    # informed priors, full sampling
    python -m mbc.mmm.train --priors weak      # sensitivity run
    python -m mbc.mmm.train --quick            # 2 chains x 200 draws, for checking code
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mbc  # noqa: F401  (PyTensor compiler fix; must precede pymc imports)
from mbc.mmm import evaluate as ev  # noqa: E402
from mbc.mmm import plots  # noqa: E402
from mbc.mmm.data import (  # noqa: E402
    CHANNELS,
    DATE,
    ROOT,
    SPEND_COLUMNS,
    load_synthetic,
    spend_share,
    split_holdout,
)
from mbc.mmm.model import build_mmm  # noqa: E402
from mbc.plotting import apply_style  # noqa: E402

REPORTS = ROOT / "reports" / "mmm"
MODELS = ROOT / "models"
MLFLOW_DB = ROOT / "mlflow.db"
MLFLOW_ARTIFACTS = ROOT / "mlruns"
EXPERIMENT = "mmm-synthetic"


def _to_original(samples, y_max: float, target_scale: float) -> np.ndarray:
    """Predictive samples as a (n_samples, n_dates) array in sales units.

    Depending on the call, PyMC-Marketing returns y scaled (max 1) or in sales units.
    Decide from the magnitude, then convert if needed.
    """
    da = samples["y"] if hasattr(samples, "data_vars") else samples
    if "sample" not in da.dims:
        da = da.stack(sample=("chain", "draw"))
    arr = da.transpose("sample", "date").values
    if np.nanmedian(np.abs(arr)) < 0.1 * y_max:  # scaled space
        arr = arr * target_scale
    return arr


def run(priors: str, draws: int, tune: int, chains: int, target_accept: float,
        run_name: str, use_mlflow: bool, seed: int = 42) -> dict:
    apply_style()
    out_dir = REPORTS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    truth = json.loads((ROOT / "data" / "ground_truth.json").read_text())
    components = pd.read_csv(ROOT / "data" / "true_components.csv", index_col="date",
                             parse_dates=True)
    df = load_synthetic()
    split = split_holdout(df, truth["config"]["holdout_weeks"])
    X_train, y_train = split.xy("train")
    X_hold, y_hold = split.xy("holdout")

    mmm = build_mmm(priors, spend_share(split.train))

    # 1. Prior predictive: what sales do the priors alone consider plausible?
    prior_idata = mmm.sample_prior_predictive(X_train, y_train, samples=500, random_seed=seed)
    target_scale = float(mmm.scalers._target)
    prior_y = _to_original(prior_idata.prior_predictive if hasattr(prior_idata, "prior_predictive")
                           else mmm.idata.prior_predictive, y_train.max(), target_scale)
    plots.plot_predictive(
        X_train[DATE], y_train, ev.predictive_interval(prior_y), None, None, None,
        "Prior predictive check: sales the priors allow, before seeing data",
        out_dir / "prior_predictive.png", band_label="prior predictive (90% band)",
    )
    prior_params = mmm.idata.prior

    # 2. Fit
    t0 = time.time()
    mmm.fit(X_train, y_train, draws=draws, tune=tune, chains=chains,
            target_accept=target_accept, random_seed=seed, progressbar=False)
    fit_seconds = time.time() - t0
    idata = mmm.idata

    # 3. Posterior predictive, in-sample and on the holdout
    pp_train = mmm.sample_posterior_predictive(X_train, extend_idata=True, combined=True,
                                               random_seed=seed, progressbar=False)
    pp_hold = mmm.sample_posterior_predictive(X_hold, include_last_observations=True,
                                              extend_idata=False, combined=True,
                                              random_seed=seed, progressbar=False)
    yhat_train = _to_original(pp_train, y_train.max(), target_scale)
    yhat_hold = _to_original(pp_hold, y_train.max(), target_scale)
    pi_train, pi_hold = ev.predictive_interval(yhat_train), ev.predictive_interval(yhat_hold)

    # 4. Numbers
    diag = ev.diagnostics(idata)
    contrib = ev.contributions_original_scale(idata)
    roi = ev.roi_samples(contrib, split.train)
    roi_table = ev.summarise_draws(roi, "roi")
    model_scales = dict(zip(SPEND_COLUMNS, idata.constant_data["channel_scale"].values.tolist(),
                            strict=True))
    truth_window = ev.true_values_for_window(truth, components, split.train, model_scales,
                                             target_scale)
    recovery = ev.recovery_table(idata.posterior, roi, truth_window,
                                 idata.constant_data["channel_scale"], target_scale)
    metrics = {
        **{k: v for k, v in diag.items() if not isinstance(v, str)},
        "mape_train": ev.mape(y_train, pi_train["mean"]),
        "mape_holdout": ev.mape(y_hold, pi_hold["mean"]),
        "coverage90_train": ev.coverage(y_train, pi_train),
        "coverage90_holdout": ev.coverage(y_hold, pi_hold),
        "recovery_coverage": float(recovery["covered"].mean()),
        "roi_coverage": float(recovery.query("parameter == 'roi'")["covered"].mean()),
        "fit_seconds": fit_seconds,
    }
    for c in CHANNELS:
        metrics[f"roi_{c}"] = float(roi_table.loc[c, "roi_mean"])
        metrics[f"roi_{c}_true"] = float(truth_window.loc[c, "roi"])

    # 5. Figures and tables
    plots.plot_predictive(X_train[DATE], y_train, pi_train, X_hold[DATE], y_hold, pi_hold,
                          f"Posterior predictive: holdout MAPE {metrics['mape_holdout']:.1%}",
                          out_dir / "posterior_predictive.png")
    plots.plot_contributions_vs_truth(contrib, components, X_train[DATE].to_numpy(),
                                      out_dir / "contributions_vs_truth.png")
    plots.plot_recovery(recovery, out_dir / "parameter_recovery.png")
    plots.plot_roi(roi_table, truth_window["roi"], out_dir / "roi.png")
    plots.plot_prior_vs_posterior(prior_params, idata.posterior, "adstock_alpha",
                                  "Adstock decay α: prior vs posterior",
                                  out_dir / "prior_vs_posterior_alpha.png")
    plots.plot_prior_vs_posterior(prior_params, idata.posterior, "saturation_lam",
                                  "Saturation λ: prior vs posterior",
                                  out_dir / "prior_vs_posterior_lam.png")
    recovery.to_csv(out_dir / "parameter_recovery.csv", index=False)
    roi_table.to_csv(out_dir / "roi.csv")
    (out_dir / "metrics.json").write_text(json.dumps(
        {**metrics, "worst_r_hat_param": diag["worst_r_hat_param"]}, indent=2))
    model_path = MODELS / f"{run_name}.nc"
    mmm.save(str(model_path))

    if use_mlflow:
        _log_mlflow(run_name, priors, draws, tune, chains, target_accept, seed, metrics, out_dir,
                    model_path)
    return metrics


def _log_mlflow(run_name, priors, draws, tune, chains, target_accept, seed, metrics, out_dir,
                model_path) -> None:
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=MLFLOW_ARTIFACTS.as_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"priors": priors, "draws": draws, "tune": tune, "chains": chains,
                           "target_accept": target_accept, "seed": seed, "adstock": "geometric",
                           "l_max": 8, "saturation": "logistic", "yearly_fourier": 2,
                           "holdout_weeks": 12})
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(out_dir), artifact_path="report")
        mlflow.log_artifact(str(model_path), artifact_path="model")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--priors", choices=["informed", "weak"], default="informed")
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1500)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--target-accept", type=float, default=0.97)
    p.add_argument("--quick", action="store_true", help="2 chains x 200 draws")
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-mlflow", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.draws, a.tune, a.chains = 200, 300, 2
    name = a.run_name or f"{a.priors}{'-quick' if a.quick else ''}"
    m = run(a.priors, a.draws, a.tune, a.chains, a.target_accept, name, not a.no_mlflow)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
                     indent=2))


if __name__ == "__main__":
    main()
