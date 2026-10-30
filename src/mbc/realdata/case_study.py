"""Real-data case study: one Facebook + Instagram apparel brand from the Conjura data.

    python -m mbc.realdata.case_study            # all four fits (about 15 minutes)
    python -m mbc.realdata.case_study --quick    # small samples, for checking code

Four fits, four questions:
1. semi-synthetic: plant known effects on the brand's real spend. Can the model
   find them? (Is this data informative at all?)
2. real, informed priors: the actual answer.
3. real, weak priors: does the answer depend on our assumptions?
4. real, first 75% of weeks: is the answer stable over time?
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

import mbc  # noqa: F401
from mbc.mmm import evaluate as ev  # noqa: E402
from mbc.mmm import plots  # noqa: E402
from mbc.mmm.data import ROOT, split_holdout  # noqa: E402
from mbc.mmm.model import build_mmm  # noqa: E402
from mbc.mmm.train import _to_original  # noqa: E402
from mbc.plotting import REAL_CHANNEL_COLORS, TEXT_SECONDARY, TRUTH_COLOR, apply_style  # noqa: E402
from mbc.realdata.prepare import REAL_CONTROLS, load_brand  # noqa: E402
from mbc.simulate.generate import yearly_fourier  # noqa: E402
from mbc.transforms import media_response  # noqa: E402

CHANNELS = ["spend_facebook", "spend_instagram"]
COLORS = {"spend_facebook": REAL_CHANNEL_COLORS["meta_facebook"],
          "spend_instagram": REAL_CHANNEL_COLORS["meta_instagram"]}
# Digital social ads: short carry-over expected (prior mean 0.33).
ALPHA_PRIOR = {"spend_facebook": (2.0, 4.0), "spend_instagram": (2.0, 4.0)}
YEARLY_FOURIER = 3  # Q4 peak is sharp; 3 harmonics fit it better than 2 (checked by OLS)
OUT = ROOT / "reports" / "realdata"

# Planted truth for the semi-synthetic check: plausible, and different per channel.
PLANTED = {"spend_facebook": {"alpha": 0.3, "lam": 1.0, "roi": 1.5},
           "spend_instagram": {"alpha": 0.2, "lam": 1.5, "roi": 1.0}}


def semi_synthetic(df: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Same spend, same calendar, same noise level as the brand; media effects we choose.

    Baseline = the brand's revenue explained by calendar controls and seasonality only
    (OLS fit), so it keeps the real shape. Noise = OLS residual spread.
    """
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(df["date"])
    X = np.column_stack([df[REAL_CONTROLS].to_numpy(), yearly_fourier(dates, YEARLY_FOURIER)])
    ols = sm.OLS(df["sales"].to_numpy(), sm.add_constant(X)).fit()
    base = ols.fittedvalues
    noise_sd = float(np.std(ols.resid))

    truth, media = {}, np.zeros(len(df))
    for col, p in PLANTED.items():
        spend = df[col].to_numpy()
        scale = float(spend.max())
        unit = media_response(spend, alpha=p["alpha"], lam=p["lam"], beta=1.0, scale=scale)
        beta = p["roi"] * spend.sum() / unit.sum()
        contrib = beta * unit
        media += contrib
        truth[col] = {**p, "beta": beta, "contribution": contrib}
    out = df.copy()
    out["sales"] = np.clip(base, 0, None) + media + rng.normal(0, noise_sd, len(df))
    return out, {"truth": truth, "noise_sd": noise_sd}


def fit_and_report(df: pd.DataFrame, run_name: str, priors: str, sampler: dict,
                   holdout_weeks: int = 12, seed: int = 42) -> dict:
    out = OUT / run_name
    out.mkdir(parents=True, exist_ok=True)
    split = split_holdout(df, holdout_weeks)
    X_train, y_train = split.xy("train")
    X_hold, y_hold = split.xy("holdout")
    mmm = build_mmm(priors, spend_share_cols(split.train), channels=CHANNELS,
                    controls=REAL_CONTROLS, alpha_prior=ALPHA_PRIOR,
                    yearly_seasonality=YEARLY_FOURIER)
    t0 = time.time()
    mmm.fit(X_train, y_train, random_seed=seed, progressbar=False, **sampler)
    secs = time.time() - t0
    idata = mmm.idata
    target_scale = float(idata.constant_data["target_scale"])

    pp_train = mmm.sample_posterior_predictive(X_train, extend_idata=True, combined=True,
                                               random_seed=seed, progressbar=False)
    pp_hold = mmm.sample_posterior_predictive(X_hold, include_last_observations=True,
                                              extend_idata=False, combined=True,
                                              random_seed=seed, progressbar=False)
    pi_train = ev.predictive_interval(_to_original(pp_train, y_train.max(), target_scale))
    pi_hold = ev.predictive_interval(_to_original(pp_hold, y_train.max(), target_scale))

    contrib = ev.contributions_original_scale(idata)
    roi = ev.roi_samples(contrib, split.train, CHANNELS)
    roi_table = ev.summarise_draws(roi, "roi")
    diag = ev.diagnostics(idata)
    metrics = {
        **{k: v for k, v in diag.items() if not isinstance(v, str)},
        "mape_train": ev.mape(y_train, pi_train["mean"]),
        "mape_holdout": ev.mape(y_hold, pi_hold["mean"]),
        "coverage90_holdout": ev.coverage(y_hold, pi_hold),
        "media_share": float(contrib.sum("channel").sum("date").mean() / y_train.sum()),
        "fit_seconds": secs,
        "train_weeks": len(split.train),
    }
    plots.plot_predictive(X_train["date"], y_train, pi_train, X_hold["date"], y_hold, pi_hold,
                          f"{run_name}: holdout MAPE {metrics['mape_holdout']:.1%}",
                          out / "posterior_predictive.png")
    _plot_decomposition(contrib, y_train, X_train["date"], out / "decomposition.png", run_name)
    roi_table.to_csv(out / "roi.csv")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    mmm.save(str(ROOT / "models" / f"realdata-{run_name}.nc"))
    _log_mlflow(run_name, priors, sampler, metrics, roi_table, out)
    return {"metrics": metrics, "roi": roi_table, "roi_draws": roi, "split": split}


def spend_share_cols(train: pd.DataFrame) -> pd.Series:
    totals = train[CHANNELS].sum()
    return totals / totals.sum()


def _plot_decomposition(contrib, y, dates, path, title) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(dates, y, color=TRUTH_COLOR, lw=1, label="revenue")
    bottom = np.zeros(len(dates))
    for col in CHANNELS:
        m = contrib.sel(channel=col).mean(("chain", "draw")).values
        ax.fill_between(dates, bottom, bottom + m, color=COLORS[col], alpha=0.85, lw=0,
                        label=f"{col.removeprefix('spend_')} (estimated)")
        bottom = bottom + m
    ax.set_title(f"{title}: estimated media contribution under weekly revenue")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _log_mlflow(run_name, priors, sampler, metrics, roi_table, out) -> None:
    import mlflow

    from mbc.mmm.train import MLFLOW_ARTIFACTS, MLFLOW_DB

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    exp = "mmm-realdata"
    if mlflow.get_experiment_by_name(exp) is None:
        mlflow.create_experiment(exp, artifact_location=MLFLOW_ARTIFACTS.as_uri())
    mlflow.set_experiment(exp)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"priors": priors, **sampler, "yearly_fourier": YEARLY_FOURIER,
                           "controls": ",".join(REAL_CONTROLS)})
        mlflow.log_metrics(metrics)
        mlflow.log_metrics({f"roi_{c}": float(roi_table.loc[c, "roi_mean"])
                            for c in roi_table.index})
        mlflow.log_artifacts(str(out), artifact_path="report")


def plot_roi_runs(results: dict[str, pd.DataFrame], planted: dict | None, path) -> None:
    """ROI with 90% HDI for every run, per channel, so stability is visible at a glance."""
    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.6 * len(names) + 1.4), sharey=True)
    for ax, col in zip(axes, CHANNELS, strict=True):
        ch = col.removeprefix("spend_")
        for yi, name in enumerate(names[::-1]):
            r = results[name].loc[ch]
            ax.plot([r["roi_hdi_low"], r["roi_hdi_high"]], [yi, yi], color=COLORS[col], lw=6,
                    alpha=0.7, solid_capstyle="round")
            ax.scatter(r["roi_mean"], yi, s=45, color=COLORS[col], edgecolor="white", zorder=3)
            ax.text(r["roi_hdi_high"], yi, f"  {r['roi_mean']:.2f}", va="center", fontsize=8,
                    color=TEXT_SECONDARY)
            if planted and name.startswith("semi"):
                ax.scatter(planted[col]["roi"], yi, marker="|", s=250, color=TRUTH_COLOR, lw=2,
                           zorder=4)
        ax.axvline(1, color=TEXT_SECONDARY, ls="--", lw=0.8)
        ax.set_title(ch.capitalize())
        ax.set_yticks(range(len(names)), names[::-1])
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("ROI (revenue per $ spent), 90% HDI")
    fig.suptitle("Real brand: ROI across fits (black tick = planted truth, semi-synthetic only)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    sampler = ({"draws": 200, "tune": 300, "chains": 2, "target_accept": 0.95} if a.quick else
               {"draws": 1000, "tune": 1500, "chains": 4, "target_accept": 0.95})
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    df = load_brand()
    df.to_csv(ROOT / "data" / "realdata_brand_weekly.csv", index=False)

    semi_df, semi = semi_synthetic(df)
    runs = {
        "semi-synthetic": (semi_df, "informed"),
        "real-informed": (df, "informed"),
        "real-weak": (df, "weak"),
        "real-first-75pct": (df.iloc[: int(len(df) * 0.75)].reset_index(drop=True), "informed"),
    }
    results, summary = {}, {}
    for name, (data, priors) in runs.items():
        r = fit_and_report(data, name, priors, sampler)
        results[name] = r["roi"]
        summary[name] = {"metrics": r["metrics"], "roi": r["roi"].round(3).to_dict("index")}
        if name == "semi-synthetic":
            train_idx = r["split"].train.index
            summary[name]["planted_roi_train"] = {
                c: float(semi["truth"][c]["contribution"][train_idx].sum()
                         / data.loc[train_idx, c].sum())
                for c in CHANNELS}
        print(name, json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                for k, v in r["metrics"].items()}), "\n", r["roi"].round(2))
    plot_roi_runs(results, PLANTED, OUT / "roi_across_runs.png")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
