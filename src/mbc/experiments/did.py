"""Difference-in-differences for a geo-lift test.

DiD compares the change in treated regions (before -> during the test) with the
change in control regions over the same weeks. Anything that hits all regions alike
(season, holidays, national campaigns) cancels out; what's left is the test's effect.

    sales[r, t] = region_effect[r] + week_effect[t] + tau * treated[r] * post[t] + noise

tau is the average extra weekly sales per treated region during the test (the ATT).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


@dataclass
class DiDResult:
    tau: float  # extra sales per treated region per test week
    se: float  # cluster-robust standard error (clustered by region)
    ci_low: float  # 95% confidence interval
    ci_high: float
    p_value: float
    n_treated: int
    n_post_weeks: int

    @property
    def total(self) -> tuple[float, float, float]:
        """Total lift across treated regions and test weeks: (estimate, ci_low, ci_high)."""
        k = self.n_treated * self.n_post_weeks
        return self.tau * k, self.ci_low * k, self.ci_high * k


def analysis_window(panel: pd.DataFrame, pre_weeks: int) -> pd.DataFrame:
    """Keep `pre_weeks` before the test plus the test weeks themselves.

    Weeks after the test are dropped: carry-over would contaminate the "post" contrast.
    """
    post_weeks = panel.loc[panel["post"] == 1, "week"].unique()
    start, end = post_weeks.min() - pre_weeks, post_weeks.max()
    return panel[(panel["week"] >= start) & (panel["week"] <= end)].copy()


def did_estimate(panel: pd.DataFrame, pre_weeks: int = 26) -> DiDResult:
    """Two-way fixed effects DiD with standard errors clustered by region.

    Clustering: weeks within one region share shocks, so they aren't independent
    observations. Treating them as independent would make the CI far too narrow.
    """
    df = analysis_window(panel, pre_weeks).assign(
        treat_post=lambda d: d["treated"] * d["post"]
    )
    groups = pd.factorize(df["region"])[0]
    fit = smf.ols("sales ~ treat_post + C(region) + C(week)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    lo, hi = fit.conf_int().loc["treat_post"]
    return DiDResult(
        tau=float(fit.params["treat_post"]), se=float(fit.bse["treat_post"]),
        ci_low=float(lo), ci_high=float(hi), p_value=float(fit.pvalues["treat_post"]),
        n_treated=int(df.loc[df["treated"] == 1, "region"].nunique()),
        n_post_weeks=int(df.loc[df["post"] == 1, "week"].nunique()),
    )


def event_study(panel: pd.DataFrame, pre_weeks: int = 12, post_weeks: int | None = None
                ) -> pd.DataFrame:
    """Treated-minus-control difference for each week relative to the test start.

    Week -1 is the reference (fixed at 0). If trends are parallel, the pre-test
    coefficients hover around 0; the test-week coefficients show the effect.
    """
    start = int(panel.loc[panel["post"] == 1, "week"].min())
    n_post = post_weeks or int(panel["post"].sum() / panel["region"].nunique())
    df = panel[(panel["week"] >= start - pre_weeks) & (panel["week"] < start + n_post)].copy()
    df["rel"] = df["week"] - start
    names = {}
    for k in sorted(df["rel"].unique()):
        if k == -1:
            continue
        name = f"d_m{-k}" if k < 0 else f"d_p{k}"
        df[name] = ((df["rel"] == k) & (df["treated"] == 1)).astype(int)
        names[name] = k
    formula = "sales ~ " + " + ".join(names) + " + C(region) + C(week)"
    fit = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df["region"])[0]}
    )
    ci = fit.conf_int()
    out = pd.DataFrame({
        "rel_week": list(names.values()),
        "coef": [fit.params[n] for n in names],
        "ci_low": [ci.loc[n, 0] for n in names],
        "ci_high": [ci.loc[n, 1] for n in names],
    })
    ref = pd.DataFrame({"rel_week": [-1], "coef": [0.0], "ci_low": [0.0], "ci_high": [0.0]})
    return pd.concat([out, ref]).sort_values("rel_week").reset_index(drop=True)


def permutation_pvalue(panel: pd.DataFrame, pre_weeks: int = 26, n_perm: int = 500,
                       seed: int = 0) -> tuple[float, np.ndarray]:
    """Randomisation inference: how unusual is our estimate among random assignments?

    Reassign 'treated' to random sets of regions (same count), re-estimate each time.
    The p-value is the share of placebo estimates at least as large as the real one.
    It needs no assumption about the error distribution, which helps with only 20
    regions.
    """
    rng = np.random.default_rng(seed)
    df = analysis_window(panel, pre_weeks)
    # Fast version of the same estimator: average treated-minus-control change.
    wide = df.pivot_table(index="region", columns="post", values="sales", aggfunc="mean")
    change = (wide[1] - wide[0]).to_numpy()
    is_t = df.groupby("region")["treated"].first().reindex(wide.index).to_numpy().astype(bool)
    actual = change[is_t].mean() - change[~is_t].mean()
    n_t = is_t.sum()
    placebo = np.empty(n_perm)
    for i in range(n_perm):
        m = np.zeros(len(change), bool)
        m[rng.choice(len(change), n_t, replace=False)] = True
        placebo[i] = change[m].mean() - change[~m].mean()
    return float(np.mean(np.abs(placebo) >= abs(actual))), placebo


def lift_test_for_mmm(panel: pd.DataFrame, result: DiDResult, national_spend: pd.Series,
                      lift_factor: float, channel_column: str) -> pd.DataFrame:
    """Express the geo result as one national lift-test row for PyMC-Marketing.

    The MMM is national, the test ran in some regions. Treated regions make up a share
    W of the business, estimated from their share of *pre-test* sales (observable, no
    ground truth needed). Dividing the treated-group lift by W gives the lift the
    national business would see from the same relative spend change.

    Row semantics (weekly, original units): moving spend from x to x + delta_x raises
    weekly sales by delta_y, with standard error sigma.
    """
    pre = panel[panel["post"] == 0]
    share = pre.loc[pre["treated"] == 1, "sales"].sum() / pre["sales"].sum()
    weekly_lift = result.tau * result.n_treated  # treated group, per week
    x = float(national_spend.mean())
    return pd.DataFrame({
        "channel": [channel_column],
        "x": [x],
        "delta_x": [x * (lift_factor - 1)],
        "delta_y": [weekly_lift / share],
        "sigma": [result.se * result.n_treated / share],
        "treated_share": [share],
    })
