"""Generate the synthetic dataset, its ground truth and figures.

    python -m mbc.simulate
"""

from __future__ import annotations

import json
from pathlib import Path

from mbc.simulate import simulate
from mbc.simulate.plots import make_all

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "reports" / "figures"


def main() -> None:
    sim = simulate()
    DATA_DIR.mkdir(exist_ok=True)
    sim.data.to_csv(DATA_DIR / "mmm_weekly.csv", index=False)
    sim.components.to_csv(DATA_DIR / "true_components.csv")
    (DATA_DIR / "ground_truth.json").write_text(json.dumps(sim.truth, indent=2))
    figures = make_all(sim, FIG_DIR)

    print(f"Wrote {len(sim.data)} weeks to {DATA_DIR / 'mmm_weekly.csv'}")
    print(f"Media share of sales: {sim.truth['media_share_of_sales']:.1%}")
    print(f"{'channel':<8} {'alpha':>6} {'ROI':>6} {'mROI':>6}")
    for c, t in sim.truth["channels"].items():
        print(f"{c:<8} {t['alpha']:>6.2f} {t['roi']:>6.2f} {t['marginal_roi']:>6.2f}")
    for f in figures:
        print(f"Figure: {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
