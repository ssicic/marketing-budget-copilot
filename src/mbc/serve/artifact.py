"""The serving artifact: what the API needs from a trained MMM, in two small files.

    model.npz   posterior draws of each channel's response curve (alpha, lam, beta)
    model.json  metadata and precomputed summaries (ROI, contributions, current spend)

Loading needs only NumPy and the standard library, so the API container doesn't
install PyMC.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mbc.mmm.optimize import Curves

ARTIFACT_VERSION = 1


@dataclass
class ModelArtifact:
    channels: list[str]
    curves: Curves  # beta in sales units, lam on spend / scale
    alpha: np.ndarray  # (n_draws, n_channels) adstock decay
    meta: dict  # everything in model.json

    @property
    def current_weekly_spend(self) -> dict[str, float]:
        return self.meta["current_weekly_spend"]

    @property
    def roi(self) -> dict[str, dict]:
        return self.meta["roi"]

    @property
    def contributions(self) -> dict[str, dict]:
        return self.meta["contributions"]


def save_artifact(out_dir: Path, channels: list[str], alpha: np.ndarray, lam: np.ndarray,
                  beta: np.ndarray, scale: np.ndarray, meta: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "model.npz", alpha=alpha.astype(np.float32),
                        lam=lam.astype(np.float32), beta=beta.astype(np.float32),
                        scale=scale.astype(np.float64))
    meta = {"artifact_version": ARTIFACT_VERSION, "channels": channels, **meta}
    (out_dir / "model.json").write_text(json.dumps(meta, indent=2))


def load_artifact(path: Path | str) -> ModelArtifact:
    path = Path(path)
    meta = json.loads((path / "model.json").read_text())
    if meta.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(f"unsupported artifact version {meta.get('artifact_version')}")
    with np.load(path / "model.npz") as z:
        alpha, lam, beta, scale = (z[k].astype(np.float64) for k in ("alpha", "lam", "beta",
                                                                    "scale"))
    channels = meta["channels"]
    return ModelArtifact(channels=channels, alpha=alpha, meta=meta,
                         curves=Curves(channels=channels, beta=beta, lam=lam, scale=scale))
