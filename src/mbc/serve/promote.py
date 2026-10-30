"""Promote a registered model version to "champion" and stage its files for the API.

    python -m mbc.serve.promote --version 1

Moves the MLflow alias `champion` to that version, then copies its model.npz and
model.json into artifacts/champion/, which the API (and its Docker image) loads.
Rolling back = promoting the previous version.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from mbc.mmm.data import ROOT
from mbc.serve.export import REGISTERED_MODEL

CHAMPION_DIR = ROOT / "artifacts" / "champion"
ALIAS = "champion"


def promote(version: str, tracking_uri: str, out_dir: Path = CHAMPION_DIR) -> Path:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL, ALIAS, version)
    local = Path(mlflow.artifacts.download_artifacts(f"models:/{REGISTERED_MODEL}@{ALIAS}"))
    files = {p.name: p for p in local.rglob("model.*") if p.suffix in (".npz", ".json")}
    if set(files) != {"model.npz", "model.json"}:
        raise FileNotFoundError(f"artifact files missing in {local}: found {sorted(files)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        shutil.copy(src, out_dir / name)
    (out_dir / "VERSION").write_text(f"{REGISTERED_MODEL} v{version}\n")
    return out_dir


def main() -> None:
    from mbc.mmm.train import MLFLOW_DB

    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True)
    a = p.parse_args()
    out = promote(a.version, f"sqlite:///{MLFLOW_DB}")
    print(f"{REGISTERED_MODEL} v{a.version} is now '{ALIAS}'; files in {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
