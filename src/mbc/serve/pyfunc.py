"""MLflow pyfunc wrapper, so a registered model version can be loaded and called anywhere.

    model = mlflow.pyfunc.load_model("models:/mmm-budget@champion")
    model.predict(pd.DataFrame([{"tv": 20000, "search": 18000, "social": 15000, "radio": 2000}]))

Input: one row per scenario, weekly spend per channel (EUR).
Output: expected weekly media-driven sales, mean and 90% interval, per scenario.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel


class MMMBudgetModel(PythonModel):
    def load_context(self, context):
        import shutil
        import tempfile
        from pathlib import Path

        from mbc.serve.artifact import load_artifact

        d = Path(tempfile.mkdtemp())
        shutil.copy(context.artifacts["npz"], d / "model.npz")
        shutil.copy(context.artifacts["json"], d / "model.json")
        self.artifact = load_artifact(d)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        a = self.artifact
        spend = model_input[a.channels].to_numpy(float)
        rows = []
        for s in spend:
            total = a.curves.response(s).sum(axis=1)
            lo, hi = np.quantile(total, [0.05, 0.95])
            rows.append({"media_sales_mean": total.mean(), "media_sales_low": lo,
                         "media_sales_high": hi})
        return pd.DataFrame(rows)
