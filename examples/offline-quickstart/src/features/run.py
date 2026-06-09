"""Feature engineering for the offline quickstart.

Reads ``data/raw/train.csv`` (a tiny synthetic "did the student pass?" dataset),
adds one engineered interaction feature, and writes the model-ready table to
``data/processed/features.parquet``. The target column (``passed``) is kept in the
output — the Trainer separates it.
"""
from __future__ import annotations

import pandas as pd
from kitchen.steps import FeatureBuilder
from kitchen.store import DataStore

# Columns passed to the model (the target is excluded by the Trainer).
FEATURES: list[str] = [
    "study_hours",
    "prior_score",
    "attendance",
    "sleep_hours",
    "study_x_attendance",
]


class OfflineFeatures(FeatureBuilder):
    def build(self, raw: pd.DataFrame, params: dict) -> pd.DataFrame:
        target = params["model"]["target"]
        df = raw.copy()
        # One engineered feature so the step does real work: effort × engagement.
        df["study_x_attendance"] = df["study_hours"] * df["attendance"]
        return df[FEATURES + [target]]


def build(params: dict, store: DataStore) -> None:
    OfflineFeatures().run(store, params)
