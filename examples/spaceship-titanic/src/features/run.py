"""Feature engineering for Spaceship Titanic — predict ``Transported``.

Reads ``data/raw/train.csv`` and turns the raw passenger records into a numeric,
model-ready table at ``data/processed/features.parquet``. The engineering mirrors
what you'd actually do on the real competition data:

* split the ``deck/num/side`` ``Cabin`` into ``deck`` + ``side``;
* ``total_spend`` across the five amenity columns (a strong signal — awake spenders
  were less likely to be transported);
* ``group_size`` from the ``gggg_pp`` ``PassengerId`` (travellers moved in groups);
* fill the missing values the real data is full of, and integer-encode the
  categoricals so XGBoost can consume them.
"""
from __future__ import annotations

import pandas as pd

from kitchen.steps import FeatureBuilder
from kitchen.store import DataStore

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

# The columns handed to the model (the target is excluded by the Trainer).
FEATURES: list[str] = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "Age",
    "VIP",
    *SPEND_COLS,
    "total_spend",
    "deck",
    "side",
    "group_size",
]


class SpaceshipFeatures(FeatureBuilder):
    def build(self, raw: pd.DataFrame, params: dict) -> pd.DataFrame:
        target = params["model"]["target"]
        df = raw.copy()

        # Cabin: deck/num/side → deck + side (the side, port vs starboard, matters).
        cabin = df["Cabin"].astype("string").str.split("/", expand=True)
        df["deck"] = cabin[0]
        df["side"] = cabin[2]

        # Group travel: passengers sharing a gggg_ prefix travelled together.
        group = df["PassengerId"].astype("string").str.split("_", expand=True)[0]
        df["group_size"] = group.map(group.value_counts())

        # Spend: fill missing with 0, then total across amenities.
        df[SPEND_COLS] = df[SPEND_COLS].fillna(0.0)
        df["total_spend"] = df[SPEND_COLS].sum(axis=1)

        # Booleans → 0/1 (missing = not-in-cryo / not-VIP).
        for col in ("CryoSleep", "VIP"):
            df[col] = df[col].map({True: 1, False: 0}).fillna(0).astype(int)

        # Age: fill with the median.
        df["Age"] = df["Age"].fillna(df["Age"].median())

        # Categoricals → integer codes (XGBoost needs numbers); NaN → -1.
        for col in ("HomePlanet", "Destination", "deck", "side"):
            df[col] = df[col].astype("category").cat.codes

        out = df[FEATURES + [target]].copy()
        out[target] = out[target].astype(int)  # Transported True/False → 1/0
        return out


def build(params: dict, store: DataStore) -> None:
    SpaceshipFeatures().run(store, params)
