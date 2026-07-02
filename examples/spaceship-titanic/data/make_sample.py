"""Generate a small **synthetic** Spaceship Titanic sample (data/raw/train.csv).

The real competition data lives on Kaggle (accept the rules, then
`kitchen init --source kaggle --competition spaceship-titanic`). To keep this
showcase runnable offline with zero credentials, we generate a tiny dataset that
mirrors the real schema — same columns, same quirks (a `deck/num/side` Cabin,
grouped PassengerIds, missing values) — with a **realistic-strength** signal: a
tuned XGBoost lands ≈0.81, right where real Spaceship Titanic baselines do, and
just under the ~0.83 information ceiling (so feature work + tuning still move it).

`Transported` is driven by several features at once — CryoSleep, total spend,
HomePlanet, Cabin deck/side, Age, VIP — plus Gaussian noise, so no single column
gives it away. Deterministic (seeded), so the committed CSV is reproducible:

    python examples/spaceship-titanic/data/make_sample.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_GROUPS = 600
NOISE = 1.6  # sets the difficulty: bigger → lower ceiling (see the module docstring)
SEED = 42
OUT = Path(__file__).resolve().parent / "raw" / "train.csv"

HOME_PLANETS = ["Earth", "Europa", "Mars"]
DESTINATIONS = ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"]
DECKS = list("ABCDEFG")
DECK_EFFECT = {"A": 0.5, "B": 0.6, "C": 0.5, "D": 0.1, "E": -0.2, "F": -0.3, "G": -0.4}
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    pid = 0
    while len({r["PassengerId"].split("_")[0] for r in rows}) < N_GROUPS:
        pid += 1
        group = f"{pid:04d}"
        size = int(rng.choice([1, 1, 1, 2, 2, 3, 4]))
        deck = rng.choice(DECKS, p=[0.05, 0.1, 0.12, 0.18, 0.2, 0.2, 0.15])
        side = rng.choice(["P", "S"])
        home = rng.choice(HOME_PLANETS, p=[0.5, 0.25, 0.25])
        for member in range(1, size + 1):
            cryo = bool(rng.random() < 0.35)
            age = float(round(np.clip(rng.normal(28, 14), 0, 79)))
            vip = bool(rng.random() < 0.02)
            spend = {c: 0.0 if cryo else float(round(rng.exponential(220))) for c in SPEND_COLS}
            total = sum(spend.values())
            # Multi-feature signal (real SST is roughly this learnable).
            signal = (
                2.2 * cryo
                - 0.0016 * total
                + 0.9 * (home == "Europa")
                - 0.5 * (home == "Earth")
                + DECK_EFFECT[deck]
                + 0.4 * (side == "S")
                - 0.9 * vip
                + 0.015 * (30 - age)
            )
            rows.append(
                {
                    "PassengerId": f"{group}_{member:02d}",
                    "HomePlanet": home,
                    "CryoSleep": cryo,
                    "Cabin": f"{deck}/{pid}/{side}",
                    "Destination": rng.choice(DESTINATIONS, p=[0.7, 0.2, 0.1]),
                    "Age": age,
                    "VIP": vip,
                    **spend,
                    "Name": f"Test Passenger{pid}{member}",
                    "_z": signal + rng.normal(0, NOISE),  # signal + noise; dropped below
                }
            )

    df = pd.DataFrame(rows)
    # Label at the median of (signal + noise) → a balanced ~50/50 target.
    df["Transported"] = df["_z"] > df["_z"].median()
    df = df.drop(columns="_z")

    # Sprinkle realistic missing values (the real data has ~2% NaN per column).
    for col in ["HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", "VIP", *SPEND_COLS]:
        mask = rng.random(len(df)) < 0.02
        df.loc[mask, col] = np.nan

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} rows → {OUT}  (Transported rate: {df['Transported'].mean():.2f})")


if __name__ == "__main__":
    main()
