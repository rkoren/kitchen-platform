"""Generate a small **synthetic** Spaceship Titanic sample (data/raw/train.csv).

The real competition data lives on Kaggle (accept the rules, then
`kitchen init --source kaggle --competition spaceship-titanic`). To keep this
showcase runnable offline with zero credentials, we generate a tiny dataset that
mirrors the real schema — same columns, same quirks (a `deck/num/side` Cabin,
grouped PassengerIds, missing values) — with a deliberately learnable signal so
the pipeline trains a real (if toy) model.

Deterministic (seeded), so the committed CSV is reproducible:
    python examples/spaceship-titanic/data/make_sample.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_GROUPS = 220
SEED = 42
OUT = Path(__file__).resolve().parent / "raw" / "train.csv"

HOME_PLANETS = ["Earth", "Europa", "Mars"]
DESTINATIONS = ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"]
DECKS = list("ABCDEFG")
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
            # Spend is zero in cryosleep; otherwise a heavy-tailed amount.
            spend = {
                c: 0.0 if cryo else float(round(rng.exponential(220)))
                for c in SPEND_COLS
            }
            total = sum(spend.values())
            # The learnable signal: cryosleep ↑, spending ↓, Europa ↑, starboard ↑.
            logit = (
                1.6 * cryo
                - 0.0009 * total
                + 0.7 * (home == "Europa")
                + 0.3 * (side == "S")
                - 0.3
                + rng.normal(0, 0.5)
            )
            transported = bool(1 / (1 + np.exp(-logit)) > rng.random())
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
                    "Transported": transported,
                }
            )

    df = pd.DataFrame(rows)
    # Sprinkle realistic missing values (the real data has ~2% NaN per column).
    for col in ["HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", "VIP", *SPEND_COLS]:
        mask = rng.random(len(df)) < 0.02
        df.loc[mask, col] = np.nan

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} rows → {OUT}  (Transported rate: {df['Transported'].mean():.2f})")


if __name__ == "__main__":
    main()
