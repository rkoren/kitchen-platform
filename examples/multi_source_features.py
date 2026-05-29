"""VAL-003: Multi-source FeatureBuilder validation.

Demonstrates that FeatureBuilder.sources() correctly routes multiple raw CSVs
into build() as a dict[filename, DataFrame], and that run() produces the same
output parquet as a single-file implementation would.

This example uses synthetic data so it can be run without any project setup:

    python examples/multi_source_features.py

Acceptance criteria:
- build() receives a dict keyed by filename when sources() returns > 1 file
- run() saves the merged output to data/processed/features.parquet
- DataStore.list("raw") reports both source files
- DataStore.list("processed") reports the output file
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from kitchen.steps import FeatureBuilder
from kitchen.store import DataStore


# ---------------------------------------------------------------------------
# Example: two-source FeatureBuilder (teams + games, generic sports/competition shape)
# ---------------------------------------------------------------------------


class MultiSourceFeatures(FeatureBuilder):
    """Merges two raw CSVs into a single feature table.

    Override sources() to declare both input files; build() receives a
    dict[filename, DataFrame] and is responsible for the merge logic.
    """

    def sources(self, params: dict) -> list[str]:
        return ["teams.csv", "games.csv"]

    def build(self, raw: pd.DataFrame | dict[str, pd.DataFrame], params: dict) -> pd.DataFrame:
        assert isinstance(raw, dict), "Expected dict when sources() returns multiple files"
        teams = raw["teams.csv"]
        games = raw["games.csv"]
        merged = games.merge(teams, on="team_id", how="left")
        # Drop non-feature columns for the model; keep target
        return merged.drop(columns=["team_id"])


# ---------------------------------------------------------------------------
# Run the example
# ---------------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(42)

    teams = pd.DataFrame(
        {
            "team_id": range(1, 11),
            "strength": rng.uniform(0.3, 0.9, size=10).round(3),
            "conference": rng.choice(["A", "B", "C"], size=10),
        }
    )
    games = pd.DataFrame(
        {
            "team_id": rng.integers(1, 11, size=300),
            "season": rng.choice(range(2018, 2024), size=300),
            "home": rng.integers(0, 2, size=300),
            "won": rng.integers(0, 2, size=300),
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        store = DataStore(root=root)
        store.raw_dir.mkdir(parents=True)

        teams.to_csv(store.raw_dir / "teams.csv", index=False)
        games.to_csv(store.raw_dir / "games.csv", index=False)

        print(f"Raw files: {store.list('raw')}")

        MultiSourceFeatures().run(store, params={"processed_file": "features.parquet"})

        result = store.load_parquet("features.parquet")

        print(f"Processed files: {store.list('processed')}")
        print(f"Output shape: {result.shape}")
        print(f"Output columns: {list(result.columns)}")

        # Acceptance checks
        assert "features.parquet" in store.list("processed")
        assert "strength" in result.columns, "team feature should be present after merge"
        assert "conference" in result.columns, "team feature should be present after merge"
        assert "won" in result.columns, "target column should be present"
        assert "team_id" not in result.columns, "join key should be dropped"
        assert len(result) == len(games), "row count should match games table"

    print("\nVAL-003 passed: multi-source FeatureBuilder produces correct output.")


if __name__ == "__main__":
    main()
