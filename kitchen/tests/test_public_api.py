"""The top-level ``kitchen`` namespace is a public surface — exercise its contract (CBB-024)."""

from __future__ import annotations

import kitchen


def test_all_names_are_real_attributes():
    """Every name in ``kitchen.__all__`` must resolve on the package."""
    missing = [name for name in kitchen.__all__ if not hasattr(kitchen, name)]
    assert not missing, f"names in __all__ not importable: {missing}"


def test_new_helpers_surfaced_at_top_level():
    """CBB-024: the TRUST-epic helpers are importable from the top-level namespace,
    matching the convention of the established helpers (not submodule-only)."""
    from kitchen import cached_fetch, require_external, score_run_holdout

    for name in ("cached_fetch", "require_external", "score_run_holdout"):
        assert name in kitchen.__all__

    # They are the same objects as their submodule definitions (re-export, not a shadow).
    from kitchen.holdout import score_run_holdout as _srh
    from kitchen.ingest import cached_fetch as _cf
    from kitchen.ingest import require_external as _re

    assert cached_fetch is _cf
    assert require_external is _re
    assert score_run_holdout is _srh


# --- REL-002: freeze the public surface for 1.0 (see docs/kitchen/api-stability.md) ---

# The frozen public API. Changing this set is a deliberate API change — update it together with
# docs/kitchen/api-stability.md and the CHANGELOG (MINOR bump to add a name, MAJOR to remove one).
PUBLIC_API: frozenset[str] = frozenset(
    {
        # pipeline building blocks
        "DataStore", "DriftReport", "Evaluator", "FeatureBuilder", "KitchenConfig",
        "Tracker", "Trainer",
        # experiment tracking
        "experiment", "init_run", "load_params",
        # modeling — metrics / CV / ensembling / calibration / utils
        "classification_metrics", "regression_metrics",
        "cross_validate", "time_series_cv", "loto_cv",
        "blend_predictions", "make_stack_features", "rank_average", "voting_predict",
        "calibrate_model", "clip_predictions", "clip_proba", "compute_calibration_curve",
        "set_seed", "train_val_split",
        # hyperparameter search
        "grid_search", "random_search", "bayes_search",
        # data ingestion
        "cached_fetch", "require_external",
        # holdout scoring
        "score_run_holdout",
        # submodules
        "tracking", "evaluate", "registry", "search",
    }
)


def test_public_api_is_frozen():
    """`kitchen.__all__` matches the pinned surface — no accidental additions/removals slip into
    the 1.x compatibility promise."""
    actual = set(kitchen.__all__)
    assert actual == set(PUBLIC_API), (
        "kitchen.__all__ changed. If intentional, update PUBLIC_API here + "
        "docs/kitchen/api-stability.md + the CHANGELOG.\n"
        f"  added:   {sorted(actual - PUBLIC_API)}\n"
        f"  removed: {sorted(PUBLIC_API - actual)}"
    )


def test_all_is_free_of_duplicates():
    """No duplicate exports (a duplicate hints at a merge slip)."""
    assert len(kitchen.__all__) == len(set(kitchen.__all__))


def test_provisioning_is_not_a_python_import_surface():
    """Provisioning is a CLI/schema surface (`kitchen recipes`, `kitchen menu schema`), not a
    top-level import — so `kitchen.recipes` internals stay out of the public contract (REL-002)."""
    assert "recipes" not in kitchen.__all__
