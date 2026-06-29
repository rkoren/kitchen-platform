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
