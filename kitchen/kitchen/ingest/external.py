"""External-API feature inputs — the "third data category" (CBB-018).

The platform's two built-in data categories are ``data/raw`` (immutable source, restored in CI
by ``dvc pull``) and ``data/processed`` (regenerable from raw by the features stage). Features
pulled from a *live API* — KenPom ratings, an odds feed, weather — fit neither: they aren't
regenerable from raw (they need the API, and they're point-in-time) and they aren't Kaggle raw.

This module gives that third category a first-class pattern without the platform dictating
*how* to fetch (irreducibly source-specific) or *where* to cache it (the project's choice):

- :func:`cached_fetch` — fetch once, cache to a project-chosen path, reuse on every later call.
  Skip-if-present is the snapshot-stability guarantee: once a season/date-keyed file exists it
  is never silently re-fetched to a different "now". Point the path at a DVC-tracked directory
  (``dvc add`` it once) so CI's ``dvc pull`` restores it instead of hitting the API.
- :func:`require_external` — the absence guard. The footgun this exists for: a features stage
  that merges external columns *only when the cache is present* trains baseline-only, silently,
  when the cache is missing (e.g. a gitignored dir DVC never restored in CI — the cbb bug).
  Call this at the merge site so a missing input is a loud, explained failure, never a skip.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd


def cached_fetch(
    fetch: Callable[[], pd.DataFrame],
    path: str | Path,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch an external dataset once and cache it as Parquet at a project-chosen path.

    When ``path`` already exists (and ``refresh`` is false) the cached file is read and
    returned **without calling** ``fetch`` — so an external snapshot is pulled exactly once and
    every later run (including CI after ``dvc pull``) reuses it instead of hitting the API. The
    project supplies ``fetch`` (the source-specific call) and chooses ``path`` (a DVC-tracked
    location it ``dvc add``s); the platform owns only the cache/skip/snapshot semantics.

    Snapshot stability comes from the *path*: encode the as-of key (season, date) in the
    filename so a new snapshot is a new file. ``refresh=True`` overwrites the cache at ``path``
    — for a genuinely new snapshot, use a new path rather than refreshing in place.

    Args:
        fetch: Zero-arg callable returning the external data as a DataFrame. Only called on a
            cache miss (or ``refresh``).
        path: Where to cache it (``.parquet``). Parent directories are created.
        refresh: Force a re-fetch, overwriting any existing cache at ``path``.

    Returns:
        The fetched (or cached) DataFrame.
    """
    target = Path(path)
    if target.exists() and not refresh:
        return pd.read_parquet(target)

    df = fetch()
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"cached_fetch: fetch() must return a pandas DataFrame, got {type(df).__name__}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False)
    return df


def require_external(path: str | Path, *, hint: str | None = None) -> Path:
    """Assert an external feature input exists, failing **loudly** when it doesn't (CBB-018).

    This is the absence guard for the silent-baseline footgun: a features stage that merges
    external columns only when their cache is present trains baseline-only — undetected — when
    the cache is missing (the classic case: a gitignored external dir that DVC never restored in
    CI). Call this at the merge site instead of an ``if path.exists()`` skip, so a missing input
    becomes a hard, explained error rather than a quietly degraded model.

    Deliberately has **no** ``optional``/soft mode: a toggle that downgrades absent → warning is
    the same silent footgun behind a log line. A project that truly wants dev tolerance writes an
    explicit ``try/except FileNotFoundError`` at the call site — a per-site decision, not a
    default escape hatch.

    Args:
        path: The external input that must exist.
        hint: Optional extra guidance appended to the error (e.g. how to populate the cache).

    Returns:
        The path as a :class:`~pathlib.Path`, when it exists.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
    """
    p = Path(path)
    if p.exists():
        return p
    lines = [
        f"required external input not found: {p}",
        "  External/API feature inputs (CBB-018) must be cached to a DVC-tracked path so CI "
        "restores them — otherwise the features stage silently trains baseline-only.",
        "  Populate it with `kitchen.ingest.cached_fetch(<your fetch>, path)` and `dvc add` "
        "its directory so `dvc pull` restores it in CI.",
    ]
    if hint:
        lines.insert(1, f"  {hint}")
    raise FileNotFoundError("\n".join(lines))
