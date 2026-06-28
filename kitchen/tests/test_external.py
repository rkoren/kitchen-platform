"""Tests for kitchen.ingest.external — the external-API "third data category" (CBB-018)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kitchen.ingest import cached_fetch, require_external


def _df(n=3):
    return pd.DataFrame({"team": list(range(n)), "rating": [1.0] * n})


# ── cached_fetch ────────────────────────────────────────────────────────────────


def test_cache_miss_fetches_and_writes(tmp_path):
    path = tmp_path / "kenpom_2026.parquet"
    calls = []

    def fetch():
        calls.append(1)
        return _df()

    out = cached_fetch(fetch, path)
    assert calls == [1]
    assert path.exists()
    pd.testing.assert_frame_equal(out, _df())


def test_cache_hit_skips_fetch(tmp_path):
    path = tmp_path / "kenpom_2026.parquet"
    _df().to_parquet(path, index=False)  # pre-seed the cache

    def fetch():
        raise AssertionError("fetch must not be called on a cache hit")

    out = cached_fetch(fetch, path)
    pd.testing.assert_frame_equal(out, _df())


def test_refresh_overwrites_cache(tmp_path):
    path = tmp_path / "kenpom_2026.parquet"
    _df(n=2).to_parquet(path, index=False)

    out = cached_fetch(lambda: _df(n=5), path, refresh=True)
    assert len(out) == 5
    assert len(pd.read_parquet(path)) == 5  # cache replaced on disk


def test_creates_parent_dirs(tmp_path):
    path = tmp_path / "data" / "kenpom" / "kenpom_2026.parquet"
    cached_fetch(_df, path)
    assert path.exists()


def test_fetch_must_return_dataframe(tmp_path):
    path = tmp_path / "bad.parquet"
    with pytest.raises(TypeError, match="DataFrame"):
        cached_fetch(lambda: {"not": "a frame"}, path)
    assert not path.exists()  # nothing cached on a bad fetch


# ── require_external (absence guard) ────────────────────────────────────────────


def test_require_present_returns_path(tmp_path):
    path = tmp_path / "kenpom_2026.parquet"
    _df().to_parquet(path, index=False)
    assert require_external(path) == Path(path)


def test_require_absent_raises_loudly(tmp_path):
    path = tmp_path / "missing.parquet"
    with pytest.raises(FileNotFoundError) as ei:
        require_external(path)
    msg = str(ei.value)
    assert "required external input not found" in msg
    assert "CBB-018" in msg and "dvc" in msg.lower()  # explains the fix, never a silent skip


def test_require_absent_includes_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="set KENPOM_API_KEY"):
        require_external(tmp_path / "missing.parquet", hint="set KENPOM_API_KEY and re-run ingest")
