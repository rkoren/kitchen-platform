"""Which kitchen is installed — honest even on an editable install (CBB-044).

An editable install (``pip install -e``) records the version from the ``pyproject.toml`` it was
installed from and never refreshes it, so ``importlib.metadata`` can report ``0.1.0`` while the
source tree is at ``1.2.0``. For an editable install the source tree's ``pyproject.toml`` is the
truth; the stale metadata version is kept so callers can say "re-run ``pip install -e``".
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

DIST = "rkoren-kitchen"


@dataclass(frozen=True)
class VersionInfo:
    version: str  # the code actually running (source pyproject for an editable install)
    metadata_version: str | None = None  # what the installed dist-info says
    editable_path: Path | None = None  # source tree, when installed with ``pip install -e``

    @property
    def stale_metadata(self) -> bool:
        """Editable install whose recorded version lags the source tree."""
        return self.metadata_version is not None and self.metadata_version != self.version


def _editable_path(dist: metadata.Distribution) -> Path | None:
    try:
        raw = dist.read_text("direct_url.json")
        info = json.loads(raw) if raw else {}
    except (OSError, ValueError):
        return None
    if not info.get("dir_info", {}).get("editable"):
        return None
    url = urlparse(info.get("url", ""))
    return Path(unquote(url.path)) if url.scheme == "file" else None


def _source_version(root: Path) -> str | None:
    try:
        with (root / "pyproject.toml").open("rb") as fh:
            project = tomllib.load(fh).get("project", {})
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return project.get("version") if project.get("name") == DIST else None


def version_info() -> VersionInfo:
    """The running kitchen's version, install mode and (for editable installs) source path."""
    try:
        dist = metadata.distribution(DIST)
    except metadata.PackageNotFoundError:
        # Running from a source checkout that was never installed.
        root = Path(__file__).resolve().parent.parent
        return VersionInfo(version=_source_version(root) or "unknown", editable_path=root)
    installed = dist.version
    editable = _editable_path(dist)
    if editable is None:
        return VersionInfo(version=installed)
    source = _source_version(editable)
    return VersionInfo(
        version=source or installed,
        metadata_version=installed if source else None,
        editable_path=editable,
    )


def describe(info: VersionInfo | None = None) -> str:
    """One-line ``kitchen --version`` output."""
    info = info or version_info()
    line = f"kitchen {info.version}"
    if info.editable_path is not None:
        line += f" (editable: {info.editable_path})"
    if info.stale_metadata:
        line += (
            f"\nwarning: installed metadata says {info.metadata_version} — re-run "
            f"`pip install -e {info.editable_path}` so version pins resolve against it"
        )
    return line
