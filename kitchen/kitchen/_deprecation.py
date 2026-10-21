"""Deprecation shims for the public API (CBB-037).

Policy (``docs/kitchen/deprecation-policy.md``): anything exported from ``kitchen.__all__``, a
documented submodule, the CLI or the menu schema is removed only after **one minor release**
of a :class:`DeprecationWarning` shim plus a CHANGELOG ``Deprecated`` entry, then a
``Removed`` entry when it goes. These helpers make the shim a few lines::

    # kitchen/experiment.py
    from kitchen._deprecation import Deprecated, deprecated_getattr

    __getattr__ = deprecated_getattr(__name__, {
        "ExperimentConfig": Deprecated(
            lambda: _ExperimentConfigShim, since="1.2", removal="1.3",
            replacement="kitchen.experiment()",
        ),
    })
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


def warn_deprecated(
    name: str,
    *,
    since: str,
    removal: str,
    replacement: str | None = None,
    stacklevel: int = 3,
) -> None:
    """Emit the standard :class:`DeprecationWarning` for ``name``.

    The default ``stacklevel`` points at the caller of the deprecated function (one frame
    above the function that calls this).
    """
    message = f"{name} is deprecated since kitchen {since} and will be removed in {removal}"
    if replacement:
        message += f"; use {replacement} instead"
    warnings.warn(message + ".", DeprecationWarning, stacklevel=stacklevel)


@dataclass(frozen=True)
class Deprecated:
    """One module attribute kept alive by a shim until ``removal``."""

    target: Callable[[], Any]  # lazily returns the object handed back (or a compatibility shim)
    since: str
    removal: str
    replacement: str | None = None


def deprecated_getattr(module: str, table: dict[str, Deprecated]) -> Callable[[str], Any]:
    """A module-level ``__getattr__`` (PEP 562) serving deprecated names with a warning.

    Assign it as ``__getattr__`` in the module that used to define the names. Only names
    missing from the module reach it, so live attributes are unaffected; anything not in
    ``table`` raises the usual :class:`AttributeError`.
    """

    def __getattr__(name: str) -> Any:
        entry = table.get(name)
        if entry is None:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        warn_deprecated(
            f"{module}.{name}",
            since=entry.since,
            removal=entry.removal,
            replacement=entry.replacement,
        )
        return entry.target()

    return __getattr__
