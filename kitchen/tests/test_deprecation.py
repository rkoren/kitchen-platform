"""Deprecation shim helpers (CBB-037)."""

from __future__ import annotations

import types
import warnings

import pytest

from kitchen._deprecation import Deprecated, deprecated_getattr, warn_deprecated


def test_warn_deprecated_message_and_category():
    with pytest.warns(DeprecationWarning) as rec:
        warn_deprecated("kitchen.old", since="1.2", removal="1.3", replacement="kitchen.new")
    assert str(rec[0].message) == (
        "kitchen.old is deprecated since kitchen 1.2 and will be removed in 1.3; "
        "use kitchen.new instead."
    )


def test_warn_deprecated_points_at_the_caller():
    def old_fn():
        warn_deprecated("kitchen.old_fn", since="1.2", removal="1.3")

    with pytest.warns(DeprecationWarning) as rec:
        old_fn()  # the warning should be attributed to this line, not to old_fn's body
    assert rec[0].filename == __file__
    assert "will be removed in 1.3." in str(rec[0].message)


def _shim_module():
    mod = types.ModuleType("kitchen._fake")
    mod.live = "live"
    mod.__getattr__ = deprecated_getattr(
        "kitchen._fake",
        {"Old": Deprecated(lambda: "new-object", since="1.2", removal="1.3", replacement="New")},
    )
    return mod


def test_deprecated_getattr_serves_old_name_with_warning():
    mod = _shim_module()
    with pytest.warns(DeprecationWarning, match=r"kitchen\._fake\.Old .*use New instead"):
        assert mod.Old == "new-object"


def test_deprecated_getattr_leaves_live_names_alone():
    mod = _shim_module()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert mod.live == "live"


def test_deprecated_getattr_unknown_name_is_attribute_error():
    mod = _shim_module()
    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        _ = mod.nope
    assert not hasattr(mod, "nope")
