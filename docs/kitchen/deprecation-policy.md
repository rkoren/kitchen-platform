# Deprecation Policy

`rkoren-kitchen` follows SemVer, and since 1.0 every public surface carries a compatibility
promise. Removing or renaming something public goes through **one minor release of a
deprecation shim** first, so a downstream project gets a warning and a release's worth of time
instead of an `ImportError`.

## What counts as public

- Everything in `kitchen.__all__` (pinned by `kitchen/tests/test_public_api.py`).
- Documented submodules and their documented names (`kitchen.serve`, `kitchen.tracking`,
  `kitchen.registry`, `kitchen.modeling`, …) — anything shown in these docs.
- The `kitchen` CLI: commands, options and their defaults.
- The `menu.yaml` schema (`kitchen menu schema`), including legacy `params.yaml` loading.

Underscore-prefixed names (`kitchen._cli`, `_helper`) are internal and can change any time.

## The rule

1. **Deprecate in a minor release (`1.N`).** Keep the old name working through a shim that
   emits `DeprecationWarning` naming the replacement and the removal version. Add a
   CHANGELOG `### Deprecated` entry.
2. **Remove no earlier than the next minor (`1.N+1`).** Add a CHANGELOG `### Removed` entry that
   says what to use instead.
3. **A break that can't be shimmed** (a semantic change behind the same name) waits for a major
   release, or ships as a new name alongside the old one.

CLI options and menu keys follow the same cycle: accept the old spelling, print a warning to
stderr, and document it under `Deprecated`.

## Writing a shim

`kitchen._deprecation` keeps it to a few lines. For a module attribute that moved or was
replaced, use a PEP 562 module `__getattr__`:

```python
from kitchen._deprecation import Deprecated, deprecated_getattr

__getattr__ = deprecated_getattr(__name__, {
    "OldThing": Deprecated(lambda: NewThing, since="1.2", removal="1.3",
                           replacement="kitchen.NewThing"),
})
```

For a function or method that stays callable but should no longer be used, call
`warn_deprecated("kitchen.old_fn", since="1.2", removal="1.3", replacement="kitchen.new_fn")`
at the top of it.

Keep the shim's name in `kitchen.__all__` (and `PUBLIC_API` in the guard test) until removal.

## Why

`ExperimentConfig` / `log_config` were deleted outright when `kitchen.experiment()` replaced
them (NB-001, `5e2ca0d`), with no shim and no CHANGELOG entry. cbb's experiment scripts broke
on upgrade with nothing pointing at the replacement (CBB-037).
