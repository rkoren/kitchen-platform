# API Stability & Versioning

What `rkoren-kitchen` promises to keep stable, and what it reserves the right to change. This
is the contract that lets you depend on the package without a 1.x upgrade breaking you.

## The public API is `kitchen.__all__`

The stable, supported Python surface is exactly the names exported at the top level —
`kitchen.__all__`. Import these directly from `kitchen`:

```python
from kitchen import Trainer, Evaluator, FeatureBuilder, DataStore, KitchenConfig
from kitchen import experiment, train_val_split, classification_metrics
```

Grouped:

- **Pipeline building blocks** — `FeatureBuilder`, `Trainer`, `Evaluator`, `DataStore`, `Tracker`, `KitchenConfig`, `DriftReport`
- **Experiment tracking** — `experiment`, `init_run`, `load_params`
- **Modeling — metrics** — `classification_metrics`, `regression_metrics`
- **Modeling — cross-validation** — `cross_validate`, `time_series_cv`, `loto_cv`
- **Modeling — ensembling** — `blend_predictions`, `make_stack_features`, `rank_average`, `voting_predict`
- **Modeling — calibration** — `calibrate_model`, `clip_predictions`, `clip_proba`, `compute_calibration_curve`
- **Modeling — utilities** — `set_seed`, `train_val_split`
- **Hyperparameter search** — `grid_search`, `random_search`, `bayes_search`
- **Data ingestion** — `cached_fetch`, `require_external`
- **Holdout scoring** — `score_run_holdout`
- **Submodules** — `tracking`, `evaluate`, `registry`, `search` (their *documented* members)

A test (`tests/test_public_api.py`) pins this set, so it can only change by an intentional edit
— an accidental re-export never silently becomes part of the 1.x compatibility promise.

## What is *internal* (may change in any release)

Anything not in `kitchen.__all__` is internal, even if it's importable:

- Underscore-prefixed names (`_foo`), and any module member not re-exported at the top level.
- `kitchen._cli.*` — the CLI implementation. Depend on the **commands** (below), not these functions.
- `kitchen.recipes.*` internals — provisioning is a **CLI/schema** surface (below), not a Python import API. `kitchen.recipes.schema` types are used *through* the menu, not imported by consumers.
- `kitchen.menu_run`, `kitchen.menu_resolve`, and other internal modules not surfaced in `__all__`.

Reaching into internals is allowed but unsupported: it can break in a minor or patch release.

## Non-Python public surfaces

Two more things are part of the stability promise, versioned the same way:

- **The CLI** — the documented `kitchen …` commands (`run`, `menu`, `secrets`, `serve`,
  `dashboard`, `recipes`, `experiments`) and their options. The `recipes` standalone alias is
  the same surface under a second entry point.
- **The `menu.yaml` schema** — the manifest fields, exported as JSON Schema by
  `kitchen menu schema` (committed at `docs/kitchen/menu.schema.json`). A breaking schema change
  is a breaking release.

## Versioning policy (SemVer, from 1.0.0)

`rkoren-kitchen` follows [SemVer](https://semver.org/). Given a public change:

- **MAJOR** — a breaking change to the public API, the documented CLI, or the `menu.yaml`
  schema: removing/renaming a `__all__` name, changing a documented signature incompatibly,
  removing a CLI command/option, or a manifest field change that invalidates an existing menu.
- **MINOR** — backwards-compatible additions: a new `__all__` name, a new CLI command/option,
  a new optional manifest field.
- **PATCH** — backwards-compatible bug fixes with no surface change.

Deprecations are announced in the changelog and release notes and kept working for at least
one minor release before removal in the next major. Roadmap milestones (`v0.6.0`, …) are
project-level labels and are distinct from the package version (see
`docs/decisions/versioning.md`).
