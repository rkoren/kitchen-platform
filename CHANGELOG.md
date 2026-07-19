# Changelog

All notable changes to `rkoren-kitchen` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Promotion **validation-scheme guard** (S6E7-002) — `kitchen promote <metric>` and
  `kitchen run train --auto-promote` now refuse to rank/compare runs that declare different
  `validation_scheme` run tags (e.g. an 80/20 holdout vs out-of-fold CV): their metrics aren't
  comparable, and a scheme-advantaged run could silently crown the wrong champion. Projects opt
  in by tagging runs (`mlflow.set_tag("validation_scheme", "...")`); untagged runs are unaffected
  (backward compatible). Narrow a ranking to one scheme with `kitchen promote <metric> --scheme
  <name>`, or target a specific run with `--run-id`.
- `kitchen submit --notebook` (GEN-005a) — assemble a Kaggle **submission notebook** from a
  notebook-safe submission script (default `flows/generate_submission.py`): a header, the inlined
  source, an explicit call to its entry function, and a validation cell (`validate_submission`).
  `--execute` runs the cells top-to-bottom (as Kaggle would) and validates the produced CSV. The
  builder enforces the notebook-safe contract via AST (rejects module-level `__file__` /
  `os.chdir` / `sys.path` and excludes the `__main__` block). Offline-dependency bundling for
  internet-off code competitions is a separate follow-up (GEN-005b).
- `kitchen submit --wait` polls longer and reports precisely (S6E7-003) — Kaggle lags tens of
  seconds behind an upload, so `--wait` now waits up to 5 min (tunable with `--wait-timeout`),
  polling on an exponential backoff instead of a fixed short window. The result is no longer a
  bare "score not available": it distinguishes **scored** (writes `metrics.json`), **still
  scoring** (re-run / raise the timeout), **errored** (submission failed on Kaggle), and
  **unavailable** (auth/API problem). Submission-status matching now tolerates enum-style values.
  New `poll_submission_score()` returns a structured `ScoreResult`; `fetch_score()` stays as a
  thin `float | None` wrapper.
- `kitchen init --kind pipeline` (GEN-007) — scaffold a **lean** project built around a command
  stage (GEN-002/003) instead of the tabular `FeatureBuilder`/`Trainer`/`Evaluator` ABCs: just
  `menu.yaml` + a `src/pipeline/run.py` stub (which writes its metric to `$KITCHEN_METRICS_FILE`),
  no `serve`/`experiments`/`flows`/dashboard. For inference-only / non-tabular projects. `--source
  kaggle` wires `kitchen ingest`. The default `--kind tabular` is unchanged; the model `--template`,
  `--ci`, and `--with-dvc` (tabular-specific) error clearly if combined with `--kind pipeline`.
- Generic command sweeps — `kitchen sweep --run "<cmd with {a} {b}>" --param a=… --param b=… --metric <m>`
  (GEN-004). Sweeps a param grid over an arbitrary command (not just the train loop), pointing each
  combo at a per-combo metrics file via `KITCHEN_METRICS_FILE`, logging every combo to the run store
  (GEN-001), and ranking them. Composes with `kitchen leaderboard --store`. `kitchen score` and other
  metric writers honor `KITCHEN_METRICS_FILE`. The train-loop sweep (`--override`, MLflow-ranked)
  stays the default.
- `kitchen score` + a `scorer:` menu section (GEN-006) — register a project's scoring callable
  (any `{name: value}`-returning function, taking `(params, store)` or no args) as the metric
  source, so `thresholds`/`leaderboard`/`promote` ride on the real domain score **without** the
  `Trainer`/`Evaluator` ABCs. Fits inference-only / non-tabular pipelines. Logs a distinct MLflow
  run + writes `metrics.json`; non-scalar returns are rejected before a run is opened. `score` is
  also a `menu.yaml` pipeline verb (`pipeline: [score]`).
- `kitchen log` + `kitchen leaderboard --store <path>` (GEN-001) — framework-agnostic run tracking.
  Log `{params → metrics}` to a dependency-light local JSON Lines store from **any** process, env,
  or venv (no MLflow, no `Trainer` ABC), then rank the runs. The JSONL line format is the cross-env
  contract (a mlflow-free venv can append directly); `kitchen log` stamps id/timestamp/git-sha and
  takes a file lock so concurrent sweep writers serialize instead of corrupting the store. MLflow
  stays the default rich backend.
- Command / subprocess stages + per-stage environments (GEN-002/003). A `kind: stage` recipe can
  declare a `cmd:` (a subprocess argv — a list used verbatim, or a shlex-split string, no shell)
  instead of an in-process `source:` callable, with an optional `python:` interpreter (the stage's
  own venv — `cmd:` is then the args to it) and declared `inputs:`/`outputs:` (inputs checked
  before, missing outputs warn after). `kitchen menu run` runs command stages in the pipeline;
  `kitchen stage <name>` runs one in isolation (`--dry-run` previews the argv, `-C` runs from a
  dir). Metrics are the stage's job (`kitchen log` / `metrics.json`). Fits inference-only /
  non-tabular / separate-interpreter pipelines without the `FeatureBuilder`/`Trainer`/`Evaluator`
  ABCs.

### Fixed
- Scaffolded CI (`kitchen init --ci`) now installs the platform correctly (SCF-021). Both
  "Install kitchen" steps ran `pip install "kitchen @ git+…"`, but the PEP 508 name `kitchen`
  no longer matches the renamed distribution `rkoren-kitchen`, so pip resolved an unrelated
  PyPI `kitchen` and **every scaffolded project's first CI run failed at install**. Now
  `pip install rkoren-kitchen`.
- Scaffolded Kaggle CI now authenticates via `~/.kaggle/kaggle.json` (SCF-022). The ingest +
  submit steps exported `KAGGLE_USERNAME`/`KAGGLE_KEY`, but current kaggle clients **401 on
  env-var auth** against `api.kaggle.com`; the steps now write `kaggle.json` from the secrets
  (and `chmod 600` it) before `kitchen ingest` / `kitchen submit`.

## [1.0.2] - 2026-07-07

### Changed
- Trimmed the sdist to ship only the package, tests, README, and LICENSE — the stray
  project-runtime dirs (`docs/`, `flows/`, `monitoring/`, `submissions/`) and dev config
  (`.pylintrc`, `infra.yaml`, `uv.lock`) no longer leak into it (~635 KB → ~327 KB).
- Bumped all GitHub Actions off the deprecated Node 20 runtime to their first Node 24 major
  (`checkout@v5`, `setup-python@v6`, `upload-artifact@v6`, `download-artifact@v7`,
  `configure-aws-credentials@v6`, `setup-terraform@v4`, `gitleaks-action@v3`).

### Fixed
- Corrected stale pre-merge install instructions in the AWS/Lambda deploy docs — they told
  users to `pip install -e .../recipes` as a separate package; provisioning ships as the
  `kitchen.recipes` sub-package of `rkoren-kitchen`.
- `kitchen version` and the MLflow package-version logging looked up the distribution as
  `kitchen` instead of `rkoren-kitchen` (`importlib.metadata` on the wrong, unrelated package) —
  now resolve the correct distribution.

## [1.0.1] - 2026-07-07

### Fixed
- **Scaffolded projects declared the wrong dependency.** `kitchen init` generated
  `dependencies = ["kitchen"]`, but the published distribution is `rkoren-kitchen` (and
  `kitchen` is an unrelated package on PyPI) — so a scaffolded project's `pip install` pulled
  the wrong package. The scaffold now declares `rkoren-kitchen>=1.0`, and the generated
  `dvc.yaml` install hints use `rkoren-kitchen[dvc]`.
- Corrected `rkoren-kitchen[...]` install commands in the README and docs (they said
  `kitchen[dvc]` / `kitchen[postgres]` / `kitchen @ git+...`).

## [1.0.0] - 2026-07-07

First public release.

### Changed
- **Unified platform.** `recipes` (the IaC CLI) merged into `rkoren-kitchen` as the
  `kitchen.recipes` sub-package — one distribution, one `kitchen` CLI (with `recipes` kept as a
  back-compat alias). The full ML stack ships in the base install.
- One `menu.yaml` reader: `Menu.to_recipe_spec()` projects the manifest into infrastructure, and
  infra fields are validated at manifest load.
- Provisioning runs in-process from `kitchen menu run` (no cross-CLI shell-out); the recipes
  workspace resolves from a manifest's `project`.
- Stage code loads from each recipe's declared `source` (falling back to `src/<stage>/run.py`).

### Added
- `kitchen menu schema` — export the `menu.yaml` JSON Schema (draft 2020-12).
- `-C/--project DIR` (like `git -C`) on `kitchen run`, `kitchen menu run`, `kitchen ingest`, and
  `kitchen submit` — drive a project from any directory.
- `kitchen submit --dry-run` — validate a submission and report what would be uploaded, without
  credentials or an upload.
- Segment-scoped holdout scoring: `holdout.segments:` logs `holdout_<metric>_<segment>` for named
  subpopulations, so a segment gain a combined metric averages away is promotable via
  `--promote-metric` (which ranks on any logged metric).
- Spaceship Titanic end-to-end example — the whole loop on a real Kaggle competition
  (ingest → train → evaluate → promote → submit).
- Packaging metadata for PyPI (long-description README, bundled license).
