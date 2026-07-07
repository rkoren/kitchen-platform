# Changelog

All notable changes to `rkoren-kitchen` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
