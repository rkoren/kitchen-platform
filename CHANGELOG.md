# Changelog

All notable changes to `rkoren-kitchen` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/) (see `docs/kitchen/api-stability.md` for what the
public surface covers).

## [Unreleased]

Working toward the first public `1.0.0` release.

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
- Packaging metadata for PyPI (long-description README, bundled license).

_This section will be split into a dated `1.0.0` entry when the release is cut._
