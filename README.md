# kitchen-platform

> **Status: kitchen 2.0 is being rebuilt from scratch.** 1.x is frozen — install
> `rkoren-kitchen<2` from PyPI or use the [`1.x` branch](https://github.com/rkoren/kitchen-platform/tree/1.x).

kitchen 2.0 is a small CLI harness that AI coding agents (Claude Code first) work inside when
competing in ML competitions and similar prediction projects. It does not wrap your training
code — it records it:

- **Record any run** — a script, notebook, heredoc, Kaggle kernel or remote GPU job — with git
  provenance, through a plain file contract (`$KITCHEN_RUN_DIR`). Projects never import kitchen.
- **Score honestly** — kitchen computes every score from out-of-fold predictions against frozen
  splits and compares runs with one paired, cluster-bootstrap verdict.
- **Gate the irreversible steps** — promoting a champion, submitting, pushing a Kaggle kernel —
  behind human approval, with lineage from submission back to the run and commit.
- **Do the Kaggle and GPU plumbing** — code-competition kernels, offline wheel/asset packs,
  leaderboard capture, ssh GPU hosts, batch predictions with the champion.
- **Ship the agent playbook** — a `/kitchen` skill, a session-start brief of the run history,
  and permission rules, so interactive and autonomous sessions share one way of working.

It is held to a hard size budget (`budget.toml`: 3,500 package lines, 6 dependencies, 16
commands), enforced in CI.

## Repository

| Path | What |
|---|---|
| `kitchen/` | kitchen 2.0 (under construction) |
| `recipes/` | `rkoren-recipes`: standalone YAML -> Terraform generator (optional, separate) |
| `tools/check_budget.py` | size-budget check (CI + pre-commit) |
| `tools/adoption.py` | measures how much of an agent's train/eval work runs through kitchen |

## Roadmap

| Milestone | Scope |
|---|---|
| M0 | Freeze 1.x, split out `recipes`, budgets + adoption baseline |
| M1 | Kaggle code-competition submission spike |
| M2 | Core (ledger, runs, splits, scoring, verdicts, promote) + interactive agent layer |
| M3 | Kaggle path (kernels, packs, submit, LB calibration) + blending |
| M4 | ssh GPU hosts + batch predictions |
| M5 | Headless autonomous loop |
| M6 | 2.0.0 release |

Each milestone closes on measured adoption in a real project, not on internal polish.

## Development

```bash
make budget   # size budgets
make test     # package tests
```

MIT licensed.
