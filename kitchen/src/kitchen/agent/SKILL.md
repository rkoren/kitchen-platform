---
name: kitchen
description: Run, record, score and compare ML experiments with the kitchen CLI in a project that has a kitchen.yaml. Use whenever you train, evaluate, tune, debug or compare models, read experiment history, or propose a champion or a submission.
---

# kitchen — the experiment loop

kitchen records every training/evaluation run, scores it itself against a frozen split, and
compares runs with a paired bootstrap. You do the thinking and write the code; kitchen keeps the
memory and keeps results honest. The session brief (`kitchen show --brief`) lists each track's
champion, recent verdicts (negatives included), runs awaiting promotion, and anything in flight.

## Hard rules
1. Run all train/eval work through `kitchen run`: a job (`kitchen run <job> -p key=value`) or any
   command (`kitchen run -m "<hypothesis>" -- <command>`). For a heredoc or one-off script use
   `kitchen run --scratch -m "<hypothesis>" -- python - <<'EOF'`. Never fit or score models outside it.
2. Never report a number kitchen did not compute. Quote the verdict — `better` / `worse` / `tie`
   with its delta and CI — not your own arithmetic. A `tie` is not a win.
3. One hypothesis per run: `-m "<what changed and why>" --parent <run> --op <operator>`.
4. Record what every run taught you, especially negatives: `kitchen mark <run> --note "<finding>"`.
5. Splits are immutable. Never edit a split definition to make a result look better; a new
   validation scheme is a new split name, and the champion must be re-run on it.
6. Promotion and submission are human decisions. Propose with `kitchen compare <run> --md`, then
   stop and ask. (Under `KITCHEN_AUTONOMOUS=1` kitchen refuses them outright.)
7. Long runs: start with `--detach`, then `kitchen status --wait <run> --timeout 9m` (repeat as
   needed) or `kitchen status --watch <run>`. Never walk away from a run you started without saying so.
8. Call the bare `kitchen` command and add `--json` when you parse output. Never call
   `.venv/bin/kitchen`: in some projects that is a different, legacy tool.

## The cycle
1. **Collect** — `kitchen status` closes finished or lost runs; read the brief.
2. **Orient** — `kitchen show` (board) and `kitchen show <run> --lineage`: what the parent and its
   siblings already tried. Do not repeat a sibling's change.
3. **Pick one operator**
   - `debug`: the latest run failed — `kitchen show <run>` has the log tail; fix and re-run.
   - `improve`: one targeted change to the champion (or best run), `--parent <champion>`.
   - `draft`: after ~3 ties or losses in a row on one line of work, try a genuinely different approach.
   - `confirm`: re-run a promising winner with new seeds or from a clean commit before proposing it.
4. **Run** — `kitchen run <job> -p ... -m "..." --parent <run> --op <operator>`.
5. **Read the verdict** — printed when the run ends; also in `kitchen show <run>`.
6. **Note it** — `kitchen mark <run> --note "..."`. If it beat the champion, propose promotion.

## What a run must write
Each run gets `KITCHEN_RUN_DIR` (write outputs here), `KITCHEN_SPLIT` (parquet with the split's id
column, targets, group/cluster columns and `fold`: >= 0 is a validation fold, -1 is train-only),
`KITCHEN_PARAMS` (JSON), `KITCHEN_RUN_ID` and `KITCHEN_TRACK`.
- Out-of-fold predictions: `$KITCHEN_RUN_DIR/oof.parquet` (or `.csv`) with the split's id column and
  a `pred` column (one column per target if there are several). Predict every validation row —
  partial coverage makes a run incomparable.
- If the project has its own evaluator (e.g. per-video scores): `$KITCHEN_RUN_DIR/scores.csv` with
  `group,<value>`, one row per validation group.
- Anything else (`metrics.json`, models under `artifacts/`) is kept with the run, never used in verdicts.

## Commands
| Command | Purpose |
|---|---|
| `kitchen run <job> [-p k=v,v2] [-m HYP] [--parent R] [--op OP] [--detach] [--timeout 30m]` | run a job (commas make a grid) |
| `kitchen run [--scratch] -m HYP -- <command>` | run anything; `--import DIR` records outputs made elsewhere |
| `kitchen status [--wait R --timeout 9m \| --watch R]` | collect finished or lost runs |
| `kitchen show [R] [--lineage] [--brief] [--champion]` | board, run detail, lineage, champion paths |
| `kitchen compare A [B] [--md]` | verdict for A vs B (default B: the champion) |
| `kitchen mark R --note TEXT \| --quarantine REASON \| --retire \| --baseline` | memory and hygiene |
| `kitchen promote R` | human-approved: make R the champion |
| `kitchen split NAME` / `kitchen split --list` | build or list frozen splits |

Exit codes: 0 ok · 2 config error · 3 needs a human · 5 incomparable · 10 the run's command failed ·
124 timeout. How verdicts, noise and leakage work: [stats.md](stats.md).
