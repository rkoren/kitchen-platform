<!-- kitchen -->
## kitchen (experiment loop)

This project records ML experiments with kitchen (`kitchen.yaml`; state in `.kitchen/`).
- Run every train/eval through `kitchen run`; quote kitchen's verdicts, never your own numbers.
- One hypothesis per run (`-m`, `--parent`, `--op`); note every outcome with `kitchen mark`.
- Promotion and submission are human decisions: propose with `kitchen compare <run> --md`.
- Use the bare `kitchen` command (add `--json` to parse output). Details: the `/kitchen` skill.
