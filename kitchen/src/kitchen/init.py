"""`kitchen init`: scaffold kitchen.yaml; `--agent` installs the Claude Code skill, hook and permissions."""

import json
from importlib.resources import files
from pathlib import Path

from kitchen import __version__

TEMPLATE = """# kitchen.yaml — runs are recorded under .kitchen/ (see `kitchen --help`)
project: {name}
python: .venv/bin/python        # interpreter used for {{python}} in job commands
pythonpath: [src]               # prepended to PYTHONPATH for every run
# env_file: .env

splits:                         # built once and frozen; a new definition needs a new name
  cv:
    data: data/train.csv        # .parquet or .csv
    id: id
    target: target
    scheme: kfold               # kfold | stratified | group | logo | forward | holdout
    k: 5

tracks:                         # brier logloss auc auc_macro accuracy balanced_accuracy f1 rmse mae r2
  main: {{split: cv, metric: auc}}

jobs:                           # a job writes $KITCHEN_RUN_DIR/oof.parquet (id column + pred)
  baseline:
    cmd: "{{python}} src/train.py --seed {{seed}}"
    params: {{seed: 0}}

run: {{timeout: 30m}}
"""
MARKER = "<!-- kitchen -->"


def scaffold(root: Path) -> list[str]:
    wrote = []
    if not (root / "kitchen.yaml").exists():
        (root / "kitchen.yaml").write_text(TEMPLATE.format(name=root.name))
        wrote.append("kitchen.yaml")
    gitignore = root / ".gitignore"
    lines = gitignore.read_text().splitlines() if gitignore.exists() else []
    if not {".kitchen", ".kitchen/"} & set(lines):
        gitignore.write_text("\n".join([*lines, ".kitchen/"]) + "\n")
        wrote.append(".gitignore")
    return wrote


def install_agent(root: Path) -> list[str]:
    """Copy the skill, merge permissions + the SessionStart hook, add the CLAUDE.md section."""
    src = files("kitchen") / "agent"
    skill = root / ".claude" / "skills" / "kitchen"
    skill.mkdir(parents=True, exist_ok=True)
    stamp = f"\n<!-- kitchen-version: {__version__} -->\n"
    (skill / "SKILL.md").write_text((src / "SKILL.md").read_text() + stamp)
    (skill / "stats.md").write_text((src / "stats.md").read_text())

    path = root / ".claude" / "settings.json"
    settings = json.loads(path.read_text()) if path.exists() else {}
    template = json.loads((src / "settings.json").read_text())
    perms = settings.setdefault("permissions", {})
    for kind, rules in template["permissions"].items():
        perms[kind] = list(dict.fromkeys([*perms.get(kind, []), *rules]))
    hooks = settings.setdefault("hooks", {}).setdefault("SessionStart", [])
    commands = {h.get("command") for entry in hooks for h in entry.get("hooks", [])}
    hooks += [
        e for e in template["hooks"]["SessionStart"] if e["hooks"][0]["command"] not in commands
    ]
    path.write_text(json.dumps(settings, indent=2) + "\n")

    claude_md = root / "CLAUDE.md"
    text = claude_md.read_text() if claude_md.exists() else ""
    if MARKER not in text:
        claude_md.write_text(
            text + ("\n\n" if text.strip() else "") + (src / "claude-md.md").read_text()
        )
    return [
        ".claude/skills/kitchen/SKILL.md",
        ".claude/skills/kitchen/stats.md",
        ".claude/settings.json",
        "CLAUDE.md",
    ]
