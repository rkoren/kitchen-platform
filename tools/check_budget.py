#!/usr/bin/env python3
"""Enforce the size budgets in budget.toml. Exits 1 on any violation.

Stdlib only (Python >= 3.11). Counts physical lines. `--json` prints machine-readable results.
"""

import ast
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def count(paths) -> int:
    return sum(len(p.read_bytes().splitlines()) for p in paths)


def files(rel: str, globs: list[str]) -> list[Path]:
    base = ROOT / rel
    return sorted({p for g in globs for p in base.glob(g) if p.is_file()}) if base.is_dir() else []


def check(budget: dict) -> tuple[list[dict], list[str]]:
    rows, errors = [], []

    def row(name, measured, cap):
        rows.append({"check": name, "measured": measured, "cap": cap})
        if measured > cap:
            errors.append(f"{name}: {measured} > cap {cap}")

    pkg = budget["package"]
    modules = files(pkg["path"], ["**/*.py"])
    caps = pkg["modules"]
    for p in modules:
        key = p.relative_to(ROOT / pkg["path"]).with_suffix("").as_posix()
        if key not in caps:
            errors.append(
                f"unbudgeted module: {key}.py (add it to [package.modules] via an owner PR)"
            )
            continue
        row(f"module {key}", count([p]), caps[key])
    milestone = budget["milestone"]
    cap = min(pkg["total"], pkg["milestone_caps"][milestone])
    row(f"package total ({milestone})", count(modules), cap)

    row("tests total", count(files(budget["tests"]["path"], ["**/*.py"])), budget["tests"]["total"])
    sk = budget["skills"]
    row("skills total", count(files(sk["path"], [sk["glob"]])), sk["total"])

    pyproject = ROOT / budget["deps"]["pyproject"]
    if pyproject.exists():
        deps = tomllib.loads(pyproject.read_text())["project"].get("dependencies", [])
        row("core dependencies", len(deps), budget["deps"]["max"])

    cli = ROOT / pkg["path"] / "cli.py"
    if cli.exists():
        paths = None
        for node in ast.walk(ast.parse(cli.read_text())):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "COMMAND_PATHS" for t in node.targets
            ):
                paths = ast.literal_eval(node.value)
        if paths is None:
            errors.append("cli.py must define COMMAND_PATHS (a literal tuple of command paths)")
        else:
            row("command paths", len(paths), budget["commands"]["max"])

    rc = budget["recipes"]
    row("recipes src (frozen)", count(files(rc["path"], rc["globs"])), rc["total"])
    row("recipes tests", count(files(rc["tests_path"], ["**/*.py"])), rc["tests_total"])
    return rows, errors


def main() -> int:
    rows, errors = check(tomllib.loads((ROOT / "budget.toml").read_text()))
    if "--json" in sys.argv:
        print(json.dumps({"ok": not errors, "rows": rows, "errors": errors}, indent=2))
    else:
        for r in rows:
            mark = "FAIL" if r["measured"] > r["cap"] else "ok"
            print(f"{mark:>4}  {r['check']:<32} {r['measured']:>6} / {r['cap']}")
        for e in errors:
            print(f"ERROR {e}")
        print("budget: OK" if not errors else f"budget: {len(errors)} violation(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
