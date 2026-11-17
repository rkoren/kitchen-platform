.PHONY: budget test adoption hooks

# Size budgets (budget.toml) — the same check CI runs.
budget:
	python3 tools/check_budget.py

# Tests for each package present.
test:
	uv run --directory recipes --extra dev python -m pytest -q
	@if [ -f kitchen/pyproject.toml ]; then uv run --directory kitchen --all-extras python -m pytest -q; fi

# Adoption on a real project: make adoption PROJECT=~/cbb-model [SINCE=2026-11-01]
adoption:
	python3 tools/adoption.py --project $(PROJECT) $(if $(SINCE),--since $(SINCE),)

# Install pre-commit hooks into the local git repo
hooks:
	pre-commit install
