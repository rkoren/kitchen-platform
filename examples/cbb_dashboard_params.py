"""VAL-006: dashboard param-comparison validation after a CBB sweep.

Verifies that ``kitchen dashboard generate`` (DASH-001) renders the data and
template hooks required by the VAL-006 acceptance criteria:

- the ``model.max_depth`` column appears when ``--show-params model.max_depth``
  is passed (param columns rely on ``params`` in the results JSON — LML-010),
- the promoted champion run is flagged for distinct highlighting,
- a Kaggle LB-score column appears when at least one run carries an ``lb_score``.

The script runs ``kitchen dashboard generate`` itself, then parses the embedded
``RESULTS`` / ``PARAM_KEYS`` / ``HAS_LB`` data out of the generated HTML. It can
verify the *data and template hooks* (the param key list, the ``champion`` flag,
the ``tr.champion`` highlight rule, the LB column gate) — it does not render the
page, so visual highlighting is asserted via the template hook, not pixels.

Prerequisites
-------------
- ``kitchen`` installed (``pip install -e kitchen/``).
- The VAL-002 sweep completed (three ``kitchen run train --override
  model.max_depth=N`` runs).
- ``kitchen push`` run for **each** run so the local ``results`` branch holds a
  ``results/<sha>.json`` per run — otherwise ``kitchen dashboard generate`` exits
  with "branch 'results' not found locally". One of those pushes must record the
  promoted champion (``champion: true``).
- Optional: a ``kitchen submit`` so an ``lb_score`` is present; the LB-column
  check is skipped (not failed) when no run has a score.
- Run from the project root (or set ``PROJECT_DIR``).

Usage
-----
    cd /path/to/cbb-model
    kitchen run train --override model.max_depth=4 --auto-promote
    kitchen push
    kitchen run train --override model.max_depth=6
    kitchen push
    kitchen run train --override model.max_depth=8
    kitchen push
    python /path/to/kitchen-platform/examples/cbb_dashboard_params.py

Acceptance criteria (VAL-006)
-----------------------------
1. ``kitchen dashboard generate --show-params model.max_depth`` writes the HTML
   and the ``model.max_depth`` param column is present.
2. The champion row is flagged for distinct highlighting.
3. The Kaggle LB-score column appears when any run has an ``lb_score`` (skipped
   otherwise).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

_PARAM = "model.max_depth"


def _extract_js_value(html: str, var_name: str):
    """Pull `var <name> = <json>;` out of the generated dashboard HTML."""
    m = re.search(rf"var {re.escape(var_name)} = (.+);", html)
    if not m:
        return None
    return json.loads(m.group(1))


def main() -> int:
    project_dir = os.environ.get("PROJECT_DIR", os.getcwd())
    output = os.path.join(project_dir, "dashboard", "index.html")

    # --- Generate the dashboard (exercises the command under test) --------------
    proc = subprocess.run(
        [
            sys.executable, "-m", "kitchen.cli", "dashboard", "generate",
            "--show-params", _PARAM,
            "--output", output,
        ],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )
    print(proc.stdout, end="")
    if proc.returncode != 0:
        print(proc.stderr, end="")
        print("\n[error] `kitchen dashboard generate` failed.")
        print("  Did you run `kitchen push` after each training run? The results")
        print("  branch must contain results/*.json for the dashboard to build.")
        return 1

    if not os.path.exists(output):
        print(f"[error] Expected generated dashboard at {output}, not found.")
        return 1

    with open(output) as f:
        html = f.read()

    results = _extract_js_value(html, "RESULTS") or []
    param_keys = _extract_js_value(html, "PARAM_KEYS") or []
    has_lb = _extract_js_value(html, "HAS_LB")

    n_runs = len(results)
    n_champ = sum(1 for r in results if r.get("champion"))
    n_with_params = sum(1 for r in results if r.get("params"))
    n_with_lb = sum(1 for r in results if r.get("lb_score") is not None)

    print(f"\nGenerated: {output}")
    print(f"  runs            : {n_runs}")
    print(f"  with params     : {n_with_params}")
    print(f"  champion-flagged: {n_champ}")
    print(f"  with lb_score   : {n_with_lb}")

    # --- Acceptance checks ------------------------------------------------------
    print("\n--- Acceptance checks ---")

    ok_runs = n_runs >= 3
    print(f"[{'OK' if ok_runs else 'FAIL'}] At least 3 runs plotted (found {n_runs})")

    ok_param = _PARAM in param_keys
    print(f"[{'OK' if ok_param else 'FAIL'}] '{_PARAM}' param column present (PARAM_KEYS={param_keys})")
    if not ok_param and n_with_params == 0:
        print("        No run carries `params` — re-run `kitchen push` after LML-010")
        print("        so the results JSON includes logged params.")

    ok_champ = n_champ >= 1 and "tr.champion" in html
    print(f"[{'OK' if ok_champ else 'FAIL'}] Champion row flagged + 'tr.champion' highlight rule present")
    if n_champ == 0:
        print("        No run has champion=true — promote one (`--auto-promote`) then `kitchen push`.")

    # LB column is conditional: only meaningful once a submission exists.
    if n_with_lb > 0:
        ok_lb = has_lb is True and "LB Score" in html
        print(f"[{'OK' if ok_lb else 'FAIL'}] LB-score column shown (HAS_LB={has_lb})")
    else:
        ok_lb = True
        print("[SKIP] LB-score column — no run has an lb_score (run `kitchen submit` to test)")

    passed = ok_runs and ok_param and ok_champ and ok_lb
    print(f"\n{'PASS' if passed else 'FAIL'}: VAL-006")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
