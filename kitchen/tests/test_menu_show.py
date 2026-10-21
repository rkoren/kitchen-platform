"""`kitchen menu show` — a variant's effective params with owned/inherited markers (CBB-035)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from kitchen.cli import app
from kitchen.menu import VariantNotFound, variant_provenance

runner = CliRunner()

_MENU = """\
project: p
experiment: p
pipeline: []
recipes: {}
feature_candidates: [a, b, c]
model:
  eta: 0.05
  calib:
    method: isotonic
  num_rounds: 700
  target: win
variants:
  reg:
    model:
      num_rounds: 700
      target: margin
      holdout_season: null
      calib: null
    feature_candidates:
      add: [d]
      remove: [a]
models:
  tournament: {}
  regm:
    variant: reg
"""


def _params():
    import yaml

    return yaml.safe_load(_MENU)


def _by_key(rows):
    return {r.key: r for r in rows}


def test_variant_provenance_marks_each_key():
    rows = _by_key(variant_provenance(_params(), "reg"))
    assert rows["model.eta"].status == "inherited"
    assert rows["model.num_rounds"].status == "same"  # set by the variant, equal to base
    assert (rows["model.target"].status, rows["model.target"].base) == ("overridden", "win")
    assert rows["model.holdout_season"].status == "added"
    assert rows["feature_candidates"].status == "overridden"
    assert rows["feature_candidates"].value == ["b", "c", "d"]
    # a scalar replacing a whole base subtree shows as one overridden leaf
    assert (rows["model.calib"].status, rows["model.calib"].value) == ("overridden", None)
    assert "model.calib.method" not in rows
    assert "variants" not in rows and not any(k.startswith("variants.") for k in rows)


def test_variant_provenance_none_is_base():
    rows = variant_provenance(_params(), None)
    assert {r.status for r in rows} == {"inherited"}
    assert _by_key(rows)["model.target"].value == "win"


def test_variant_provenance_unknown_variant():
    with pytest.raises(VariantNotFound):
        variant_provenance(_params(), "nope")


@pytest.fixture
def menu(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "menu.yaml").write_text(_MENU)
    return tmp_path


def test_menu_show_variant_markers(menu):
    result = runner.invoke(app, ["menu", "show", "--variant", "reg"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert any(line.startswith("  = model.num_rounds") for line in lines)
    assert any(line.startswith("  * model.target") and "(base: win)" in line for line in lines)
    assert any(line.startswith("  + model.holdout_season") for line in lines)
    assert any(line.startswith("    model.eta") for line in lines)  # inherited: no marker
    assert any(line.startswith("  * feature_candidates") and "+d -a" in line for line in lines)


def test_menu_show_diff_only_owned(menu):
    result = runner.invoke(app, ["menu", "show", "--variant", "reg", "--diff"])
    assert result.exit_code == 0, result.output
    assert "model.target" in result.output
    assert "model.eta" not in result.output


def test_menu_show_section_filter(menu):
    result = runner.invoke(app, ["menu", "show", "--variant", "reg", "--section", "model"])
    assert "model.eta" in result.output
    assert "feature_candidates" not in result.output


def test_menu_show_model_implies_variant(menu):
    result = runner.invoke(app, ["menu", "show", "--model", "regm", "--diff"])
    assert result.exit_code == 0, result.output
    assert "variant 'reg'" in result.output


def test_menu_show_model_without_variant_shows_base(menu):
    result = runner.invoke(app, ["menu", "show", "--model", "tournament"])
    assert result.exit_code == 0, result.output
    assert "declares no variant" in result.output
    assert "Base params" in result.output


def test_menu_show_model_variant_conflict(menu):
    result = runner.invoke(app, ["menu", "show", "--model", "regm", "--variant", "other"])
    assert result.exit_code == 1
    assert "conflicts" in result.output


def test_menu_show_base_lists_variants(menu):
    result = runner.invoke(app, ["menu", "show"])
    assert result.exit_code == 0, result.output
    assert "Base params" in result.output
    assert "variants: reg" in result.output
    assert "*" not in result.output.split("variants:")[0].split("\n", 2)[-1]


def test_menu_show_unknown_variant(menu):
    result = runner.invoke(app, ["menu", "show", "--variant", "nope"])
    assert result.exit_code == 1
    assert "available: reg" in result.output


def test_menu_show_model_invalid_menu_errors_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "menu.yaml").write_text(_MENU + "holdout:\n  season: 2024\n")
    result = runner.invoke(app, ["menu", "show", "--model", "regm"])
    assert result.exit_code == 1
    assert "error:" in result.output
