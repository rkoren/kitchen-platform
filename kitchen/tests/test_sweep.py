"""Tests for the `kitchen sweep` CLI command (SWEEP-005).

`train_pipeline` and the MLflow client are mocked so the sweep logic (combo
expansion, ranking, error handling, tagging) is exercised without training a
real model. `itertools.product` runs for real.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import yaml
from typer.testing import CliRunner

from kitchen.cli import app

runner = CliRunner()

# Maps a sweep run_id to the metric value its fake run reports.
SCORES = {
    "run-d4": 0.30,
    "run-d6": 0.10,  # best when lower_is_better
    "run-d8": 0.20,
}


def _write_params(tmp_path, thresholds=True):
    params = {"experiment": "sweep-exp", "model": {"max_depth": 3}}
    if thresholds:
        params["thresholds"] = {"val_brier": {"max": 0.5}}  # lower-is-better
    path = tmp_path / "params.yaml"
    path.write_text(yaml.dump(params))
    return str(path)


def _fake_client(metric_name="val_brier", scores=None):
    """An MlflowClient stand-in whose get_run returns the mapped metric value."""
    scores = scores if scores is not None else SCORES
    client = MagicMock()

    def get_run(run_id):
        run = MagicMock()
        run.data.metrics = {metric_name: scores[run_id]} if run_id in scores else {}
        return run

    client.get_run.side_effect = get_run
    return client


def _patches(train_side_effect, client):
    """Patch train_pipeline, MlflowClient, and configure_from_env for run_sweep."""
    return (
        patch("kitchen.flows.train_flow.train_pipeline", side_effect=train_side_effect),
        patch("mlflow.tracking.MlflowClient", return_value=client),
        patch("kitchen.tracking.configure_from_env"),
    )


# ── Combo expansion + ranking ──────────────────────────────────────────────────


def test_sweep_runs_one_pipeline_per_combo(tmp_path):
    params_file = _write_params(tmp_path)
    run_ids = ["run-d4", "run-d6", "run-d8"]
    train = MagicMock(side_effect=run_ids)
    client = _fake_client()
    p1, p2, p3 = _patches(train, client)
    with p1 as mock_train, p2, p3:
        result = runner.invoke(
            app,
            ["sweep", "--params", params_file, "--override", "model.max_depth=4,6,8"],
            catch_exceptions=False,
        )
    assert result.exit_code == 0
    assert mock_train.call_count == 3
    # Each call shadowed model.max_depth with one grid value.
    depths = sorted(c.kwargs["overrides"]["model.max_depth"] for c in mock_train.call_args_list)
    assert depths == [4, 6, 8]


def test_sweep_cartesian_product_of_two_keys(tmp_path):
    params_file = _write_params(tmp_path)
    train = MagicMock(side_effect=[f"run-{i}" for i in range(4)])
    client = _fake_client(scores={f"run-{i}": 0.1 * (i + 1) for i in range(4)})
    p1, p2, p3 = _patches(train, client)
    with p1 as mock_train, p2, p3:
        result = runner.invoke(
            app,
            [
                "sweep", "--params", params_file,
                "--override", "model.max_depth=4,6",
                "--override", "model.eta=0.05,0.1",
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0
    assert mock_train.call_count == 4  # 2 × 2


def test_sweep_ranks_best_by_lower_is_better(tmp_path):
    params_file = _write_params(tmp_path)  # thresholds → val_brier, lower-is-better
    train = MagicMock(side_effect=["run-d4", "run-d6", "run-d8"])
    client = _fake_client()
    p1, p2, p3 = _patches(train, client)
    with p1, p2, p3:
        result = runner.invoke(
            app,
            ["sweep", "--params", params_file, "--override", "model.max_depth=4,6,8"],
            catch_exceptions=False,
        )
    assert result.exit_code == 0
    # run-d6 has the lowest val_brier (0.10) → best.
    assert "Best: run-d6" in result.stdout
    assert "kitchen promote --run-id run-d6" in result.stdout


def test_sweep_higher_is_better_flips_winner(tmp_path):
    params_file = _write_params(tmp_path, thresholds=False)
    train = MagicMock(side_effect=["run-d4", "run-d6", "run-d8"])
    client = _fake_client(metric_name="val_acc", scores={"run-d4": 0.7, "run-d6": 0.9, "run-d8": 0.8})
    p1, p2, p3 = _patches(train, client)
    with p1, p2, p3:
        result = runner.invoke(
            app,
            [
                "sweep", "--params", params_file,
                "--override", "model.max_depth=4,6,8",
                "--metric", "val_acc", "--higher-is-better",
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0
    assert "Best: run-d6" in result.stdout  # 0.9 is highest


# ── sweep.group tagging ─────────────────────────────────────────────────────────


def test_sweep_tags_every_run_with_group(tmp_path):
    params_file = _write_params(tmp_path)
    train = MagicMock(side_effect=["run-d4", "run-d6", "run-d8"])
    client = _fake_client()
    p1, p2, p3 = _patches(train, client)
    with p1, p2, p3:
        runner.invoke(
            app,
            ["sweep", "--params", params_file, "--override", "model.max_depth=4,6,8"],
            catch_exceptions=False,
        )
    tagged = [c.args for c in client.set_tag.call_args_list]
    assert len(tagged) == 3
    # Same group id applied to every run.
    groups = {args[2] for args in tagged}
    assert len(groups) == 1
    assert all(args[1] == "sweep.group" for args in tagged)


# ── Metric auto-detection ───────────────────────────────────────────────────────


def test_sweep_autodetects_metric_from_thresholds(tmp_path):
    params_file = _write_params(tmp_path)  # val_brier in thresholds
    train = MagicMock(side_effect=["run-d4", "run-d6", "run-d8"])
    client = _fake_client()  # reports val_brier
    p1, p2, p3 = _patches(train, client)
    with p1, p2, p3:
        result = runner.invoke(
            app,
            ["sweep", "--params", params_file, "--override", "model.max_depth=4,6,8"],
            catch_exceptions=False,
        )
    assert result.exit_code == 0
    assert "metric=val_brier (lower=better)" in result.stdout


def test_sweep_errors_when_no_metric_and_no_thresholds(tmp_path):
    params_file = _write_params(tmp_path, thresholds=False)
    train = MagicMock(side_effect=["run-d4", "run-d6"])
    client = _fake_client()
    p1, p2, p3 = _patches(train, client)
    with p1, p2, p3:
        result = runner.invoke(
            app,
            ["sweep", "--params", params_file, "--override", "model.max_depth=4,6"],
            catch_exceptions=False,
        )
    assert result.exit_code == 1
    assert "--metric is required" in result.stderr


# ── Error handling ──────────────────────────────────────────────────────────────


def test_sweep_requires_override(tmp_path):
    params_file = _write_params(tmp_path)
    result = runner.invoke(app, ["sweep", "--params", params_file], catch_exceptions=False)
    assert result.exit_code == 1
    assert "--override is required" in result.stderr


def test_sweep_rejects_override_without_equals(tmp_path):
    params_file = _write_params(tmp_path)
    result = runner.invoke(
        app,
        ["sweep", "--params", params_file, "--override", "model.max_depth"],
        catch_exceptions=False,
    )
    assert result.exit_code == 1
    assert "key=value format" in result.stderr


def test_sweep_rejects_single_combination(tmp_path):
    params_file = _write_params(tmp_path)
    result = runner.invoke(
        app,
        ["sweep", "--params", params_file, "--override", "model.max_depth=4"],
        catch_exceptions=False,
    )
    assert result.exit_code == 1
    assert "more than one combination" in result.stderr


def test_sweep_enforces_max_combos(tmp_path):
    params_file = _write_params(tmp_path)
    result = runner.invoke(
        app,
        [
            "sweep", "--params", params_file,
            "--override", "model.max_depth=1,2,3,4,5",
            "--max-combos", "3",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 1
    assert "limit 3" in result.stderr


def test_sweep_aborts_on_first_combo_failure(tmp_path):
    params_file = _write_params(tmp_path)
    train = MagicMock(side_effect=ModuleNotFoundError("No module named 'src'"))
    client = _fake_client()
    p1, p2, p3 = _patches(train, client)
    with p1 as mock_train, p2, p3:
        result = runner.invoke(
            app,
            ["sweep", "--params", params_file, "--override", "model.max_depth=4,6,8"],
            catch_exceptions=False,
        )
    assert result.exit_code == 1
    assert "first sweep run failed" in result.stderr
    assert mock_train.call_count == 1  # aborted, did not try combo 2 or 3


def test_sweep_continues_after_later_combo_failure(tmp_path):
    params_file = _write_params(tmp_path)
    # First succeeds, second raises, third succeeds.
    train = MagicMock(side_effect=["run-d4", RuntimeError("bad param"), "run-d8"])
    client = _fake_client(scores={"run-d4": 0.30, "run-d8": 0.20})
    p1, p2, p3 = _patches(train, client)
    with p1 as mock_train, p2, p3:
        result = runner.invoke(
            app,
            ["sweep", "--params", params_file, "--override", "model.max_depth=4,6,8"],
            catch_exceptions=False,
        )
    assert result.exit_code == 0
    assert mock_train.call_count == 3  # did not abort
    assert "run failed, skipping" in result.stderr
    assert "Best: run-d8" in result.stdout  # 0.20 < 0.30


def test_sweep_errors_when_no_run_logged_metric(tmp_path):
    params_file = _write_params(tmp_path)
    train = MagicMock(side_effect=["run-d4", "run-d6"])
    # Client returns runs with no matching metric.
    client = _fake_client(scores={})
    p1, p2, p3 = _patches(train, client)
    with p1, p2, p3:
        result = runner.invoke(
            app,
            ["sweep", "--params", params_file, "--override", "model.max_depth=4,6"],
            catch_exceptions=False,
        )
    assert result.exit_code == 1
    assert "nothing to rank" in result.stderr
