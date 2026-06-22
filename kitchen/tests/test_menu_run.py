"""Tests for kitchen.menu_run — the pipeline runner (INT-005)."""

from __future__ import annotations

import os
import subprocess
from unittest.mock import patch

import pytest

from kitchen.menu import Menu
from kitchen.menu_run import PipelineError, run_pipeline

FULL = Menu.model_validate(
    {
        "project": "p",
        "pipeline": ["provision", "train", "serve", "monitor"],
        "recipes": {
            "mlflow-backend": {"kind": "rds", "role": "mlflow-backend"},
            "train": {"kind": "stage", "source": "src/train/run.py"},
            "serve": {"kind": "lambda", "role": "serving", "source": "src/serve/"},
        },
    }
)


class _Recorder:
    def __init__(self):
        self.cmds: list[list[str]] = []

    def __call__(self, cmd):
        self.cmds.append(cmd)


def test_sequences_provision_stage_monitor_and_skips_serve():
    rec = _Recorder()
    with patch("kitchen.menu_run.resolve_mlflow_env", return_value={}):
        run_pipeline(FULL, menu_path="menu.yaml", state_bucket="b", run=rec)
    assert rec.cmds == [
        ["recipes", "apply", "menu.yaml", "--state-bucket", "b", "--yes"],
        ["kitchen", "run", "train"],
        ["kitchen", "run", "monitor"],
    ]  # serve (lambda) is recognised but not run


def test_dry_run_executes_nothing_but_plans():
    rec = _Recorder()
    lines: list[str] = []
    run_pipeline(FULL, menu_path="menu.yaml", state_bucket="b", dry_run=True, run=rec, echo=lines.append)
    assert rec.cmds == []
    assert any("provision" in line for line in lines)
    assert any("train" in line for line in lines)


def test_provision_without_state_bucket_raises():
    menu = Menu.model_validate({"project": "p", "pipeline": ["provision"]})
    with pytest.raises(PipelineError, match="state bucket"):
        run_pipeline(menu, menu_path="m", run=_Recorder())


def test_provision_materializes_env_into_process(monkeypatch):
    menu = Menu.model_validate(
        {
            "project": "p",
            "pipeline": ["provision"],
            "recipes": {"db": {"kind": "rds", "role": "backend"}},
        }
    )
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with patch(
        "kitchen.menu_run.resolve_mlflow_env", return_value={"MLFLOW_TRACKING_URI": "postgresql://x"}
    ):
        run_pipeline(menu, menu_path="m", state_bucket="b", run=_Recorder())
    try:
        assert os.environ["MLFLOW_TRACKING_URI"] == "postgresql://x"  # inherited by later stages
    finally:
        os.environ.pop("MLFLOW_TRACKING_URI", None)


def test_fail_fast_propagates_step_error():
    menu = Menu.model_validate(
        {
            "project": "p",
            "pipeline": ["train"],
            "recipes": {"train": {"kind": "stage", "source": "src/train/run.py"}},
        }
    )

    def boom(cmd):
        raise subprocess.CalledProcessError(1, cmd)

    with pytest.raises(subprocess.CalledProcessError):
        run_pipeline(menu, menu_path="m", state_bucket="b", run=boom)
