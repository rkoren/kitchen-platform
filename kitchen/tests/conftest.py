import logging
import os
import sys
from unittest.mock import MagicMock

import mlflow
import pytest


def pytest_configure(config):  # pylint: disable=unused-argument
    # Prefect's ephemeral server tries to log to a Rich console that pytest has
    # already closed during teardown, producing a harmless but noisy ValueError.
    # logging.raiseExceptions controls whether handler errors are printed;
    # disabling it silences the traceback without hiding real test failures.
    logging.raiseExceptions = False

    # kaggle.__init__ calls api.authenticate() at import time, which calls exit(1)
    # when no credentials are configured, crashing pytest collection. Pre-inject a
    # mock so test_submit.py's top-level `import kaggle` succeeds; individual tests
    # override api methods via patch.object as needed.
    if "kaggle" not in sys.modules:
        sys.modules["kaggle"] = MagicMock()


@pytest.fixture(autouse=True)
def _end_mlflow_run():
    """Ensure no MLflow run or tracking-URI change leaks between tests.

    _log_feature_importances may auto-start a run when called outside a tracked
    context (mocked tracker tests). Ending any active run after each test
    prevents 'run already active' errors in later tests.

    mlflow.set_experiment() (called by Tracker.__init__ and by test_tracking.py
    tests directly) changes THREE pieces of global state that must all be
    restored:
      1. mlflow.tracking.fluent._active_experiment_id  — in-process cache
      2. MLFLOW_EXPERIMENT_ID env var — set explicitly by set_experiment() for
         subprocess inheritance; leaks into subsequent tests via _get_experiment_id_from_env()
      3. MLFLOW_TRACKING_URI (handled by set_tracking_uri calls in some tests)
    Restoring only the URI left the experiment ID pointing at an experiment that
    does not exist in the restored SQLite store, causing MlflowException in any
    test that calls mlflow.start_run() afterward.
    """
    # Snapshot ALL MLflow global state that tests can mutate.
    # Both mlflow.set_tracking_uri() and mlflow.set_experiment() write to
    # os.environ as a subprocess-inheritance side effect, so we must capture and
    # restore the env vars too — not just the in-process mlflow state.
    orig_tracking_uri_env = os.environ.get("MLFLOW_TRACKING_URI")
    orig_exp_id_env = os.environ.get("MLFLOW_EXPERIMENT_ID")
    original_uri = mlflow.get_tracking_uri()
    try:
        orig_active_exp = mlflow.tracking.fluent._active_experiment_id  # pylint: disable=protected-access
    except AttributeError:
        orig_active_exp = None

    yield

    mlflow.end_run()

    # Restore tracking URI without side-effecting os.environ.  We manage the env
    # var ourselves below so that subsequent tests see the exact same env they
    # started with — not one that has MLFLOW_TRACKING_URI injected by this teardown.
    try:
        from mlflow.tracking._tracking_service import (
            utils as _ts_utils,  # pylint: disable=import-outside-toplevel
        )
        _ts_utils._tracking_uri = original_uri  # pylint: disable=protected-access
    except Exception:  # pylint: disable=broad-exception-caught
        mlflow.set_tracking_uri(original_uri)  # fallback if internals change

    # Restore MLFLOW_TRACKING_URI env var.
    if orig_tracking_uri_env is None:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    else:
        os.environ["MLFLOW_TRACKING_URI"] = orig_tracking_uri_env

    # Restore MLFLOW_EXPERIMENT_ID env var (set_experiment() writes it for subprocess inheritance).
    if orig_exp_id_env is None:
        os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
    else:
        os.environ["MLFLOW_EXPERIMENT_ID"] = orig_exp_id_env

    # Restore in-process experiment ID cache.
    try:
        mlflow.tracking.fluent._active_experiment_id = orig_active_exp  # pylint: disable=protected-access
    except AttributeError:
        pass  # guard against future mlflow internal refactors
