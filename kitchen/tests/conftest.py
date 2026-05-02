import logging

import mlflow
import pytest


def pytest_configure(config):
    # Prefect's ephemeral server tries to log to a Rich console that pytest has
    # already closed during teardown, producing a harmless but noisy ValueError.
    # logging.raiseExceptions controls whether handler errors are printed;
    # disabling it silences the traceback without hiding real test failures.
    logging.raiseExceptions = False


@pytest.fixture(autouse=True)
def _end_mlflow_run():
    """Ensure no MLflow run leaks between tests.

    _log_feature_importances may auto-start a run when called outside a tracked
    context (mocked tracker tests). Ending any active run after each test
    prevents 'run already active' errors in later tests.
    """
    yield
    mlflow.end_run()
