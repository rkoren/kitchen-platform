"""MLflow tracking setup and run wrapper.

Usage::

    from kitchen.tracking import configure_from_env, init_experiment, Tracker

    configure_from_env()
    init_experiment("my-project")

    tracker = Tracker("my-experiment")
    with tracker.run(params=params) as run:
        model = train(...)
        tracker.log_metrics({"accuracy": 0.92})
        tracker.log_model(model, "model", flavour="xgboost")

Environment variables:
    MLFLOW_TRACKING_URI    — tracking store (default: sqlite:///mlruns.db)
    MLFLOW_ARTIFACT_BUCKET — S3 bucket name; when set, new experiments store artifacts
                             at s3://<bucket>/mlflow-artifacts/<experiment-name>
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import mlflow

_TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
_ARTIFACT_BUCKET_ENV = "MLFLOW_ARTIFACT_BUCKET"
_ARTIFACT_PREFIX = "mlflow-artifacts"


_FLAVOURS: dict[str, Any] = {
    "sklearn": mlflow.sklearn,
    "xgboost": mlflow.xgboost,
    "lightgbm": mlflow.lightgbm,
    "pyfunc": mlflow.pyfunc,
}


_TRACKED_PACKAGES = [
    "kitchen",
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "mlflow",
]


def _git_sha() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _package_versions(packages: list[str]) -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for pkg in packages:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def _dict_hash(d: dict) -> str:
    import hashlib
    import json

    canonical = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _file_hash(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def log_run_context(
    params: dict | None = None,
    data_path: Path | str | None = None,
) -> None:
    """Best-effort: tag the active MLflow run with git SHA, package versions, and data/params hashes."""
    import sys

    try:
        tags: dict[str, str] = {}
        sha = _git_sha()
        if sha:
            tags["kitchen.git_sha"] = sha
        tags["kitchen.python"] = (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        for pkg, ver in _package_versions(_TRACKED_PACKAGES).items():
            tags[f"kitchen.pkg.{pkg}"] = ver
        if params is not None:
            tags["kitchen.params_sha256"] = _dict_hash(params)
        if data_path is not None:
            p = Path(data_path)
            if p.exists():
                tags["kitchen.data_sha256"] = _file_hash(p)
        mlflow.set_tags(tags)
    except Exception:
        pass


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict to dot-separated keys for MLflow log_params."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


class Tracker:
    def __init__(self, experiment: str, tracking_uri: str | None = None) -> None:
        """Set the MLflow experiment, optionally pointing at a remote tracking server."""
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        # NB-008: Trainer.run() honours this flag to skip model-artifact logging
        # for throwaway sessions (set False by kitchen.init_run(log_model=False)).
        self.log_model_enabled: bool = True

    @contextlib.contextmanager
    def run(
        self,
        run_name: str | None = None,
        params: dict | None = None,
    ) -> Generator[mlflow.ActiveRun, None, None]:
        """Context manager that starts an MLflow run and logs flattened params on entry."""
        with mlflow.start_run(run_name=run_name) as active_run:
            if params:
                mlflow.log_params(_flatten(params))
            log_run_context(params=params)
            yield active_run

    @staticmethod
    def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
        """Log a dict of scalar metrics to the active MLflow run."""
        mlflow.log_metrics(metrics, step=step)

    @staticmethod
    def log_model(model: Any, artifact_path: str, flavour: str = "sklearn") -> None:
        """Persist a model artifact using the named MLflow flavour (sklearn, xgboost, lightgbm, pyfunc)."""
        mod = _FLAVOURS.get(flavour)
        if mod is None:
            raise ValueError(f"Unknown flavour: {flavour!r}. Choose from: {list(_FLAVOURS)}")
        mod.log_model(model, artifact_path)


# ── Functional API for env-driven setup ───────────────────────────────────────


class MlflowSchemaError(RuntimeError):
    """Raised when the tracking store's DB schema is older than the installed MLflow."""


def _schema_remediation(tracking_uri: str) -> str:
    """Actionable guidance for an out-of-date tracking-store schema."""
    archive = ""
    if tracking_uri.startswith("sqlite:"):
        db_path = tracking_uri.split("sqlite:///", 1)[-1] or "mlruns.db"
        archive = (
            f"  2. If that fails with \"Can't locate revision\", the database predates this\n"
            f"     MLflow's migration history — archive it and start fresh (local run history\n"
            f"     is lost; use a remote tracking server to keep history across upgrades):\n"
            f"         mv {db_path} {db_path}.bak\n"
        )
    return (
        f"MLflow tracking store schema is out of date for the installed MLflow version.\n"
        f"This usually means {tracking_uri} was created by an older MLflow.\n\n"
        f"To fix:\n"
        f"  1. Try upgrading the schema (preserves run history):\n"
        f"         mlflow db upgrade {tracking_uri}\n"
        f"{archive}"
        f"\nThen re-run the command."
    )


def _is_schema_outdated(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out-of-date database schema" in msg or "mlflow db upgrade" in msg


def _verify_store_schema(tracking_uri: str) -> None:
    """Translate MLflow's opaque out-of-date-schema error into actionable guidance.

    Only DB-backed stores have an alembic schema that can drift; remote (http) stores
    and not-yet-created SQLite files are skipped so this stays a cheap no-op on the
    common paths. Any non-schema failure is ignored here and left for the actual command.
    """
    if tracking_uri.startswith("sqlite:"):
        db_path = tracking_uri.split("sqlite:///", 1)[-1]
        if db_path and not Path(db_path).exists():
            return  # a fresh DB is created with the current schema — nothing to verify
    elif not tracking_uri.startswith(("postgresql:", "mysql:", "mssql:")):
        return  # http/remote or file store — no local schema to verify

    try:
        mlflow.search_experiments(max_results=1)
    except MlflowSchemaError:
        raise
    except Exception as exc:  # noqa: BLE001 — only the schema case is ours to translate
        if _is_schema_outdated(exc):
            raise MlflowSchemaError(_schema_remediation(tracking_uri)) from exc
        # Anything else (server unreachable, permissions, …) is the command's to surface.


def configure(tracking_uri: str, artifact_bucket: str | None = None) -> None:
    """Set MLflow tracking URI and optional S3 artifact bucket."""
    mlflow.set_tracking_uri(tracking_uri)
    if artifact_bucket:
        os.environ[_ARTIFACT_BUCKET_ENV] = artifact_bucket
    _verify_store_schema(tracking_uri)


def configure_from_env() -> None:
    """Configure MLflow from standard environment variables.

    Falls back to a local SQLite store (sqlite:///mlruns.db) when
    MLFLOW_TRACKING_URI is not set, so local dev works without a running server.
    """
    tracking_uri = os.environ.get(_TRACKING_URI_ENV, "sqlite:///mlruns.db")
    artifact_bucket = os.environ.get(_ARTIFACT_BUCKET_ENV)
    configure(tracking_uri=tracking_uri, artifact_bucket=artifact_bucket)


def init_experiment(name: str) -> str:
    """Get or create an MLflow experiment by name.

    When MLFLOW_ARTIFACT_BUCKET is set, new experiments are created with
    artifacts stored at s3://<bucket>/mlflow-artifacts/<name>. Existing
    experiments keep their original artifact location.

    Returns the experiment ID.
    """
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id

    artifact_bucket = os.environ.get(_ARTIFACT_BUCKET_ENV)
    artifact_location = (
        f"s3://{artifact_bucket}/{_ARTIFACT_PREFIX}/{name}" if artifact_bucket else None
    )
    return mlflow.create_experiment(name, artifact_location=artifact_location)
