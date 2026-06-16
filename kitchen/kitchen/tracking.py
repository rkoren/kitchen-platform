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
import re
from collections.abc import Generator
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import mlflow
import mlflow.tracking

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


class ArtifactLocationError(RuntimeError):
    """Raised when a model's stored artifact location can't be reached from this env.

    The common cause (MNT-003) is migrating the tracking store — e.g. local SQLite →
    a remote MLflow server — or moving the project: the registered model version's
    ``source`` still records its original, local artifact path, which no longer
    exists here. ``ModelVersion.source`` is immutable, so it can't be edited in place.
    """


#: A URI scheme is a letter followed by 1+ scheme chars and a colon, e.g. ``s3:``,
#: ``mlflow-artifacts:`` (note: single-slash), ``http:``. Requiring 2+ chars before
#: the colon avoids matching a Windows drive letter (``C:\...``) as a scheme.
_URI_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]+:")


def _local_path_from_uri(uri: str) -> Path | None:
    """Return the local filesystem path a source URI points to, or None if not local.

    ``file://`` URIs and bare/relative/absolute paths are local; any other scheme
    (``s3://``, ``http(s)://``, proxied ``mlflow-artifacts:``) is not — those are
    handled separately by the caller.
    """
    if uri.startswith("file://"):
        return Path(unquote(urlparse(uri).path))
    if _URI_SCHEME.match(uri):
        return None
    return Path(uri)


def _artifact_storage_location(model_uri: str) -> str | None:
    """Best-effort: the real artifact storage location backing a ``models:/`` URI.

    Resolves the registered version's ``source`` for ``models:/<name>@<alias>`` or
    ``models:/<name>/<version>``. In MLflow 3.x that ``source`` is itself a
    logged-model URI (``models:/m-<id>``) whose ``artifact_location`` is the real
    filesystem/S3 path, so this follows that one level of indirection. Never raises.
    """
    if not model_uri.startswith("models:/"):
        return None
    ref = model_uri[len("models:/") :]
    try:
        client = mlflow.tracking.MlflowClient()
        if "@" in ref:
            name, alias = ref.split("@", 1)
            source = client.get_model_version_by_alias(name, alias).source
        elif "/" in ref:
            name, version = ref.rsplit("/", 1)
            source = client.get_model_version(name, version).source
        else:
            return None
        if source and source.startswith("models:/"):
            model_id = source[len("models:/") :]
            try:
                return getattr(client.get_logged_model(model_id), "artifact_location", None) or source
            except Exception:
                return source
        return source
    except Exception:
        return None


_ARTIFACT_MISSING_HINTS = (
    "no such file",
    "does not exist",
    "not found",
    "could not find",
    "failed to download",
    "unable to download",
)


def _looks_like_artifact_missing(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(hint in msg for hint in _ARTIFACT_MISSING_HINTS)


def _artifact_remediation(model_uri: str, source: str) -> str:
    """Actionable guidance for a champion whose artifact location is unreachable."""
    return (
        f"the champion's stored artifact location is not reachable from this environment:\n"
        f"    {model_uri}  →  source: {source}\n\n"
        f"This usually means the tracking store was migrated (e.g. local SQLite → a remote\n"
        f"MLflow server) or the project moved: the registered model version still records\n"
        f"its original, local artifact path, which doesn't exist here. A model version's\n"
        f"`source` is immutable, so it can't be edited in place. To fix, choose one:\n"
        f"  1. Re-train and re-promote against the current store:\n"
        f"         kitchen run train --auto-promote --promote-metric <metric>\n"
        f"  2. Re-register a run whose artifacts already live in the current store and\n"
        f"     re-point the alias:  kitchen promote <metric>  (or `--run-id <id>`).\n"
        f"  3. If you still have the original artifacts, copy them to the location above\n"
        f"     so the recorded `source` resolves again."
    )


def explain_model_load_error(model_uri: str, exc: Exception) -> ArtifactLocationError | None:
    """Translate an artifact-location-drift load failure into actionable guidance.

    Returns an :class:`ArtifactLocationError` (with remediation) when a model load
    failed because the registered version's ``source`` is unreachable from this
    environment — the SQLite→remote migration footgun (MNT-003) — and ``None``
    otherwise, so a healthy champion or a genuine S3-permissions / network error is
    left for the caller to surface unchanged.

    Detection is structural and best-effort: it confirms the version's ``source``
    resolves to a local path that does not exist here (definitive), or, for a proxied
    ``mlflow-artifacts:`` source (resolvable only by the server that wrote it), that
    the failure looks like a missing-artifact error.
    """
    source = _artifact_storage_location(model_uri)
    if source is None:
        return None
    local = _local_path_from_uri(source)
    if local is not None:
        unreachable = not local.exists()
    elif source.startswith("mlflow-artifacts:"):
        unreachable = _looks_like_artifact_missing(exc)
    else:
        unreachable = False  # s3:// / http(s):// — not our migration case
    if not unreachable:
        return None
    return ArtifactLocationError(_artifact_remediation(model_uri, source))


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
