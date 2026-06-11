"""Structured predictor loading for kitchen-serve.

A predictor module must expose::

    def predict(payload: dict) -> dict: ...

Optionally it may also export ``RequestModel`` and ``ResponseModel`` (both
Pydantic ``BaseModel`` subclasses) to enable typed OpenAPI docs on the
``/predict`` endpoint.  Use :func:`load_predictor_bundle` to retrieve the
full bundle; use :func:`load_predictor` when you only need the callable.

The loader resolves ``predictor.py`` by explicit path, environment variable,
or ``sys.path`` scan (in that order) and validates the ``predict`` callable
before returning it.

Compared to the previous bare ``from predictor import predict`` at module level,
this module:

- Provides a clear error when a predictor is found but malformed (missing
  ``predict``, ``predict`` is not callable, or the module raises on import),
  instead of silently producing a 501 response.
- Allows the load path to be injected via ``KITCHEN_PREDICTOR_DIR`` for
  deterministic resolution in ``kitchen serve local`` and in Docker/Lambda.
- Ensures the predictor's parent directory is on ``sys.path`` so sibling
  imports inside ``predictor.py`` (e.g. ``from utils import featurize``)
  resolve correctly.
- Optionally extracts ``RequestModel`` and ``ResponseModel`` (Pydantic
  BaseModel subclasses) to enable typed ``/predict`` OpenAPI schemas.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PredictFn = Callable[[dict], dict]


class LazyModel:
    """A model proxy that defers an expensive load until first use (S-005).

    On Lambda, loading the champion at predictor-module import runs during cold
    start before the handler is ready. Wrapping the load in ``lazy_model`` moves
    it to the first prediction and caches it for the life of the (warm) process::

        from kitchen.serve import lazy_model
        import mlflow

        model = lazy_model(lambda: mlflow.pyfunc.load_model("models:/proj@champion"))

        def predict(payload: dict) -> dict:
            features = [[payload["a"], payload["b"]]]
            return {"label": int(model.predict(features)[0])}  # loads on first call

    Attribute access proxies transparently to the underlying model, so
    ``model.predict(...)`` works as if it were the model itself. The loader runs
    at most once; access ``loaded`` to check status or ``unwrap()`` to force it.
    """

    def __init__(self, loader: Callable[[], Any]) -> None:
        object.__setattr__(self, "_loader", loader)
        object.__setattr__(self, "_model", None)

    def _ensure(self) -> Any:
        if self._model is None:
            object.__setattr__(self, "_model", self._loader())
        return self._model

    def unwrap(self) -> Any:
        """Force the load and return the underlying model."""
        return self._ensure()

    @property
    def loaded(self) -> bool:
        """True once the model has been loaded."""
        return self._model is not None

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not found normally (e.g. ``predict``).
        return getattr(self._ensure(), name)


def lazy_model(loader: Callable[[], Any]) -> LazyModel:
    """Wrap a model-loading callable so the load is deferred until first use.

    See :class:`LazyModel`.
    """
    return LazyModel(loader)

_DEFAULT_FILENAME = "predictor.py"

#: Environment variable that pins the predictor directory.
#: Set automatically by ``kitchen serve local`` alongside ``PYTHONPATH``.
ENV_KEY = "KITCHEN_PREDICTOR_DIR"


class PredictorLoadError(Exception):
    """Raised when a predictor module is found but cannot be loaded or validated.

    Distinct from ``ImportError`` (no predictor present) so callers can
    distinguish "nothing configured yet" from "it broke".
    """


@dataclass
class PredictorBundle:
    """All artefacts extracted from a loaded predictor module.

    Attributes:
        predict_fn:      The validated ``predict`` callable.
        request_model:   Optional Pydantic ``RequestModel`` class; ``None`` if
                         the predictor does not export one.
        response_model:  Optional Pydantic ``ResponseModel`` class; ``None`` if
                         the predictor does not export one.
        features:        Optional list of feature name strings exported as
                         ``FEATURES`` from the predictor module.  Surfaced on
                         ``GET /metadata`` so callers know which input keys the
                         model expects.
        model_name:      Optional model name exported as ``MODEL_NAME`` from the
                         predictor module; surfaced on ``GET /metadata``.
        model_version:   Optional model version/alias exported as ``MODEL_VERSION``
                         from the predictor module; surfaced on ``GET /metadata``.

    When both *request_model* and *response_model* are present the
    ``/predict`` endpoint in ``kitchen.serve.app`` registers a typed FastAPI
    route with full OpenAPI schema.  If either is absent the endpoint falls
    back to the ``dict`` → ``dict`` contract.
    """

    predict_fn: PredictFn
    request_model: type | None = None
    response_model: type | None = None
    features: list[str] | None = None
    model_name: str | None = None
    model_version: str | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_predictor_bundle(
    predictor_dir: Path | str | None = None,
) -> PredictorBundle | None:
    """Load a predictor module and return a :class:`PredictorBundle`.

    Resolution order:
      1. ``predictor_dir / predictor.py``  — when *predictor_dir* is supplied.
      2. ``$KITCHEN_PREDICTOR_DIR / predictor.py`` — environment variable.
      3. Any ``predictor.py`` reachable on ``sys.path`` — honours the
         ``PYTHONPATH`` set by ``kitchen serve local`` and the working-directory
         convention used in the Lambda Dockerfile (``WORKDIR /var/task``).

    Args:
        predictor_dir: Optional explicit directory that contains
                       ``predictor.py``.  Overrides the environment variable.

    Returns:
        A :class:`PredictorBundle` if a predictor was found, or ``None`` if no
        ``predictor.py`` exists anywhere in the resolution chain.

    Raises:
        PredictorLoadError: A ``predictor.py`` was found but could not be used:
                            the module raised on import, ``predict`` is absent,
                            ``predict`` is not callable, or an optional model
                            attribute is present but is not a Pydantic BaseModel
                            subclass.
    """
    path = _resolve(predictor_dir)
    if path is None:
        return None
    return _load_bundle_from_path(path)


def load_predictor(predictor_dir: Path | str | None = None) -> PredictFn | None:
    """Load and validate ``predict`` from a predictor module.

    Thin convenience wrapper around :func:`load_predictor_bundle` that returns
    only the callable.  Prefer :func:`load_predictor_bundle` when you also
    need the Pydantic schema models.

    Resolution order:
      1. ``predictor_dir / predictor.py``  — when *predictor_dir* is supplied.
      2. ``$KITCHEN_PREDICTOR_DIR / predictor.py`` — environment variable.
      3. Any ``predictor.py`` reachable on ``sys.path``.

    Args:
        predictor_dir: Optional explicit directory containing ``predictor.py``.

    Returns:
        The ``predict`` callable, or ``None`` if no ``predictor.py`` is found.

    Raises:
        PredictorLoadError: A ``predictor.py`` was found but could not be used.
    """
    bundle = load_predictor_bundle(predictor_dir)
    return bundle.predict_fn if bundle is not None else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve(predictor_dir: Path | str | None) -> Path | None:
    """Return the resolved absolute path to ``predictor.py``, or ``None``."""
    # 1. Explicit argument — highest priority.
    if predictor_dir is not None:
        p = Path(predictor_dir) / _DEFAULT_FILENAME
        return p.resolve() if p.exists() else None

    # 2. Environment variable.
    env_dir = os.environ.get(ENV_KEY)
    if env_dir:
        p = Path(env_dir) / _DEFAULT_FILENAME
        return p.resolve() if p.exists() else None

    # 3. sys.path scan — covers PYTHONPATH entries and the Lambda /var/task CWD.
    for entry in sys.path:
        base = Path.cwd() if not entry else Path(entry)
        p = base / _DEFAULT_FILENAME
        if p.exists():
            return p.resolve()

    return None


def _load_module(path: Path):
    """Import a predictor module from an absolute file path.

    The predictor's parent directory is prepended to ``sys.path`` (if not
    already present) so that sibling imports inside ``predictor.py`` resolve
    correctly.  The entry is kept for the process lifetime so lazy imports
    within the predictor work across requests.

    Returns:
        The loaded module object.

    Raises:
        PredictorLoadError: The module raised during import or no spec could
                            be created.
    """
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    spec = importlib.util.spec_from_file_location("predictor", path)
    if spec is None or spec.loader is None:
        raise PredictorLoadError(f"Cannot create module spec for {path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise PredictorLoadError(
            f"Error loading predictor from {path}: {exc}"
        ) from exc

    return module


def _validate_predict(module, path: Path) -> PredictFn:
    """Extract and validate the ``predict`` callable from *module*.

    Raises:
        PredictorLoadError: ``predict`` is absent or not callable.
    """
    predict_fn = getattr(module, "predict", None)
    if predict_fn is None:
        raise PredictorLoadError(
            f"predictor module at {path} has no 'predict' function — "
            "add `def predict(payload: dict) -> dict: ...`"
        )
    if not callable(predict_fn):
        raise PredictorLoadError(
            f"predictor.predict at {path} is not callable "
            f"(got {type(predict_fn).__name__})"
        )
    return predict_fn


def _extract_model(module, attr_name: str, path: Path) -> type | None:
    """Extract an optional Pydantic BaseModel subclass from *module*.

    If *attr_name* is not present on the module, returns ``None`` (absence is
    allowed — the endpoint falls back to ``dict`` mode).  If the attribute IS
    present but is not a Pydantic ``BaseModel`` subclass, raises
    :class:`PredictorLoadError`.

    Args:
        module:    The loaded predictor module.
        attr_name: Attribute name to look for (``"RequestModel"`` or
                   ``"ResponseModel"``).
        path:      File path used in error messages.

    Returns:
        The class if found and valid, or ``None``.

    Raises:
        PredictorLoadError: The attribute exists but is not a valid Pydantic
                            BaseModel subclass.
    """
    obj = getattr(module, attr_name, None)
    if obj is None:
        return None

    try:
        from pydantic import BaseModel  # noqa: PLC0415

        if not (isinstance(obj, type) and issubclass(obj, BaseModel)):
            raise PredictorLoadError(
                f"predictor.{attr_name} at {path} must be a pydantic.BaseModel "
                f"subclass — got {type(obj).__name__!r}"
            )
    except ImportError:
        # pydantic not installed; treat as if not present (dict mode).
        return None

    return obj


def _extract_features(module, path: Path) -> list[str] | None:
    """Extract an optional ``FEATURES`` list from *module*.

    If ``FEATURES`` is not present on the module, returns ``None``.  If it IS
    present but is not a ``list`` of ``str`` elements, raises
    :class:`PredictorLoadError`.

    Args:
        module:  The loaded predictor module.
        path:    File path used in error messages.

    Returns:
        The list of feature name strings, or ``None`` if absent.

    Raises:
        PredictorLoadError: ``FEATURES`` exists but is not a list of strings.
    """
    obj = getattr(module, "FEATURES", None)
    if obj is None:
        return None
    if not isinstance(obj, list):
        raise PredictorLoadError(
            f"predictor.FEATURES at {path} must be a list of strings "
            f"(got {type(obj).__name__!r})"
        )
    if not all(isinstance(f, str) for f in obj):
        raise PredictorLoadError(
            f"predictor.FEATURES at {path} must be a list of strings — "
            f"all elements must be str"
        )
    return obj


def _extract_str(module, name: str, path: Path) -> str | None:
    """Extract an optional string constant (e.g. ``MODEL_NAME``) from *module*."""
    obj = getattr(module, name, None)
    if obj is None:
        return None
    if not isinstance(obj, str):
        raise PredictorLoadError(
            f"predictor.{name} at {path} must be a string (got {type(obj).__name__!r})"
        )
    return obj


def _load_bundle_from_path(path: Path) -> PredictorBundle:
    """Load all artefacts from *path* into a :class:`PredictorBundle`."""
    module = _load_module(path)
    predict_fn = _validate_predict(module, path)
    request_model = _extract_model(module, "RequestModel", path)
    response_model = _extract_model(module, "ResponseModel", path)
    features = _extract_features(module, path)
    return PredictorBundle(
        predict_fn=predict_fn,
        request_model=request_model,
        response_model=response_model,
        features=features,
        model_name=_extract_str(module, "MODEL_NAME", path),
        model_version=_extract_str(module, "MODEL_VERSION", path),
    )


def _load_from_path(path: Path) -> PredictFn:
    """Legacy thin wrapper kept for internal back-compat."""
    module = _load_module(path)
    return _validate_predict(module, path)
