"""Generic FastAPI serving app — deployed to Lambda via ECR/Mangum.

Projects implement predictions by placing a ``predictor.py`` module alongside
this file (or in a directory on ``sys.path`` / ``$KITCHEN_PREDICTOR_DIR``) that
exposes::

    def predict(payload: dict) -> dict: ...

Optionally, ``predictor.py`` may also export Pydantic ``BaseModel`` subclasses
named ``RequestModel`` and ``ResponseModel``.  When *both* are present the
``/predict`` endpoint is registered with full typed OpenAPI schema.  When
either is absent the endpoint falls back to the ``dict`` → ``dict`` contract.

The predictor is loaded at startup via :func:`kitchen.serve.loader.load_predictor_bundle`,
which validates the contract and surfaces clear errors when the module is
present but malformed.  If no predictor is found the ``/predict`` endpoint
returns 501 Not Implemented.

Typed-route testing
-------------------
The module-level ``_predict_fn`` variable is intentionally exposed so that
existing tests can stub it with ``unittest.mock.patch``.  For tests that need
to exercise the typed route (with Pydantic request/response models), use the
:func:`build_app` factory to create an isolated app — it does not touch
module-level state.
"""

import os
import subprocess
import warnings

from fastapi import FastAPI, HTTPException
from pydantic import create_model

from kitchen.serve.loader import PredictorBundle, PredictorLoadError, load_predictor_bundle

#: Default maximum number of items accepted in a single ``POST /predict/batch``
#: request.  Override via the ``KITCHEN_BATCH_MAX_ITEMS`` environment variable.
#: Lambda's 6 MB payload limit means unbounded batches can silently OOM; we
#: surface a 413 instead.
_DEFAULT_BATCH_MAX_ITEMS = 1000

# ---------------------------------------------------------------------------
# Module-level predictor bootstrap
# ---------------------------------------------------------------------------

_bundle: PredictorBundle | None = None
_predict_fn = None  # kept module-level so patch("kitchen.serve.app._predict_fn", …) works

try:
    _bundle = load_predictor_bundle()
    _predict_fn = _bundle.predict_fn if _bundle is not None else None
except PredictorLoadError as _exc:
    warnings.warn(
        f"kitchen-serve: predictor load error — {_exc}",
        RuntimeWarning,
        stacklevel=1,
    )

# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _resolve_git_sha() -> str | None:
    """Return the current git SHA, trying env vars before spawning a subprocess.

    Resolution order:
      1. ``GITHUB_SHA`` — set by GitHub Actions.
      2. ``GIT_SHA``    — set by CI or deployment tooling.
      3. ``git rev-parse --short HEAD`` subprocess — local dev fallback only;
         Lambda images do not have a ``git`` binary so this will return ``None``
         in production.

    Returns:
        A non-empty string if any source resolves, or ``None``.
    """
    for env_key in ("GITHUB_SHA", "GIT_SHA"):
        val = os.environ.get(env_key)
        if val:
            return val
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            return sha if sha else None
    except Exception:
        pass
    return None


def _collect_metadata(bundle: PredictorBundle | None) -> dict:
    """Build the ``/metadata`` response payload.

    Model identity is resolved (at *call time*, so tests can monkeypatch) in order:
    the predictor's ``MODEL_NAME`` / ``MODEL_VERSION`` exports, then the env vars —
    ``KITCHEN_MODEL_NAME`` or ``MLFLOW_MODEL_NAME`` (the convention used by the
    registry/evaluate/predictor) for the name, and ``KITCHEN_MODEL_VERSION`` for the
    version.  The ``git_sha`` field is resolved via :func:`_resolve_git_sha`.

    Args:
        bundle: The loaded :class:`PredictorBundle`, or ``None`` if no
                predictor is configured.

    Returns:
        A plain ``dict`` suitable for JSON serialisation.
    """
    model_name = (
        (bundle.model_name if bundle is not None else None)
        or os.environ.get("KITCHEN_MODEL_NAME")
        or os.environ.get("MLFLOW_MODEL_NAME")
    )
    model_version = (
        (bundle.model_version if bundle is not None else None)
        or os.environ.get("KITCHEN_MODEL_VERSION")
    )
    return {
        "model_name": model_name,
        "model_version": model_version,
        "git_sha": _resolve_git_sha(),
        "features": bundle.features if bundle is not None else None,
    }


# ---------------------------------------------------------------------------
# Global app — used by uvicorn / Mangum / TestClient in existing tests
# ---------------------------------------------------------------------------

app = FastAPI(title="kitchen-serve", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict:
    """Return model metadata: name, version, git SHA, and feature list."""
    return _collect_metadata(_bundle)


if (
    _bundle is not None
    and _bundle.request_model is not None
    and _bundle.response_model is not None
):
    # Typed route — registered when predictor exports both Pydantic models.
    _RequestModel = _bundle.request_model
    _ResponseModel = _bundle.response_model

    @app.post("/predict", response_model=_ResponseModel)
    def predict(payload: _RequestModel) -> _ResponseModel:  # type: ignore[valid-type]
        """Typed prediction endpoint — schema generated from predictor models."""
        return _predict_fn(payload.model_dump())  # type: ignore[misc]

else:
    # Dict route — plain dict in / dict out; backward-compatible with all
    # existing tests that patch ``_predict_fn`` at module level.
    @app.post("/predict")
    def predict(payload: dict) -> dict:  # type: ignore[no-redef]
        """Prediction endpoint (untyped) — accepts and returns arbitrary JSON."""
        if _predict_fn is None:
            raise HTTPException(status_code=501, detail="No predictor implemented")
        return _predict_fn(payload)


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def _batch_max_items() -> int:
    """Return the configured batch size cap.

    Reads ``KITCHEN_BATCH_MAX_ITEMS`` from the environment at call time so
    tests can monkeypatch it.  Falls back to :data:`_DEFAULT_BATCH_MAX_ITEMS`.
    """
    raw = os.environ.get("KITCHEN_BATCH_MAX_ITEMS")
    if raw is not None:
        try:
            val = int(raw)
            if val > 0:
                return val
        except ValueError:
            pass
    return _DEFAULT_BATCH_MAX_ITEMS


def _register_batch_routes(app: FastAPI, predict_fn, bundle: PredictorBundle | None) -> None:
    """Register ``POST /predict/batch`` on *app*.

    When *bundle* carries both a ``request_model`` and a ``response_model`` the
    endpoint is typed: each item is validated against ``RequestModel`` and each
    result is validated against ``ResponseModel``.  Otherwise the endpoint
    accepts and returns plain dicts.

    The endpoint calls ``predict_fn`` *once per item* so that existing
    single-item predictors work without modification.  Results are returned in
    the same order as the input items.

    If any single item's ``predict_fn`` call raises the whole request fails with
    HTTP 500 (fail-fast, v1).  No partial results are returned.

    A 413 is returned when the number of items exceeds ``KITCHEN_BATCH_MAX_ITEMS``
    (default: 1000).  A 501 is returned when no predictor is configured.
    """
    if (
        bundle is not None
        and bundle.request_model is not None
        and bundle.response_model is not None
    ):
        _ReqM = bundle.request_model
        _RespM = bundle.response_model
        BatchReq = create_model("BatchRequest", items=(list[_ReqM], ...))  # type: ignore[valid-type]
        BatchResp = create_model("BatchResponse", results=(list[_RespM], ...))  # type: ignore[valid-type]

        @app.post("/predict/batch", response_model=BatchResp)
        def _predict_batch_typed(payload: BatchReq) -> BatchResp:  # type: ignore[valid-type]
            """Batch prediction endpoint — validates each item against RequestModel."""
            max_items = _batch_max_items()
            if len(payload.items) > max_items:  # type: ignore[attr-defined]
                raise HTTPException(
                    status_code=413,
                    detail=f"Batch too large: {len(payload.items)} items exceeds limit of {max_items}",  # type: ignore[arg-type]
                )
            results = [predict_fn(item.model_dump()) for item in payload.items]  # type: ignore[attr-defined]
            return {"results": results}

    else:

        @app.post("/predict/batch")
        def _predict_batch_dict(payload: dict) -> dict:
            """Batch prediction endpoint (untyped) — accepts ``{\"items\": [...]}``, returns ``{\"results\": [...]}``.

            Calls ``predict_fn`` once per item.  Results preserve input order.
            """
            if predict_fn is None:
                raise HTTPException(status_code=501, detail="No predictor implemented")
            items = payload.get("items")
            if not isinstance(items, list):
                raise HTTPException(
                    status_code=422,
                    detail="Request body must contain an 'items' key with a list value",
                )
            max_items = _batch_max_items()
            if len(items) > max_items:
                raise HTTPException(
                    status_code=413,
                    detail=f"Batch too large: {len(items)} items exceeds limit of {max_items}",
                )
            return {"results": [predict_fn(item) for item in items]}


# ---------------------------------------------------------------------------
# Register batch route on global app
# ---------------------------------------------------------------------------

_register_batch_routes(app, _predict_fn, _bundle)


# ---------------------------------------------------------------------------
# Factory — for typed-route tests and programmatic app construction
# ---------------------------------------------------------------------------


def build_app(bundle: PredictorBundle | None = None) -> FastAPI:
    """Create a fresh FastAPI app wired to *bundle*.

    This factory is the recommended way to test the typed ``/predict`` route
    without modifying module-level state.  Pass a :class:`PredictorBundle`
    loaded from a temporary predictor directory::

        bundle = load_predictor_bundle(predictor_dir=tmp_path)
        fresh_app = build_app(bundle)
        client = TestClient(fresh_app)

    Args:
        bundle: A :class:`PredictorBundle` returned by
                :func:`~kitchen.serve.loader.load_predictor_bundle`, or
                ``None`` to build a stub app that returns 501 for all requests.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    fresh = FastAPI(title="kitchen-serve", version="0.1.0")
    _pf = bundle.predict_fn if bundle is not None else None

    @fresh.get("/health")
    def _health() -> dict:
        return {"status": "ok"}

    @fresh.get("/metadata")
    def _metadata() -> dict:
        return _collect_metadata(bundle)

    if (
        bundle is not None
        and bundle.request_model is not None
        and bundle.response_model is not None
    ):
        _ReqM = bundle.request_model
        _RespM = bundle.response_model

        @fresh.post("/predict", response_model=_RespM)
        def _predict_typed(payload: _ReqM) -> _RespM:  # type: ignore[valid-type]
            return _pf(payload.model_dump())  # type: ignore[misc]

    else:
        @fresh.post("/predict")
        def _predict_dict(payload: dict) -> dict:
            if _pf is None:
                raise HTTPException(status_code=501, detail="No predictor implemented")
            return _pf(payload)

    _register_batch_routes(fresh, _pf, bundle)

    return fresh


# ---------------------------------------------------------------------------
# Lambda handler — only available when mangum is installed
# ---------------------------------------------------------------------------

try:
    from mangum import Mangum

    handler = Mangum(app)
except ImportError:
    handler = None
