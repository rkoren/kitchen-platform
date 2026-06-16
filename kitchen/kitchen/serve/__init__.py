"""kitchen serving layer.

Projects place a ``predictor.py`` alongside their serving code (or in a
directory on ``PYTHONPATH`` / ``KITCHEN_PREDICTOR_DIR``) and the loader
resolves it at startup.

Public API::

    from kitchen.serve import load_predictor, load_predictor_bundle
    from kitchen.serve import PredictorBundle, PredictorLoadError, PredictFn
    from kitchen.serve import lazy_model  # defer model load to first prediction
    from kitchen.serve import load_champion  # load champion + explain artifact drift
"""

from kitchen.serve.loader import (
    LazyModel,
    PredictFn,
    PredictorBundle,
    PredictorLoadError,
    lazy_model,
    load_champion,
    load_predictor,
    load_predictor_bundle,
)

__all__ = [
    "load_predictor",
    "load_predictor_bundle",
    "PredictorBundle",
    "PredictorLoadError",
    "PredictFn",
    "lazy_model",
    "LazyModel",
    "load_champion",
]
