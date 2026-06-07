"""Hyperparameter search helpers (SWEEP-002, SWEEP-003, SWEEP-004).

These wrap :func:`kitchen.modeling.cross_validate` so a sweep needs no custom
training loop: each parameter combination is scored by K-fold CV and logged as a
child MLflow run under the current run/experiment, and the best combination is
returned.

    from kitchen.search import grid_search, random_search
    from kitchen.modeling import classification_metrics
    from xgboost import XGBClassifier

    best = grid_search(
        trainer_fn=lambda p: XGBClassifier(**p),
        param_grid={"max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1]},
        df=train_df,
        target_col="target",
        metric="accuracy",
        return_proba=True,
    )
    # best == {"max_depth": 6, "learning_rate": 0.05}

The caller is responsible for configuring MLflow (``configure_from_env()`` or
running inside ``kitchen.experiment()``); these helpers do not touch the
tracking URI.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Any, Callable, Mapping, Sequence

import mlflow

from kitchen.modeling import classification_metrics, cross_validate
from kitchen.tracking import _flatten


def _resolve_metric_key(metric: str, cv_result: "dict[str, float]") -> str:
    """Map a user-supplied metric name onto a key in the cross_validate result.

    ``cross_validate`` returns ``{base}_mean`` / ``{base}_std`` keys, so a bare
    base name (``"accuracy"``) resolves to ``"accuracy_mean"``. An exact key
    (``"accuracy_mean"``) is used as-is.
    """
    if metric in cv_result:
        return metric
    mean_key = f"{metric}_mean"
    if mean_key in cv_result:
        return mean_key
    raise ValueError(
        f"metric {metric!r} not found in cross-validation result; "
        f"available keys: {sorted(cv_result)}"
    )


def _run_search(
    trainer_fn: "Callable[[dict[str, Any]], Any]",
    combos: "list[dict[str, Any]]",
    df: "Any",
    target_col: str,
    metric: str,
    *,
    metric_fn: "Callable[..., dict[str, float]]",
    n_splits: int,
    higher_is_better: bool,
    seed: int,
    stratify: bool,
    return_proba: bool,
    run_name: str,
    kind: str,
) -> "dict[str, Any]":
    """Score each combination by CV, log it as a nested run, return the best params.

    Shared engine for :func:`grid_search` and :func:`random_search` (and, later,
    SWEEP-004 ``bayes_search``) — they differ only in how *combos* is produced.
    Ties on the score are broken in favour of the first-encountered combination.
    """
    best_params: dict[str, Any] | None = None
    best_score: float | None = None
    resolved_key: str | None = None

    parent = mlflow.active_run()
    with contextlib.ExitStack() as stack:
        if parent is None:
            stack.enter_context(mlflow.start_run(run_name=run_name))

        for i, combo in enumerate(combos):
            with mlflow.start_run(run_name=f"{run_name}-{i}", nested=True):
                mlflow.log_params(_flatten(dict(combo)))
                mlflow.set_tag("sweep.trial", str(i))
                cv = cross_validate(
                    df,
                    target_col=target_col,
                    estimator_fn=lambda c=combo: trainer_fn(dict(c)),
                    metric_fn=metric_fn,
                    n_splits=n_splits,
                    seed=seed,
                    stratify=stratify,
                    return_proba=return_proba,
                )
                if resolved_key is None:
                    resolved_key = _resolve_metric_key(metric, cv)
                mlflow.log_metrics(cv)
                score = cv[resolved_key]

            better = (
                best_score is None
                or (score > best_score if higher_is_better else score < best_score)
            )
            if better:
                best_score, best_params = score, dict(combo)

        # Summarise the sweep on the parent run so it is comparable on the
        # leaderboard to a normal run scored on the same metric.
        assert best_params is not None and resolved_key is not None
        mlflow.set_tags(
            {
                "sweep.kind": kind,
                "sweep.metric": metric,
                "sweep.n_trials": str(len(combos)),
                **{f"sweep.best.{k}": str(v) for k, v in best_params.items()},
            }
        )
        mlflow.log_metric(resolved_key, float(best_score))

    return best_params


def grid_search(
    trainer_fn: "Callable[[dict[str, Any]], Any]",
    param_grid: "Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]]",
    df: "Any",
    target_col: str,
    metric: str,
    *,
    metric_fn: "Callable[..., dict[str, float]] | None" = None,
    n_splits: int = 5,
    higher_is_better: bool = True,
    seed: int = 42,
    stratify: bool = True,
    return_proba: bool = False,
    run_name: str = "grid-search",
) -> "dict[str, Any]":
    """Exhaustive grid search over ``param_grid``, scored by K-fold CV.

    Iterates over every combination in *param_grid* (Cartesian product), scores
    each with :func:`kitchen.modeling.cross_validate`, logs it as a nested MLflow
    run, and returns the best-scoring combination's params.

    The signature extends the SWEEP-002 spec ``(trainer_fn, param_grid, df,
    metric, n_splits)`` with ``target_col`` and ``metric_fn``, both required by
    ``cross_validate()`` under the hood. :func:`random_search` follows the same
    shape.

    Args:
        trainer_fn: Callable mapping a single param combination (a flat dict,
            e.g. ``{"max_depth": 6}``) to a new, unfitted sklearn-compatible
            estimator. Called once per fold per combination, so each call must
            return a fresh estimator. Example: ``lambda p: XGBClassifier(**p)``.
        param_grid: Either a mapping of param name to a list of values (a
            Cartesian product is taken) or a list of such mappings (each a
            sub-grid) — accepts anything ``sklearn.model_selection.ParameterGrid``
            does.
        df: Full dataset including the target column.
        target_col: Name of the target column.
        metric: Metric to optimise. Either a base name returned by *metric_fn*
            (``"accuracy"`` → the CV ``accuracy_mean`` aggregate) or an exact CV
            key (``"accuracy_mean"``).
        metric_fn: Metric function ``(y_true, y_pred, **kwargs) -> dict``.
            Defaults to :func:`kitchen.modeling.classification_metrics`. Use
            :func:`kitchen.modeling.regression_metrics` for regression.
        n_splits: Number of CV folds per combination (default 5).
        higher_is_better: When ``True`` (default) the highest *metric* wins;
            set ``False`` for loss-style metrics (log_loss, brier, rmse). Ties
            are broken in favour of the first-encountered combination.
        seed: Seed forwarded to the CV splitter.
        stratify: Forwarded to ``cross_validate`` (use ``False`` for regression).
        return_proba: Forwarded to ``cross_validate``; set ``True`` when *metric*
            needs probabilities (roc_auc, log_loss, brier).
        run_name: Name of the parent MLflow run (and prefix for child runs). Only
            a parent run is started when no MLflow run is already active; when
            called inside an active run, trials nest under it.

    Returns:
        The best-scoring parameter combination as a plain ``dict``. Every trial's
        full CV metrics are logged to MLflow; the parent run additionally carries
        the best score (under the resolved metric key) and ``sweep.*`` tags.

    Raises:
        ValueError: If *param_grid* yields no combinations, or *metric* is not
            present in the cross-validation result.

    Note:
        Uses stratified K-fold CV. For temporal or grouped data, prefer
        :func:`kitchen.modeling.loto_cv` / :func:`kitchen.modeling.time_series_cv`
        directly — random K-fold leaks information across time/groups.
    """
    from sklearn.model_selection import ParameterGrid

    combos = list(ParameterGrid(param_grid))
    if not combos:
        raise ValueError("param_grid yielded no parameter combinations to search.")

    return _run_search(
        trainer_fn,
        combos,
        df,
        target_col,
        metric,
        metric_fn=metric_fn if metric_fn is not None else classification_metrics,
        n_splits=n_splits,
        higher_is_better=higher_is_better,
        seed=seed,
        stratify=stratify,
        return_proba=return_proba,
        run_name=run_name,
        kind="grid",
    )


def random_search(
    trainer_fn: "Callable[[dict[str, Any]], Any]",
    param_distributions: "Mapping[str, Sequence[Any] | Any]",
    n_iter: int,
    df: "Any",
    target_col: str,
    metric: str,
    *,
    metric_fn: "Callable[..., dict[str, float]] | None" = None,
    n_splits: int = 5,
    higher_is_better: bool = True,
    seed: int = 42,
    stratify: bool = True,
    return_proba: bool = False,
    run_name: str = "random-search",
) -> "dict[str, Any]":
    """Randomised search: sample *n_iter* combinations, scored by K-fold CV.

    Same interface as :func:`grid_search`, but instead of enumerating the full
    grid it draws *n_iter* samples from *param_distributions* — useful when the
    grid is too large to enumerate. Each sampled combination is scored by
    :func:`kitchen.modeling.cross_validate`, logged as a nested MLflow run, and
    the best-scoring combination's params are returned.

    Args:
        trainer_fn: Callable mapping a param combination to a fresh, unfitted
            estimator (see :func:`grid_search`).
        param_distributions: Mapping of param name to either a list of discrete
            values or a scipy-stats distribution exposing ``rvs`` — anything
            ``sklearn.model_selection.ParameterSampler`` accepts. When every
            value is a list, sampling is without replacement and *n_iter* is
            silently capped at the grid size (a warning is emitted).
        n_iter: Number of parameter settings to sample (must be >= 1).
        df: Full dataset including the target column.
        target_col: Name of the target column.
        metric: Metric to optimise (base name or exact CV key — see
            :func:`grid_search`).
        metric_fn: Metric function; defaults to
            :func:`kitchen.modeling.classification_metrics`.
        n_splits: Number of CV folds per combination.
        higher_is_better: Direction of *metric* (ties → first sampled wins).
        seed: Seed for both the sampler (reproducible draws) and the CV splitter.
        stratify: Forwarded to ``cross_validate``.
        return_proba: Forwarded to ``cross_validate``.
        run_name: Parent MLflow run name / child-run prefix.

    Returns:
        The best-scoring parameter combination as a plain ``dict``.

    Raises:
        ValueError: If *n_iter* < 1, no combinations are sampled, or *metric* is
            absent from the cross-validation result.
    """
    from sklearn.model_selection import ParameterSampler

    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}.")

    combos = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=seed))
    if not combos:
        raise ValueError("param_distributions yielded no parameter combinations to sample.")
    if len(combos) < n_iter:
        warnings.warn(
            f"random_search requested n_iter={n_iter} but the discrete grid only has "
            f"{len(combos)} unique combinations; sampling without replacement capped the "
            "search at that size.",
            stacklevel=2,
        )

    return _run_search(
        trainer_fn,
        combos,
        df,
        target_col,
        metric,
        metric_fn=metric_fn if metric_fn is not None else classification_metrics,
        n_splits=n_splits,
        higher_is_better=higher_is_better,
        seed=seed,
        stratify=stratify,
        return_proba=return_proba,
        run_name=run_name,
        kind="random",
    )


def _suggest_param(trial: "Any", name: str, spec: "Any") -> "Any":
    """Map a param_space entry to an optuna ``trial.suggest_*`` call.

    Supported spec forms:
      - ``("int", low, high)``
      - ``("float", low, high)`` or ``("float", low, high, "log")``
      - ``("categorical", [choices])``
      - a plain ``list`` of values -> categorical
    """
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)
    if isinstance(spec, tuple) and spec and isinstance(spec[0], str):
        kind = spec[0]
        if kind == "int":
            return trial.suggest_int(name, spec[1], spec[2])
        if kind == "float":
            log = len(spec) > 3 and spec[3] == "log"
            return trial.suggest_float(name, spec[1], spec[2], log=log)
        if kind == "categorical":
            return trial.suggest_categorical(name, list(spec[1]))
        raise ValueError(
            f"unknown param spec kind {kind!r} for {name!r}; use 'int', 'float', or 'categorical'"
        )
    raise ValueError(
        f"invalid param_space spec for {name!r}: {spec!r}. Use a typed tuple "
        "(e.g. ('int', 3, 10)) or a list of categorical values."
    )


def bayes_search(
    trainer_fn: "Callable[[dict[str, Any]], Any]",
    param_space: "Mapping[str, Any]",
    n_trials: int,
    df: "Any",
    target_col: str,
    metric: str,
    *,
    metric_fn: "Callable[..., dict[str, float]] | None" = None,
    n_splits: int = 5,
    higher_is_better: bool = True,
    seed: int = 42,
    stratify: bool = True,
    return_proba: bool = False,
    run_name: str = "bayes-search",
) -> "dict[str, Any]":
    """Bayesian (TPE) hyperparameter search via optuna.

    Same contract as :func:`grid_search` / :func:`random_search`, but instead of
    enumerating or randomly sampling a grid, optuna proposes each trial's params
    adaptively from past results — more sample-efficient for high-dimensional
    spaces. Each trial is a nested MLflow run; the best combination is returned.

    Requires the optional ``optuna`` dependency: ``pip install 'kitchen[search]'``.

    Args:
        trainer_fn: Callable mapping a param combination to a fresh estimator
            (see :func:`grid_search`).
        param_space: Mapping of param name to a search spec consumed by
            :func:`_suggest_param` — ``("int", low, high)``,
            ``("float", low, high[, "log"])``, ``("categorical", choices)``, or a
            plain list of categorical values.
        n_trials: Number of optuna trials to run (must be >= 1).
        df: Full dataset including the target column.
        target_col: Name of the target column.
        metric: Metric to optimise (base name or exact CV key — see
            :func:`grid_search`).
        metric_fn: Metric function; defaults to ``classification_metrics``.
        n_splits: CV folds per trial.
        higher_is_better: Direction — maximise (default) or minimise *metric*.
        seed: Seed for the TPE sampler (reproducible proposals) and the CV splitter.
        stratify: Forwarded to ``cross_validate``.
        return_proba: Forwarded to ``cross_validate``.
        run_name: Parent MLflow run name / child-run prefix.

    Returns:
        The best-scoring parameter combination as a plain ``dict``.

    Raises:
        ImportError: If optuna is not installed.
        ValueError: If *n_trials* < 1, *param_space* is empty, a spec is invalid,
            or *metric* is absent from the cross-validation result.
    """
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "bayes_search requires optuna — install with `pip install 'kitchen[search]'`"
        ) from exc

    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}.")
    if not param_space:
        raise ValueError("param_space is empty — nothing to search.")

    mf = metric_fn if metric_fn is not None else classification_metrics
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    state: dict[str, str | None] = {"resolved_key": None}

    parent = mlflow.active_run()
    with contextlib.ExitStack() as stack:
        if parent is None:
            stack.enter_context(mlflow.start_run(run_name=run_name))

        def objective(trial: "Any") -> float:
            combo = {name: _suggest_param(trial, name, spec) for name, spec in param_space.items()}
            with mlflow.start_run(run_name=f"{run_name}-{trial.number}", nested=True):
                mlflow.log_params(_flatten(dict(combo)))
                mlflow.set_tag("sweep.trial", str(trial.number))
                cv = cross_validate(
                    df,
                    target_col=target_col,
                    estimator_fn=lambda c=combo: trainer_fn(dict(c)),
                    metric_fn=mf,
                    n_splits=n_splits,
                    seed=seed,
                    stratify=stratify,
                    return_proba=return_proba,
                )
                if state["resolved_key"] is None:
                    state["resolved_key"] = _resolve_metric_key(metric, cv)
                mlflow.log_metrics(cv)
                return cv[state["resolved_key"]]

        study = optuna.create_study(
            direction="maximize" if higher_is_better else "minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=n_trials)

        best_params = dict(study.best_params)
        resolved_key = state["resolved_key"]
        mlflow.set_tags(
            {
                "sweep.kind": "bayes",
                "sweep.metric": metric,
                "sweep.n_trials": str(n_trials),
                **{f"sweep.best.{k}": str(v) for k, v in best_params.items()},
            }
        )
        if resolved_key is not None:
            mlflow.log_metric(resolved_key, float(study.best_value))

    return best_params
