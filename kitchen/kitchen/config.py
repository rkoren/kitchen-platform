"""Typed configuration model for kitchen params.yaml files.

Load and validate with::

    from kitchen.config import KitchenConfig
    cfg = KitchenConfig.from_yaml("params.yaml")

Or validate inline::

    cfg = KitchenConfig(**yaml.safe_load(open("params.yaml")))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DataConfig(BaseModel):
    """``data:`` section — describes where raw data lives."""

    model_config = ConfigDict(extra="allow")

    source: Literal["kaggle", "s3", "local"]
    competition: str | None = None  # required when source=kaggle
    bucket: str | None = None  # required when source=s3
    prefix: str = ""  # s3 key prefix
    path: str | None = None  # required when source=local

    @model_validator(mode="after")
    def _validate_source_fields(self) -> "DataConfig":
        if self.source == "kaggle" and not self.competition:
            raise ValueError("data.competition is required when source is 'kaggle'")
        if self.source == "s3" and not self.bucket:
            raise ValueError("data.bucket is required when source is 's3'")
        if self.source == "local" and not self.path:
            raise ValueError("data.path is required when source is 'local'")
        return self


class MLflowConfig(BaseModel):
    """``mlflow:`` section — experiment tracking backend."""

    model_config = ConfigDict(extra="allow")

    tracking_uri: str = "sqlite:///mlruns.db"
    artifact_bucket: str | None = None
    # Name the project logs its model under — i.e. the ``name`` passed to
    # ``mlflow.<flavor>.log_model(model, name)``. ``kitchen promote`` /
    # ``--auto-promote`` register this logged model. Default ``"model"`` matches
    # the scaffolded templates; set it (e.g. ``cbb_model``) when a project logs
    # under a different name.
    model_artifact_path: str = "model"


class MonitorConfig(BaseModel):
    """``monitor:`` section — drift monitoring configuration."""

    model_config = ConfigDict(extra="allow")

    reference_file: str = "reference.parquet"
    current_file: str = "current.parquet"
    report_bucket: str = ""
    report_key: str = "monitoring/drift_report.html"
    local_path: str = ""
    # MON-007: per-column p-value threshold; fail the run when the drifted share
    # reaches max_drift_share (only enforced when fail_on_drift is true).
    drift_threshold: float = 0.05
    fail_on_drift: bool = False
    max_drift_share: float = 0.5
    # MON-006: log drift summary metrics + the HTML/JSON report to MLflow.
    log_to_mlflow: bool = False
    mlflow_experiment: str | None = None  # default: "<experiment>-monitoring"

    @model_validator(mode="after")
    def _require_output(self) -> "MonitorConfig":
        if not self.report_bucket and not self.local_path:
            raise ValueError(
                "monitor config must specify at least one of: "
                "report_bucket (S3 upload) or local_path (local file)."
            )
        return self


class NotificationsConfig(BaseModel):
    """``ci.notifications:`` — where to send CI run notifications.

    Declarative only — the scaffolded workflow reads these to decide whether to
    fire a notify step. ``slack_webhook_secret`` names the GitHub *secret* that
    holds the incoming-webhook URL (the URL itself never lives in params.yaml).
    """

    model_config = ConfigDict(extra="allow")

    slack_webhook_secret: str | None = None
    # NB: not ``on`` — YAML 1.1 parses a bare ``on:`` key as boolean True.
    when: Literal["failure", "success", "always"] = "failure"


class CIConfig(BaseModel):
    """``ci:`` section — one home for CI behavior knobs.

    Read by the scaffolded GitHub Actions workflow and by ``kitchen report``.
    Metric thresholds themselves stay in the top-level ``thresholds:`` map;
    ``fail_on_threshold`` controls whether a breach fails the CI job.
    """

    model_config = ConfigDict(extra="allow")

    auto_submit: bool = False  # submit to Kaggle after evaluate on a main-branch push
    fail_on_threshold: bool = True  # whether a threshold breach exits `kitchen report` non-zero
    notifications: NotificationsConfig | None = None


class CheckConfig(BaseModel):
    """``check:`` section — extra pre-flight validations for ``kitchen check``.

    ``required_env`` lists environment variables the project needs to run (e.g. an
    API key`). ``kitchen check`` hard-fails when one is
    absent from both the process environment and a local ``.env`` — so a project
    fed by an external API gates on its own secrets without editing platform code.
    """

    model_config = ConfigDict(extra="allow")

    required_env: list[str] = Field(default_factory=list)


class SubmissionConfig(BaseModel):
    """``submission:`` section — Kaggle submission configuration."""

    model_config = ConfigDict(extra="allow")

    id_col: str = "Id"
    target_col: str = "target"
    competition: str | None = None
    message: str = "kitchen submit"
    sample_submission: str = "sample_submission.csv"


class ThresholdSpec(BaseModel):
    """Explicit min/max threshold for a single metric.

    Use ``min`` for higher-is-better metrics (accuracy, AUC).
    Use ``max`` for lower-is-better metrics (logloss, RMSE).
    Both can be set to define a valid range.
    """

    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _at_least_one(self) -> "ThresholdSpec":
        if self.min is None and self.max is None:
            raise ValueError("ThresholdSpec must specify at least one of: min, max")
        return self


#: Holdout metrics that score model *probabilities* (positive class). ``accuracy`` also needs
#: probabilities (thresholded at 0.5) → it's classification but not in this set; the regression
#: metrics (rmse/mae/r2) score point predictions and are the default ``predict`` path.
HOLDOUT_PROBA_METRICS: frozenset[str] = frozenset({"brier", "log_loss", "roc_auc"})
HOLDOUT_CLASSIFICATION_METRICS: frozenset[str] = HOLDOUT_PROBA_METRICS | {"accuracy"}


class HoldoutSpec(BaseModel):
    """``holdout:`` — a frozen, never-trained-on evaluation set scored as a distinct metric (CBB-017).

    The project produces a parity-matched parquet (the model's features + the realized
    ``label``), leak-free by construction because it lives outside ``data/raw`` where training
    can't read it. The platform scores **every run's** model against it and logs
    ``holdout_<metric>`` — a trusted generalization number distinct from the in-CV metric, so a
    project iterating on its CV score can tell CV-overfit from real generalization. Treat it
    like a Kaggle private leaderboard: iterate on CV, check the holdout sparingly.

    An absent ``path`` is a no-op (the pipeline runs unchanged until results exist) unless
    ``optional`` is ``False``. ``features`` defaults to the project's ``feature_candidates``
    (the training feature list); they are parity-checked against the parquet before scoring and
    a break **skips** scoring loudly rather than zero-filling — a trusted metric must never be
    silently wrong. ``predict_method`` overrides the default model call (``predict_proba`` for a
    classification metric, ``predict`` for a regression one) for models with a custom interface.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    label: str
    metric: Literal["brier", "log_loss", "roc_auc", "accuracy", "rmse", "mae", "r2"] = "brier"
    features: list[str] | None = None
    predict_method: str | None = None
    optional: bool = True


class ModelSpec(BaseModel):
    """One entry in the ``models:`` map — a distinct model in a multi-model project (CBB-020).

    A project that grows a second model (e.g. cbb's tournament Brier model **and** a
    regular-season margin/total model) declares each here so the platform can scope
    auto-promote, the champion, and the read commands per-model instead of guessing from a
    single shared experiment + the first ``thresholds`` key. ``--model <name>`` on the
    ``run train``/``leaderboard``/``status`` commands selects an entry.

    All fields default sensibly so a minimal entry works: ``experiment`` falls back to the
    project's top-level ``experiment``; ``model_name`` (the registered champion) to
    ``f"{experiment}-model"``; ``metric``/``lower_is_better`` (the promote/rank metric) to the
    first ``thresholds`` key. Single-model projects don't need a ``models:`` map at all.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    experiment: str | None = None  # MLflow experiment for this model's runs (default: project experiment)
    model_name: str | None = None  # registered champion model name (default: f"{experiment}-model")
    metric: str | None = None  # promote/rank metric (default: first thresholds key)
    lower_is_better: bool = False  # metric direction for promotion/ranking


class ModelNotFound(KeyError):
    """A ``--model`` name has no entry in the menu's ``models:`` map (or one is required)."""


@dataclass(frozen=True)
class ResolvedModel:
    """The concrete identity a ``--model`` selection resolves to (CBB-020)."""

    name: str | None  # the models: key, or None for a single-model project
    experiment: str
    model_name: str
    metric: str | None
    lower_is_better: bool


def resolve_model(cfg: "KitchenConfig", model: str | None = None) -> ResolvedModel:
    """Resolve ``--model <name>`` (or its absence) to a concrete model identity (CBB-020).

    - ``model`` given → that ``models:`` entry (raises :class:`ModelNotFound` if absent).
    - ``model`` omitted + exactly one entry → that entry (convenience).
    - ``model`` omitted + multiple entries → :class:`ModelNotFound` (ambiguous; pass ``--model``).
    - ``model`` omitted + no ``models:`` map → the legacy single-model identity
      (``cfg.experiment`` + ``MLFLOW_MODEL_NAME``/``f"{experiment}-model"``), so existing
      projects are unchanged.
    """
    import os

    models = cfg.models
    spec: ModelSpec | None
    name: str | None
    if model is not None:
        if model not in models:
            have = ", ".join(sorted(models)) or "(no models: map defined)"
            raise ModelNotFound(f"no model {model!r} in menu — available: {have}")
        spec, name = models[model], model
    elif len(models) == 1:
        name, spec = next(iter(models.items()))
    elif len(models) > 1:
        raise ModelNotFound(
            f"project defines multiple models ({', '.join(sorted(models))}) — pass --model <name>"
        )
    else:
        spec, name = None, None

    experiment = (spec.experiment if spec else None) or cfg.experiment
    model_name = (
        (spec.model_name if spec else None)
        or os.environ.get("MLFLOW_MODEL_NAME")
        or f"{experiment}-model"
    )
    metric = spec.metric if spec else None
    lower = spec.lower_is_better if spec else False
    return ResolvedModel(name=name, experiment=experiment, model_name=model_name, metric=metric, lower_is_better=lower)


class FeatureSchemaSpec(BaseModel):
    """``feature_schema:`` — the expected schema of a processed feature file (KG-014).

    Declares the column → pandas-dtype contract for a ``data/processed/`` parquet (e.g. the
    feature matrix the training stage reads). ``kitchen check`` validates the file against it,
    catching feature-matrix schema drift before training — the feature-side analogue of the
    submission/parity checks (KG-006..013). An absent file is a soft warning (not built yet);
    a missing column or dtype mismatch is a hard check failure. Validation reuses the same
    ``DataStore`` schema machinery (DS-002) a stage gets from ``load_parquet(schema=...)``.
    """

    model_config = ConfigDict(extra="forbid")

    file: str  # processed parquet to validate, relative to data/processed/
    columns: dict[str, str]  # column name → expected pandas dtype (e.g. "float64", "int64")


class SecretSpec(BaseModel):
    """One entry in the ``secrets:`` manifest — declares where a secret resolves from.

    Pick one source (or none, for env-only):
      - **SM JSON bundle:** ``aws_secret`` (bundle name/ARN) + ``key`` (field within the JSON)
      - **SSM Parameter Store:** ``ssm`` (parameter path)
      - **omit both:** the secret must come from the environment / ``.env``

    ``required`` (default True) gates ``kitchen check``. This manifest is the single
    declarative source of truth the resolver (SECR-002) reads; it supersedes the deprecated
    ``check.required_env`` list.
    """

    model_config = ConfigDict(extra="allow")

    aws_secret: str | None = None
    key: str | None = None
    ssm: str | None = None
    required: bool = True

    @model_validator(mode="after")
    def _coherent_source(self) -> "SecretSpec":
        if self.aws_secret and self.ssm:
            raise ValueError("a secret declares `aws_secret` or `ssm`, not both")
        if self.key and not self.aws_secret:
            raise ValueError("`key` selects a field within an SM bundle — it requires `aws_secret`")
        return self

    @property
    def source(self) -> str:
        """Human-readable source label for check/validate output (``env`` when undeclared)."""
        if self.aws_secret:
            return f"SM {self.aws_secret}" + (f"#{self.key}" if self.key else "")
        if self.ssm:
            return f"SSM {self.ssm}"
        return "env"


def resolve_params_path(params_file: str = "params.yaml") -> str:
    """Resolve the config path, honoring the menu fallback (INT-007). When the default
    ``params.yaml`` is requested but absent, a sibling ``menu.yaml`` stands in — so every
    command works in a menu-only project with no ``--params`` flag. Resolve-only: returns the
    path string (caller decides how to handle a still-missing file); an explicit non-default
    path is returned unchanged.
    """
    from pathlib import Path

    p = Path(params_file)
    if not p.exists() and p.name == "params.yaml":
        sibling = p.with_name("menu.yaml")
        if sibling.exists():
            return str(sibling)
    return str(p)


_LEGACY_PARAMS_WARNED = False


def _warn_legacy_params(path: "object") -> None:
    """Warn once per process that a legacy ``params.yaml`` was read (INT-007b).

    ``kitchen init`` now scaffolds a unified ``menu.yaml``; ``params.yaml`` still loads
    transparently during the deprecation window. Fired only on the legacy branch, so a
    ``menu.yaml`` project never sees it."""
    global _LEGACY_PARAMS_WARNED
    if _LEGACY_PARAMS_WARNED:
        return
    _LEGACY_PARAMS_WARNED = True
    import warnings

    warnings.warn(
        f"{path}: loading a legacy params.yaml. The platform now uses a unified menu.yaml "
        "(run `kitchen init` to see the new layout); params.yaml still works for now.",
        UserWarning,
        stacklevel=3,
    )


class KitchenConfig(BaseModel):
    """Top-level model for params.yaml.

    Framework-owned sections (``data``, ``mlflow``, ``monitor``) are typed and
    validated.  Project-defined sections (``model``, ``features``, ``train``,
    etc.) are passed through without validation.
    """

    model_config = ConfigDict(extra="allow")

    experiment: str
    data: DataConfig | None = None
    mlflow: MLflowConfig = MLflowConfig()
    monitor: MonitorConfig | None = None
    submission: SubmissionConfig | None = None
    check: CheckConfig | None = None
    secrets: dict[str, SecretSpec] = Field(default_factory=dict)
    ci: CIConfig | None = None
    run_name: str | None = None
    metrics_file: str = "metrics.json"
    thresholds: dict[str, float | ThresholdSpec] = Field(default_factory=dict)
    holdout: HoldoutSpec | None = None
    feature_schema: FeatureSchemaSpec | None = None
    models: dict[str, ModelSpec] = Field(default_factory=dict)

    @property
    def uses_legacy_required_env(self) -> bool:
        """True when the deprecated ``check.required_env`` list is populated."""
        return bool(self.check and self.check.required_env)

    def effective_secrets(self) -> dict[str, SecretSpec]:
        """The unified secrets manifest.

        Returns the ``secrets:`` map with any legacy ``check.required_env`` names folded in
        as env-only required secrets (deprecated). ``secrets:`` wins on a name conflict, so a
        project can migrate one secret at a time without losing the legacy guard.
        """
        merged = dict(self.secrets)
        legacy = (self.check.required_env if self.check else None) or []
        for name in legacy:
            merged.setdefault(name, SecretSpec(required=True))
        return merged

    @classmethod
    def from_yaml(cls, path: str = "params.yaml") -> "KitchenConfig":
        """Load and validate a params.yaml *or* a menu.yaml file (INT-007).

        A unified ``menu.yaml`` is detected by content and bridged via
        ``Menu.to_kitchen_config()`` so every command transparently accepts either. When the
        default ``params.yaml`` is absent, a sibling ``menu.yaml`` is used instead — so a
        migrated project needs no ``--params`` flag.
        """
        from pathlib import Path

        import yaml

        # Path discovery: fall back to a sibling menu.yaml only when the default
        # params.yaml was requested and is missing (never auto-prefer when both exist).
        target = Path(path)
        if not target.exists() and target.name == "params.yaml":
            sibling = target.with_name("menu.yaml")
            if sibling.exists():
                target = sibling

        with open(target, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        from kitchen.menu import Menu, is_menu

        if is_menu(raw):
            return Menu.model_validate(raw).to_kitchen_config()
        _warn_legacy_params(target)
        return cls(**raw)
