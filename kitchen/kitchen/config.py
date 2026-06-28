"""Typed configuration model for kitchen params.yaml files.

Load and validate with::

    from kitchen.config import KitchenConfig
    cfg = KitchenConfig.from_yaml("params.yaml")

Or validate inline::

    cfg = KitchenConfig(**yaml.safe_load(open("params.yaml")))
"""

from __future__ import annotations

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
