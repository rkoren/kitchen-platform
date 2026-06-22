"""The unified project manifest — ``menu.yaml`` (INT-002).

One file is the platform's source of truth: a `pipeline` (the ordered deploy/run/test
sequence) and `recipes` (the definition of each unit — its `kind`, `role`, `source`, and
kind-specific fields), plus the ML settings. See
``docs/decisions/recipes-kitchen-integration.md``.

This module owns the *schema and cross-references only* — not resolution (INT-003), the
pipeline runner (INT-005), or any deploy wiring. It deliberately does **not** import
``recipes`` (kitchen installs without it); the kind-specific fields of an infra recipe are
validated per-kind by recipes at deploy time (INT-004), where they become an actual
``RDSSpec``/``S3Spec``/… via ``{type=kind, name=<key>, **fields}``.
"""

from __future__ import annotations

from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kitchen.config import CIConfig, DataConfig, MonitorConfig, SecretSpec, ThresholdSpec

# Infra kinds must stay in lockstep with recipes' ``ResourceSpec`` discriminator (a test
# asserts equality). ``stage`` is the runtime-only kind recipes has no equivalent for.
INFRA_KINDS: tuple[str, ...] = ("rds", "s3", "security_group", "iam_role", "ecr", "lambda")
RUNTIME_KINDS: tuple[str, ...] = ("stage",)

# Pipeline steps that are platform actions rather than references into ``recipes``:
#   provision — apply the infra recipes (recipes/Terraform resolves their internal order)
#   monitor   — run the drift-monitoring step
PLATFORM_VERBS: frozenset[str] = frozenset({"provision", "monitor"})


class NetworkSpec(BaseModel):
    """Networking declared once and inherited by infra recipes (R-017 at the menu scope)."""

    model_config = ConfigDict(extra="forbid")

    vpc_id: str | None = None  # omit → the account's default VPC
    subnets: list[str] = Field(default_factory=list)  # ≥2 AZs → an RDS DB subnet group


class RoleRef(BaseModel):
    """A reference to a recipe's ``role`` — resolved to a concrete value at deploy time
    (INT-003), e.g. ``tracking_uri: {from_role: mlflow-backend}``."""

    model_config = ConfigDict(extra="forbid")

    from_role: str


class RecipeEntry(BaseModel):
    """One entry in the ``recipes:`` map — keyed by name, defining what to deploy/run.

    ``kind`` (not recipes' ``type``) is the discriminator, per the unified-platform
    decision; the same value is handed to recipes as ``type`` when an infra recipe is
    deployed (INT-004). Kind-specific fields (e.g. an rds ``instance_class``, a stage's
    extras) sit flat alongside ``kind``/``role``/``source`` and are collected into
    :pyattr:`fields`; they are validated **per kind** downstream (recipes for infra, the
    runner for ``stage``), not here.
    """

    model_config = ConfigDict(extra="allow")  # kind-specific fields land in model_extra

    kind: Literal["rds", "s3", "security_group", "iam_role", "ecr", "lambda", "stage"]
    role: str | None = None  # discovery marker; what `{from_role}` references resolve to
    source: str | None = None  # where this recipe's code lives (stage/serve)

    @property
    def fields(self) -> dict[str, Any]:
        """The kind-specific fields (everything beyond kind/role/source)."""
        return dict(self.model_extra or {})


class MlflowSettings(BaseModel):
    """Menu-local MLflow settings — like ``KitchenConfig.mlflow`` but the infra-coupled
    values may be a :class:`RoleRef` resolved from a recipe (INT-003). Kept separate so the
    back-compat ``KitchenConfig`` stays untouched until the INT-007 bridge."""

    model_config = ConfigDict(extra="allow")

    tracking_uri: str | RoleRef = "sqlite:///mlruns.db"
    artifact_bucket: str | RoleRef | None = None
    model_artifact_path: str = "model"


class Menu(BaseModel):
    """The unified project manifest (``menu.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    project: str
    region: str = "us-east-1"
    aws_account: str | None = None
    network: NetworkSpec | None = None

    pipeline: list[str] = Field(default_factory=list)
    recipes: dict[str, RecipeEntry] = Field(default_factory=dict)

    # ML settings (kitchen sub-models reused as-is; mlflow gets the RoleRef-aware variant).
    experiment: str | None = None  # defaults to `project`
    mlflow: MlflowSettings = Field(default_factory=MlflowSettings)
    data: DataConfig | None = None
    monitor: MonitorConfig | None = None
    ci: CIConfig | None = None
    secrets: dict[str, SecretSpec] = Field(default_factory=dict)
    thresholds: dict[str, float | ThresholdSpec] = Field(default_factory=dict)
    run_name: str | None = None

    @model_validator(mode="after")
    def _default_experiment(self) -> "Menu":
        if self.experiment is None:
            object.__setattr__(self, "experiment", self.project)
        return self

    @model_validator(mode="after")
    def _validate_references(self) -> "Menu":
        errors: list[str] = []

        # Every pipeline step is a platform verb or a key in `recipes`.
        for step in self.pipeline:
            if step not in PLATFORM_VERBS and step not in self.recipes:
                errors.append(
                    f"pipeline step '{step}' is neither a platform verb "
                    f"({', '.join(sorted(PLATFORM_VERBS))}) nor a recipe in `recipes:`."
                )

        # A stage recipe is pure code — the manifest must say where it lives (INT-006).
        for name, entry in self.recipes.items():
            if entry.kind == "stage" and not entry.source:
                errors.append(f"stage recipe '{name}' must declare a `source` (where its code lives).")

        # Every `{from_role}` reference resolves to some recipe's `role`.
        roles = {r.role for r in self.recipes.values() if r.role is not None}
        for label, value in (
            ("mlflow.tracking_uri", self.mlflow.tracking_uri),
            ("mlflow.artifact_bucket", self.mlflow.artifact_bucket),
        ):
            if isinstance(value, RoleRef) and value.from_role not in roles:
                have = ", ".join(sorted(roles)) or "(none)"
                errors.append(
                    f"{label}: from_role '{value.from_role}' matches no recipe role (have: {have})."
                )

        if errors:
            raise ValueError("\n".join(errors))
        return self

    @property
    def source_map(self) -> dict[str, str]:
        """Each recipe that declares where its code lives → its ``source`` path. The manifest
        is the single place mapping a stage/deploy to its code (INT-006)."""
        return {name: r.source for name, r in self.recipes.items() if r.source}

    @classmethod
    def from_yaml(cls, path: str = "menu.yaml") -> "Menu":
        """Load and validate a ``menu.yaml`` manifest."""
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)
