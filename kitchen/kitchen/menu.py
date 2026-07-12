"""The unified project manifest — ``menu.yaml`` (INT-002).

One file is the platform's source of truth: a `pipeline` (the ordered deploy/run/test
sequence) and `recipes` (the definition of each unit — its `kind`, `role`, `source`, and
kind-specific fields), plus the ML settings. See
``docs/decisions/recipes-kitchen-integration.md``.

This module owns the *schema, cross-references, and the projection to recipes* — not
resolution (INT-003), the pipeline runner (INT-005), or any deploy wiring. Since the package
merge (INT-013) it imports ``kitchen.recipes.schema`` (lazily) to validate each infra
recipe's kind-specific fields against the real ``RDSSpec``/``S3Spec``/… at load (S-2, via
``{type=kind, name=<key>, **fields}``) and to project the menu into a ``RecipeSpec`` (S-1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from kitchen.config import (
    CheckConfig,
    CIConfig,
    DataConfig,
    FeatureSchemaSpec,
    HoldoutSpec,
    ModelSpec,
    MonitorConfig,
    ScorerConfig,
    SecretSpec,
    SubmissionConfig,
    ThresholdSpec,
)
from kitchen.recipes.schema import ResourceSpec

if TYPE_CHECKING:
    from kitchen.config import KitchenConfig
    from kitchen.recipes.schema import RecipeSpec

# Infra kinds are derived directly from recipes' ``ResourceSpec`` discriminator — the single
# source of truth (S-3, INT-016; possible now recipes is the ``kitchen.recipes`` sub-package).
# ``stage`` is the runtime-only kind recipes has no equivalent for.
INFRA_KINDS: tuple[str, ...] = tuple(
    get_args(spec.model_fields["type"].annotation)[0]
    for spec in get_args(get_args(ResourceSpec)[0])  # Annotated[Union[...], Field] → the specs
)

# Pipeline steps that are platform actions rather than references into ``recipes``:
#   provision — apply the infra recipes (recipes/Terraform resolves their internal order)
#   monitor   — run the drift-monitoring step
PLATFORM_VERBS: frozenset[str] = frozenset({"provision", "monitor", "score"})


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
    # GEN-002/003: a command stage runs a subprocess instead of an in-process `source` callable.
    # `cmd` is the argv (a list, used verbatim; or a string, shlex-split — no shell). `python`,
    # when set, is the interpreter and `cmd` is the *args* passed to it (e.g. python: a venv, cmd:
    # "-m pipeline.run") — a per-stage environment. `inputs`/`outputs` are declared paths: inputs
    # are checked to exist before running (fail fast); missing outputs warn after.
    cmd: str | list[str] | None = None
    python: str | None = None
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)

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


class FeatureCandidatesOverlay(BaseModel):
    """A variant's change to the base ``feature_candidates`` list (CBB-016).

    ``add``/``remove`` are a delta on the base list; ``replace`` swaps it wholesale. A variant
    may instead give a plain list (the obvious full-replacement case). Structured rather than
    ``+``/``-`` prefixes so a feature name can never be mistaken for a marker."""

    model_config = ConfigDict(extra="forbid")

    add: list[str] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)
    replace: list[str] | None = None


class VariantSpec(BaseModel):
    """One named experiment variant — an overlay of arbitrary config keys (CBB-016).

    Non-list keys (``model.max_depth``, …) deep-merge over the base; ``feature_candidates``
    takes the typed overlay above (or a plain replacement list)."""

    model_config = ConfigDict(extra="allow")  # arbitrary overlay keys (model, run_name, …)

    feature_candidates: list[str] | FeatureCandidatesOverlay | None = None


class Menu(BaseModel):
    """The unified project manifest (``menu.yaml``).

    ``extra="allow"`` intentionally — the menu is a *superset* of ``params.yaml``, so a
    project's free-form stage sections (``model:``, ``features:``, ``train:``, …) live at the
    top level exactly as before and pass through to ``model_extra`` (INT-007). The stage code
    reads them from the raw YAML (``params["model"]["target"]``), so they must stay here, not
    nested. This matches ``KitchenConfig``'s posture and trades top-level-key typo safety for
    that passthrough.
    """

    model_config = ConfigDict(extra="allow")

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
    submission: SubmissionConfig | None = None  # Kaggle submit config (CBB and other comps)
    check: CheckConfig | None = None  # deprecated `required_env` legacy guard (superseded by `secrets`)
    secrets: dict[str, SecretSpec] = Field(default_factory=dict)
    thresholds: dict[str, float | ThresholdSpec] = Field(default_factory=dict)
    holdout: HoldoutSpec | None = None  # CBB-017: frozen-holdout generalization metric
    scorer: ScorerConfig | None = None  # GEN-006: project scoring callable as the metric source
    feature_schema: FeatureSchemaSpec | None = None  # KG-014: processed-feature schema contract
    models: dict[str, ModelSpec] = Field(default_factory=dict)  # CBB-020: multi-model map
    metrics_file: str = "metrics.json"
    run_name: str | None = None
    variants: dict[str, VariantSpec] = Field(default_factory=dict)  # CBB-016: --variant overlays

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

        # A stage recipe is pure code — the manifest must say where it lives: an in-process
        # `source` callable (INT-006) or a `cmd` subprocess (GEN-002), exactly one.
        for name, entry in self.recipes.items():
            if entry.kind == "stage" and not entry.source and entry.cmd is None:
                errors.append(
                    f"stage recipe '{name}' must declare a `source` (in-process code) or a `cmd` "
                    "(subprocess command)."
                )
            if entry.kind == "stage" and entry.source and entry.cmd is not None:
                errors.append(
                    f"stage recipe '{name}' declares both `source` and `cmd` — pick one "
                    "(in-process callable or subprocess command)."
                )

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

    @model_validator(mode="after")
    def _validate_infra_fields(self) -> "Menu":
        """Validate each infra recipe's kind-specific fields against the real recipes spec at
        ``Menu`` load (S-2, INT-015). A typo like ``instance_clas`` on an ``rds`` recipe now
        errors here — in *every* command that loads the manifest (``kitchen run``, a notebook,
        ``recipes generate``) — instead of only surfacing at provision time.

        Possible now that recipes is the ``kitchen.recipes`` sub-package (INT-013). We build the
        *whole* projected ``RecipeSpec`` (via :meth:`to_recipe_spec`) rather than validating each
        entry in isolation, because the recipes specs carry **cross-resource** rules — a
        lambda's ``ecr_repo``/``iam_role`` and an rds's ``security_groups`` must resolve to
        sibling resources in the same spec. ``stage`` entries have no spec and stay loose — the
        runner owns them.
        """
        infra_names = [name for name, e in self.recipes.items() if e.kind in INFRA_KINDS]
        if not infra_names:
            return self
        try:
            self.to_recipe_spec()  # builds + validates the full RecipeSpec (all infra resources)
        except ValidationError as exc:
            raise ValueError(_infra_error_msg(exc, infra_names)) from exc
        return self

    @property
    def source_map(self) -> dict[str, str]:
        """Each recipe that declares where its code lives → its ``source`` path. The manifest
        is the single place mapping a stage/deploy to its code (INT-006)."""
        return {name: r.source for name, r in self.recipes.items() if r.source}

    def to_recipe_spec(self) -> "RecipeSpec":
        """Project the menu's infra recipes into a recipes :class:`RecipeSpec` (S-1, INT-014).

        The single menu reader: recipes no longer re-parses ``menu.yaml`` (the old
        ``recipes/menu.py`` re-parser is gone) — it consumes the already-validated ``Menu``
        and calls this. Only infra kinds are provisioned; runtime ``stage`` recipes are
        kitchen's and skipped. Menu-only keys (``role`` discovery marker, ``source`` code
        location) are dropped — they live on :class:`RecipeEntry`, not in ``.fields``. A serve
        ``lambda``'s ``iam_role``/``source`` are wired for provisioning (INT-006, S-4), and
        menu-level networking is inherited unless the recipe overrides it (R-017).

        Per-kind field validation already ran at ``Menu`` load (``_validate_infra_fields``,
        INT-015/S-2); ``RecipeSpec`` re-validates here as cheap defence-in-depth (a ``Menu``
        can be hand-built in a test without going through ``model_validate``).
        """
        from kitchen.recipes.schema import RecipeSpec

        resources = [
            _project_entry(name, entry, self.network)
            for name, entry in self.recipes.items()
            if entry.kind in INFRA_KINDS  # runtime kinds (e.g. `stage`) are kitchen's, not infra
        ]
        return RecipeSpec.model_validate(
            {"name": self.project, "region": self.region, "resources": resources}
        )

    def to_kitchen_config(self) -> "KitchenConfig":
        """Project the ML half of the menu onto a :class:`KitchenConfig` (INT-007).

        This is the back-compat bridge: every ``kitchen`` command loads a
        ``KitchenConfig``, so a project can switch its ``params.yaml`` for a
        ``menu.yaml`` without touching the commands. The infra half (``recipes``,
        ``network``, ``pipeline``) is read separately by recipes (INT-004) and the
        runner (INT-005) and has no place in a ``KitchenConfig``.

        ``mlflow`` ``{from_role}`` references are intentionally **dropped** here: INT-003
        (``kitchen menu materialize``) resolves them to ``MLFLOW_TRACKING_URI`` /
        ``MLFLOW_ARTIFACT_BUCKET`` in the environment, and the env value takes precedence
        over the config (see ``experiment.py``). A ``RoleRef`` therefore falls back to the
        ``MLflowConfig`` default (``sqlite:///mlruns.db``) here, which the materialized env
        overrides at run time.
        """
        from kitchen.config import KitchenConfig, MLflowConfig

        # Literal mlflow values only; spread the allowed extras first, then the typed three.
        mlflow_kwargs: dict[str, Any] = dict(self.mlflow.model_extra or {})
        mlflow_kwargs["model_artifact_path"] = self.mlflow.model_artifact_path
        if isinstance(self.mlflow.tracking_uri, str):
            mlflow_kwargs["tracking_uri"] = self.mlflow.tracking_uri
        if isinstance(self.mlflow.artifact_bucket, str):
            mlflow_kwargs["artifact_bucket"] = self.mlflow.artifact_bucket

        # Project-defined sections (model/features/train/…) ride in model_extra; spread them
        # first so the typed platform fields below always win on any name overlap.
        return KitchenConfig(
            **(self.model_extra or {}),
            experiment=self.experiment,
            data=self.data,
            mlflow=MLflowConfig(**mlflow_kwargs),
            monitor=self.monitor,
            submission=self.submission,
            check=self.check,
            ci=self.ci,
            secrets=self.secrets,
            thresholds=self.thresholds,
            holdout=self.holdout,
            scorer=self.scorer,
            feature_schema=self.feature_schema,
            models=self.models,
            metrics_file=self.metrics_file,
            run_name=self.run_name,
        )

    @classmethod
    def from_yaml(cls, path: str = "menu.yaml") -> "Menu":
        """Load and validate a ``menu.yaml`` manifest."""
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)


def is_menu(raw: Any) -> bool:
    """True when a parsed YAML mapping is a unified ``menu.yaml`` rather than a legacy
    ``params.yaml``. Neither ``recipes`` nor ``pipeline`` is a meaningful top-level key in a
    params file, so their presence identifies a menu (INT-007).

    Also the router recipes uses (S-1): a standalone recipes spec carries ``resources:`` and
    neither ``recipes:`` nor ``pipeline:``, so it correctly reads as *not* a menu.
    """
    return isinstance(raw, dict) and ("recipes" in raw or "pipeline" in raw)


def _wire_lambda_resource(resource: dict[str, Any], source: str | None) -> None:
    """Resolve a serve lambda's two menu-isms for provisioning (INT-006, S-4):

    * the menu uses ``role`` for the *discovery* marker and ``iam_role`` for the lambda's
      execution role — map ``iam_role`` onto recipes' ``LambdaSpec.role``;
    * the menu ``source`` (where the predictor lives) is injected as the deployed function's
      ``KITCHEN_PREDICTOR_DIR`` so serving finds it at runtime.
    """
    if "iam_role" in resource:
        resource["role"] = resource.pop("iam_role")
    if source:
        env = dict(resource.get("environment") or {})
        env.setdefault("KITCHEN_PREDICTOR_DIR", source)
        resource["environment"] = env


def _inherit_network(resource: dict[str, Any], kind: str, network: "NetworkSpec | None") -> None:
    """Apply menu-level networking unless the recipe overrides it (R-017 at the menu scope):
    the security group inherits ``vpc_id``; the rds inherits ``subnets`` as ``subnet_ids``."""
    if network is None:
        return
    if kind == "security_group" and network.vpc_id and "vpc_id" not in resource:
        resource["vpc_id"] = network.vpc_id
    if (
        kind == "rds"
        and network.subnets
        and "subnet_ids" not in resource
        and "db_subnet_group_name" not in resource
    ):
        resource["subnet_ids"] = list(network.subnets)


def _project_entry(name: str, entry: "RecipeEntry", network: "NetworkSpec | None") -> dict[str, Any]:
    """Project one infra ``RecipeEntry`` into a recipes resource dict: ``{type, name}`` +
    kind-specific fields, plus the serve-lambda menu-isms and menu-level network inheritance.

    The single projection, shared by ``Menu._validate_infra_fields`` (validate at load, S-2)
    and ``Menu.to_recipe_spec`` (build the RecipeSpec recipes consumes, S-1) — so both see an
    identical resource and validation can't diverge from what actually gets provisioned.
    """
    resource: dict[str, Any] = {"type": entry.kind, "name": name, **entry.fields}
    if entry.kind == "lambda":
        _wire_lambda_resource(resource, entry.source)
    _inherit_network(resource, entry.kind, network)
    return resource


def _infra_error_msg(exc: ValidationError, infra_names: list[str]) -> str:
    """Turn a projected-``RecipeSpec`` validation error into a menu-centric message that names
    the offending recipe. The error ``loc`` is ``("resources", <index>, [<kind-tag>,] <field>…)``;
    map the index back to the recipe key and drop the projection framing + discriminator tag."""
    err = exc.errors()[0]
    loc = list(err["loc"])
    recipe = None
    if len(loc) >= 2 and loc[0] == "resources" and isinstance(loc[1], int):
        recipe = infra_names[loc[1]] if loc[1] < len(infra_names) else None
        loc = loc[2:]
        if loc and loc[0] in INFRA_KINDS:  # drop the discriminated-union tag
            loc = loc[1:]
    field = ".".join(str(p) for p in loc)
    where = f"recipe {recipe!r}" if recipe else "infra recipe"
    return f"{where}: {field + ' — ' if field else ''}{err['msg']}"


def load_params(path: str) -> dict[str, Any]:
    """Load a ``params.yaml`` *or* ``menu.yaml`` as a flat params-shaped dict for the raw
    stage code (``params["model"]["target"]``, ``params["experiment"]``).

    A menu is a superset of params.yaml (INT-007), so its project sections are already flat;
    the one *derived* value raw consumers expect is ``experiment``, which the menu expresses
    as ``project``. This injects it so the raw stage path matches the bridged
    :class:`KitchenConfig` (whose ``experiment`` also defaults to ``project``) — without it a
    menu-run trains under the wrong experiment and auto-promote can't find the run.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if is_menu(raw):
        raw.setdefault("experiment", raw.get("project"))
    return raw


def stage_module_name(stage: str, params: dict[str, Any]) -> str:
    """The importable module for a ``stage``'s code (S-8, INT-019).

    Honors a menu recipe's declared ``source`` (INT-006) — e.g. ``source: src/training/main.py``
    → ``src.training.main`` — so a project's stage code can live anywhere; falls back to the
    convention ``src.<stage>.run`` when no source is declared (a ``params.yaml`` project, or a
    menu that follows the convention).

    Import-based on purpose (not path-based file loading): scaffolded stage modules import each
    other (``from src.features.run import …``), which only resolves as a package import.
    """
    source = None
    recipes = params.get("recipes")
    if isinstance(recipes, dict):
        entry = recipes.get(stage)
        if isinstance(entry, dict):
            source = entry.get("source")
    if source:
        stem = source[:-3] if source.endswith(".py") else source
        return stem.strip("/").replace("/", ".")
    return f"src.{stage}.run"


def load_stage_callable(stage: str, func: str, params: dict[str, Any]):
    """Import ``func`` from a stage's module (S-8, INT-019) — byte-identical to
    ``from <module> import <func>``.

    ``__import__(..., fromlist=[func])`` is a package import (intra-``src`` imports resolve) and
    goes through the ``builtins.__import__`` hook the missing-module tests mock. Missing module →
    ``ModuleNotFoundError``; missing attribute → ``ImportError`` (the ``AttributeError`` from the
    explicit getattr is converted, so it matches what the ``from`` statement itself would raise —
    not ``AttributeError``, which broke `test_train_flow`'s missing-module assertions).
    """
    module_name = stage_module_name(stage, params)
    module = __import__(module_name, fromlist=[func])
    try:
        return getattr(module, func)
    except AttributeError as exc:
        raise ImportError(f"cannot import name {func!r} from {module_name!r}") from exc


def load_scorer_callable(scorer: ScorerConfig):
    """Import a scorer's ``function`` from its ``source`` (GEN-006).

    ``scorer.source`` follows the same rule as a stage's ``source`` — a path (``src/score/run.py``)
    or a dotted module (``src.score.run``). Missing module → ``ModuleNotFoundError``; missing
    attribute → ``ImportError`` (mirroring :func:`load_stage_callable`).
    """
    stem = scorer.source[:-3] if scorer.source.endswith(".py") else scorer.source
    module_name = stem.strip("/").replace("/", ".")
    module = __import__(module_name, fromlist=[scorer.function])
    try:
        return getattr(module, scorer.function)
    except AttributeError as exc:
        raise ImportError(f"cannot import name {scorer.function!r} from {module_name!r}") from exc


class VariantNotFound(KeyError):
    """A ``--variant`` name has no entry in the menu's ``variants:`` map."""


def _deep_merge(base: dict, overlay: dict) -> None:
    """Recursively merge ``overlay`` into ``base`` in-place; overlay wins on a scalar/leaf
    conflict, nested dicts merge rather than replace."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _merge_feature_candidates(base: list[str], overlay: Any) -> list[str]:
    """Apply a variant's ``feature_candidates`` overlay to the base list (CBB-016).

    A plain list replaces. A mapping applies ``replace`` (wholesale) then ``remove`` then
    ``add`` (union, order-preserving, no duplicates)."""
    if isinstance(overlay, list):
        return list(overlay)
    if not isinstance(overlay, dict):
        raise ValueError("variant feature_candidates must be a list or an {add,remove,replace} map")
    result = list(overlay["replace"]) if overlay.get("replace") is not None else list(base)
    remove = set(overlay.get("remove") or [])
    result = [f for f in result if f not in remove]
    for f in overlay.get("add") or []:
        if f not in result:
            result.append(f)
    return result


def apply_variant(params: dict[str, Any], name: str) -> None:
    """Overlay the named variant onto ``params`` in-place (CBB-016).

    Non-list keys deep-merge (variant wins); ``feature_candidates`` uses list-merge semantics.
    Composes with ``--override`` by being applied first (overrides win). Raises
    :class:`VariantNotFound` (message lists the available names) when ``name`` is undeclared."""
    variants = params.get("variants") or {}
    if name not in variants:
        have = ", ".join(sorted(variants)) or "(none defined)"
        raise VariantNotFound(f"no variant {name!r} in menu — available: {have}")
    overlay = dict(variants[name] or {})
    fc_overlay = overlay.pop("feature_candidates", None)
    _deep_merge(params, overlay)
    if fc_overlay is not None:
        params["feature_candidates"] = _merge_feature_candidates(
            params.get("feature_candidates") or [], fc_overlay
        )
