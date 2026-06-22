"""Build a ``RecipeSpec`` from a ``menu.yaml`` manifest's infra recipes (INT-004).

recipes reads the menu's ``recipes:`` / ``network`` / ``project`` / ``region`` **directly**
— it does not import ``kitchen``, so recipes stays installable on its own. Only the infra
kinds are recipes' concern; runtime kinds (``stage``) belong to kitchen and are skipped.
The menu-only keys ``role`` (discovery marker) and ``source`` (code location) are kitchen
concerns and dropped here. Per-kind field validation happens via ``RecipeSpec`` (each
resource is validated against the ``extra="forbid"`` specs), so typos in a recipe's fields
surface here — this is the validation ``kitchen.menu`` (INT-002) deliberately deferred.
"""

from __future__ import annotations

from recipes.schema import RecipeSpec

# Infra kinds recipes provisions — must match the ``ResourceSpec`` discriminator (and
# ``kitchen.menu.INFRA_KINDS``, which a kitchen test locks to this same set).
_INFRA_KINDS = frozenset({"rds", "s3", "security_group", "iam_role", "ecr", "lambda"})

# Menu-only keys that are kitchen's concern, not recipes': dropped before building a resource.
_MENU_ONLY_KEYS = frozenset({"kind", "role", "source"})


def is_menu(raw: dict) -> bool:
    """A menu has a ``recipes:`` map; a standalone recipes spec has a ``resources:`` list."""
    return isinstance(raw, dict) and "recipes" in raw and "resources" not in raw


def recipe_spec_from_menu(raw: dict) -> RecipeSpec:
    """Project the infra recipes of a parsed ``menu.yaml`` into a ``RecipeSpec``."""
    recipes = raw.get("recipes") or {}
    network = raw.get("network") or {}
    resources: list[dict] = []
    for name, entry in recipes.items():
        entry = entry or {}
        if entry.get("kind") not in _INFRA_KINDS:
            continue  # stage / other runtime kinds are kitchen's, not infra
        kind = entry["kind"]
        resource = {"type": kind, "name": name}
        resource.update({k: v for k, v in entry.items() if k not in _MENU_ONLY_KEYS})
        if kind == "lambda":
            _wire_lambda(resource, entry)
        _inherit_network(resource, kind, network)
        resources.append(resource)
    return RecipeSpec.model_validate(
        {
            "name": raw.get("project"),
            "region": raw.get("region", "us-east-1"),
            "resources": resources,
        }
    )


def _wire_lambda(resource: dict, entry: dict) -> None:
    """Resolve the serve lambda's two menu-isms (INT-006, simplification S-4):

    * the menu uses ``role`` for the *discovery* marker and ``iam_role`` for the lambda's
      execution role — map ``iam_role`` onto recipes' ``LambdaSpec.role``;
    * the menu ``source`` (where the predictor lives) is injected as the deployed function's
      ``KITCHEN_PREDICTOR_DIR`` so serving finds it at runtime.
    """
    if "iam_role" in resource:
        resource["role"] = resource.pop("iam_role")
    source = entry.get("source")
    if source:
        env = dict(resource.get("environment") or {})
        env.setdefault("KITCHEN_PREDICTOR_DIR", source)
        resource["environment"] = env


def _inherit_network(resource: dict, kind: str, network: dict) -> None:
    """Apply menu-level networking unless the recipe overrides it (R-017 at the menu scope):
    the security group inherits ``vpc_id``; the rds inherits ``subnets`` as ``subnet_ids``."""
    if kind == "security_group" and network.get("vpc_id") and "vpc_id" not in resource:
        resource["vpc_id"] = network["vpc_id"]
    if (
        kind == "rds"
        and network.get("subnets")
        and "subnet_ids" not in resource
        and "db_subnet_group_name" not in resource
    ):
        resource["subnet_ids"] = list(network["subnets"])
