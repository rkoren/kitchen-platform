"""Terraform generator for AWS Lambda resources."""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from recipes.schema import IAMRoleSpec, LambdaSpec

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent.parent / "templates"),
    keep_trailing_newline=True,
)
_env.filters["tf_id"] = lambda s: s.replace("-", "_")


def generate(spec: LambdaSpec, all_resources: list = None) -> str:
    role_spec = next(
        (r for r in (all_resources or []) if isinstance(r, IAMRoleSpec) and r.name == spec.role),
        None,
    )
    policy_count = len(role_spec.policies) if role_spec else 0
    return _env.get_template("lambda.tf.j2").render(**spec.model_dump(), policy_count=policy_count)
