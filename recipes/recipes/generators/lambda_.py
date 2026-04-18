"""Terraform generator for AWS Lambda resources."""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from recipes.schema import LambdaSpec

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent.parent / "templates"),
    keep_trailing_newline=True,
)


def generate(spec: LambdaSpec) -> str:
    return _env.get_template("lambda.tf.j2").render(spec.model_dump())
