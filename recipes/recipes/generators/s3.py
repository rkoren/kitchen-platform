"""Terraform generator for S3 bucket resources."""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from recipes.schema import S3Spec

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent.parent / "templates"),
    keep_trailing_newline=True,
)
_env.filters["tf_id"] = lambda s: s.replace("-", "_")


def generate(spec: S3Spec) -> str:
    return _env.get_template("s3.tf.j2").render(spec.model_dump())
