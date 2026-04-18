"""Pydantic models for YAML spec validation."""
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class S3Spec(BaseModel):
    type: Literal["s3"]
    name: str
    versioning: bool = False


class IAMRoleSpec(BaseModel):
    type: Literal["iam_role"]
    name: str
    service: str
    policies: list[str] = []


class LambdaSpec(BaseModel):
    type: Literal["lambda"]
    name: str
    role: str
    runtime: str | None = None
    handler: str | None = None
    memory_mb: int = 128
    timeout_s: int = 3
    image_uri: str | None = None
    environment: dict[str, str] = {}


ResourceSpec = Annotated[
    Union[S3Spec, IAMRoleSpec, LambdaSpec],
    Field(discriminator="type"),
]


class RecipeSpec(BaseModel):
    name: str
    region: str = "us-east-1"
    resources: list[ResourceSpec] = []
