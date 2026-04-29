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


class ECRSpec(BaseModel):
    type: Literal["ecr"]
    name: str
    scan_on_push: bool = True
    image_tag_mutability: Literal["MUTABLE", "IMMUTABLE"] = "MUTABLE"
    lambda_access: bool = False


class LambdaSpec(BaseModel):
    type: Literal["lambda"]
    name: str
    role: str
    runtime: str | None = None
    handler: str | None = None
    memory: int = 128
    timeout: int = 3
    image_uri: str | None = None
    ecr_repo: str | None = None  # logical name of an ecr resource; generates a TF reference
    environment: dict[str, str] = {}


ResourceSpec = Annotated[
    Union[S3Spec, IAMRoleSpec, ECRSpec, LambdaSpec],
    Field(discriminator="type"),
]


class RecipeSpec(BaseModel):
    name: str
    region: str = "us-east-1"
    resources: list[ResourceSpec] = []
