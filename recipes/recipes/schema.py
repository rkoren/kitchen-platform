"""Pydantic models for YAML spec validation."""
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class S3Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["s3"]
    name: str
    versioning: bool = False


class IAMRoleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["iam_role"]
    name: str
    service: str
    policies: list[str] = []


class ECRSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["ecr"]
    name: str
    scan_on_push: bool = True
    image_tag_mutability: Literal["MUTABLE", "IMMUTABLE"] = "MUTABLE"
    lambda_access: bool = False


class LambdaSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

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

    @model_validator(mode="after")
    def _validate_package_type(self) -> "LambdaSpec":
        is_image = self.image_uri is not None or self.ecr_repo is not None
        is_zip = self.runtime is not None or self.handler is not None
        if is_image and is_zip:
            raise ValueError(
                "Lambda cannot mix image fields (image_uri/ecr_repo) with zip fields "
                "(runtime/handler) — choose one package type."
            )
        if not is_image and not is_zip:
            raise ValueError(
                "Lambda must specify either image fields (image_uri or ecr_repo) "
                "or zip fields (runtime and handler)."
            )
        if is_zip and (self.runtime is None or self.handler is None):
            raise ValueError("Zip Lambda requires both runtime and handler.")
        return self


ResourceSpec = Annotated[
    Union[S3Spec, IAMRoleSpec, ECRSpec, LambdaSpec],
    Field(discriminator="type"),
]


class RecipeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    region: str = "us-east-1"
    resources: list[ResourceSpec] = []

    @model_validator(mode="after")
    def _validate_resource_references(self) -> "RecipeSpec":
        iam_names = {r.name for r in self.resources if isinstance(r, IAMRoleSpec)}
        ecr_names = {r.name for r in self.resources if isinstance(r, ECRSpec)}
        errors: list[str] = []
        for r in self.resources:
            if not isinstance(r, LambdaSpec):
                continue
            # role must be an existing iam_role name or a literal ARN
            if r.role not in iam_names and not r.role.startswith("arn:"):
                errors.append(
                    f"Lambda '{r.name}': role '{r.role}' does not match any iam_role "
                    "resource in this spec (and is not an ARN)."
                )
            # ecr_repo must be an existing ecr name
            if r.ecr_repo is not None and r.ecr_repo not in ecr_names:
                errors.append(
                    f"Lambda '{r.name}': ecr_repo '{r.ecr_repo}' does not match any "
                    "ecr resource in this spec."
                )
        if errors:
            raise ValueError("\n".join(errors))
        return self
