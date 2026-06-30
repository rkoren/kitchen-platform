"""SEC-010: the version-controlled CI-role IAM policy is narrowed (no `s3:*`) and well-formed.

The policy itself is proven against a real `terraform plan` by `infra/verify-ci-policy.sh`
(AWS-gated, run on demand). This pytest is the cheap, always-on guard: it pins the structure
and, critically, that `ManageS3` never regresses back to the broad `s3:*`.
"""

from __future__ import annotations

import json
from pathlib import Path

_POLICY_TMPL = Path(__file__).resolve().parents[2] / "infra" / "ci-role-policy.json.tmpl"


def _load() -> dict:
    text = _POLICY_TMPL.read_text(encoding="utf-8").replace("${TF_STATE_BUCKET}", "example-tf-state")
    return json.loads(text)


def test_ci_policy_template_is_valid_and_complete():
    doc = _load()
    assert doc["Version"] == "2012-10-17"
    sids = {s["Sid"] for s in doc["Statement"]}
    assert {
        "TerraformState",
        "ManageS3",
        "ManageECR",
        "ManageIAM",
        "ManageLambda",
        "ManageLogs",
    } <= sids


def test_manage_s3_is_explicit_not_wildcard():
    """ManageS3 must be an explicit action set, never `s3:*` — the SEC-010 narrowing."""
    s3 = next(s for s in _load()["Statement"] if s["Sid"] == "ManageS3")
    actions = s3["Action"]
    assert "s3:*" not in actions
    assert actions and all(a.startswith("s3:") for a in actions)
    # the finicky aws_s3_bucket refresh reads SEC-004 flagged + bucket lifecycle ops
    for needed in (
        "s3:GetBucketVersioning",
        "s3:GetEncryptionConfiguration",
        "s3:GetBucketPublicAccessBlock",
        "s3:GetBucketTagging",
        "s3:CreateBucket",
        "s3:DeleteBucket",
    ):
        assert needed in actions, f"ManageS3 missing {needed}"


def test_other_statements_are_explicit_not_wildcard():
    """SEC-004: ECR/IAM/Lambda/Logs are explicit action sets too — no service wildcards.

    These ran only in the (gitignored, CI-skipped) bootstrap guard before SEC-010; pinning
    them here keeps them enforced in the normal gate now the policy is version-controlled.
    """
    statements = {s["Sid"]: s for s in _load()["Statement"]}
    for sid in ("ManageECR", "ManageIAM", "ManageLambda", "ManageLogs"):
        actions = statements[sid]["Action"]
        prefix = actions[0].split(":", 1)[0]
        assert f"{prefix}:*" not in actions, f"{sid} must not use {prefix}:*"
        assert all(":" in a and "*" not in a for a in actions), f"{sid} has a wildcard action"
    # Terraform reads this on aws_iam_role refresh; a missing it surfaces as a deploy-time deny.
    assert "iam:ListInstanceProfilesForRole" in statements["ManageIAM"]["Action"]
    assert {"ecr:GetAuthorizationToken", "ecr:PutImage"} <= set(statements["ManageECR"]["Action"])
