# infra/ — CI IAM policy (version-controlled) + coverage harness

## `ci-role-policy.json.tmpl`
The least-privilege IAM policy attached to the GitHub Actions CI role (`github-actions-ci`).
It is the **single source of truth** for that role's permissions — `scripts/bootstrap-aws.sh`
renders it (substituting `${TF_STATE_BUCKET}`) and applies it via `aws iam put-role-policy`.

Each statement is scoped to explicit actions (SEC-004, SEC-010). In particular **`ManageS3` is
an explicit action set, not `s3:*`** — narrowed under SEC-010, which previously couldn't be done
safely because Terraform's `aws_s3_bucket` refresh reads ~15 `GetBucket*` sub-resources and an
opaque `AccessDenied` would only surface mid-deploy.

## `verify-ci-policy.sh` — the coverage harness (SEC-010 prerequisite)
Proves the policy covers a real deploy **before** trusting the narrowing:

```bash
TF_STATE_BUCKET=<your-state-bucket> infra/verify-ci-policy.sh [infra-spec] [state-key]
```

It renders the candidate policy onto a **throwaway** IAM role, assumes it, and runs
`recipes generate` → `terraform plan` against the stack. A clean plan (no `AccessDenied`) means
the policy covers Terraform's refresh reads. The throwaway role is always deleted on exit; `plan`
is read-only (nothing is applied). Re-run it whenever the stack or the policy changes.

**Scope:** the harness validates the **read/refresh** path (`terraform plan`), which is where the
`GetBucket*` risk lives. The create/update/delete (`Put*`/`Create*`/`Delete*`) actions are
included from the AWS provider's documented apply behavior; a full guarantee of the apply path
would need an actual `terraform apply` against a throwaway stack (out of scope here).

The cheap always-on guard that `ManageS3` never regresses to `s3:*` is
`kitchen/tests/test_ci_policy.py` (runs in the normal test gate).
