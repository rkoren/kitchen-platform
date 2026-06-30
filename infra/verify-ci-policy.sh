#!/usr/bin/env bash
# SEC-010: verify the CI role's IAM policy covers a real deploy WITHOUT over-permission.
#
# The prerequisite for narrowing `ManageS3` from `s3:*` to an explicit action set: this
# applies the candidate policy (infra/ci-role-policy.json.tmpl) to a *throwaway* IAM role,
# assumes it, and runs `recipes generate` -> `terraform plan` against the real stack. A clean
# plan (no AccessDenied) proves the narrowed policy covers Terraform's finicky `aws_s3_bucket`
# refresh reads (~15 GetBucket* sub-resources) — so we narrow with confidence instead of
# risking an opaque AccessDenied on the next deploy. `plan` is read-only; nothing is applied.
#
# Requires: aws cli + terraform + the recipes CLI, and credentials that can create/assume a
# role and read the Terraform state. The throwaway role is always deleted on exit.
#
# Usage:
#   TF_STATE_BUCKET=<state-bucket> infra/verify-ci-policy.sh [infra-spec] [state-key]
set -euo pipefail

INFRA_SPEC="${1:-kitchen/infra.yaml}"
STATE_KEY="${2:-kitchen/terraform.tfstate}"
AWS_REGION="${AWS_REGION:-us-east-1}"
: "${TF_STATE_BUCKET:?set TF_STATE_BUCKET to the Terraform state bucket}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROLE="sec010-ci-policy-test-$$"
WORK="$(mktemp -d)"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
CALLER_ARN="$(aws sts get-caller-identity --query Arn --output text)"

cleanup() {
  # Drop the assumed-role creds (exported below for the plan) so these IAM deletes run as the
  # caller — the assumed role is narrowed and can't delete itself, which would leak the role.
  unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
  aws iam delete-role-policy --role-name "$ROLE" --policy-name ci-permissions >/dev/null 2>&1 || true
  aws iam delete-role --role-name "$ROLE" >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "==> Rendering candidate policy (state bucket: ${TF_STATE_BUCKET})"
TF_STATE_BUCKET="$TF_STATE_BUCKET" envsubst '${TF_STATE_BUCKET}' \
  < "${REPO_ROOT}/infra/ci-role-policy.json.tmpl" > "${WORK}/policy.json"

echo "==> Creating throwaway test role: ${ROLE}"
cat > "${WORK}/trust.json" <<JSON
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"${CALLER_ARN}"},"Action":"sts:AssumeRole"}]}
JSON
aws iam create-role --role-name "$ROLE" \
  --assume-role-policy-document "file://${WORK}/trust.json" \
  --description "TEMP SEC-010 CI policy-coverage test - safe to delete" >/dev/null
aws iam put-role-policy --role-name "$ROLE" --policy-name ci-permissions \
  --policy-document "file://${WORK}/policy.json" >/dev/null

echo "==> Generating Terraform for ${INFRA_SPEC}"
( cd "$REPO_ROOT" && uv run --directory recipes recipes generate "${REPO_ROOT}/${INFRA_SPEC}" --out "${WORK}/tf" >/dev/null )

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE}"
echo "==> Assuming ${ROLE_ARN} and planning (read-only) under the candidate policy"
# Brief retry: a freshly-created role/policy is eventually consistent.
for attempt in 1 2 3 4 5; do
  if CREDS_JSON="$(aws sts assume-role --role-arn "$ROLE_ARN" --role-session-name sec010 \
      --query Credentials --output json 2>/dev/null)"; then break; fi
  [ "$attempt" = 5 ] && { echo "ERROR: could not assume $ROLE_ARN"; exit 1; }
  sleep 3
done
export AWS_ACCESS_KEY_ID="$(echo "$CREDS_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["AccessKeyId"])')"
export AWS_SECRET_ACCESS_KEY="$(echo "$CREDS_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["SecretAccessKey"])')"
export AWS_SESSION_TOKEN="$(echo "$CREDS_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["SessionToken"])')"

terraform -chdir="${WORK}/tf" init -input=false -no-color \
  -backend-config="bucket=${TF_STATE_BUCKET}" \
  -backend-config="key=${STATE_KEY}" \
  -backend-config="region=${AWS_REGION}" >/dev/null

PLAN_LOG="${WORK}/plan.log"
# -lock=false: plan is read-only; avoid contending for the deploy lock.
if terraform -chdir="${WORK}/tf" plan -lock=false -input=false -no-color > "$PLAN_LOG" 2>&1; then
  if grep -qiE "AccessDenied|not authorized|is not authorized to perform" "$PLAN_LOG"; then
    echo "SMOKE FAILED: plan succeeded but the log shows an authorization error:" >&2
    grep -iE "AccessDenied|not authorized" "$PLAN_LOG" >&2 | head
    exit 1
  fi
  echo "OK: terraform plan succeeded under the narrowed CI policy (no AccessDenied)."
  grep -E "No changes|Plan:" "$PLAN_LOG" | head -1
else
  echo "POLICY INSUFFICIENT: terraform plan failed under the candidate policy:" >&2
  grep -iE "AccessDenied|not authorized|Error:" "$PLAN_LOG" >&2 | head
  echo "  → add the denied action(s) to infra/ci-role-policy.json.tmpl and re-run." >&2
  exit 1
fi
