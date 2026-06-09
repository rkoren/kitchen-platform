"""Guards for scripts/bootstrap-aws.sh (SEC-001 parameterised, SEC-003 idempotent).

These are static/behavioural checks that need no AWS account: a syntax check, a
scan for personal values, and a check that the required-variable guard fires
before any AWS call is made.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "bootstrap-aws.sh"

# The bootstrap script is intentionally gitignored (local-only), so it is absent in a
# fresh CI checkout. Skip these guards when it isn't present rather than fail; they run
# wherever the script exists (e.g. a maintainer's working tree).
pytestmark = [
    pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available"),
    pytest.mark.skipif(
        not SCRIPT.exists(), reason="scripts/bootstrap-aws.sh not present (gitignored, local-only)"
    ),
]


def test_script_exists_and_executable():
    assert SCRIPT.exists(), f"missing {SCRIPT}"
    assert SCRIPT.stat().st_mode & 0o111, "bootstrap script should be executable"


def test_script_has_valid_bash_syntax():
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_script_carries_no_personal_values():
    # SEC-001 acceptance: "Bootstrap script has no personal values."
    text = SCRIPT.read_text()
    for needle in ("rkoren", "light-ml", "reilly"):
        assert needle not in text, f"personal value {needle!r} still present"


@pytest.mark.parametrize(
    "var",
    ["AWS_REGION", "CI_ROLE_NAME", "POLICY_NAME", "TF_STATE_BUCKET", "OIDC_THUMBPRINT"],
)
def test_parameters_are_env_overridable(var):
    # Each tunable resolves via ${VAR:-default} so callers can override it.
    assert f"${{{var}:-" in SCRIPT.read_text()


def test_requires_github_repo_before_any_aws_call():
    # SEC-001: GITHUB_REPO has no default. With it unset the script must exit
    # non-zero on the guard, before reaching the sts/iam/s3 calls — so this
    # passes without AWS credentials or even the aws CLI installed.
    env = {"PATH": "/usr/bin:/bin"}  # deliberately omit GITHUB_REPO
    result = subprocess.run(
        ["bash", str(SCRIPT)], capture_output=True, text=True, env=env
    )
    assert result.returncode != 0
    assert "GITHUB_REPO" in result.stderr
