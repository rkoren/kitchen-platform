"""Tests for kitchen.secrets — the resolver, provider chain, and TTL cache (SECR-002)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kitchen import secrets as sec
from kitchen.config import SecretSpec


@pytest.fixture(autouse=True)
def _clear_cache():
    sec.clear_cache()
    yield
    sec.clear_cache()


# ---------------------------------------------------------------------------
# Env / .env resolution
# ---------------------------------------------------------------------------


def test_get_from_env(monkeypatch):
    monkeypatch.setenv("FOO", "from-env")
    assert sec.get("FOO", cfg=None, params_file="nope.yaml") == "from-env"


def test_get_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FOO", raising=False)
    (tmp_path / ".env").write_text("FOO=from-dotenv\n")
    assert sec.get("FOO", params_file="nope.yaml") == "from-dotenv"


def test_env_beats_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("FOO=from-dotenv\n")
    monkeypatch.setenv("FOO", "from-env")
    assert sec.get("FOO", params_file="nope.yaml") == "from-env"


def test_undeclared_name_is_env_only(monkeypatch, tmp_path):
    """An undeclared secret resolves from env — no implicit cloud lookup."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MISSING", raising=False)
    with pytest.raises(sec.SecretNotFound) as ei:
        sec.get("MISSING", params_file="nope.yaml")
    msg = str(ei.value)
    assert "MISSING" in msg
    assert "environment variable" in msg


# ---------------------------------------------------------------------------
# Cloud sources (mocked boto3)
# ---------------------------------------------------------------------------


def _cfg_with(spec_kwargs: dict, name: str = "KENPOM_API_KEY"):
    from kitchen.config import KitchenConfig

    return KitchenConfig(experiment="e", secrets={name: spec_kwargs})


def test_secrets_manager_bundle_field(monkeypatch):
    monkeypatch.delenv("KENPOM_API_KEY", raising=False)
    cfg = _cfg_with({"aws_secret": "cbb-model/prod", "key": "KENPOM_API_KEY"})
    client = MagicMock()
    client.get_secret_value.return_value = {"SecretString": '{"KENPOM_API_KEY": "sm-value"}'}
    with (
        patch.object(sec, "_aws_identity_available", return_value=True),
        patch("boto3.client", return_value=client),
    ):
        assert sec.get("KENPOM_API_KEY", cfg=cfg) == "sm-value"
    client.get_secret_value.assert_called_once_with(SecretId="cbb-model/prod")


def test_secrets_manager_plain_string_no_key(monkeypatch):
    monkeypatch.delenv("TOK", raising=False)
    cfg = _cfg_with({"aws_secret": "proj/tok"}, name="TOK")
    client = MagicMock()
    client.get_secret_value.return_value = {"SecretString": "raw-secret"}
    with (
        patch.object(sec, "_aws_identity_available", return_value=True),
        patch("boto3.client", return_value=client),
    ):
        assert sec.get("TOK", cfg=cfg) == "raw-secret"


def test_secrets_manager_missing_field_raises(monkeypatch):
    monkeypatch.delenv("KENPOM_API_KEY", raising=False)
    cfg = _cfg_with({"aws_secret": "cbb-model/prod", "key": "KENPOM_API_KEY"})
    client = MagicMock()
    client.get_secret_value.return_value = {"SecretString": '{"OTHER": "x"}'}
    with (
        patch.object(sec, "_aws_identity_available", return_value=True),
        patch("boto3.client", return_value=client),
    ):
        with pytest.raises(sec.SecretNotFound, match="field 'KENPOM_API_KEY' not found"):
            sec.get("KENPOM_API_KEY", cfg=cfg)


def test_ssm_parameter(monkeypatch):
    monkeypatch.delenv("PARAM", raising=False)
    cfg = _cfg_with({"ssm": "/cbb/param"}, name="PARAM")
    client = MagicMock()
    client.get_parameter.return_value = {"Parameter": {"Value": "ssm-value"}}
    with (
        patch.object(sec, "_aws_identity_available", return_value=True),
        patch("boto3.client", return_value=client),
    ):
        assert sec.get("PARAM", cfg=cfg) == "ssm-value"
    client.get_parameter.assert_called_once_with(Name="/cbb/param", WithDecryption=True)


def test_env_overrides_cloud_source(monkeypatch):
    """An env var of the same name wins over the declared cloud source (no boto call)."""
    monkeypatch.setenv("KENPOM_API_KEY", "env-override")
    cfg = _cfg_with({"aws_secret": "cbb-model/prod", "key": "KENPOM_API_KEY"})
    with patch("boto3.client", side_effect=AssertionError("cloud must not be called")):
        assert sec.get("KENPOM_API_KEY", cfg=cfg) == "env-override"


def test_cloud_source_without_identity_raises_actionable(monkeypatch):
    monkeypatch.delenv("KENPOM_API_KEY", raising=False)
    cfg = _cfg_with({"aws_secret": "cbb-model/prod", "key": "KENPOM_API_KEY"})
    with patch.object(sec, "_aws_identity_available", return_value=False):
        with pytest.raises(sec.SecretNotFound) as ei:
            sec.get("KENPOM_API_KEY", cfg=cfg)
    msg = str(ei.value)
    assert "no AWS identity" in msg
    assert "KENPOM_API_KEY environment variable" in msg


# ---------------------------------------------------------------------------
# TTL cache
# ---------------------------------------------------------------------------


def test_cache_avoids_second_resolution(monkeypatch):
    monkeypatch.setenv("FOO", "v1")
    assert sec.get("FOO", params_file="nope.yaml") == "v1"
    monkeypatch.setenv("FOO", "v2")  # changed underneath
    assert sec.get("FOO", params_file="nope.yaml") == "v1"  # served from cache


def test_cache_ttl_zero_reresolves(monkeypatch):
    monkeypatch.setenv("FOO", "v1")
    assert sec.get("FOO", params_file="nope.yaml", ttl=0) == "v1"
    monkeypatch.setenv("FOO", "v2")
    assert sec.get("FOO", params_file="nope.yaml", ttl=0) == "v2"


def test_clear_cache(monkeypatch):
    monkeypatch.setenv("FOO", "v1")
    sec.get("FOO", params_file="nope.yaml")
    sec.clear_cache()
    monkeypatch.setenv("FOO", "v2")
    assert sec.get("FOO", params_file="nope.yaml") == "v2"


def test_use_cache_false_bypasses(monkeypatch):
    monkeypatch.setenv("FOO", "v1")
    sec.get("FOO", params_file="nope.yaml")
    monkeypatch.setenv("FOO", "v2")
    assert sec.get("FOO", params_file="nope.yaml", use_cache=False) == "v2"


# ---------------------------------------------------------------------------
# try_get + never-log
# ---------------------------------------------------------------------------


def test_try_get_returns_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MISSING", raising=False)
    assert sec.try_get("MISSING", params_file="nope.yaml") is None


def test_try_get_returns_value(monkeypatch):
    monkeypatch.setenv("FOO", "v")
    assert sec.try_get("FOO", params_file="nope.yaml") == "v"


def test_resolver_never_prints_value(monkeypatch, capsys):
    """The resolved value must never reach stdout/stderr (never-log discipline)."""
    monkeypatch.setenv("SECRET_TOKEN", "super-secret-value")
    sec.get("SECRET_TOKEN", params_file="nope.yaml")
    out = capsys.readouterr()
    assert "super-secret-value" not in out.out
    assert "super-secret-value" not in out.err


def test_provider_interface_get_signature():
    """EnvProvider implements the SecretProvider contract."""
    assert isinstance(sec.EnvProvider(), sec.SecretProvider)
    assert sec.EnvProvider().get("NOPE", SecretSpec()) is None


# ---------------------------------------------------------------------------
# Masking + subprocess-env injection (SECR-004)
# ---------------------------------------------------------------------------


def test_mask_emits_under_github_actions(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    sec.mask("super-secret-value")
    assert capsys.readouterr().out == "::add-mask::super-secret-value\n"


def test_mask_noop_outside_github_actions(monkeypatch, capsys):
    """Outside CI there is no runner to intercept ::add-mask::, so emitting it would leak."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    sec.mask("super-secret-value")
    assert capsys.readouterr().out == ""


def test_mask_empty_value_noop(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    sec.mask("")
    assert capsys.readouterr().out == ""


def test_resolve_into_env_injects_and_masks(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("TOKEN", "tok")
    env = sec.resolve_into_env(
        ["AWS_SECRET_ACCESS_KEY", "TOKEN"], base={"PATH": "/bin"}, params_file="nope.yaml"
    )
    assert env["AWS_SECRET_ACCESS_KEY"] == "aws-secret"
    assert env["TOKEN"] == "tok"
    assert env["PATH"] == "/bin"  # base preserved
    out = capsys.readouterr().out
    assert "::add-mask::aws-secret" in out
    assert "::add-mask::tok" in out


def test_resolve_into_env_defaults_to_os_environ(monkeypatch):
    monkeypatch.setenv("TOKEN", "tok")
    monkeypatch.setenv("EXISTING", "keep")
    env = sec.resolve_into_env(["TOKEN"], params_file="nope.yaml")
    assert env["TOKEN"] == "tok"
    assert env["EXISTING"] == "keep"  # inherited from os.environ


def test_resolve_into_env_raises_on_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MISSING", raising=False)
    with pytest.raises(sec.SecretNotFound):
        sec.resolve_into_env(["MISSING"], params_file="nope.yaml")


def test_resolve_into_env_does_not_leak_outside_ci(monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("TOKEN", "leaky-value")
    sec.resolve_into_env(["TOKEN"], params_file="nope.yaml")
    out = capsys.readouterr()
    assert "leaky-value" not in out.out
    assert "leaky-value" not in out.err


# ---------------------------------------------------------------------------
# .env.example generation (SECR-005)
# ---------------------------------------------------------------------------


def _cfg(secrets=None, required_env=None):
    from kitchen.config import KitchenConfig

    kw = {"experiment": "e"}
    if secrets is not None:
        kw["secrets"] = secrets
    if required_env is not None:
        kw["check"] = {"required_env": required_env}
    return KitchenConfig(**kw)


def test_env_example_empty_manifest():
    body = sec.env_example(_cfg())
    assert "no secrets declared" in body
    assert "=" not in body.replace("# ", "")  # no NAME= lines


def test_env_example_annotates_required_and_source():
    cfg = _cfg(
        {
            "KAGGLE_KEY": {"aws_secret": "p/prod", "key": "KAGGLE_KEY", "required": True},
            "OPT": {"ssm": "/p/opt", "required": False},
            "LOCAL": {},
        }
    )
    body = sec.env_example(cfg)
    assert "# KAGGLE_KEY (required) — from SM p/prod#KAGGLE_KEY" in body
    assert "KAGGLE_KEY=" in body
    assert "# OPT (optional) — from SSM /p/opt" in body
    assert "# LOCAL (required) — env-only — set this value" in body
    # values are never written
    assert "KAGGLE_KEY=\n" in body


def test_env_example_includes_legacy_required_env():
    body = sec.env_example(_cfg(required_env=["LEGACY"]))
    assert "# LEGACY (required) — env-only" in body
    assert "LEGACY=" in body
