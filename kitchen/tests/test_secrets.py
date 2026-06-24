"""Tests for kitchen.secrets — the resolver, provider chain, and TTL cache (SECR-002)."""

from __future__ import annotations

import json
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
# Manifest discovery — menu-aware default (CBB-013)
# ---------------------------------------------------------------------------


def test_resolve_manifest_path_prefers_menu(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: p\n")
    assert sec._resolve_manifest_path(None) == "params.yaml"  # no menu → legacy
    (tmp_path / "menu.yaml").write_text("project: p\nrecipes: {}\n")
    assert sec._resolve_manifest_path(None) == "menu.yaml"  # canonical menu preferred
    assert sec._resolve_manifest_path("explicit.yaml") == "explicit.yaml"  # explicit respected


def test_secrets_manifest_read_from_menu_when_both_exist(tmp_path, monkeypatch):
    """CBB-013: get()/`_spec_for` with no path read menu.yaml's `secrets:` block (not the
    legacy params.yaml) when both exist — and no deprecation warning fires."""
    import warnings

    import kitchen.config as cfgmod

    monkeypatch.chdir(tmp_path)
    (tmp_path / "params.yaml").write_text("experiment: p\n")  # legacy, no secrets:
    (tmp_path / "menu.yaml").write_text(
        "project: p\nrecipes: {}\nsecrets:\n  K:\n    aws_secret: bundle\n    key: K\n"
    )
    cfgmod._LEGACY_PARAMS_WARNED = False  # reset the once-per-process guard for the assertion
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = sec._spec_for("K", None, None)
    assert spec.aws_secret == "bundle" and spec.key == "K"  # read from menu, not env-only
    assert not any("legacy params.yaml" in str(w.message) for w in caught)


def test_menu_only_project_secrets_manifest_is_visible(tmp_path, monkeypatch):
    """Before CBB-013 the params.yaml default made a menu-only project's `secrets:` invisible
    (no file → env-only); now the manifest is read."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "menu.yaml").write_text(
        "project: p\nrecipes: {}\nsecrets:\n  K:\n    ssm: /path/to/k\n"
    )
    spec = sec._spec_for("K", None, None)
    assert spec.ssm == "/path/to/k"


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


# ---------------------------------------------------------------------------
# Least-privilege IAM policy generation (SECR-006)
# ---------------------------------------------------------------------------


def test_iam_policy_sm_and_ssm_statements():
    cfg = _cfg(
        {
            "API": {"aws_secret": "proj/prod", "key": "API"},
            "DBP": {"ssm": "/proj/db"},
            "LOCAL": {},  # env-only → excluded
        }
    )
    pol = sec.iam_policy(cfg, account="123456789012", region="us-east-1")
    sids = {s["Sid"] for s in pol["Statement"]}
    assert sids == {"KitchenSecretsManagerRead", "KitchenSsmParameterRead"}
    sm = next(s for s in pol["Statement"] if s["Sid"] == "KitchenSecretsManagerRead")
    assert sm["Action"] == ["secretsmanager:GetSecretValue"]
    assert sm["Resource"] == ["arn:aws:secretsmanager:us-east-1:123456789012:secret:proj/prod-*"]
    ssm = next(s for s in pol["Statement"] if s["Sid"] == "KitchenSsmParameterRead")
    assert ssm["Resource"] == ["arn:aws:ssm:us-east-1:123456789012:parameter/proj/db"]


def test_iam_policy_default_wildcards_no_account():
    """SEC-001-style guard: default output embeds no account ID / personal value."""
    import re

    cfg = _cfg({"API": {"aws_secret": "kenpom_key", "key": "API"}})
    text = json.dumps(sec.iam_policy(cfg))
    assert "arn:aws:secretsmanager:*:*:secret:kenpom_key-*" in text
    assert re.search(r":\d{12}:", text) is None  # no 12-digit AWS account number
    for needle in ("rkoren", "reilly", "674325521451"):
        assert needle not in text


def test_iam_policy_passes_through_full_arn():
    cfg = _cfg({"API": {"aws_secret": "arn:aws:secretsmanager:us-east-1:111:secret:x-AbC", "key": "API"}})
    pol = sec.iam_policy(cfg)
    assert pol["Statement"][0]["Resource"] == ["arn:aws:secretsmanager:us-east-1:111:secret:x-AbC"]


def test_iam_policy_empty_when_no_cloud_secrets():
    cfg = _cfg({"LOCAL": {}}, required_env=["LEGACY"])
    assert sec.iam_policy(cfg)["Statement"] == []


# ---------------------------------------------------------------------------
# CI environment export (SECR-007)
# ---------------------------------------------------------------------------


def test_export_env_file_writes_heredoc_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKEN", "tok-value")
    cfg = _cfg({"TOKEN": {"required": True}})
    target = tmp_path / "gh_env"
    exported = sec.export_env_file(target, cfg=cfg, params_file="nope.yaml")
    assert exported == ["TOKEN"]
    content = target.read_text()
    # heredoc form: NAME<<DELIM \n value \n DELIM
    assert content.startswith("TOKEN<<")
    assert "\ntok-value\n" in content


def test_export_env_file_defaults_to_all_declared(tmp_path, monkeypatch):
    monkeypatch.setenv("A", "av")
    monkeypatch.setenv("B", "bv")
    cfg = _cfg({"A": {"required": True}, "B": {"required": True}})
    target = tmp_path / "gh_env"
    exported = sec.export_env_file(target, cfg=cfg, params_file="nope.yaml")
    assert sorted(exported) == ["A", "B"]


def test_export_env_file_skips_unresolved_optional(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPT", raising=False)
    cfg = _cfg({"OPT": {"required": False}})
    target = tmp_path / "gh_env"
    exported = sec.export_env_file(target, cfg=cfg, params_file="nope.yaml")
    assert exported == []
    assert not target.exists()  # nothing written when no entries resolve


def test_export_env_file_required_missing_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("REQ", raising=False)
    cfg = _cfg({"REQ": {"required": True}})
    with pytest.raises(sec.SecretNotFound):
        sec.export_env_file(tmp_path / "gh_env", cfg=cfg, params_file="nope.yaml")


def test_export_env_file_explicit_names_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("A", "av")
    monkeypatch.setenv("B", "bv")
    cfg = _cfg({"A": {"required": True}, "B": {"required": True}})
    target = tmp_path / "gh_env"
    exported = sec.export_env_file(target, ["A"], cfg=cfg, params_file="nope.yaml")
    assert exported == ["A"]
    assert "B<<" not in target.read_text()


def test_export_env_file_masks_in_ci(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("TOKEN", "tok-value")
    cfg = _cfg({"TOKEN": {"required": True}})
    sec.export_env_file(tmp_path / "gh_env", cfg=cfg, params_file="nope.yaml")
    assert "::add-mask::tok-value" in capsys.readouterr().out


def test_export_env_file_does_not_leak_outside_ci(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("TOKEN", "leaky-value")
    cfg = _cfg({"TOKEN": {"required": True}})
    target = tmp_path / "gh_env"
    sec.export_env_file(target, cfg=cfg, params_file="nope.yaml")
    out = capsys.readouterr()
    assert "leaky-value" not in out.out and "leaky-value" not in out.err
    assert "leaky-value" in target.read_text()  # but it IS written to the env file


def test_export_env_file_heredoc_delimiter_not_in_value(tmp_path, monkeypatch):
    # A value containing '=' and newlines must still round-trip safely.
    monkeypatch.setenv("MULTI", "line1\nKEY=val\nline3")
    cfg = _cfg({"MULTI": {"required": True}})
    target = tmp_path / "gh_env"
    sec.export_env_file(target, cfg=cfg, params_file="nope.yaml")
    content = target.read_text()
    delim = content.split("<<", 1)[1].splitlines()[0]
    assert delim not in "line1\nKEY=val\nline3"
    assert f"\nline1\nKEY=val\nline3\n{delim}\n" in content


# ---------------------------------------------------------------------------
# DB-URL assembly (LML-012 follow-up)
# ---------------------------------------------------------------------------


def test_build_db_url_encodes_credentials():
    from urllib.parse import unquote, urlsplit

    url = sec.build_db_url(
        {"username": "ml flow", "password": "p@ss:w/rd"}, endpoint="host:5432", db="mlflow"
    )
    parts = urlsplit(url)
    assert parts.scheme == "postgresql"
    assert parts.hostname == "host" and parts.port == 5432 and parts.path == "/mlflow"
    # Special characters survive the URI round-trip (the correctness bug to nail).
    assert unquote(parts.username) == "ml flow"
    assert unquote(parts.password) == "p@ss:w/rd"


def test_build_db_url_missing_key_raises():
    with pytest.raises(sec.SecretNotFound, match="password"):
        sec.build_db_url({"username": "mlflow"}, endpoint="host:5432")


def test_db_url_fetches_and_assembles():
    client = MagicMock()
    client.get_secret_value.return_value = {
        "SecretString": json.dumps({"username": "mlflow", "password": "s3cr@t"})
    }
    with patch("boto3.client", return_value=client):
        url = sec.db_url("arn:rds-managed", endpoint="db.rds.amazonaws.com:5432", db="mlflow")
    assert url.startswith("postgresql://mlflow:s3cr%40t@db.rds.amazonaws.com:5432/mlflow")
    client.get_secret_value.assert_called_once_with(SecretId="arn:rds-managed")


def test_db_url_non_json_secret_raises():
    client = MagicMock()
    client.get_secret_value.return_value = {"SecretString": "not-json"}
    with patch("boto3.client", return_value=client):
        with pytest.raises(sec.SecretNotFound, match="not a JSON bundle"):
            sec.db_url("arn:bad", endpoint="host:5432")


def test_terraform_outputs_missing_terraform_raises():
    with patch("shutil.which", return_value=None):
        with pytest.raises(sec.SecretNotFound, match="terraform is not installed"):
            sec.terraform_outputs("/some/dir")


def test_db_url_from_terraform_autodetects_single_backend():
    outputs = {
        "mlflow_validation_endpoint": "db.rds.amazonaws.com:5432",
        "mlflow_validation_master_user_secret_arn": "arn:rds-managed",
        "some_bucket": "irrelevant",
    }
    client = MagicMock()
    client.get_secret_value.return_value = {
        "SecretString": json.dumps({"username": "mlflow", "password": "p@ss"})
    }
    with (
        patch.object(sec, "terraform_outputs", return_value=outputs),
        patch("boto3.client", return_value=client),
    ):
        url = sec.db_url_from_terraform("/ws", db="mlflow")
    assert url.startswith("postgresql://mlflow:p%40ss@db.rds.amazonaws.com:5432/mlflow")
    client.get_secret_value.assert_called_once_with(SecretId="arn:rds-managed")


def test_db_url_from_terraform_no_backend_raises():
    with patch.object(sec, "terraform_outputs", return_value={"some_bucket": "x"}):
        with pytest.raises(sec.SecretNotFound, match="no RDS backend outputs"):
            sec.db_url_from_terraform("/ws")


def test_db_url_from_terraform_multiple_requires_rds():
    outputs = {
        "a_endpoint": "h1:5432", "a_master_user_secret_arn": "arn:a",
        "b_endpoint": "h2:5432", "b_master_user_secret_arn": "arn:b",
    }
    with patch.object(sec, "terraform_outputs", return_value=outputs):
        with pytest.raises(sec.SecretNotFound, match="multiple RDS backends"):
            sec.db_url_from_terraform("/ws")


def test_db_url_from_terraform_rds_selector_picks_one():
    outputs = {
        "mlflow_a_endpoint": "h1:5432", "mlflow_a_master_user_secret_arn": "arn:a",
        "mlflow_b_endpoint": "h2:5432", "mlflow_b_master_user_secret_arn": "arn:b",
    }
    client = MagicMock()
    client.get_secret_value.return_value = {"SecretString": json.dumps({"username": "u", "password": "p"})}
    with (
        patch.object(sec, "terraform_outputs", return_value=outputs),
        patch("boto3.client", return_value=client),
    ):
        sec.db_url_from_terraform("/ws", rds="mlflow-b")  # hyphen → tf_id underscore
    client.get_secret_value.assert_called_once_with(SecretId="arn:b")
