"""kitchen.secrets — the one secret resolver (SECR-002).

``get("API_KEY")`` resolves a secret through an ordered provider chain:

    1. process environment / local ``.env``   (explicit override; CI-injected; local dev)
    2. declared cloud source                   (AWS Secrets Manager bundle, or SSM parameter)
       — attempted only when an AWS identity resolves
    3. raise ``SecretNotFound`` naming the secret and exactly how to provide it

Where a secret lives is read from the ``secrets:`` manifest (SECR-001); an undeclared name is
treated as **env-only** (you opt into a cloud source by declaring ``aws_secret`` or ``ssm`` —
there is no implicit cloud lookup). Resolved values are cached in-process with a short TTL so a
rotated secret is picked up without a restart.

The resolver **never logs or prints secret values**, at any verbosity.
"""

from __future__ import annotations

import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from kitchen.config import KitchenConfig, SecretSpec

_DEFAULT_TTL_SECONDS = 300.0


class SecretNotFound(RuntimeError):
    """Raised when a secret cannot be resolved. The message names the secret and the fix.

    Never contains a secret value.
    """


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class SecretProvider(ABC):
    """A source kitchen can resolve a secret from.

    ``get`` returns the value, or ``None`` when this provider does not apply / has nothing for
    the secret (not an error). A provider may raise ``SecretNotFound`` for a hard failure it can
    describe better than the generic not-found message (e.g. a missing field in an SM bundle).
    """

    name: ClassVar[str]

    @abstractmethod
    def get(self, secret: str, spec: SecretSpec) -> str | None:  # pragma: no cover - interface
        ...


def _dotenv_values() -> dict[str, str | None]:
    """Read a local ``.env`` (without mutating ``os.environ``); ``{}`` if absent/unreadable."""
    if not Path(".env").exists():
        return {}
    try:
        from dotenv import dotenv_values

        return dict(dotenv_values(".env"))
    except Exception:
        return {}


class EnvProvider(SecretProvider):
    """Process environment first, then a local ``.env`` (matches how the project runs)."""

    name = "env"

    def get(self, secret: str, spec: SecretSpec) -> str | None:
        value = os.environ.get(secret)
        if value:
            return value
        value = _dotenv_values().get(secret)
        return value or None


def _region() -> str | None:
    return os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")


class AwsSecretsManagerProvider(SecretProvider):
    """AWS Secrets Manager — a JSON bundle (``aws_secret``), optionally one field (``key``)."""

    name = "aws-secretsmanager"

    def get(self, secret: str, spec: SecretSpec) -> str | None:
        if not spec.aws_secret:
            return None
        import boto3

        client = boto3.client("secretsmanager", region_name=_region())
        try:
            raw = client.get_secret_value(SecretId=spec.aws_secret)["SecretString"]
        except Exception as exc:  # botocore errors carry no secret value
            raise SecretNotFound(
                f"secret {secret!r}: could not fetch Secrets Manager bundle "
                f"{spec.aws_secret!r} ({type(exc).__name__}). Check the bundle name and IAM "
                f"read access, or set the {secret} environment variable to override."
            ) from exc
        if not spec.key:
            return raw
        import json

        try:
            bundle = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SecretNotFound(
                f"secret {secret!r}: bundle {spec.aws_secret!r} is not JSON, but a field "
                f"{spec.key!r} was requested. Drop `key` for a plain-string secret."
            ) from exc
        if spec.key not in bundle:
            raise SecretNotFound(
                f"secret {secret!r}: field {spec.key!r} not found in Secrets Manager bundle "
                f"{spec.aws_secret!r}."
            )
        return bundle[spec.key]


class AwsSsmProvider(SecretProvider):
    """AWS SSM Parameter Store — a ``SecureString`` parameter at ``ssm``."""

    name = "aws-ssm"

    def get(self, secret: str, spec: SecretSpec) -> str | None:
        if not spec.ssm:
            return None
        import boto3

        client = boto3.client("ssm", region_name=_region())
        try:
            param = client.get_parameter(Name=spec.ssm, WithDecryption=True)
        except Exception as exc:
            raise SecretNotFound(
                f"secret {secret!r}: could not fetch SSM parameter {spec.ssm!r} "
                f"({type(exc).__name__}). Check the path and IAM read access, or set the "
                f"{secret} environment variable to override."
            ) from exc
        return param["Parameter"]["Value"]


# Cloud providers, tried in order for a secret that declares a cloud source.
_ENV_PROVIDER = EnvProvider()
_CLOUD_PROVIDERS: tuple[SecretProvider, ...] = (AwsSecretsManagerProvider(), AwsSsmProvider())


def _aws_identity_available() -> bool:
    """True when the boto3 default chain resolves credentials (no network call)."""
    try:
        import boto3

        return boto3.Session().get_credentials() is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _spec_for(name: str, params_file: str, cfg: KitchenConfig | None) -> SecretSpec:
    """The declared spec for ``name``, or an env-only default when undeclared."""
    if cfg is None:
        p = Path(params_file)
        if p.exists():
            try:
                cfg = KitchenConfig.from_yaml(str(p))
            except Exception:
                cfg = None
    if cfg is not None:
        spec = cfg.effective_secrets().get(name)
        if spec is not None:
            return spec
    return SecretSpec()  # env-only: no implicit cloud lookup for undeclared names


def _resolve(name: str, spec: SecretSpec) -> str:
    """Run the provider chain for one secret. Raises SecretNotFound with a fix."""
    value = _ENV_PROVIDER.get(name, spec)
    if value:
        return value

    if spec.source != "env":
        if not _aws_identity_available():
            raise SecretNotFound(
                f"secret {name!r} is sourced from {spec.source} but no AWS identity resolved. "
                f"Set the {name} environment variable to override, or configure AWS credentials "
                f"(AWS_PROFILE, `aws configure`, or an assumed OIDC role)."
            )
        for provider in _CLOUD_PROVIDERS:
            resolved = provider.get(name, spec)  # may raise SecretNotFound with a precise message
            if resolved is not None:
                return resolved

    raise SecretNotFound(
        f"secret {name!r} not found. Provide it via the {name} environment variable or a local "
        f".env entry"
        + (
            ", or declare a source (aws_secret/ssm) for it in the `secrets:` manifest."
            if spec.source == "env"
            else "."
        )
    )


# ---------------------------------------------------------------------------
# In-process TTL cache
# ---------------------------------------------------------------------------

_CACHE: dict[str, tuple[str, float]] = {}
_CACHE_LOCK = threading.Lock()


def _default_ttl() -> float:
    try:
        return float(os.environ.get("KITCHEN_SECRETS_TTL", _DEFAULT_TTL_SECONDS))
    except ValueError:
        return _DEFAULT_TTL_SECONDS


def _cache_get(name: str) -> str | None:
    with _CACHE_LOCK:
        hit = _CACHE.get(name)
        if hit is None:
            return None
        value, expiry = hit
        if expiry > time.monotonic():
            return value
        del _CACHE[name]
        return None


def _cache_put(name: str, value: str, ttl: float | None) -> None:
    ttl = _default_ttl() if ttl is None else ttl
    with _CACHE_LOCK:
        _CACHE[name] = (value, time.monotonic() + ttl)


def clear_cache() -> None:
    """Drop all cached secret values (e.g. to force re-resolution after rotation, or in tests)."""
    with _CACHE_LOCK:
        _CACHE.clear()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get(
    name: str,
    *,
    params_file: str = "params.yaml",
    cfg: KitchenConfig | None = None,
    ttl: float | None = None,
    use_cache: bool = True,
) -> str:
    """Resolve a secret by name through the provider chain (env → cloud → error).

    The secret's source is read from the ``secrets:`` manifest in ``params_file`` (or ``cfg`` if
    given); an undeclared name is resolved env-only. Values are cached in-process for ``ttl``
    seconds (default ``KITCHEN_SECRETS_TTL`` or 300). Raises ``SecretNotFound`` — whose message
    names the secret and how to provide it — when resolution fails. Never logs the value.
    """
    if use_cache:
        cached = _cache_get(name)
        if cached is not None:
            return cached
    value = _resolve(name, _spec_for(name, params_file, cfg))
    if use_cache:
        _cache_put(name, value, ttl)
    return value


def try_get(
    name: str,
    *,
    params_file: str = "params.yaml",
    cfg: KitchenConfig | None = None,
    ttl: float | None = None,
    use_cache: bool = True,
) -> str | None:
    """Like :func:`get`, but return ``None`` instead of raising when the secret is unresolved.

    Useful for pre-flight checks that want to report many secrets without aborting on the first.
    """
    try:
        return get(name, params_file=params_file, cfg=cfg, ttl=ttl, use_cache=use_cache)
    except SecretNotFound:
        return None


# ---------------------------------------------------------------------------
# Masking (SECR-004)
# ---------------------------------------------------------------------------


def _under_github_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def mask(value: str) -> None:
    """Register ``value`` to be scrubbed from GitHub Actions logs; a no-op elsewhere.

    Emits the ``::add-mask::`` workflow command, which the runner intercepts — the value is
    registered as a secret and replaced with ``***`` in all subsequent log output (the command
    line itself included). **Outside** GitHub Actions this is intentionally a no-op: there is no
    runner to intercept the line, so emitting it would *leak* the value. Call this whenever a
    resolved secret is about to enter a process/subprocess environment that may log.
    """
    if value and _under_github_actions():
        print(f"::add-mask::{value}", flush=True)


def resolve_into_env(
    names: list[str],
    *,
    base: dict[str, str] | None = None,
    params_file: str = "params.yaml",
    cfg: KitchenConfig | None = None,
) -> dict[str, str]:
    """Resolve each secret and return an environment mapping with the values injected + masked.

    Use this to build the ``env=`` for a subprocess that needs real credentials in its
    environment (e.g. DVC needs ``AWS_*`` to reach S3). Each resolved value is registered with
    :func:`mask` so it never appears in CI logs, then set on a copy of ``base`` (defaults to the
    current environment). Raises ``SecretNotFound`` if any secret can't be resolved.
    """
    env = dict(os.environ if base is None else base)
    for name in names:
        value = get(name, params_file=params_file, cfg=cfg)
        mask(value)
        env[name] = value
    return env


# ---------------------------------------------------------------------------
# .env.example generation (SECR-005)
# ---------------------------------------------------------------------------


def _env_example_hint(spec: SecretSpec) -> str:
    """A short comment explaining where a secret comes from, for `.env.example`."""
    if spec.source == "env":
        return "env-only — set this value"
    return f"from {spec.source} in deployed envs; set here to override locally"


def env_example(cfg: KitchenConfig) -> str:
    """Render a ``.env.example`` body from the project's secrets manifest (SECR-005).

    One annotated, blank ``NAME=`` line per declared secret (required/optional + source) so a
    fresh clone self-documents what to set. Values are never filled in.
    """
    lines = [
        "# .env.example — generated by `kitchen secrets template`.",
        "# Copy to `.env` and fill in the values. Never commit `.env`.",
    ]
    secrets = cfg.effective_secrets()
    if not secrets:
        lines.append("#")
        lines.append("# (no secrets declared — add a `secrets:` section to params.yaml)")
        return "\n".join(lines) + "\n"
    for name, spec in secrets.items():
        req = "required" if spec.required else "optional"
        lines.append("")
        lines.append(f"# {name} ({req}) — {_env_example_hint(spec)}")
        lines.append(f"{name}=")
    return "\n".join(lines) + "\n"
