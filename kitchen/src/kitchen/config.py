"""Find the project and its state dir, load kitchen.yaml, render jobs."""

import os
import re
import shlex
import subprocess
from pathlib import Path

import yaml

SCHEMES = {"kfold", "stratified", "group", "logo", "forward", "holdout"}


class KitchenError(Exception):
    """A user-facing error: the message is printed and the process exits with `code`."""

    code = 1


class ConfigError(KitchenError):
    code = 2


class NeedsHuman(KitchenError):
    """Refused because the step is a human decision (promote, submit, pushes)."""

    code = 3


def duration(value) -> float | None:
    """'90s' / '30m' / '2h' / 600 -> seconds; None or 0 means no limit."""
    if value in (None, "", 0, "0"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*", str(value))
    if not m:
        raise ConfigError(f"bad duration {value!r} (use e.g. 90s, 30m, 2h)")
    return float(m[1]) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m[2]]


def git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.rstrip("\n") if out.returncode == 0 else None


def main_checkout(root: Path) -> Path:
    """`root` mapped into the repo's main worktree, so every worktree shares one .kitchen."""
    top = git(root, "rev-parse", "--show-toplevel")
    common = git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if not top or not common or Path(common).name != ".git":
        return root
    return Path(common).parent / root.resolve().relative_to(Path(top).resolve())


class Config:
    def __init__(self, root: Path, data: dict):
        self.root, self.data = root, data
        self.main = main_checkout(root)
        state = os.environ.get("KITCHEN_STATE_DIR") or data.get("state_dir")
        self.state = Path(state).expanduser() if state else self.main / ".kitchen"
        if "tracks" not in data and "split" in data:  # single-track shorthand
            data["tracks"] = {"main": {"split": data["split"], "metric": data.get("metric")}}
        self.splits = data.get("splits") or {}
        self.tracks = data.get("tracks") or {}
        self.jobs = data.get("jobs") or {}
        _validate(self)

    @classmethod
    def load(cls, start: Path | None = None) -> "Config":
        here = Path(start or Path.cwd()).resolve()
        for d in (here, *here.parents):
            if (d / "kitchen.yaml").is_file():
                data = yaml.safe_load((d / "kitchen.yaml").read_text()) or {}
                if not isinstance(data, dict):
                    raise ConfigError("kitchen.yaml must be a mapping")
                return cls(d, data)
        raise ConfigError(f"no kitchen.yaml in {here} or its parents (run `kitchen init`)")

    def path(self, p) -> Path:
        """A project path; falls back to the main checkout (a worktree may lack .venv or data)."""
        q = Path(p).expanduser()
        if q.is_absolute():
            return q
        return next((b / q for b in (self.root, self.main) if (b / q).exists()), self.root / q)

    def track(self, name: str | None) -> tuple[str, dict]:
        if name is None:
            if len(self.tracks) != 1:
                raise ConfigError("pass --track (kitchen.yaml has %d tracks)" % len(self.tracks))
            name = next(iter(self.tracks))
        if name not in self.tracks:
            raise ConfigError(f"unknown track {name!r} (have: {', '.join(self.tracks) or 'none'})")
        return name, self.tracks[name]

    def default_track(self) -> str | None:
        return next(iter(self.tracks)) if len(self.tracks) == 1 else None

    @staticmethod
    def metric_spec(track: dict) -> dict:
        m = track.get("metric")
        return {"name": m} if isinstance(m, str) else dict(m or {})

    def python(self) -> str:
        py = str(self.data.get("python", "python3"))
        return str(self.path(py)) if "/" in py and " " not in py else py

    def render(self, job: str, overrides: dict) -> tuple[str, dict, str | None, float | None]:
        """Job command with {python} and {param} filled -> (cmd, params, track, timeout)."""
        if job not in self.jobs:
            raise ConfigError(f"unknown job {job!r} (have: {', '.join(self.jobs) or 'none'})")
        spec = self.jobs[job]
        params = {**(spec.get("params") or {}), **overrides}
        values = {k: shlex.quote(str(v)) for k, v in params.items()}
        try:
            cmd = spec["cmd"].format(**values, python=self.python())
        except KeyError as e:
            raise ConfigError(f"job {job!r} uses {{{e.args[0]}}} but has no such param") from None
        timeout = spec.get("timeout") or (self.data.get("run") or {}).get("timeout")
        return cmd, params, spec.get("track") or self.default_track(), duration(timeout)

    def env(self) -> dict:
        """os.environ + env_file defaults + the configured PYTHONPATH prefix."""
        env = dict(os.environ)
        env_file = self.data.get("env_file")
        if env_file and self.path(env_file).is_file():
            for line in self.path(env_file).read_text().splitlines():
                line = line.strip().removeprefix("export ")
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env.setdefault(key.strip(), value.strip().strip("'\""))
        prefix = [str(self.path(p)) for p in self.data.get("pythonpath") or []]
        if prefix:
            env["PYTHONPATH"] = os.pathsep.join(
                prefix + [env["PYTHONPATH"]] * bool(env.get("PYTHONPATH"))
            )
        return env


def _validate(cfg: Config) -> None:
    errs = []
    for name, s in cfg.splits.items():
        if not isinstance(s, dict):
            errs.append(f"splits.{name} must be a mapping")
            continue
        errs += [
            f"splits.{name}: missing `{k}`"
            for k in ("data", "id", "target", "scheme")
            if k not in s
        ]
        if s.get("scheme") not in SCHEMES:
            errs.append(f"splits.{name}.scheme must be one of {', '.join(sorted(SCHEMES))}")
        elif s["scheme"] in ("group", "logo", "forward") and not s.get("group"):
            errs.append(f"splits.{name}: scheme {s['scheme']} needs `group`")
    for name, t in cfg.tracks.items():
        if not isinstance(t, dict) or t.get("split") not in cfg.splits:
            errs.append(f"tracks.{name}.split must name an entry in `splits`")
        elif not t.get("metric"):
            errs.append(f"tracks.{name}: missing `metric`")
    for name, j in cfg.jobs.items():
        if not isinstance(j, dict) or not isinstance(j.get("cmd"), str):
            errs.append(f"jobs.{name}: needs a `cmd` string")
        elif j.get("track") and j["track"] not in cfg.tracks:
            errs.append(f"jobs.{name}.track {j['track']!r} is not a track")
    if errs:
        raise ConfigError("kitchen.yaml: " + "; ".join(errs))
