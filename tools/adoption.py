#!/usr/bin/env python3
"""Measure what share of an agent's train/eval work goes through kitchen.

Reads Claude Code transcripts (~/.claude/projects/<slug>/**/*.jsonl, subagents included),
extracts every Bash tool call and classifies it. Shell text is parsed quote-aware (shlex) after
heredoc bodies are set aside, so text inside strings, grep patterns or heredocs never counts as
a command.

  routed    train/eval work run through kitchen
              2.0: kitchen run | predict | blend | submit | kernel push
              1.x: kitchen run | sweep | score | stage | submit | menu run
  unrouted  the same kind of work done outside kitchen
              - python script or `-m` module whose name marks a train/eval step
                (train, fit, eval, score, cv, backtest, sweep, tune, predict, infer, bench,
                 ablat, experiment, oof, holdout, calibrat, blend, ensemble)
              - python heredoc / `python -c` whose *executed* code (string literals and comments
                removed) fits or scores a model: .fit(, .predict(, cross_val*(, brier*(,
                log_loss(, roc_auc*(, lgb.train(, xgb.train(, optuna., KFold(, backtest*( ...
                A read-modify-write of a file is an edit, not training.
              - notebook execution (jupyter ... --execute, papermill)
              - raw `kaggle kernels push` / `kaggle competitions submit`
  other     everything else (tests, lint, installs, git, inspection, edits, data fetching, ...)

A call is routed if any of its commands is routed, else unrouted if any is unrouted.
adoption = routed / (routed + unrouted). Output carries counts only (no command text) unless
--sample is given for spot-checking.

usage: tools/adoption.py --project ~/cbb-model [--since YYYY-MM-DD] [--until YYYY-MM-DD]
                         [--sample N] [--json]
"""

import argparse
import io
import json
import random
import re
import shlex
import sys
import tokenize
from pathlib import Path

RULE_VERSION = "1"  # bump on ANY classification change, then recompute the baseline with it
ROUTED_VERBS = {
    "run",
    "sweep",
    "score",
    "stage",
    "submit",
    "predict",
    "blend",
    "kernel push",
    "menu run",
}
TRAIN_WORDS = re.compile(
    r"(?<![a-z])(train|fit|eval|score|cv|backtest|sweep|tun(e|ing)|predict|infer|bench|ablat"
    r"|experiment|oof|holdout|calibrat|blend|ensemble)",
    re.I,
)
FIT_CALLS = re.compile(
    r"\.fit\(|\.predict(_proba)?\(|cross_val\w*\(|train_test_split\(|brier\w*\(|log_loss\("
    r"|roc_auc\w*\(|accuracy_score\(|mean_squared_error\(|lgb\.train\(|xgb\.train\(|optuna\."
    r"|KFold\(|LeaveOneGroupOut\(|backtest\w*\("
)
EDIT = re.compile(r"(read_text\(\)|\.read\(\)).*?\.replace\(.*?(write_text\(|\.write\()", re.S)
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1")
WRAPPERS = {"time", "nohup", "caffeinate", "timeout", "env", "exec", "command", "sudo"}
RUNNERS = {"uv", "poetry", "conda", "pipx"}
VALUE_FLAGS = {"--directory", "--project", "--with", "--python", "-n", "--name", "--env-file", "-p"}
SHELLS = {"bash", "sh", "zsh"}
PYTHON = re.compile(r"python(\d+(\.\d+)?)?$")
_SKIP_TOKENS = {tokenize.STRING, tokenize.COMMENT}


def executed_code(code: str) -> str:
    """Python source minus string literals, f-strings and comments: code being *written* to a
    file lives in strings, so only what remains actually runs."""
    try:
        out, depth = [], 0
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
            name = tokenize.tok_name.get(t.type, "")
            if name == "FSTRING_START":
                depth += 1
            elif name == "FSTRING_END":
                depth -= 1
                continue
            if depth or t.type in _SKIP_TOKENS:
                continue
            out.append(t.string)
        return "".join(out)
    except (tokenize.TokenError, SyntaxError):
        code = re.sub(r"(?s)'''.*?'''|\"\"\".*?\"\"\"", "", code)
        return re.sub(r"\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", "", code)


def split_heredocs(command: str) -> tuple[str, str]:
    """(shell text with heredoc bodies removed, concatenated heredoc bodies)."""
    lines, shell, bodies, i = command.split("\n"), [], [], 0
    while i < len(lines):
        shell.append(lines[i])
        m, i = HEREDOC.search(lines[i]), i + 1
        if m:
            start = i
            while i < len(lines) and lines[i].strip() != m.group(2):
                i += 1
            bodies.append("\n".join(lines[start:i]))
            i += 1
    return "\n".join(shell), "\n".join(bodies)


def commands(shell: str) -> list[list[str]]:
    """Quote-aware split of shell text into token lists on control operators (&& || ; | & ( ))."""
    out, buf = [], ""
    for line in shell.replace("\\\n", " ").split("\n"):
        buf = f"{buf}\n{line}" if buf else line
        try:
            lex = shlex.shlex(buf, posix=True, punctuation_chars=True)
            lex.whitespace_split = True
            toks = list(lex)
        except ValueError:  # a quoted string continues on the next line
            continue
        buf, cur = "", []
        for t in toks:
            if t and set(t) <= set("();&|"):
                if cur:
                    out.append(cur)
                cur = []
            else:
                cur.append(t)
        if cur:
            out.append(cur)
    if buf:  # quotes never balanced: fall back to a naive split
        out += [s.split() for s in re.split(r"&&|\|\||;|\||\n", buf) if s.strip()]
    return out


def _head(tokens: list[str]) -> int:
    """Index of the real command word, skipping env assignments and wrapper/runner prefixes."""
    i = 0
    while i < len(tokens):
        t, prev = tokens[i], tokens[i - 1] if i else ""
        if re.match(r"^\w+=", t) or t in WRAPPERS or t in RUNNERS or re.fullmatch(r"\d+[smh]?", t):
            i += 1
        elif t == "run" and prev in RUNNERS:
            i += 1
        elif t.startswith("-") and (
            prev in WRAPPERS | RUNNERS | {"run"} or prev.startswith("-") or prev in VALUE_FLAGS
        ):
            i += 2 if t in VALUE_FLAGS else 1
        else:
            return i
    return i


def classify_command(tokens: list[str], body: str) -> str:
    i = _head(tokens)
    if i >= len(tokens):
        return "other"
    cmd, rest = tokens[i].rsplit("/", 1)[-1], tokens[i + 1 :]
    if cmd in SHELLS and "-c" in rest:
        j = rest.index("-c")
        return classify(rest[j + 1]) if j + 1 < len(rest) else "other"
    if cmd == "kitchen":
        while rest[:1] in (["-C"], ["--project"]):
            rest = rest[2:]
        verb = " ".join(rest[:2]) if rest[:1] in (["kernel"], ["menu"]) else " ".join(rest[:1])
        return "routed" if verb in ROUTED_VERBS else "other"
    if PYTHON.match(cmd):
        if rest[:1] == ["-m"] and len(rest) > 1:
            return "unrouted" if TRAIN_WORDS.search(rest[1]) and rest[1] != "pytest" else "other"
        if rest[:1] == ["-c"] and len(rest) > 1:
            code = rest[1]
        elif rest[:1] == ["-"] or any(t.startswith("<<") for t in rest):
            code = body
        else:
            script = next((t for t in rest if not t.startswith("-") and not set(t) & set("<>")), "")
            return (
                "unrouted"
                if script.endswith(".py") and TRAIN_WORDS.search(Path(script).name)
                else "other"
            )
        return (
            "unrouted"
            if FIT_CALLS.search(executed_code(code)) and not EDIT.search(code)
            else "other"
        )
    if (cmd == "jupyter" and "--execute" in rest) or cmd == "papermill":
        return "unrouted"
    if cmd == "kaggle" and (
        {"kernels", "push"} <= set(rest) or {"competitions", "submit"} <= set(rest)
    ):
        return "unrouted"
    return "other"


def classify(command: str) -> str:
    shell, body = split_heredocs(command)
    labels = {classify_command(toks, body) for toks in commands(shell)}
    return "routed" if "routed" in labels else "unrouted" if "unrouted" in labels else "other"


def bash_calls(transcript_dir: Path, since: str | None, until: str | None):
    for f in sorted(transcript_dir.rglob("*.jsonl")):
        for line in f.open(encoding="utf-8", errors="replace"):
            try:
                o = json.loads(line)
            except ValueError:
                continue
            ts = (o.get("timestamp") or "")[:10]
            if (since and ts < since) or (until and ts > until):
                continue
            msg = o.get("message")
            content = msg.get("content") if isinstance(msg, dict) else None
            for c in content if isinstance(content, list) else []:
                if isinstance(c, dict) and c.get("type") == "tool_use" and c.get("name") == "Bash":
                    yield (c.get("input") or {}).get("command", "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project", required=True, help="project directory the sessions ran in")
    ap.add_argument("--transcripts", help="override the transcript directory")
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--sample", type=int, default=0, help="print N example commands per class")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    proj = Path(a.project).expanduser().resolve()
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(proj))
    tdir = Path(a.transcripts) if a.transcripts else Path.home() / ".claude" / "projects" / slug
    if not tdir.is_dir():
        print(f"no transcripts at {tdir}", file=sys.stderr)
        return 2
    by_class: dict[str, list[str]] = {"routed": [], "unrouted": [], "other": []}
    for cmd in bash_calls(tdir, a.since, a.until):
        by_class[classify(cmd)].append(cmd)
    r, u = len(by_class["routed"]), len(by_class["unrouted"])
    out = {
        "rule_version": RULE_VERSION,
        "project": proj.name,
        "since": a.since,
        "until": a.until,
        "bash_calls": sum(map(len, by_class.values())),
        "routed": r,
        "unrouted": u,
        "other": len(by_class["other"]),
        "adoption": round(r / (r + u), 4) if r + u else None,
    }
    print(json.dumps(out, indent=2) if a.json else " ".join(f"{k}={v}" for k, v in out.items()))
    rng = random.Random(0)
    for label, cmds in by_class.items() if a.sample else []:
        print(f"\n== sample: {label} ({len(cmds)})")
        for c in rng.sample(cmds, min(a.sample, len(cmds))):
            print("  ", c.replace("\n", " ⏎ ")[:170])
    return 0


if __name__ == "__main__":
    sys.exit(main())
