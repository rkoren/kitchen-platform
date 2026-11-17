"""kitchen command line. pandas/numpy load lazily so `kitchen show --brief` stays fast."""

import argparse
import json
import os
import sys
from pathlib import Path

from kitchen import __version__
from kitchen.config import Config, KitchenError, NeedsHuman, duration, git
from kitchen.ledger import Ledger

COMMAND_PATHS = ("init", "split", "run", "status", "show", "compare", "mark", "promote")
OPS = ("draft", "improve", "debug", "blend", "confirm")
FAILED, INCOMPARABLE, TIMEOUT = 10, 5, 124
FLAG, OPT = {"action": "store_true"}, {"nargs": "?"}
ARGS = {  # command: (help, [(flag, argparse kwargs), ...])
    "init": (
        "scaffold kitchen.yaml; --agent adds the Claude Code skill, hook and permissions",
        [("--agent", FLAG)],
    ),
    "split": ("build a frozen split, or --list them", [("name", OPT), ("--list", FLAG)]),
    "run": (
        "execute and record a run: a job, or any command after --",
        [
            ("job", OPT),
            ("-p", {"dest": "params", "action": "append", "default": [], "metavar": "K=V[,V2]"}),
            ("-m", {"dest": "hypothesis", "metavar": "TEXT", "help": "what this run tests"}),
            ("--parent", {"metavar": "RUN"}),
            ("--op", {"choices": OPS}),
            ("--track", {}),
            ("--detach", {**FLAG, "help": "return at once; the run outlives this session"}),
            ("--timeout", {"metavar": "DUR", "help": "e.g. 30m (default: run.timeout)"}),
            (
                "--scratch",
                {**FLAG, "help": "ad-hoc code (e.g. a heredoc); promotable only with --force"},
            ),
            ("--import", {"dest": "import_dir", "type": Path, "metavar": "DIR"}),
        ],
    ),
    "status": (
        "close finished/lost runs; --wait for or --watch one",
        [("--wait", {"metavar": "RUN"}), ("--watch", {"metavar": "RUN"}), ("--timeout", {})],
    ),
    "show": (
        "board (default), a run, its --lineage, the --brief, or --champion paths",
        [
            ("run", OPT),
            ("--brief", FLAG),
            ("--hook", FLAG),
            ("--lineage", FLAG),
            ("--champion", FLAG),
            ("--track", {}),
        ],
    ),
    "compare": (
        "verdict for run A vs B (default B: A's champion)",
        [("a", {}), ("b", OPT), ("--md", {**FLAG, "help": "markdown for a PR body"})],
    ),
    "mark": (
        "note, quarantine, retire or mark a baseline",
        [
            ("run", {}),
            ("--note", {}),
            ("--quarantine", {"metavar": "REASON"}),
            ("--retire", FLAG),
            ("--baseline", FLAG),
        ],
    ),
    "promote": (
        "make a run its track's champion (needs a `better` verdict)",
        [("run", {}), ("--force", {"metavar": "REASON", "help": "skip the checks; recorded"})],
    ),
}


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)  # accepted before or after the subcommand
    common.add_argument("--json", **FLAG, default=argparse.SUPPRESS, help="machine-readable output")
    common.add_argument("-C", type=Path, default=argparse.SUPPRESS, metavar="DIR")
    p = argparse.ArgumentParser(
        prog="kitchen", parents=[common], description="Record and compare ML runs."
    )
    p.add_argument("--version", action="version", version=f"kitchen {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    for name, (help_, args) in ARGS.items():
        s = sub.add_parser(name, parents=[common], help=help_, description=help_)
        for flag, kw in args:
            s.add_argument(flag, **kw)
    return p


def _load(a) -> tuple[Config, Ledger]:
    cfg = Config.load(getattr(a, "C", None))
    return cfg, Ledger(cfg.state)


def emit(a, data, text: str) -> None:
    print(json.dumps(data, indent=2, default=str) if a.json else text)


def _tail(led: Ledger, rid: str, n: int = 15) -> str:
    log = led.run_dir(rid) / "log.txt"
    return (
        "\n".join(log.read_text(errors="replace").splitlines()[-n:]) if log.exists() else "(no log)"
    )


def _public(r: dict, led: Ledger) -> dict:
    out = {**r, "dir": str(led.run_dir(r["id"]))}
    if r["status"] in ("failed", "timeout", "lost"):
        out["log_tail"] = _tail(led, r["id"])
    return out


def _code(r: dict) -> int:
    return {"ok": 0, "running": 0, "timeout": TIMEOUT}.get(r["status"], FAILED)


def cmd_init(a, _) -> int:
    from kitchen import init

    root = Path(getattr(a, "C", None) or ".").resolve()
    wrote = init.scaffold(root) + (init.install_agent(root) if a.agent else [])
    emit(a, {"wrote": wrote}, "wrote: " + (", ".join(wrote) or "nothing (already set up)"))
    return 0


def cmd_split(a, _) -> int:
    cfg, led = _load(a)
    if a.list or not a.name:
        built = led.state().splits
        rows = [
            {"name": n, "scheme": s["scheme"], "id": built.get(n, {}).get("id")}
            for n, s in cfg.splits.items()
        ]
        emit(
            a,
            rows,
            "\n".join(
                f"{r['name']:<16} {r['scheme']:<10} {r['id'] or '(not built)'}" for r in rows
            ),
        )
        return 0
    from kitchen import splits

    m = splits.ensure(cfg, led, a.name)[1]
    emit(
        a, m, f"{m['id']}: {m['rows']} rows, {m['val_rows']} validation rows in {m['folds']} folds"
    )
    return 0


def cmd_run(a, command) -> int:
    from kitchen import runner
    from kitchen.context import label

    cfg, led = _load(a)
    shared = {"parent": a.parent, "op": a.op, "hypothesis": a.hypothesis}
    if a.import_dir:
        track = a.track or cfg.default_track()
        rids = [runner.import_dir(cfg, led, a.import_dir, track=track, **shared)]
    elif a.job or command:
        echo = None if a.detach else (sys.stderr if a.json else sys.stdout)
        shared |= {"detach": a.detach, "scratch": a.scratch, "job": a.job, "shell": bool(a.job)}
        shared |= {"stdin": None if a.job else runner.read_stdin(), "echo": echo}
        adhoc = (cfg.default_track(), duration((cfg.data.get("run") or {}).get("timeout")))
        rids = []
        for over in runner.grid(a.params) or [{}]:
            fields = cfg.render(a.job, over) if a.job else (command, over, *adhoc)
            run = dict(zip(("cmd", "params", "track", "timeout"), fields))
            run["track"], run["timeout"] = (
                a.track or run["track"],
                duration(a.timeout) or run["timeout"],
            )
            rids.append(runner.start(cfg, led, **run, **shared))
    else:
        raise KitchenError("give a job name or a command after `--` (see `kitchen run -h`)")
    runs, text = [led.state().get(r) for r in rids], []
    for r in runs:
        hint = f"\n  detached: kitchen status --wait {r['id']}" if r["status"] == "running" else ""
        text.append(label(r) + hint)
        if r["status"] in ("failed", "timeout"):
            text.append("--- log tail ---\n" + _tail(led, r["id"]))
    emit(a, [_public(r, led) for r in runs], "\n".join(text))
    return max(_code(r) for r in runs)


def cmd_status(a, _) -> int:
    from kitchen import runner
    from kitchen.context import label

    cfg, led = _load(a)
    lost = runner.refresh(led)
    if a.watch or a.wait:
        out = sys.stderr if a.json else sys.stdout
        r = (
            runner.watch(led, a.watch, out)
            if a.watch
            else runner.wait(led, a.wait, duration(a.timeout))
        )
        emit(
            a, _public(r, led), label(r) + ("  (still running)" if r["status"] == "running" else "")
        )
        return TIMEOUT if r["status"] == "running" else _code(r)
    runs = sorted(led.state().runs.values(), key=lambda r: r.get("ended") or r["t"])
    live, done = (
        [r for r in runs if r["status"] == "running"],
        [r for r in runs if r["status"] != "running"],
    )
    text = [f"marked lost: {', '.join(lost)}"] * bool(lost) + [
        label(r) for r in live + done[:-6:-1]
    ]
    emit(a, {"lost": lost, "running": live, "recent": done[-5:]}, "\n".join(text) or "no runs yet")
    return 0


def cmd_show(a, _) -> int:
    from kitchen import context

    try:
        cfg, led = _load(a)
    except KitchenError:
        if a.hook:  # SessionStart hook outside a kitchen project: print nothing
            return 0
        raise
    st = led.state()
    if a.brief or a.hook:
        text = context.brief(cfg, st)
        print(text) if a.hook else emit(a, {"brief": text}, text)  # hook stdout -> session context
    elif a.champion:
        rows = {
            t: {"id": c, "path": str(led.dir / "champion" / t), "score": st.runs[c].get("score")}
            for t, c in st.champions.items()
        }
        emit(
            a,
            rows,
            "\n".join(f"{t}: {v['id']}  {v['path']}" for t, v in rows.items())
            or "no champions yet",
        )
    elif a.run and a.lineage:
        lin = context.lineage(st, a.run)
        emit(a, lin, context.render_lineage(lin))
    elif a.run:
        r = st.get(a.run)
        keys = ("job", "cmd", "params", "track", "split", "git", "score", "verdict", "notes")
        detail = json.dumps({k: r.get(k) for k in keys}, indent=2, default=str)
        emit(
            a,
            _public(r, led),
            f"{context.label(r)}\n{detail}\n--- log tail ---\n{_tail(led, a.run)}",
        )
    else:
        rows, text = context.board(cfg, st, a.track)
        emit(a, rows, text or "no tracks configured")
    return 0


def cmd_compare(a, _) -> int:
    from kitchen import stats

    cfg, led = _load(a)
    st = led.state()
    ra = st.get(a.a)
    b = a.b or st.champions.get(ra.get("track") or "")
    if not b:
        raise KitchenError(f"{a.a}'s track has no champion yet; name the run to compare against")
    rb, v = st.get(b), stats.compare(cfg, led, a.a, b)
    pa, pb = ra.get("params") or {}, rb.get("params") or {}
    params = {k: [pb.get(k), pa.get(k)] for k in sorted({*pa, *pb}) if pa.get(k) != pb.get(k)}
    shas = [(r.get("git") or {}).get("sha") for r in (rb, ra)]
    code = git(cfg.root, "diff", "--stat", *shas) if all(shas) and shas[0] != shas[1] else None
    sd, n = stats.seed_sd(st, ra)
    s = ra.get("score") or {}
    better = {"min": "lower", "max": "higher"}.get(s.get("direction"), "?")
    body = (
        v.get("reason")
        or f"{stats.ci_text(v)}, P(better) {v['p_better']}, {v['clusters']} clusters"
    )
    changed = "; ".join(f"{k}: {old} -> {new}" for k, (old, new) in params.items()) or "none"
    head = f"`{a.a}` vs `{b}` on `{s.get('split')}` ({s.get('metric')}, {better} is better)"
    if a.md:
        text = (
            f"**kitchen verdict:** {head}\n\n- verdict: **{v['verdict']}** — {body}\n- scores: "
            f"{s.get('value')} vs {(rb.get('score') or {}).get('value')}\n- params changed: {changed}"
            + (f"\n\n```\n{code}\n```" if code else "")
            + (
                f"\n\nAfter review, a human runs `kitchen promote {a.a}`."
                if v["verdict"] == "better"
                else ""
            )
        )
    else:
        text = f"{head}: {v['verdict']}\n{body}\nparams changed: {changed}" + (
            f"\nseed SD {sd:.5f} ({n} runs)" if sd else ""
        )
    emit(
        a,
        {"a": a.a, "b": b, **v, "params": params, "seed_sd": sd, "seed_runs": n, "code": code},
        text,
    )
    return INCOMPARABLE if v["verdict"] == "incomparable" else 0


def cmd_mark(a, _) -> int:
    cfg, led = _load(a)
    led.state().get(a.run)
    ev = {
        "ev": "mark",
        "id": a.run,
        **{k: v for k, v in (("note", a.note), ("quarantine", a.quarantine)) if v},
        **{k: True for k in ("retire", "baseline") if getattr(a, k)},
    }
    if len(ev) == 2:
        raise KitchenError("give --note, --quarantine, --retire or --baseline")
    led.append(ev)
    emit(a, ev, f"marked {a.run}")
    return 0


def cmd_promote(a, _) -> int:
    if os.environ.get("KITCHEN_AUTONOMOUS") == "1":
        raise NeedsHuman(
            "promotion is a human decision (propose it with `kitchen compare RUN --md`)"
        )
    from kitchen import stats

    cfg, led = _load(a)
    st = led.state()
    track = st.get(a.run).get("track")
    prev = st.champions.get(track or "")
    if not track or prev == a.run:
        raise KitchenError(
            f"{a.run} " + (f"is already the {track} champion" if track else "has no track")
        )
    problem = None if a.force else stats.blocker(cfg, led, a.run)
    if problem:
        raise KitchenError(f"won't promote {a.run}: {problem}; pass --force REASON to override")
    led.append({"ev": "promote", "id": a.run, "track": track, "prev": prev, "forced": a.force})
    link = led.set_champion(track, a.run)
    data = {"track": track, "champion": a.run, "previous": prev, "path": str(link)}
    emit(a, data, f"{track} champion: {a.run} (was {prev or 'none'}) -> {link}")
    return 0


COMMANDS = {c: globals()[f"cmd_{c}"] for c in COMMAND_PATHS}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = None
    if "--" in argv:
        i = argv.index("--")
        argv, command = argv[:i], argv[i + 1 :]
    a = parser().parse_args(argv)
    a.json = getattr(a, "json", False) or os.environ.get("KITCHEN_JSON") == "1"
    try:
        return COMMANDS[a.command](a, command)
    except KitchenError as e:
        print(
            json.dumps({"error": str(e), "code": e.code}) if a.json else f"kitchen: {e}",
            file=sys.stdout if a.json else sys.stderr,
        )
        return e.code
