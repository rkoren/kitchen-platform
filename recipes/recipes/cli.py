"""recipes CLI entry point."""

import json
import shutil
import subprocess
from pathlib import Path

import typer
import yaml
from jinja2 import Environment, FileSystemLoader
from rich.console import Console

from recipes.generators import generate_resource
from recipes.menu import is_menu, recipe_spec_from_menu
from recipes.schema import RecipeSpec

app = typer.Typer(help="YAML spec → Terraform config generator and provisioner.")
console = Console()

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_WORKSPACE_ROOT = Path.home() / ".recipes" / "tf"


# ── Shared helpers ─────────────────────────────────────────────────────────────


def _load_spec(spec_path: str) -> RecipeSpec:
    spec_file = Path(spec_path)
    if not spec_file.exists():
        console.print(f"[red]error:[/red] spec file not found: {spec_path}")
        raise typer.Exit(1)
    raw = yaml.safe_load(spec_file.read_text(encoding="utf-8"))
    # A menu.yaml (unified manifest) carries a `recipes:` map; project its infra recipes
    # into a RecipeSpec, keyed by `project` for state. A standalone spec has `resources:`.
    if is_menu(raw):
        return recipe_spec_from_menu(raw)
    return RecipeSpec.model_validate(raw)


def _generate_to(spec: RecipeSpec, out_dir: Path) -> None:
    """Write provider.tf + one .tf per resource into out_dir."""
    _write_provider(spec.region, out_dir)
    for resource in spec.resources:
        tf_content = generate_resource(resource, all_resources=spec.resources)
        filename = f"{resource.type.replace('_', '-')}-{resource.name}.tf"
        (out_dir / filename).write_text(tf_content, encoding="utf-8")


def _workspace(spec_name: str) -> Path:
    """Persistent workspace directory for a named spec.

    Provider plugins are cached here across runs; only .tf files are refreshed.
    """
    ws = _WORKSPACE_ROOT / spec_name
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _refresh_tf_files(spec: RecipeSpec, workspace: Path) -> None:
    """Replace only *.tf files in the workspace, preserving .terraform cache."""
    for f in workspace.glob("*.tf"):
        f.unlink()
    _generate_to(spec, workspace)


def _run_tf(args: list[str], workspace: Path) -> int:
    """Stream terraform output line-by-line. Returns exit code."""
    tf = shutil.which("terraform")
    if not tf:
        console.print(
            "[red]error:[/red] terraform not found on PATH. Install via: brew install hashicorp/tap/terraform"
        )
        return 1

    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        [tf] + args,
        cwd=workspace,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        console.print(line, end="")
    proc.wait()
    return proc.returncode


def _terraform_or_exit() -> str:
    """Return the terraform binary path, or exit 1 with an install hint."""
    tf = shutil.which("terraform")
    if not tf:
        console.print(
            "[red]error:[/red] terraform not found on PATH. Install via: brew install hashicorp/tap/terraform"
        )
        raise typer.Exit(1)
    return tf


def _fmt_check(out_dir: Path) -> None:
    """Fail if any generated .tf in out_dir is not canonically formatted (R-004)."""
    tf = _terraform_or_exit()
    rc = subprocess.run([tf, "fmt", "-check", "-diff", str(out_dir)], check=False).returncode
    if rc != 0:
        console.print("[red]✗[/red] generated HCL is not canonically formatted (see diff above)")
        raise typer.Exit(rc)
    console.print("  [green]✓[/green] terraform fmt")


def _validate_hcl(out_dir: Path) -> None:
    """Run `terraform validate` on out_dir (R-004) — downloads providers, needs network."""
    tf = _terraform_or_exit()
    init = subprocess.run(
        [tf, f"-chdir={out_dir}", "init", "-backend=false", "-input=false", "-no-color"],
        check=False,
        capture_output=True,
        text=True,
    )
    if init.returncode != 0:
        console.print(f"[red]✗[/red] terraform init failed:\n{init.stderr}")
        raise typer.Exit(init.returncode)
    rc = subprocess.run([tf, f"-chdir={out_dir}", "validate", "-no-color"], check=False).returncode
    if rc != 0:
        console.print("[red]✗[/red] terraform validate failed")
        raise typer.Exit(rc)
    console.print("  [green]✓[/green] terraform validate")


def _tf_init(spec: RecipeSpec, workspace: Path, state_bucket: str) -> bool:
    """Run terraform init with S3 backend config. Returns True on success."""
    console.print(f"\n[bold]→ terraform init[/bold]  (workspace: {workspace})\n")
    state_key = f"{spec.name}/terraform.tfstate"
    rc = _run_tf(
        [
            "init",
            f"-backend-config=bucket={state_bucket}",
            f"-backend-config=key={state_key}",
            f"-backend-config=region={spec.region}",
            "-reconfigure",
        ],
        workspace,
    )
    return rc == 0


# ── Commands ───────────────────────────────────────────────────────────────────


@app.command()
def generate(
    spec_path: str = typer.Argument(..., metavar="SPEC", help="Path to YAML spec file"),
    out: str = typer.Option("./tf", help="Output directory for generated configs"),
    check: bool = typer.Option(
        False, "--check", help="Run `terraform fmt -check` on the generated HCL"
    ),
    validate: bool = typer.Option(
        False,
        "--validate",
        help="Run `terraform validate` on the generated HCL (downloads providers; needs network)",
    ),
):
    """Generate Terraform configs from a YAML spec."""
    spec = _load_spec(spec_path)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_provider(spec.region, out_dir)
    for resource in spec.resources:
        tf_content = generate_resource(resource, all_resources=spec.resources)
        filename = f"{resource.type.replace('_', '-')}-{resource.name}.tf"
        tf_file = out_dir / filename
        tf_file.write_text(tf_content, encoding="utf-8")
        console.print(f"  [green]✓[/green] {filename} [dim]({resource.type})[/dim]")
    total = len(spec.resources) + 1  # +1 for provider.tf
    console.print(f"\n[bold]Generated {total} file(s) → {out_dir}[/bold]")

    # R-004: verify the generated HCL is canonical (and optionally valid).
    if check or validate:
        _fmt_check(out_dir)
    if validate:
        _validate_hcl(out_dir)


@app.command()
def validate(
    spec_path: str = typer.Argument(..., metavar="SPEC", help="Path to YAML spec file"),
):
    """Validate a YAML spec without generating any files."""
    _load_spec(spec_path)
    console.print("[green]✓[/green] spec is valid")


@app.command()
def doctor(
    state_bucket: str = typer.Option(
        None,
        envvar="RECIPES_STATE_BUCKET",
        help="Optional: check read access to this Terraform state bucket",
    ),
):
    """Pre-flight checks: Terraform, AWS credentials, and (optionally) state-bucket access.

    Exits non-zero if a hard requirement (Terraform, AWS credentials) is missing, so it
    can gate a CI job before `recipes plan`/`apply`.
    """
    ok = True

    # Terraform — required, and >= 1.10 for the S3-native state locking the backend uses.
    tf = shutil.which("terraform")
    if tf:
        ver = subprocess.run(
            [tf, "version", "-json"], check=False, capture_output=True, text=True
        )
        version = ""
        if ver.returncode == 0:
            try:
                version = json.loads(ver.stdout).get("terraform_version", "")
            except json.JSONDecodeError:
                version = ""
        console.print(f"[green]✓[/green] terraform{f' {version}' if version else ''}")
        if version and tuple(int(p) for p in version.split(".")[:2]) < (1, 10):
            console.print(
                "  [yellow]⚠[/yellow] terraform < 1.10 — the generated backend uses S3-native "
                "state locking (use_lockfile), which needs >= 1.10"
            )
    else:
        console.print("[red]✗[/red] terraform not found on PATH")
        ok = False

    # AWS credentials — required for plan/apply; checked via the AWS CLI.
    aws = shutil.which("aws")
    if not aws:
        console.print(
            "[yellow]⚠[/yellow] aws CLI not found — skipping credential check "
            "(Terraform can still use environment credentials)"
        )
    else:
        ident = subprocess.run(
            [aws, "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
            check=False,
            capture_output=True,
            text=True,
        )
        if ident.returncode == 0:
            console.print(f"[green]✓[/green] AWS credentials (account {ident.stdout.strip()})")
        else:
            console.print("[red]✗[/red] AWS credentials not found or invalid")
            ok = False

    # State bucket — optional; only checked when configured.
    if not state_bucket:
        console.print(
            "[yellow]⚠[/yellow] no state bucket configured "
            "(pass --state-bucket or set RECIPES_STATE_BUCKET to check access)"
        )
    elif not aws:
        console.print("[yellow]⚠[/yellow] state-bucket check skipped (aws CLI not found)")
    else:
        head = subprocess.run(
            [aws, "s3api", "head-bucket", "--bucket", state_bucket],
            check=False,
            capture_output=True,
            text=True,
        )
        if head.returncode == 0:
            console.print(f"[green]✓[/green] state bucket accessible ({state_bucket})")
        else:
            console.print(f"[red]✗[/red] cannot access state bucket: {state_bucket}")
            ok = False

    if not ok:
        raise typer.Exit(1)
    console.print("\n[bold green]All checks passed[/bold green]")


def _recipe_json_schema() -> dict:
    """RecipeSpec as a standalone JSON Schema document (draft 2020-12)."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **RecipeSpec.model_json_schema(),
    }


@app.command()
def schema(
    out: str = typer.Option(
        None, "--out", "-o", metavar="PATH", help="Write schema to a file instead of stdout"
    ),
):
    """Export the recipe YAML JSON Schema (draft 2020-12).

    With no --out, prints the schema to stdout so it can be redirected or piped
    (e.g. `recipes schema > recipe.schema.json`). Use the emitted file as a
    `$schema` reference for editor validation and autocompletion of spec YAML.
    """
    text = json.dumps(_recipe_json_schema(), indent=2) + "\n"
    if out:
        Path(out).write_text(text, encoding="utf-8")
        console.print(f"[green]✓[/green] wrote JSON Schema → {out}")
    else:
        # Plain stdout only — no decoration — so the output is a valid schema file.
        typer.echo(text, nl=False)


@app.command()
def plan(
    spec_path: str = typer.Argument(..., metavar="SPEC", help="Path to YAML spec file"),
    state_bucket: str = typer.Option(
        ...,
        envvar="RECIPES_STATE_BUCKET",
        help="S3 bucket for Terraform state (or set RECIPES_STATE_BUCKET)",
    ),
):
    """Preview the changes a YAML spec would make, without applying them.

    Generates Terraform configs, initialises the S3 backend, and runs
    `terraform plan` — a read-only preview of what `recipes apply` would do.
    State is read from s3://<state-bucket>/<spec-name>/terraform.tfstate.
    """
    spec = _load_spec(spec_path)
    workspace = _workspace(spec.name)

    console.print(
        f"\n[bold]recipes plan[/bold]  spec=[cyan]{spec_path}[/cyan]  project=[cyan]{spec.name}[/cyan]"
    )
    console.print(f"[dim]state: s3://{state_bucket}/{spec.name}/terraform.tfstate[/dim]")
    console.print(f"[dim]workspace: {workspace}[/dim]\n")

    _refresh_tf_files(spec, workspace)

    for resource in spec.resources:
        console.print(f"  [green]✓[/green] {resource.type}  [dim]{resource.name}[/dim]")
    console.print()

    if not _tf_init(spec, workspace, state_bucket):
        raise typer.Exit(1)

    console.print("\n[bold]→ terraform plan[/bold]\n")
    rc = _run_tf(["plan"], workspace)
    if rc != 0:
        console.print("\n[red]plan failed[/red]")
        raise typer.Exit(rc)

    console.print(f"\n[bold green]✓ plan complete[/bold green]  [{spec.name}]")


@app.command()
def apply(
    spec_path: str = typer.Argument(..., metavar="SPEC", help="Path to YAML spec file"),
    state_bucket: str = typer.Option(
        ...,
        envvar="RECIPES_STATE_BUCKET",
        help="S3 bucket for Terraform state (or set RECIPES_STATE_BUCKET)",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Provision AWS resources defined in a YAML spec.

    Generates Terraform configs, initialises the S3 backend, and applies.
    Resources are provisioned in dependency order — Terraform resolves the graph.
    State is stored at s3://<state-bucket>/<spec-name>/terraform.tfstate.
    """
    spec = _load_spec(spec_path)
    workspace = _workspace(spec.name)

    console.print(
        f"\n[bold]recipes apply[/bold]  spec=[cyan]{spec_path}[/cyan]  project=[cyan]{spec.name}[/cyan]"
    )
    console.print(f"[dim]state: s3://{state_bucket}/{spec.name}/terraform.tfstate[/dim]")
    console.print(f"[dim]workspace: {workspace}[/dim]\n")

    _refresh_tf_files(spec, workspace)

    for resource in spec.resources:
        console.print(f"  [green]✓[/green] {resource.type}  [dim]{resource.name}[/dim]")
    console.print()

    if not yes:
        typer.confirm("Apply these changes?", abort=True)

    if not _tf_init(spec, workspace, state_bucket):
        raise typer.Exit(1)

    console.print("\n[bold]→ terraform apply[/bold]\n")
    rc = _run_tf(["apply", "-auto-approve"], workspace)
    if rc != 0:
        console.print("\n[red]apply failed[/red]")
        raise typer.Exit(rc)

    console.print(f"\n[bold green]✓ apply complete[/bold green]  [{spec.name}]")


@app.command()
def destroy(
    spec_path: str = typer.Argument(..., metavar="SPEC", help="Path to YAML spec file"),
    state_bucket: str = typer.Option(
        ...,
        envvar="RECIPES_STATE_BUCKET",
        help="S3 bucket for Terraform state (or set RECIPES_STATE_BUCKET)",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Destroy all AWS resources defined in a YAML spec.

    Always prompts for confirmation unless --yes is passed.
    """
    spec = _load_spec(spec_path)
    workspace = _workspace(spec.name)

    console.print(
        f"\n[bold red]recipes destroy[/bold red]  spec=[cyan]{spec_path}[/cyan]  project=[cyan]{spec.name}[/cyan]"
    )
    console.print(f"[dim]state: s3://{state_bucket}/{spec.name}/terraform.tfstate[/dim]\n")

    _refresh_tf_files(spec, workspace)

    if not yes:
        typer.confirm(
            f"[bold red]Destroy all resources in '{spec.name}'?[/bold red] This cannot be undone.",
            abort=True,
        )

    if not _tf_init(spec, workspace, state_bucket):
        raise typer.Exit(1)

    console.print("\n[bold]→ terraform destroy[/bold]\n")
    rc = _run_tf(["destroy", "-auto-approve"], workspace)
    if rc != 0:
        console.print("\n[red]destroy failed[/red]")
        raise typer.Exit(rc)

    console.print(f"\n[bold green]✓ destroy complete[/bold green]  [{spec.name}]")


# ── Internal ───────────────────────────────────────────────────────────────────


def _write_provider(region: str, out_dir: Path) -> None:
    env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR), keep_trailing_newline=True)
    content = env.get_template("provider.tf.j2").render(region=region)
    (out_dir / "provider.tf").write_text(content, encoding="utf-8")
    console.print("  [green]✓[/green] provider.tf")


if __name__ == "__main__":
    app()
