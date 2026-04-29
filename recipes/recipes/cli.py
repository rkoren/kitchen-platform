"""recipes CLI entry point."""
import shutil
import subprocess
import sys
from pathlib import Path

import typer
import yaml
from jinja2 import Environment, FileSystemLoader
from rich.console import Console

from recipes.generators import generate_resource
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
        console.print("[red]error:[/red] terraform not found on PATH. Install via: brew install hashicorp/tap/terraform")
        return 1

    proc = subprocess.Popen(
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


def _tf_init(spec: RecipeSpec, workspace: Path, state_bucket: str) -> bool:
    """Run terraform init with S3 backend config. Returns True on success."""
    console.print(f"\n[bold]→ terraform init[/bold]  (workspace: {workspace})\n")
    state_key = f"{spec.name}/terraform.tfstate"
    rc = _run_tf([
        "init",
        f"-backend-config=bucket={state_bucket}",
        f"-backend-config=key={state_key}",
        f"-backend-config=region={spec.region}",
        "-reconfigure",
    ], workspace)
    return rc == 0


# ── Commands ───────────────────────────────────────────────────────────────────

@app.command()
def generate(
    spec_path: str = typer.Argument(..., metavar="SPEC", help="Path to YAML spec file"),
    out: str = typer.Option("./tf", help="Output directory for generated configs"),
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


@app.command()
def validate(
    spec_path: str = typer.Argument(..., metavar="SPEC", help="Path to YAML spec file"),
):
    """Validate a YAML spec without generating any files."""
    _load_spec(spec_path)
    console.print("[green]✓[/green] spec is valid")


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

    console.print(f"\n[bold]recipes apply[/bold]  spec=[cyan]{spec_path}[/cyan]  project=[cyan]{spec.name}[/cyan]")
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

    console.print(f"\n[bold]→ terraform apply[/bold]\n")
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

    console.print(f"\n[bold red]recipes destroy[/bold red]  spec=[cyan]{spec_path}[/cyan]  project=[cyan]{spec.name}[/cyan]")
    console.print(f"[dim]state: s3://{state_bucket}/{spec.name}/terraform.tfstate[/dim]\n")

    _refresh_tf_files(spec, workspace)

    if not yes:
        typer.confirm(
            f"[bold red]Destroy all resources in '{spec.name}'?[/bold red] This cannot be undone.",
            abort=True,
        )

    if not _tf_init(spec, workspace, state_bucket):
        raise typer.Exit(1)

    console.print(f"\n[bold]→ terraform destroy[/bold]\n")
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
