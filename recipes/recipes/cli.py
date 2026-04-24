"""recipes CLI entry point."""
from pathlib import Path

import typer
import yaml
from jinja2 import Environment, FileSystemLoader
from rich.console import Console

from recipes.generators import generate_resource
from recipes.schema import RecipeSpec

app = typer.Typer(help="YAML spec → Terraform config generator.")
console = Console()

_TEMPLATES_DIR = Path(__file__).parent / "templates"


@app.command()
def generate(
    spec_path: str = typer.Argument(..., metavar="SPEC", help="Path to YAML spec file"),
    out: str = typer.Option("./tf", help="Output directory for generated configs"),
):
    """Generate Terraform configs from a YAML spec."""
    spec_file = Path(spec_path)
    if not spec_file.exists():
        console.print(f"[red]error:[/red] spec file not found: {spec_path}")
        raise typer.Exit(1)

    raw = yaml.safe_load(spec_file.read_text(encoding="utf-8"))
    spec = RecipeSpec.model_validate(raw)

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
    spec_file = Path(spec_path)
    if not spec_file.exists():
        console.print(f"[red]error:[/red] spec file not found: {spec_path}")
        raise typer.Exit(1)

    raw = yaml.safe_load(spec_file.read_text(encoding="utf-8"))
    RecipeSpec.model_validate(raw)
    console.print("[green]✓[/green] spec is valid")


def _write_provider(region: str, out_dir: Path) -> None:
    env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR), keep_trailing_newline=True)
    content = env.get_template("provider.tf.j2").render(region=region)
    (out_dir / "provider.tf").write_text(content, encoding="utf-8")
    console.print("  [green]✓[/green] provider.tf")


if __name__ == "__main__":
    app()
