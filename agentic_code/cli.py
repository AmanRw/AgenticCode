"""
CLI Interface for AgenticCode using Typer.
Provides entry points for issue resolution, benchmark evaluation, webhook server, and auditing.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from agentic_code import (
    SWEBenchEvaluator,
    audit_logger,
    get_standard_reference_instances,
    load_swebench_dataset,
    run_workflow,
    settings,
)

app = typer.Typer(
    name="agentic-code",
    help="AgenticCode: End-to-End Autonomous GitHub Issue Resolution Engine",
    add_completion=False,
)
console = Console()


@app.command()
def resolve(
    title: str = typer.Argument(..., help="Issue title or description"),
    repo_path: str = typer.Option(".", "--repo", "-r", help="Path to local target git repository"),
    issue_number: int = typer.Option(1, "--issue-number", "-i", help="GitHub issue number"),
    max_iterations: int = typer.Option(3, "--max-iterations", "-k", help="Max iterative repair loops"),
) -> None:
    """Resolve an issue on a local repository using the multi-agent cyclic state machine."""
    console.print(f"[bold cyan]Starting AgenticCode issue resolution for: '{title}'[/bold cyan]")
    issue = {
        "title": title,
        "body": title,
        "issue_number": issue_number,
        "repo_url": repo_path,
    }
    final_state = run_workflow(issue, repo_path=repo_path, max_iterations=max_iterations)
    passed = final_state.get("test_results", {}).get("passed", False)
    if passed:
        console.print(f"[bold green]Fix verified and applied! PR URL: {final_state.get('pr_url')}[/bold green]")
    else:
        console.print("[bold red]Automated repair could not verify fix within iteration limit.[/bold red]")


@app.command()
def evaluate(
    dataset_path: Optional[str] = typer.Option(
        None, "--dataset", "-d", help="Path to SWE-bench JSON or JSONL file"
    ),
    max_iterations: int = typer.Option(3, "--max-iterations", "-k", help="Max iterative repair loops"),
    repo_path: str = typer.Option(".", "--repo", "-r", help="Target repository path"),
) -> None:
    """Run SWE-bench benchmark evaluation and report Reproduction Rate, Pass@k, and Regression Rate."""
    if dataset_path:
        instances = load_swebench_dataset(dataset_path)
    else:
        console.print("[yellow]No dataset path provided; evaluating on standard reference SWE-bench instances (requests, flask, scikit-learn, sympy)...[/yellow]")
        instances = get_standard_reference_instances()

    console.print(f"[bold blue]Evaluating {len(instances)} SWE-bench instances (k_max={max_iterations})...[/bold blue]")
    evaluator = SWEBenchEvaluator(max_iterations=max_iterations)
    metrics = evaluator.evaluate_batch(instances, repo_path=repo_path)

    table = Table(title="SWE-bench Evaluation Results")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="bold green")
    table.add_row("Total Instances", str(metrics.total_instances))
    table.add_row("Reproduction Rate", f"{metrics.reproduction_rate:.1f}%")
    table.add_row("Pass@1", f"{metrics.pass_at_1:.1f}%")
    table.add_row("Pass@2", f"{metrics.pass_at_2:.1f}%")
    table.add_row("Pass@3", f"{metrics.pass_at_3:.1f}%")
    table.add_row("Regression Retention Rate", f"{metrics.regression_rate:.1f}%")
    table.add_row("Mean Duration", f"{metrics.mean_duration_seconds:.2f}s")
    console.print(table)


@app.command()
def audit(
    export_file: Optional[str] = typer.Option(
        None, "--export", "-e", help="Path to write JSON audit log"
    ),
) -> None:
    """View active audit log traces and export statistics."""
    traces = audit_logger.export_audit_log(export_file)
    console.print(f"[bold green]Total audit runs recorded: {traces.get('runs_count', 0)}[/bold green]")
    if export_file:
        console.print(f"[cyan]Saved audit log to: {Path(export_file).resolve()}[/cyan]")


@app.command()
def server(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
) -> None:
    """Start the FastAPI GitHub Webhook ingestion server."""
    import uvicorn
    console.print(f"[bold green]Starting GitHub Webhook Ingestion Server on {host}:{port}...[/bold green]")
    uvicorn.run("agentic_code.webhook:app", host=host, port=port, reload=False)


@app.command()
def info() -> None:
    """Display AgenticCode configuration and capabilities."""
    table = Table(title="AgenticCode System Configuration")
    table.add_column("Component", style="bold yellow")
    table.add_column("Configured Value", style="white")
    table.add_row("Default LLM Provider", settings.DEFAULT_LLM_PROVIDER)
    table.add_row("Default Model", settings.DEFAULT_MODEL)
    table.add_row("Planner Model", settings.PLANNER_MODEL)
    table.add_row("Code Writer Model", settings.CODE_WRITER_MODEL)
    table.add_row("Sandbox Image", settings.SANDBOX_IMAGE)
    table.add_row("Max Iterations", str(settings.MAX_ITERATIONS))
    console.print(table)


if __name__ == "__main__":
    app()
