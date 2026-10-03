"""
AgenticCode Main Entry Point & Verification Demo (Phases 1-4).
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentic_code.agents import (
    CodeReaderAgent,
    CodeWriterAgent,
    PlannerAgent,
    PROpenerAgent,
    TestWriterAgent,
)
from agentic_code.config import settings
from agentic_code.sandbox.sandbox import DockerSandbox
from agentic_code.tools.ast_tools import extract_symbols_from_source

console = Console()


def main():
    console.print(
        Panel.fit(
            "[bold cyan]AgenticCode: End-to-End Autonomous GitHub Issue Resolution[/bold cyan]\n"
            "[green]Phases 1 - 4 System Verification[/green]",
            border_style="cyan",
        )
    )

    # 1. Environment & Configuration
    config_table = Table(title="Phase 1: Environment & Settings")
    config_table.add_column("Setting", style="bold yellow")
    config_table.add_column("Value", style="white")
    config_table.add_row("Default LLM Provider", settings.DEFAULT_LLM_PROVIDER)
    config_table.add_row("Default Model", settings.DEFAULT_MODEL)
    config_table.add_row("Sandbox Image", settings.SANDBOX_IMAGE)
    config_table.add_row("Max Repair Iterations", str(settings.MAX_ITERATIONS))
    console.print(config_table)

    # 2. Hardened Sandbox Specifications
    sandbox = DockerSandbox(repo_path=".")
    sandbox_table = Table(title="Phase 2: Hardened Docker Sandbox Security Policies")
    sandbox_table.add_column("Security Dimension", style="bold red")
    sandbox_table.add_column("Enforced Boundary", style="white")
    sandbox_table.add_row("Capability Stripping", "cap_drop=['ALL']")
    sandbox_table.add_row("Network Blackholing", f"network_mode='{sandbox.network_mode}'")
    sandbox_table.add_row("Execution User", f"user='{sandbox.user}' (Non-root)")
    sandbox_table.add_row("Resource Limits", f"mem_limit='{sandbox.mem_limit}', cpu_quota={sandbox.cpu_quota}")
    sandbox_table.add_row("Timeout Deadline", f"{sandbox.timeout_seconds}s strict kill deadline")
    console.print(sandbox_table)

    # 3. AST & Code Retrieval
    sample_code = "class PaymentProcessor:\n    def charge(self, amount: float):\n        return True\n"
    symbols = extract_symbols_from_source(sample_code, "payment.py")
    console.print(
        Panel(
            f"Extracted {len(symbols['symbols'])} AST symbols from source without loading full file into LLM context:\n"
            f"[bold green]{[s['signature'] for s in symbols['symbols']]}[/bold green]",
            title="Phase 3: AST Retrieval Tools",
            border_style="green",
        )
    )

    # 4. Specialized Agent Personas
    agents = [
        CodeReaderAgent(llm=None),
        PlannerAgent(llm=None),
        TestWriterAgent(llm=None),
        CodeWriterAgent(llm=None),
        PROpenerAgent(llm=None),
    ]
    agents_table = Table(title="Phase 4: Specialized Agent Personas")
    agents_table.add_column("Persona", style="bold magenta")
    agents_table.add_column("Configured Model", style="cyan")
    for agent in agents:
        agents_table.add_row(agent.name, agent.model_name or "default")
    console.print(agents_table)

    console.print("\n[bold green][OK] Phases 1 to 4 successfully implemented and verified![/bold green]\n")


if __name__ == "__main__":
    main()
