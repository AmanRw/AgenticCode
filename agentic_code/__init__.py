"""
AgenticCode: End-to-End Autonomous GitHub Issue Resolution.
"""

from agentic_code.agents import (
    BasePersona,
    CodeReaderAgent,
    CodeWriterAgent,
    PlannerAgent,
    PROpenerAgent,
    TestWriterAgent,
)
from agentic_code.config import settings
from agentic_code.sandbox import DockerSandbox, LocalSubprocessSandbox, SandboxExecutionError
from agentic_code.state import AgentState, CodeContext, IssueMetadata, SymbolInfo, TestResults
from agentic_code.tools import (
    build_repository_call_graph,
    extract_code_snippet,
    extract_symbols_from_source,
    generate_repository_symbol_outline,
    search_codebase,
    search_traceback_symbols,
)
from agentic_code.vcs import GitHubClient, GitHubClientError, GitManager, GitOperationError

__version__ = "0.1.0"

__all__ = [
    # Config & State
    "settings",
    "AgentState",
    "IssueMetadata",
    "CodeContext",
    "TestResults",
    "SymbolInfo",
    # Sandbox
    "DockerSandbox",
    "LocalSubprocessSandbox",
    "SandboxExecutionError",
    # Tools
    "extract_symbols_from_source",
    "generate_repository_symbol_outline",
    "extract_code_snippet",
    "search_codebase",
    "search_traceback_symbols",
    "build_repository_call_graph",
    # Agents
    "BasePersona",
    "CodeReaderAgent",
    "PlannerAgent",
    "TestWriterAgent",
    "CodeWriterAgent",
    "PROpenerAgent",
    # VCS
    "GitManager",
    "GitOperationError",
    "GitHubClient",
    "GitHubClientError",
]
