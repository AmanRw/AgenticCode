"""
Shared Orchestrator State definition for AgenticCode.
"""

from typing import Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class IssueMetadata(BaseModel):
    """Normalized metadata for a GitHub Issue."""
    title: str = Field(description="Issue title")
    body: str = Field(description="Issue description/body")
    issue_number: int = Field(description="Issue number (#)")
    repo_url: str = Field(description="GitHub repository URL or owner/repo")
    author: Optional[str] = Field(default=None, description="Issue creator username")
    labels: List[str] = Field(default_factory=list, description="Labels on the issue")
    comments: List[str] = Field(default_factory=list, description="Issue discussion comments")


class SymbolInfo(BaseModel):
    """Extracted AST/Tree-Sitter symbol information."""
    name: str
    kind: str  # class, method, function, variable
    file_path: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    signature: Optional[str] = None


class CodeContext(BaseModel):
    """Structured code context extracted by Agent 01 (Code Reader)."""
    target_files: List[str] = Field(default_factory=list)
    symbols: List[SymbolInfo] = Field(default_factory=list)
    file_snippets: Dict[str, str] = Field(default_factory=dict)
    call_graph: Dict[str, List[str]] = Field(default_factory=dict)
    summary: str = Field(default="")


class TestResults(BaseModel):
    """Execution output from the Docker Sandbox Engine."""
    __test__ = False
    exit_code: int
    stdout: str
    stderr: str
    passed: bool
    duration_seconds: float = 0.0
    command: str = ""
    error_summary: Optional[str] = None


class AgentState(TypedDict):
    """
    Central typed state definition stored and updated throughout the LangGraph workflow.
    """
    issue: dict                    # title, body, issue_number, repo_url
    repo_path: str                 # local clone path
    code_context: dict             # relevant files, ast symbols, call graphs
    plan: str                      # root-cause analysis and repair steps
    tests: str                     # reproduction & regression test code
    patch: str                     # unified diff or file updates
    test_results: dict             # exit_code, stdout, stderr, passed
    error_feedback: Optional[str]  # traceback/failure logs for iterative repair
    iteration: int                 # current retry count k
    max_iterations: int            # k_max threshold (e.g., 3)
    pr_url: Optional[str]          # created PR link
