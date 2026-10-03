"""
Tests for AgentState and configuration.
"""

from agentic_code.config import Settings
from agentic_code.state import AgentState, IssueMetadata, TestResults


def test_agent_state_instantiation():
    state: AgentState = {
        "issue": {
            "title": "Fix ZeroDivisionError in math_utils.py",
            "body": "When denominator is 0, raise ValueError instead of crashing.",
            "issue_number": 42,
            "repo_url": "https://github.com/example/repo",
        },
        "repo_path": "/tmp/repo",
        "code_context": {
            "target_files": ["math_utils.py"],
            "symbols": [],
        },
        "plan": "Diagnose ZeroDivisionError and wrap check.",
        "tests": "def test_zero(): pass",
        "patch": "diff --git a/math_utils.py b/math_utils.py ...",
        "test_results": {
            "exit_code": 0,
            "stdout": "1 passed",
            "stderr": "",
            "passed": True,
        },
        "error_feedback": None,
        "iteration": 0,
        "max_iterations": 3,
        "pr_url": None,
    }

    assert state["issue"]["issue_number"] == 42
    assert state["iteration"] == 0
    assert state["test_results"]["passed"] is True


def test_models():
    issue = IssueMetadata(
        title="Bug in parser",
        body="Unexpected token encountered.",
        issue_number=101,
        repo_url="owner/repo",
    )
    assert issue.issue_number == 101
    assert issue.labels == []

    results = TestResults(
        exit_code=0,
        stdout="OK",
        stderr="",
        passed=True,
    )
    assert results.passed is True


def test_settings_mock():
    s = Settings(DEFAULT_LLM_PROVIDER="mock")
    llm = s.get_llm()
    assert llm is not None


def test_settings_gemini():
    s = Settings(DEFAULT_LLM_PROVIDER="gemini", GEMINI_API_KEY="dummy_key_for_unit_tests")
    llm = s.get_llm(model_name="gemini-2.5-flash")
    assert llm is not None
    assert getattr(llm, "model", None) == "gemini-2.5-flash"

