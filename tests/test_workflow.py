"""
Tests for Phase 6: LangGraph State Machine & Cyclic Orchestration.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import git

from agentic_code.state import AgentState
from agentic_code.workflow import (
    build_agentic_code_graph,
    node_code_reader,
    node_code_writer,
    node_escalation,
    node_planner,
    node_pr_opener,
    node_reproduction_verifier,
    node_sandbox_verifier,
    node_test_writer,
    route_after_verification,
)


def test_routing_after_verification():
    # 1. Test passed -> PR Opener
    state_passed: AgentState = {
        "issue": {},
        "repo_path": ".",
        "code_context": {},
        "plan": "",
        "tests": "",
        "patch": "",
        "test_results": {"passed": True, "exit_code": 0, "stdout": "", "stderr": ""},
        "error_feedback": None,
        "iteration": 1,
        "max_iterations": 3,
        "pr_url": None,
    }
    assert route_after_verification(state_passed) == "node_pr_opener"

    # 2. Test failed & iteration < max_iterations -> Retry via Code Writer
    state_retry: AgentState = {
        "issue": {},
        "repo_path": ".",
        "code_context": {},
        "plan": "",
        "tests": "",
        "patch": "",
        "test_results": {"passed": False, "exit_code": 1, "stdout": "", "stderr": "AssertError"},
        "error_feedback": "Traceback",
        "iteration": 1,
        "max_iterations": 3,
        "pr_url": None,
    }
    assert route_after_verification(state_retry) == "node_code_writer"

    # 3. Test failed & iteration >= max_iterations -> Escalation
    state_escalate: AgentState = {
        "issue": {},
        "repo_path": ".",
        "code_context": {},
        "plan": "",
        "tests": "",
        "patch": "",
        "test_results": {"passed": False, "exit_code": 1, "stdout": "", "stderr": "AssertError"},
        "error_feedback": "Traceback",
        "iteration": 3,
        "max_iterations": 3,
        "pr_url": None,
    }
    assert route_after_verification(state_escalate) == "node_escalation"


def test_graph_compilation():
    graph = build_agentic_code_graph()
    assert graph is not None


def test_individual_nodes_execution():
    from agentic_code.agents import BasePersona

    def mock_invoke(self, prompt: str):
        if "target files" in prompt.lower() or "files sample" in prompt.lower():
            return '{"target_files": ["calc.py"], "symbols": []}'
        if "diagnostic" in prompt.lower():
            return "### Identified Bug / Failure Mode\nNeeds docstring.\n### Invariants and Existing Behaviors\nKeep add.\n### Step-by-Step Logic Modifications\nAdd docstring.\n### Requirements for Reproduction and Regression Tests\nTest add."
        if "pytest" in prompt.lower():
            return "def test_calc(): assert True"
        if "diff" in prompt.lower() or "patch" in prompt.lower():
            return "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,3 @@\n def add(a, b):\n+    '''Docstring'''\n     return a + b\n"
        return "## Summary\nFixes #5"

    with patch.object(BasePersona, "invoke_prompt", mock_invoke):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            repo_path = Path(tmpdir)
            repo = git.Repo.init(repo_path)
            repo.config_writer().set_value("user", "name", "Test Runner").release()
            repo.config_writer().set_value("user", "email", "test@example.com").release()

            # Target file
            calc = repo_path / "calc.py"
            calc.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            repo.git.add("-A")
            repo.git.commit("-m", "Initial commit")
            repo.close()

            state: AgentState = {
            "issue": {
                "title": "Fix bug in calc",
                "body": "add function needs docstring",
                "issue_number": 5,
                "repo_url": "https://github.com/test/repo",
            },
            "repo_path": str(repo_path),
            "code_context": {},
            "plan": "",
            "tests": "",
            "patch": "",
            "test_results": {"passed": False, "exit_code": -1, "stdout": "", "stderr": ""},
            "error_feedback": None,
            "iteration": 0,
            "max_iterations": 3,
            "pr_url": None,
        }

            # 1. Test Code Reader Node
            cr_res = node_code_reader(state)
            assert "code_context" in cr_res
            state.update(cr_res)

            # 2. Test Planner Node
            pl_res = node_planner(state)
            assert "plan" in pl_res
            assert len(pl_res["plan"]) > 0
            state.update(pl_res)

            # 3. Test Test Writer Node
            tw_res = node_test_writer(state)
            assert "tests" in tw_res
            state.update(tw_res)

            # 4. Test Reproduction Verifier Node
            rep_res = node_reproduction_verifier(state)
            assert "test_results" in rep_res

            # 5. Test Code Writer Node with Patch
            cw_res = node_code_writer(state)
            assert "patch" in cw_res
            state.update(cw_res)

            # 6. Test Sandbox Verifier Node
            sb_res = node_sandbox_verifier(state)
            assert "test_results" in sb_res

            # 7. Test PR Opener Node
            pr_res = node_pr_opener(state)
            assert "pr_url" in pr_res

            # 8. Test Escalation Node
            state["iteration"] = 3
            esc_res = node_escalation(state)
            assert "error_feedback" in esc_res
            assert "Escalated" in esc_res["error_feedback"]
