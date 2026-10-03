"""
Tests for Phase 4 Specialized Agent Personas.
"""

import tempfile
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agentic_code.agents.base import BasePersona
from agentic_code.agents.code_reader import CodeReaderAgent
from agentic_code.agents.code_writer import CodeWriterAgent
from agentic_code.agents.planner import PlannerAgent
from agentic_code.agents.pr_opener import PROpenerAgent
from agentic_code.agents.test_writer import TestWriterAgent


def test_base_persona_utilities():
    persona = BasePersona(
        name="TestPersona",
        system_prompt="System instructions",
        llm=FakeListChatModel(responses=["Hello World"]),
    )
    res = persona.invoke_prompt("Hi")
    assert "Hello World" in res

    # JSON extraction
    json_text = '```json\n{"target_files": ["foo.py"], "key": 123}\n```'
    parsed = BasePersona.extract_json_from_response(json_text)
    assert parsed["target_files"] == ["foo.py"]
    assert parsed["key"] == 123

    # Code block extraction
    code_text = '```python\ndef test(): pass\n```'
    code = BasePersona.extract_code_block(code_text, language="python")
    assert code == "def test(): pass"


def test_code_reader_agent():
    mock_llm = FakeListChatModel(
        responses=[
            """```json
{
  "target_files": ["math_ops.py"],
  "symbols": [
    {
      "name": "safe_divide",
      "file_path": "math_ops.py",
      "line_start": 1,
      "line_end": 5,
      "rationale": "Divide by zero unhandled"
    }
  ],
  "search_terms_to_explore": ["divide"],
  "summary": "Target function is safe_divide in math_ops.py"
}
```"""
        ]
    )
    agent = CodeReaderAgent(llm=mock_llm)

    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "math_ops.py"
        f.write_text("def safe_divide(a, b):\n    return a / b\n", encoding="utf-8")

        issue = {
            "title": "Fix division by zero in safe_divide",
            "body": "When b=0, ZeroDivisionError is thrown.",
            "issue_number": 12,
            "repo_url": "https://github.com/example/repo",
        }

        context = agent.analyze(issue, repo_path=tmpdir)
        assert "math_ops.py" in context["target_files"]
        assert "math_ops.py" in context["file_snippets"]
        assert "safe_divide" in context["file_snippets"]["math_ops.py"]


def test_planner_agent():
    expected_plan = """### Identified Bug / Failure Mode
safe_divide crashes on b=0 with ZeroDivisionError.

### Invariants and Existing Behaviors
Normal division with b != 0 must remain unchanged.

### Step-by-Step Logic Modifications
1. Add check `if b == 0: raise ValueError('Cannot divide by zero')`.

### Requirements for Reproduction and Regression Tests
1. Reproduction: safe_divide(10, 0) raises ValueError.
2. Regression: safe_divide(10, 2) returns 5.0.
"""
    mock_llm = FakeListChatModel(responses=[expected_plan])
    agent = PlannerAgent(llm=mock_llm)

    issue = {
        "title": "Fix division by zero in safe_divide",
        "body": "When b=0, ZeroDivisionError is thrown.",
        "issue_number": 12,
    }
    code_context = {
        "target_files": ["math_ops.py"],
        "file_snippets": {"math_ops.py": "def safe_divide(a, b):\n    return a / b"},
    }

    plan_output = agent.plan(issue, code_context)
    assert "### Identified Bug / Failure Mode" in plan_output
    assert "### Invariants and Existing Behaviors" in plan_output
    assert "### Step-by-Step Logic Modifications" in plan_output
    assert "### Requirements for Reproduction and Regression Tests" in plan_output


def test_test_writer_agent():
    mock_test_code = """```python
import pytest
from math_ops import safe_divide

def test_reproduce_division_by_zero():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        safe_divide(10, 0)

def test_regression_normal_division():
    assert safe_divide(10, 2) == 5.0
```"""
    mock_llm = FakeListChatModel(responses=[mock_test_code])
    agent = TestWriterAgent(llm=mock_llm)

    plan = "Add check if b == 0 raise ValueError."
    code_context = {"target_files": ["math_ops.py"], "file_snippets": {}}
    issue = {"title": "Fix division by zero", "body": "...", "issue_number": 12}

    tests = agent.write_tests(plan, code_context, issue)
    assert "def test_reproduce_division_by_zero():" in tests
    assert "def test_regression_normal_division():" in tests


def test_code_writer_agent():
    mock_diff = """```diff
--- a/math_ops.py
+++ b/math_ops.py
@@ -1,2 +1,4 @@
 def safe_divide(a, b):
+    if b == 0:
+        raise ValueError("Cannot divide by zero")
     return a / b
```"""
    mock_llm = FakeListChatModel(responses=[mock_diff])
    agent = CodeWriterAgent(llm=mock_llm)

    patch = agent.write_patch(
        plan="Raise ValueError on 0",
        code_context={"target_files": ["math_ops.py"], "file_snippets": {}},
        tests="def test_reproduce(): pass",
    )
    assert "--- a/math_ops.py" in patch
    assert "+++ b/math_ops.py" in patch
    assert "+        raise ValueError" in patch


def test_pr_opener_agent():
    mock_llm = FakeListChatModel(responses=["## Summary\nFixes #12\n\n### Root Cause\nZeroDivisionError"])
    agent = PROpenerAgent(llm=mock_llm)
    issue = {
        "title": "Fix division by zero",
        "issue_number": 12,
        "repo_url": "https://github.com/example/repo",
    }
    plan = "Identified bug: ZeroDivisionError"
    patch = "--- a/math_ops.py\n+++ b/math_ops.py"
    test_results = {"passed": True, "exit_code": 0, "stdout": "2 passed"}

    content = agent.generate_pr_content(issue, plan, patch, test_results)
    assert "Fixes #12" in content["body"]
    assert "fix:" in content["title"].lower()
    assert "branch_name" in content
    assert content["branch_name"] == "agentic-fix/issue-12"
