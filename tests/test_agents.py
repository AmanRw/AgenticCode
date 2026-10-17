"""
Tests for Phase 4 Specialized Agent Personas.
"""

import tempfile
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agentic_code.agents import (
    BasePersona,
    CodeReaderAgent,
    CodeWriterAgent,
    PlannerAgent,
    PROpenerAgent,
    TestWriterAgent,
)


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


def test_gemini_chat_afc_solution_a():
    from unittest.mock import MagicMock

    from agentic_code.config import settings

    def sample_tool(file_name: str) -> str:
        """Sample function for AFC testing."""
        return f"content of {file_name}"

    # Verify create_gemini_chat instantiates a stateful Chat session (Solution A)
    chat = settings.create_gemini_chat(
        model_name="gemini-2.5-flash",
        tools=[sample_tool],
        system_instruction="You are a helpful coding assistant.",
    )
    assert hasattr(chat, "send_message")

    # Verify persona works with use_gemini_chat and Chat.send_message
    persona = BasePersona(
        name="AFCPersona",
        system_prompt="Test system prompt",
        model_name="gemini-2.5-flash",
        tools=[sample_tool],
        use_gemini_chat=True,
    )

    mock_chat = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"target_files": ["main.py"]}'
    mock_chat.send_message.return_value = mock_response

    persona._chat_session = mock_chat
    res = persona.invoke_prompt("Find the relevant files")

    # Verify Chat.send_message was called instead of Models.generate_content
    mock_chat.send_message.assert_called_once_with("Find the relevant files")
    assert '{"target_files": ["main.py"]}' in res


def test_code_writer_extract_code_from_plan():
    plan_with_snippet = """1. Identified Bug / Failure Mode
ZeroDivisionError on division by zero.
3. Step-by-Step Logic Modifications
Modified Code Snippet:

   1: def operation(a,b,op):
   2:     if op=='+':
   3:         return a+b
   4:     elif op=='-':
   5:         return a-b
   6:     elif op=='*':
   7:         return a*b
   8:     elif op=='/':
   9:         if b == 0:
  10:             return "Cannot divide by zero"
  11:         else:
  12:             return a/b
  13:     else:
  14:         return 'Invalid operation'

4. Requirements for Reproduction and Regression Tests
"""
    extracted = CodeWriterAgent.extract_code_from_plan(plan_with_snippet)
    assert "def operation(a,b,op):" in extracted
    assert 'return "Cannot divide by zero"' in extracted
    assert "   1:" not in extracted
    assert "  10:" not in extracted


def test_code_writer_fallback_when_llm_fails():
    from unittest.mock import patch
    plan_with_snippet = """3. Step-by-Step Logic Modifications
```python
def operation(a,b,op):
    if b == 0 and op == '/':
        return "Cannot divide by zero"
    return a / b
```"""
    agent = CodeWriterAgent()
    with patch.object(BasePersona, "invoke_prompt", side_effect=RuntimeError("Rate limit / 429")):
        patch_result = agent.write_patch(
            plan=plan_with_snippet,
            code_context={"target_files": ["simpleCalc.py"], "file_snippets": {}},
            tests="",
        )
    assert "def operation(a,b,op):" in patch_result
    assert "Fix applied" not in patch_result


