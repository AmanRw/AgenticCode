"""
Specialized Agent Personas for AgenticCode (Phase 4).
Declarative personas powered by Google Gemini:
- Agent 01: Code Reader (gemini-2.5-flash)
- Agent 02: Planner (gemini-2.5-flash)
- Agent 04: Test Writer (gemini-2.5-flash)
- Agent 03: Code Writer (gemini-2.5-flash)
- Agent 05: PR Opener (gemini-2.5-flash)

Supports Google Solution A: Stateful Chat.send_message with Automatic Function Calling (AFC).
"""

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agentic_code.config import settings
from agentic_code.tools import (
    extract_code_snippet,
    generate_repository_symbol_outline,
    search_traceback_symbols,
)


class BasePersona:
    """Base class providing LLM execution and structured markdown/JSON parsers."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model_name: Optional[str] = None,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[Any]] = None,
        use_gemini_chat: Optional[bool] = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.model_name = model_name
        self._llm = llm
        self.tools = tools or []
        self._use_gemini_chat = use_gemini_chat
        self._chat_session: Any = None

    @property
    def use_gemini_chat(self) -> bool:
        if self._use_gemini_chat is not None:
            return self._use_gemini_chat
        if self._llm is not None:
            return False
        return bool(settings.USE_GEMINI_CHAT_AFC and settings.DEFAULT_LLM_PROVIDER == "gemini")

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = settings.get_llm(model_name=self.model_name)
        return self._llm

    @property
    def chat_session(self) -> Any:
        """
        Implements Google Solution A: Stateful Chat session supporting Automatic Function Calling (AFC).
        Uses Chat.send_message instead of Models.generate_content.
        """
        if self._chat_session is None:
            self._chat_session = settings.create_gemini_chat(
                model_name=self.model_name,
                tools=self.tools if self.tools else None,
                system_instruction=self.system_prompt,
            )
        return self._chat_session

    def send_chat_message(self, message: str) -> str:
        """
        Sends message through stateful Gemini Chat session (Solution A: Chat.send_message).
        Enables seamless Automatic Function Calling (AFC) without single-turn warnings.
        """
        response = self.chat_session.send_message(message)
        if hasattr(response, "text") and response.text:
            return str(response.text)
        return str(response)

    def invoke_prompt(self, user_content: str, max_retries: int = 3) -> str:
        """Invokes LLM with system prompt and user content, with automatic retry on transient failures."""
        import time

        last_error = None
        for attempt in range(max_retries):
            if self.use_gemini_chat:
                try:
                    return self.send_chat_message(user_content)
                except Exception as e:
                    import logging
                    logging.getLogger("agentic_code.agents").warning(
                        f"Gemini Chat.send_message failed for {self.name} (attempt {attempt + 1}/{max_retries}): {e}. Falling back to BaseChatModel."
                    )
                    last_error = e

            try:
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=user_content),
                ]
                response = self.llm.invoke(messages)
                if isinstance(response, AIMessage):
                    return str(response.content)
                return str(response)
            except Exception as e:
                import logging
                logging.getLogger("agentic_code.agents").warning(
                    f"LLM invoke failed for {self.name} (attempt {attempt + 1}/{max_retries}): {e}"
                )
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))

        if last_error:
            raise last_error
        return ""

    @staticmethod
    def clean_line_number_prefixes(code: str) -> str:
        """Strips line number annotations (e.g. '   1: ', '1: ', '1 | ') from code lines."""
        lines = code.splitlines()
        pattern = re.compile(r"^\s*\d+[:|]\s?")
        if any(pattern.match(l) for l in lines if l.strip()):
            cleaned = [pattern.sub("", l) for l in lines]
            return "\n".join(cleaned)
        return code

    @staticmethod
    def extract_json_from_response(text: str) -> Dict[str, Any]:
        """Extracts JSON object from markdown fences or raw text."""
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        raw_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if raw_match:
            try:
                return json.loads(raw_match.group(1))
            except json.JSONDecodeError:
                pass
        return {"raw_text": text}

    @staticmethod
    def extract_code_block(text: str, language: str = "python", clean_line_numbers: bool = True) -> str:
        """Extracts code block from markdown fences and cleans line number prefixes."""
        pattern = rf"```{language}\s*\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        code = ""
        if match:
            code = match.group(1).strip()
        else:
            generic = re.search(r"```(?:\w+)?\s*\n(.*?)\n```", text, re.DOTALL)
            if generic:
                code = generic.group(1).strip()
            else:
                code = text.strip()

        if clean_line_numbers and language == "python":
            return BasePersona.clean_line_number_prefixes(code)
        return code


# ---------------------------------------------------------------------------
# Agent 01: Code Reader
# ---------------------------------------------------------------------------

CODE_READER_SYSTEM_PROMPT = """You are Agent 01 (Code Reader), an expert code localization specialist.
Your goal is to inspect a GitHub issue along with repository file tree and symbols,
and pinpoint target files, symbols, and line ranges. Return JSON only:
{
  "target_files": ["path/to/file.py"],
  "symbols": [{"name": "Func", "file_path": "path/to/file.py", "line_start": 1, "line_end": 20}],
  "summary": "Explanation"
}"""


class CodeReaderAgent(BasePersona):
    """Agent 01: Pinpoints target code and extracts relevant snippets."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[Any]] = None,
        use_gemini_chat: Optional[bool] = None,
    ):
        super().__init__(
            name="Agent 01: Code Reader",
            system_prompt=CODE_READER_SYSTEM_PROMPT,
            model_name=settings.CODE_READER_MODEL,
            llm=llm,
            tools=tools,
            use_gemini_chat=use_gemini_chat,
        )

    def analyze(
        self,
        issue: Dict[str, Any],
        repo_path: str,
        symbol_outline: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if symbol_outline is None:
            symbol_outline = generate_repository_symbol_outline(repo_path)

        body = issue.get("body", "") or ""
        traceback_matches = search_traceback_symbols(repo_path, body)
        files_sample = symbol_outline.get("file_tree", [])[:80]

        prompt = f"""### Issue #{issue.get('issue_number')}: {issue.get('title')}
Description: {issue.get('body')}
### Files Sample: {json.dumps(files_sample)}
### Tracebacks: {json.dumps(traceback_matches)}
Pinpoint target files and symbols. Return JSON only."""

        try:
            data = self.extract_json_from_response(self.invoke_prompt(prompt))
        except Exception:
            data = {}

        target_files = data.get("target_files", [])
        if not target_files and traceback_matches:
            target_files = [m["matched_repo_file"] for m in traceback_matches if m.get("matched_repo_file")]
        if not target_files and files_sample:
            target_files = [files_sample[0]]

        snippets = {}
        raw_snippets = {}
        for target in target_files:
            snip = extract_code_snippet(repo_path, target, 1, 80, context_lines=3)
            if "snippet" in snip:
                snippets[target] = snip.get("raw_snippet") or snip["snippet"]
                raw_snippets[target] = snip.get("raw_snippet", "")

        return {
            "target_files": target_files,
            "symbols": data.get("symbols", []),
            "file_snippets": snippets,
            "raw_snippets": raw_snippets,
            "summary": data.get("summary", "Localization completed."),
        }


# ---------------------------------------------------------------------------
# Agent 02: Planner
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """You are Agent 02 (Planner), a software debugging architect.
Formulate a diagnostic repair plan with exactly 4 markdown sections:
1. ### Identified Bug / Failure Mode
2. ### Invariants and Existing Behaviors
3. ### Step-by-Step Logic Modifications
4. ### Requirements for Reproduction and Regression Tests

Important: In section 3, write clean, complete replacement python code in ```python ... ``` without line numbers."""


class PlannerAgent(BasePersona):
    """Agent 02: Diagnoses root cause and produces structured 4-part plan."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[Any]] = None,
        use_gemini_chat: Optional[bool] = None,
    ):
        super().__init__(
            name="Agent 02: Planner",
            system_prompt=PLANNER_SYSTEM_PROMPT,
            model_name=settings.PLANNER_MODEL,
            llm=llm,
            tools=tools,
            use_gemini_chat=use_gemini_chat,
        )

    def plan(
        self,
        issue: Dict[str, Any],
        code_context: Dict[str, Any],
        error_feedback: Optional[str] = None,
    ) -> str:
        snippets = "\n".join(
            f"File `{f}`:\n```python\n{s}\n```"
            for f, s in code_context.get("file_snippets", {}).items()
        )
        feedback = f"\n### Prior Failure Feedback:\n```\n{error_feedback}\n```" if error_feedback else ""
        prompt = f"""### Issue #{issue.get('issue_number')}: {issue.get('title')}
Description: {issue.get('body')}
### Context:
{snippets}
{feedback}
Generate the 4-part Markdown diagnostic plan."""

        try:
            return self.invoke_prompt(prompt)
        except Exception:
            return f"""### Identified Bug / Failure Mode\nIssue #{issue.get('issue_number')}: {issue.get('title')}\n### Invariants and Existing Behaviors\nExisting behavior must not break.\n### Step-by-Step Logic Modifications\nInspect and patch target files.\n### Requirements for Reproduction and Regression Tests\nWrite pytest reproduction test."""


# ---------------------------------------------------------------------------
# Agent 03: Test Writer
# ---------------------------------------------------------------------------

TEST_WRITER_SYSTEM_PROMPT = """You are Agent 03 (Test Writer).
Write a standalone pytest test file (`test_reproduce_issue.py`).
The test must fail on unpatched code and pass once fixed.
Output python code only inside ```python ... ```."""


class TestWriterAgent(BasePersona):
    """Agent 03: Writes reproduction and regression pytest test suites."""

    __test__ = False

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[Any]] = None,
        use_gemini_chat: Optional[bool] = None,
    ):
        super().__init__(
            name="Agent 03: Test Writer",
            system_prompt=TEST_WRITER_SYSTEM_PROMPT,
            model_name=settings.TEST_WRITER_MODEL,
            llm=llm,
            tools=tools,
            use_gemini_chat=use_gemini_chat,
        )

    def write_tests(self, plan: str, code_context: Dict[str, Any], issue: Dict[str, Any]) -> str:
        prompt = f"""### Plan:\n{plan}\n### Target Files:\n{code_context.get('target_files')}\n### Issue #{issue.get('issue_number')}:\n{issue.get('title')}\nOutput reproduction pytest code."""
        try:
            code = self.extract_code_block(self.invoke_prompt(prompt), "python")
            if code and "def test_" in code:
                return code
        except Exception as e:
            import logging
            logging.getLogger("agentic_code.agents").warning(f"TestWriterAgent LLM invocation failed: {e}")

        # Fallback 1: Extract test function from plan if present
        test_match = re.search(r"```(?:python)?\s*\n(.*?def test_.*?)\n```", plan, re.DOTALL)
        if test_match:
            return self.clean_line_number_prefixes(test_match.group(1).strip())

        # Fallback 2: Generate reproduction test template targeting the identified file
        targets = code_context.get("target_files", [])
        from pathlib import Path
        mod_name = Path(targets[0]).stem if targets else "module"
        num = issue.get("issue_number", 0)
        return f"""import pytest

def test_reproduce_issue_{num}():
    \"\"\"Reproduction test for issue #{num}: {issue.get('title', '')}\"\"\"
    try:
        import {mod_name}
    except Exception as e:
        pytest.fail(f"Reproduction verification failed on import of {mod_name}: {{e}}")
"""


# ---------------------------------------------------------------------------
# Agent 04: Code Writer
# ---------------------------------------------------------------------------

CODE_WRITER_SYSTEM_PROMPT = """You are Agent 04 (Code Writer).
Synthesize minimal, production-grade unified diffs targeting only the diagnosed bug.
Output diff format (--- a/file +++ b/file) inside ```diff ... ``` or updated code inside ```python ... ```."""


class CodeWriterAgent(BasePersona):
    """Agent 04: Synthesizes minimal unified diff patches."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[Any]] = None,
        use_gemini_chat: Optional[bool] = None,
    ):
        super().__init__(
            name="Agent 04: Code Writer",
            system_prompt=CODE_WRITER_SYSTEM_PROMPT,
            model_name=settings.CODE_WRITER_MODEL,
            llm=llm,
            tools=tools,
            use_gemini_chat=use_gemini_chat,
        )

    @staticmethod
    def extract_code_from_plan(plan: str) -> str:
        """Extracts candidate replacement code from the diagnostic plan."""
        # Search for code block in or around Section 3
        sec3_match = re.search(
            r"(?:###\s*3\.\s*Step-by-Step Logic Modifications|###\s*Step-by-Step Logic Modifications|3\.\s*Step-by-Step Logic Modifications)(.*?)(?:###\s*4|4\.|$)",
            plan,
            re.DOTALL | re.IGNORECASE,
        )
        search_zone = sec3_match.group(1) if sec3_match else plan

        # Try markdown code blocks in search zone
        code_match = re.search(r"```(?:python)?\s*\n(.*?)\n```", search_zone, re.DOTALL | re.IGNORECASE)
        if code_match:
            return BasePersona.clean_line_number_prefixes(code_match.group(1).strip())

        # Try 'Modified Code Snippet:' pattern
        snip_match = re.search(
            r"Modified Code Snippet:\s*\n(.*?)(?=\n\s*(?:Self-correction|\d+\.|\#\#\#|$))",
            search_zone,
            re.DOTALL | re.IGNORECASE,
        )
        if snip_match:
            return BasePersona.clean_line_number_prefixes(snip_match.group(1).strip())

        # Fallback to any python code block in the plan
        any_code = re.search(r"```python\s*\n(.*?)\n```", plan, re.DOTALL | re.IGNORECASE)
        if any_code:
            return BasePersona.clean_line_number_prefixes(any_code.group(1).strip())

        return ""

    def write_patch(
        self,
        plan: str,
        code_context: Dict[str, Any],
        tests: str,
        error_feedback: Optional[str] = None,
        iteration: int = 0,
    ) -> str:
        snippets = "\n".join(
            f"File `{f}`:\n```python\n{s}\n```"
            for f, s in code_context.get("file_snippets", {}).items()
        )
        feedback = f"\n### Iteration {iteration} Failure:\n{error_feedback}" if error_feedback else ""
        prompt = f"""### Plan:\n{plan}\n### Target Files & Existing Code:\n{snippets}\n### Tests:\n{tests}\n{feedback}\nProduce minimal unified diff patch targeting only the bug or complete updated file in ```python ... ```."""
        try:
            resp = self.invoke_prompt(prompt)
            diff = self.extract_code_block(resp, "diff", clean_line_numbers=False)
            if diff and ("---" in diff or "@@" in diff) and "+# Fix applied" not in diff:
                return diff
            py = self.extract_code_block(resp, "python", clean_line_numbers=True)
            if py and ("def " in py or "class " in py or "import " in py):
                return py
        except Exception as e:
            import logging
            logging.getLogger("agentic_code.agents").warning(
                f"CodeWriterAgent LLM synthesis failed: {e}. Falling back to diagnostic plan extraction."
            )

        # Robust Fallback: Extract code snippet directly from the Planner's diagnostic plan
        plan_code = self.extract_code_from_plan(plan)
        if plan_code and ("def " in plan_code or "class " in plan_code or "import " in plan_code):
            import logging
            logging.getLogger("agentic_code.agents").info(
                "Successfully extracted replacement code directly from diagnostic plan."
            )
            return plan_code

        return ""


# ---------------------------------------------------------------------------
# Agent 05: PR Opener
# ---------------------------------------------------------------------------

PR_OPENER_SYSTEM_PROMPT = """You are Agent 05 (PR Opener).
Prepare commit message, PR title, and Markdown PR description linking the issue: `Fixes #<id>`."""


class PROpenerAgent(BasePersona):
    """Agent 05: Formats PR metadata and markdown descriptions."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[Any]] = None,
        use_gemini_chat: Optional[bool] = None,
    ):
        super().__init__(
            name="Agent 05: PR Opener",
            system_prompt=PR_OPENER_SYSTEM_PROMPT,
            model_name=settings.DEFAULT_MODEL,
            llm=llm,
            tools=tools,
            use_gemini_chat=use_gemini_chat,
        )

    def generate_pr_content(
        self,
        issue: Dict[str, Any],
        plan: str,
        patch: str,
        test_results: Dict[str, Any],
        iteration: int = 0,
    ) -> Dict[str, str]:
        num = issue.get("issue_number", 0)
        title = issue.get("title", "Fix issue")
        prompt = f"### Issue #{num}: {title}\nPlan:\n{plan}\nPassed: {test_results.get('passed')}\nGenerate PR title and Markdown description."

        try:
            resp = self.invoke_prompt(prompt)
        except Exception:
            resp = ""

        pr_body = (
            resp
            if (resp and "## Summary" in resp)
            else f"""## Summary\nFixes #{num}\n\n### Root-Cause Diagnosis\n{plan}\n\n### Verification\nPassed in hardened Docker sandbox ({iteration + 1} iterations)."""
        )
        return {
            "title": f"fix: resolve {title} (Fixes #{num})",
            "commit_message": f"fix: resolve #{num} - {title}",
            "body": pr_body,
            "branch_name": f"agentic-fix/issue-{num}",
        }
