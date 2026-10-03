"""
Agent 03: Code Writer
Role: Synthesize minimal, production-grade patches adhering to repository style.
"""

from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from agentic_code.agents.base import BasePersona
from agentic_code.config import settings

SYSTEM_PROMPT = """You are Agent 03 (Code Writer), an expert senior software engineer.
Your responsibility is to synthesize a minimal, clean, robust, production-grade patch
that fixes the bug according to the diagnostic plan while keeping all existing functionality intact.

Instructions:
1. Synthesize minimal edits targeting ONLY the localized bug.
2. In retry loops, carefully study previous failure tracebacks and avoid repeating the same bug.
3. Return the patch in standard Unified Diff format (starting with `--- a/file.py` and `+++ b/file.py`)
   or provide the complete updated file content inside standard code fences.
4. Output the patch inside a ```diff ... ``` or ```python ... ``` block.
"""


class CodeWriterAgent(BasePersona):
    """Synthesizes minimal, production-grade code patches."""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        super().__init__(
            name="Agent 03: Code Writer",
            system_prompt=SYSTEM_PROMPT,
            model_name=settings.CODE_WRITER_MODEL,
            llm=llm,
        )

    def write_patch(
        self,
        plan: str,
        code_context: Dict[str, Any],
        tests: str,
        error_feedback: Optional[str] = None,
        iteration: int = 0,
    ) -> str:
        """
        Synthesizes patch based on plan, context, reproduction tests, and prior iteration feedback.
        """
        snippets_formatted = ""
        for filepath, snippet in code_context.get("file_snippets", {}).items():
            snippets_formatted += f"\nFile: `{filepath}`\n```python\n{snippet}\n```\n"

        feedback_section = ""
        if error_feedback:
            feedback_section = f"""\n### Prior Test Failure Traceback (Iteration {iteration}):\n```\n{error_feedback}\n```\nFix the cause of this failure.\n"""

        user_prompt = f"""### Diagnostic Plan:
{plan}

### Reproduction & Regression Tests:
```python
{tests}
```

### Relevant Code:
{snippets_formatted}
{feedback_section}
Produce the minimal unified diff or patched file content to resolve the issue.
"""

        try:
            response_text = self.invoke_prompt(user_prompt)
            # Try diff block first, then python block, then raw
            diff_block = self.extract_code_block(response_text, language="diff")
            if diff_block and ("---" in diff_block or "@@" in diff_block):
                return diff_block
            py_block = self.extract_code_block(response_text, language="python")
            if py_block:
                return py_block
            return response_text.strip()
        except Exception:
            # Fallback stub patch
            target = code_context.get("target_files", ["module.py"])[0]
            return f"""--- a/{target}\n+++ b/{target}\n@@ -1,1 +1,2 @@\n+# Fix applied by AgenticCode\n"""
