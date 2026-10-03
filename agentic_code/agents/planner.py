"""
Agent 02: Planner
Role: Diagnose root cause and formulate targeted fixes into a structured Markdown plan.
"""

from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from agentic_code.agents.base import BasePersona
from agentic_code.config import settings

SYSTEM_PROMPT = """You are Agent 02 (Planner), a principal software architect and debugging expert.
Your role is to diagnose the root cause of the reported GitHub issue using the provided code snippets
and context, and produce a structured, high-precision repair plan.

Your diagnostic plan MUST be formatted in Markdown with the following 4 sections:
1. ### Identified Bug / Failure Mode
   - Detailed diagnosis of what goes wrong and why the current code fails.
2. ### Invariants and Existing Behaviors
   - List the critical invariants, edge cases, and contracts that must not break.
3. ### Step-by-Step Logic Modifications
   - Exact algorithms, condition checks, and modifications required in the target files.
4. ### Requirements for Reproduction and Regression Tests
   - Concrete instructions for the test writer: inputs, assertions, expected exceptions, and regression checks.
"""


class PlannerAgent(BasePersona):
    """Diagnoses root cause and produces a structured diagnostic plan."""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        super().__init__(
            name="Agent 02: Planner",
            system_prompt=SYSTEM_PROMPT,
            model_name=settings.PLANNER_MODEL,
            llm=llm,
        )

    def plan(
        self,
        issue: Dict[str, Any],
        code_context: Dict[str, Any],
        error_feedback: Optional[str] = None,
    ) -> str:
        """
        Formulate diagnostic plan based on issue description, localized snippets, and past test traces.
        """
        snippets_formatted = ""
        for filepath, snippet in code_context.get("file_snippets", {}).items():
            snippets_formatted += f"\nFile: `{filepath}`\n```python\n{snippet}\n```\n"

        feedback_section = ""
        if error_feedback:
            feedback_section = f"""\n### Prior Iteration Failure Feedback:\n```\n{error_feedback}\n```\nAnalyze why the previous attempt failed and revise the repair plan accordingly.\n"""

        user_prompt = f"""### Issue #{issue.get('issue_number')}: {issue.get('title')}
Description:
{issue.get('body')}

### Localized Code Context:
{snippets_formatted}
{feedback_section}
Formulate the diagnostic repair plan adhering strictly to the 4 required sections.
"""

        try:
            return self.invoke_prompt(user_prompt)
        except Exception:
            # Fallback plan if offline/mock
            return f"""### Identified Bug / Failure Mode
Failure reported in issue #{issue.get('issue_number')}: {issue.get('title')}.

### Invariants and Existing Behaviors
Existing functionality must remain backward-compatible without breaking existing tests.

### Step-by-Step Logic Modifications
1. Inspect target files: {code_context.get('target_files', [])}.
2. Apply boundary validation and exception handling.

### Requirements for Reproduction and Regression Tests
- Create reproduction test verifying the failure condition.
- Assert correct output or expected error handling.
"""
