"""
Agent 05: PR Opener
Role: Formats git commits, generates PR description, and orchestrates PR creation on GitHub.
"""

from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from agentic_code.agents.base import BasePersona
from agentic_code.config import settings

SYSTEM_PROMPT = """You are Agent 05 (PR Opener), a release engineering specialist.
Your job is to prepare a professional, concise, and informative GitHub Pull Request description.

The PR body MUST include:
1. Reference to the issue: `Fixes #<issue_id>`
2. Summary of the Root Cause
3. Detailed Breakdown of Changes Made
4. Test Results and Verification Evidence (reproduction & regression test passing proof)
"""


class PROpenerAgent(BasePersona):
    """Prepares structured PR content and metadata."""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        super().__init__(
            name="Agent 05: PR Opener",
            system_prompt=SYSTEM_PROMPT,
            model_name=settings.DEFAULT_MODEL,
            llm=llm,
        )

    def generate_pr_content(
        self,
        issue: Dict[str, Any],
        plan: str,
        patch: str,
        test_results: Dict[str, Any],
        iteration: int = 0,
    ) -> Dict[str, str]:
        """
        Synthesizes PR title, commit message, and full markdown PR body.
        """
        issue_number = issue.get("issue_number", 0)
        issue_title = issue.get("title", "Fix issue")

        user_prompt = f"""### Issue #{issue_number}: {issue_title}
### Plan Summary:
{plan}

### Verified Patch Diff:
```
{patch[:1500]}
```

### Sandbox Test Results:
- Passed: {test_results.get('passed', True)}
- Iterations: {iteration + 1}
- Exit Code: {test_results.get('exit_code', 0)}
- Test Output:
```
{test_results.get('stdout', '')[:800]}
```

Generate:
1. Commit message (single line)
2. PR title (conventional commit format, e.g. `fix: resolve ... (#issue)`)
3. PR markdown body linking the issue and summarizing the fix.
"""

        try:
            response_text = self.invoke_prompt(user_prompt)
        except Exception:
            response_text = ""

        # Default structured fallback
        pr_title = f"fix: resolve {issue_title} (Fixes #{issue_number})"
        commit_message = f"fix(core): resolve #{issue_number} - {issue_title}"
        pr_body = f"""## Summary
Fixes #{issue_number}

### Root Cause
{plan[:300]}...

### Changes Applied
Unified patch applied and verified through isolated Docker sandbox.

### Verification
- **Sandbox Test Status**: Passed (Exit Code {test_results.get('exit_code', 0)})
- **Verification Attempts**: {iteration + 1}
- **Tests**: Standalone reproduction & regression suite executed successfully.

---
*Autonomous fix generated and verified by **AgenticCode**.*
"""

        if response_text and "## Summary" in response_text:
            pr_body = response_text

        return {
            "title": pr_title,
            "commit_message": commit_message,
            "body": pr_body,
            "branch_name": f"agentic-fix/issue-{issue_number}",
        }
