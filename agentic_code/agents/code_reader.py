"""
Agent 01: Code Reader
Role: Pinpoint target files and extract relevant code snippets without overflowing context.
"""

import json
from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from agentic_code.agents.base import BasePersona
from agentic_code.config import settings
from agentic_code.tools.ast_tools import (
    extract_code_snippet,
    generate_repository_symbol_outline,
)
from agentic_code.tools.search_tools import (
    search_traceback_symbols,
)

SYSTEM_PROMPT = """You are Agent 01 (Code Reader), an expert code localization specialist.
Your goal is to inspect a GitHub issue (title, description, tracebacks) along with a high-level file tree
and symbol outline of the repository, and determine the exact target files, symbols, and line ranges
that are directly responsible for or affected by the issue.

Return ONLY a JSON object with this exact structure:
{
  "target_files": ["path/to/file1.py", "path/to/file2.py"],
  "symbols": [
    {
      "name": "ClassOrFunctionName",
      "file_path": "path/to/file1.py",
      "line_start": 10,
      "line_end": 50,
      "rationale": "Why this symbol is relevant"
    }
  ],
  "search_terms_to_explore": ["term1", "term2"],
  "summary": "Brief explanation of how the localized files relate to the issue."
}
"""


class CodeReaderAgent(BasePersona):
    """Specialized persona to pinpoint target code and extract relevant snippets."""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        super().__init__(
            name="Agent 01: Code Reader",
            system_prompt=SYSTEM_PROMPT,
            model_name=settings.CODE_READER_MODEL,
            llm=llm,
        )

    def analyze(
        self,
        issue: Dict[str, Any],
        repo_path: str,
        symbol_outline: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Reads issue and repo outline, performs targeted searches, and returns structured code context.
        """
        if symbol_outline is None:
            symbol_outline = generate_repository_symbol_outline(repo_path)

        # Look for tracebacks in the issue body
        body = issue.get("body", "") or ""
        traceback_matches = search_traceback_symbols(repo_path, body)

        # Prepare summary of available files & symbols for prompt
        files_sample = symbol_outline.get("file_tree", [])[:80]
        symbols_summary = {}
        for f in files_sample[:30]:
            syms = symbol_outline.get("symbol_catalog", {}).get(f, [])
            symbols_summary[f] = [
                s.get("signature") or s.get("name") for s in syms if isinstance(s, dict)
            ][:10]

        user_prompt = f"""### Issue #{issue.get('issue_number')}: {issue.get('title')}
Description:
{issue.get('body')}

### Repository File Tree (Sample):
{json.dumps(files_sample, indent=2)}

### Key Symbols Detected in Files:
{json.dumps(symbols_summary, indent=2)}

### Traceback Locations Detected:
{json.dumps(traceback_matches, indent=2)}

Determine the target files and symbols for this issue. Return JSON only.
"""

        try:
            response_text = self.invoke_prompt(user_prompt)
            data = self.extract_json_from_response(response_text)
        except Exception:
            # Fallback if LLM is unavailable or offline
            data = {}

        target_files = data.get("target_files", [])
        if not target_files and traceback_matches:
            target_files = [m["matched_repo_file"] for m in traceback_matches if m.get("matched_repo_file")]

        if not target_files and files_sample:
            # Fallback heuristic: file with same name or first python file
            target_files = [files_sample[0]]

        # Extract file snippets for the localized targets
        snippets = {}
        for target in target_files:
            syms = data.get("symbols", [])
            target_syms = [s for s in syms if s.get("file_path") == target]
            if target_syms:
                s_start = min(s.get("line_start", 1) for s in target_syms)
                s_end = max(s.get("line_end", 50) for s in target_syms)
            else:
                s_start, s_end = 1, 80

            snip_info = extract_code_snippet(repo_path, target, s_start, s_end, context_lines=5)
            if "snippet" in snip_info:
                snippets[target] = snip_info["snippet"]

        return {
            "target_files": target_files,
            "symbols": data.get("symbols", []),
            "file_snippets": snippets,
            "summary": data.get("summary", "Code reading and localization completed."),
            "traceback_matches": traceback_matches,
        }
