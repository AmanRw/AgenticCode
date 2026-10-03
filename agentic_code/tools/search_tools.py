"""
Targeted lexical and structural search tools across repository code.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def search_codebase(
    repo_path: str,
    query: str,
    file_extensions: Optional[List[str]] = None,
    is_regex: bool = False,
    ignore_dirs: Optional[List[str]] = None,
    max_results: int = 50,
) -> List[Dict[str, Any]]:
    """
    Scans files in repository for query or regex pattern.
    Returns matched file, line number, line text, and surrounding snippet.
    """
    root = Path(repo_path).resolve()
    exts = tuple(file_extensions or [".py", ".toml", ".yaml", ".json", ".md"])
    ignored = set(ignore_dirs or [
        ".git", ".venv", "venv", "__pycache__", "build", "dist",
        ".egg-info", "node_modules", ".pytest_cache"
    ])

    results: List[Dict[str, Any]] = []

    try:
        flags = re.IGNORECASE
        pattern = re.compile(query if is_regex else re.escape(query), flags)
    except re.error:
        # Fallback to literal if regex compilation fails
        pattern = re.compile(re.escape(query), re.IGNORECASE)

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignored and not d.startswith(".")]

        for f in filenames:
            if not f.endswith(exts):
                continue

            full_path = Path(dirpath) / f
            rel_path = str(full_path.relative_to(root)).replace("\\", "/")

            try:
                lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
                for idx, line in enumerate(lines, start=1):
                    if pattern.search(line):
                        # Extract 2 lines before and after
                        start_c = max(0, idx - 3)
                        end_c = min(len(lines), idx + 2)
                        context_snippet = "\n".join(
                            f"{j+1:4d}: {lines[j]}" for j in range(start_c, end_c)
                        )

                        results.append({
                            "file": rel_path,
                            "line_number": idx,
                            "line_content": line.strip(),
                            "context": context_snippet,
                        })

                        if len(results) >= max_results:
                            return results
            except Exception:
                continue

    return results


def search_traceback_symbols(
    repo_path: str,
    traceback_text: str,
) -> List[Dict[str, Any]]:
    """
    Extracts file paths, function names, and line numbers from an error traceback
    and matches them against the repository codebase.
    """
    # Regex to capture: File "...path...", line 123, in func_name
    tb_pattern = re.compile(r'File "([^"]+)", line (\d+)(?:, in (\w+))?')
    matches = tb_pattern.findall(traceback_text)

    found_locations: List[Dict[str, Any]] = []
    root = Path(repo_path).resolve()

    for path_str, line_str, func_name in matches:
        norm_path = path_str.replace("\\", "/")
        # Find if file exists in repo or is relative
        matched_file = None
        for p in root.rglob("*.py"):
            rel = str(p.relative_to(root)).replace("\\", "/")
            if rel in norm_path or norm_path.endswith(rel):
                matched_file = rel
                break

        found_locations.append({
            "raw_path": path_str,
            "matched_repo_file": matched_file,
            "line": int(line_str),
            "function": func_name or "",
        })

    return found_locations
