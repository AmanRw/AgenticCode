"""
AST and Tree-Sitter tools for code retrieval and symbol extraction without loading full file bodies.
"""

import ast
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_code.state import SymbolInfo


class SymbolExtractor(ast.NodeVisitor):
    """
    Extracts classes, methods, functions, docstrings, and imports from a Python AST.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.symbols: List[SymbolInfo] = []
        self.imports: List[str] = []
        self.current_class: Optional[str] = None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        docstring = ast.get_docstring(node)
        bases = [ast.unparse(b) for b in node.bases]
        sig = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"

        sym = SymbolInfo(
            name=node.name,
            kind="class",
            file_path=self.file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=docstring,
            signature=sig,
        )
        self.symbols.append(sym)

        prev_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_func(node, is_async=False)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_func(node, is_async=True)
        self.generic_visit(node)

    def _record_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool) -> None:
        docstring = ast.get_docstring(node)
        kind = "method" if self.current_class else "function"
        full_name = f"{self.current_class}.{node.name}" if self.current_class else node.name

        args_str = ast.unparse(node.args)
        returns_str = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        async_prefix = "async " if is_async else ""
        sig = f"{async_prefix}def {node.name}({args_str}){returns_str}"

        sym = SymbolInfo(
            name=full_name,
            kind=kind,
            file_path=self.file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=docstring,
            signature=sig,
        )
        self.symbols.append(sym)


def extract_symbols_from_source(source_code: str, file_path: str = "") -> Dict[str, Any]:
    """
    Parses source code into AST and extracts top-level classes, methods, signatures, and imports.
    """
    try:
        tree = ast.parse(source_code)
        extractor = SymbolExtractor(file_path=file_path)
        extractor.visit(tree)
        return {
            "file_path": file_path,
            "symbols": [s.model_dump() for s in extractor.symbols],
            "imports": extractor.imports,
            "error": None,
        }
    except SyntaxError as e:
        # Graceful fallback on syntax error
        return {
            "file_path": file_path,
            "symbols": [],
            "imports": [],
            "error": f"SyntaxError: {str(e)}",
        }


def generate_repository_symbol_outline(
    repo_path: str,
    ignore_dirs: Optional[List[str]] = None,
    max_files: int = 500,
) -> Dict[str, Any]:
    """
    Generates a lightweight outline of Python files and their exported symbols across the repo.
    Avoids loading full file bodies into memory or LLM context.
    """
    root = Path(repo_path).resolve()
    ignored = set(ignore_dirs or [
        ".git", ".venv", "venv", "__pycache__", "build", "dist",
        ".egg-info", "node_modules", ".pytest_cache", ".ruff_cache", ".mypy_cache"
    ])

    file_tree = []
    symbol_catalog: Dict[str, List[Dict[str, Any]]] = {}
    import_catalog: Dict[str, List[str]] = {}

    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignored and not d.startswith(".")]
        for f in filenames:
            if not f.endswith(".py"):
                continue

            full_p = Path(dirpath) / f
            rel_p = str(full_p.relative_to(root)).replace("\\", "/")
            file_tree.append(rel_p)

            try:
                content = full_p.read_text(encoding="utf-8", errors="replace")
                info = extract_symbols_from_source(content, file_path=rel_p)
                symbol_catalog[rel_p] = info["symbols"]
                import_catalog[rel_p] = info["imports"]
            except Exception as e:
                symbol_catalog[rel_p] = [{"error": str(e)}]

            count += 1
            if count >= max_files:
                break
        if count >= max_files:
            break

    return {
        "file_tree": sorted(file_tree),
        "symbol_catalog": symbol_catalog,
        "import_catalog": import_catalog,
    }


def extract_code_snippet(
    repo_path: str,
    file_path: str,
    start_line: int,
    end_line: int,
    context_lines: int = 5,
) -> Dict[str, Any]:
    """
    Extracts an exact slice of lines from a file with surrounding context.
    """
    full_path = (Path(repo_path) / file_path).resolve()
    if not full_path.exists():
        return {"error": f"File {file_path} does not exist", "snippet": ""}

    try:
        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
        total_lines = len(lines)

        actual_start = max(1, start_line - context_lines)
        actual_end = min(total_lines, end_line + context_lines)

        selected = lines[actual_start - 1 : actual_end]
        annotated = [
            f"{i:4d}: {line}" for i, line in enumerate(selected, start=actual_start)
        ]
        return {
            "file_path": file_path,
            "start_line": actual_start,
            "end_line": actual_end,
            "total_lines": total_lines,
            "snippet": "\n".join(annotated),
            "raw_snippet": "\n".join(selected),
        }
    except Exception as e:
        return {"error": str(e), "snippet": ""}
