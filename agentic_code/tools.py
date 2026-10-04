"""
Unified Code Retrieval & AST Analysis Tools (Phase 3).
Provides AST parsing, symbol extraction, precise snippet slicing,
lexical/traceback search, and caller/callee dependency graphs.
"""

import ast
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from agentic_code.state import SymbolInfo

# ---------------------------------------------------------------------------
# 1. AST Symbol Extractor
# ---------------------------------------------------------------------------

class SymbolExtractor(ast.NodeVisitor):
    """Extracts top-level classes, methods, signatures, docstrings, and imports."""

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

        self.symbols.append(
            SymbolInfo(
                name=node.name,
                kind="class",
                file_path=self.file_path,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                docstring=docstring,
                signature=sig,
            )
        )
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

        self.symbols.append(
            SymbolInfo(
                name=full_name,
                kind=kind,
                file_path=self.file_path,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                docstring=docstring,
                signature=sig,
            )
        )


def extract_symbols_from_source(source_code: str, file_path: str = "") -> Dict[str, Any]:
    """Parses Python source into AST and extracts symbol outlines and imports."""
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
        return {"file_path": file_path, "symbols": [], "imports": [], "error": f"SyntaxError: {e}"}


def generate_repository_symbol_outline(
    repo_path: str,
    ignore_dirs: Optional[List[str]] = None,
    max_files: int = 500,
) -> Dict[str, Any]:
    """Extracts compact repository file tree and symbols without loading full bodies into LLM context."""
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
    """Extracts exact lines with context boundaries."""
    full_path = (Path(repo_path) / file_path).resolve()
    if not full_path.exists():
        return {"error": f"File {file_path} does not exist", "snippet": ""}

    try:
        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
        total_lines = len(lines)
        actual_start = max(1, start_line - context_lines)
        actual_end = min(total_lines, end_line + context_lines)
        selected = lines[actual_start - 1 : actual_end]
        annotated = [f"{i:4d}: {line}" for i, line in enumerate(selected, start=actual_start)]
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


# ---------------------------------------------------------------------------
# 2. Targeted Search & Traceback Matcher
# ---------------------------------------------------------------------------

def search_codebase(
    repo_path: str,
    query: str,
    file_extensions: Optional[List[str]] = None,
    is_regex: bool = False,
    ignore_dirs: Optional[List[str]] = None,
    max_results: int = 50,
) -> List[Dict[str, Any]]:
    """Fast lexical and pattern scanner across repository code."""
    root = Path(repo_path).resolve()
    exts = tuple(file_extensions or [".py", ".toml", ".yaml", ".json", ".md"])
    ignored = set(ignore_dirs or [
        ".git", ".venv", "venv", "__pycache__", "build", "dist", ".egg-info", "node_modules"
    ])
    results: List[Dict[str, Any]] = []

    try:
        pattern = re.compile(query if is_regex else re.escape(query), re.IGNORECASE)
    except re.error:
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
                        start_c = max(0, idx - 3)
                        end_c = min(len(lines), idx + 2)
                        context = "\n".join(f"{j+1:4d}: {lines[j]}" for j in range(start_c, end_c))
                        results.append({
                            "file": rel_path,
                            "line_number": idx,
                            "line_content": line.strip(),
                            "context": context,
                        })
                        if len(results) >= max_results:
                            return results
            except Exception:
                continue
    return results


def search_traceback_symbols(repo_path: str, traceback_text: str) -> List[Dict[str, Any]]:
    """Extracts file paths, function names, and line numbers from Python error tracebacks."""
    tb_pattern = re.compile(r'File "([^"]+)", line (\d+)(?:, in (\w+))?')
    matches = tb_pattern.findall(traceback_text)
    root = Path(repo_path).resolve()
    found = []

    for path_str, line_str, func_name in matches:
        norm = path_str.replace("\\", "/")
        matched_file = None
        for p in root.rglob("*.py"):
            rel = str(p.relative_to(root)).replace("\\", "/")
            if rel in norm or norm.endswith(rel):
                matched_file = rel
                break
        found.append({
            "raw_path": path_str,
            "matched_repo_file": matched_file,
            "line": int(line_str),
            "function": func_name or "",
        })
    return found


# ---------------------------------------------------------------------------
# 3. Call Graph Generator
# ---------------------------------------------------------------------------

class CallVisitor(ast.NodeVisitor):
    """Visits function definitions and maps call relationships."""

    def __init__(self, current_module: str):
        self.current_module = current_module
        self.current_func: str = ""
        self.calls: Dict[str, Set[str]] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev = self.current_func
        self.current_func = f"{self.current_module}:{node.name}"
        self.calls.setdefault(self.current_func, set())
        self.generic_visit(node)
        self.current_func = prev

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        prev = self.current_func
        self.current_func = f"{self.current_module}:{node.name}"
        self.calls.setdefault(self.current_func, set())
        self.generic_visit(node)
        self.current_func = prev

    def visit_Call(self, node: ast.Call) -> None:
        if self.current_func:
            name = self._resolve_name(node.func)
            if name:
                self.calls[self.current_func].add(name)
        self.generic_visit(node)

    def _resolve_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self._resolve_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return ""


def build_repository_call_graph(repo_path: str, max_files: int = 150) -> Dict[str, List[str]]:
    """Constructs caller -> callees mapping across repository files."""
    root = Path(repo_path).resolve()
    graph: Dict[str, List[str]] = {}
    count = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ["venv", ".venv"]]
        for f in filenames:
            if not f.endswith(".py"):
                continue
            file_path = Path(dirpath) / f
            rel_mod = str(file_path.relative_to(root)).replace("\\", "/").replace(".py", "")
            try:
                tree = ast.parse(file_path.read_text(encoding="utf-8", errors="replace"))
                visitor = CallVisitor(rel_mod)
                visitor.visit(tree)
                for caller, callees in visitor.calls.items():
                    graph[caller] = sorted(list(callees))
            except Exception:
                pass
            count += 1
            if count >= max_files:
                break
        if count >= max_files:
            break
    return graph
