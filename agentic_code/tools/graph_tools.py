"""
Lightweight dependency and call graph generator to locate caller/callee relationships across modules.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Set


class CallVisitor(ast.NodeVisitor):
    """
    Visits AST function definitions to record call expressions within them.
    """

    def __init__(self, current_module: str):
        self.current_module = current_module
        self.current_func: str = ""
        # Caller -> set of callees
        self.calls: Dict[str, Set[str]] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev = self.current_func
        self.current_func = f"{self.current_module}:{node.name}"
        if self.current_func not in self.calls:
            self.calls[self.current_func] = set()
        self.generic_visit(node)
        self.current_func = prev

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        prev = self.current_func
        self.current_func = f"{self.current_module}:{node.name}"
        if self.current_func not in self.calls:
            self.calls[self.current_func] = set()
        self.generic_visit(node)
        self.current_func = prev

    def visit_Call(self, node: ast.Call) -> None:
        if self.current_func:
            func_name = self._resolve_call_name(node.func)
            if func_name:
                self.calls[self.current_func].add(func_name)
        self.generic_visit(node)

    def _resolve_call_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self._resolve_call_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return ""


def build_repository_call_graph(
    repo_path: str,
    max_files: int = 150,
) -> Dict[str, List[str]]:
    """
    Constructs caller -> callees mapping for functions across python files in repo_path.
    """
    root = Path(repo_path).resolve()
    call_graph: Dict[str, List[str]] = {}

    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ["venv", ".venv"]]
        for f in filenames:
            if not f.endswith(".py"):
                continue

            file_path = Path(dirpath) / f
            rel_mod = str(file_path.relative_to(root)).replace("\\", "/").replace(".py", "")

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(content)
                visitor = CallVisitor(current_module=rel_mod)
                visitor.visit(tree)

                for caller, callees in visitor.calls.items():
                    call_graph[caller] = sorted(list(callees))
            except Exception:
                pass

            count += 1
            if count >= max_files:
                break
        if count >= max_files:
            break

    return call_graph
