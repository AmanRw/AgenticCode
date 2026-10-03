"""
Tests for Phase 3 Code Retrieval and AST Analysis tools.
"""

import tempfile
from pathlib import Path

from agentic_code.tools.ast_tools import (
    extract_code_snippet,
    extract_symbols_from_source,
    generate_repository_symbol_outline,
)
from agentic_code.tools.graph_tools import (
    build_repository_call_graph,
)
from agentic_code.tools.search_tools import (
    search_codebase,
    search_traceback_symbols,
)

SAMPLE_PYTHON_CODE = '''"""Sample module docstring."""
import os
from sys import version_info

class Calculator:
    """A simple calculator class."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return a / b

def compute(x: int) -> int:
    calc = Calculator()
    return calc.add(x, 10)
'''


def test_extract_symbols_from_source():
    res = extract_symbols_from_source(SAMPLE_PYTHON_CODE, file_path="sample.py")
    assert res["file_path"] == "sample.py"
    assert "os" in res["imports"]
    assert "sys.version_info" in res["imports"]

    names = [s["name"] for s in res["symbols"]]
    assert "Calculator" in names
    assert "Calculator.add" in names
    assert "Calculator.divide" in names
    assert "compute" in names

    calc_sym = next(s for s in res["symbols"] if s["name"] == "Calculator")
    assert calc_sym["kind"] == "class"
    assert calc_sym["docstring"] == "A simple calculator class."

    add_sym = next(s for s in res["symbols"] if s["name"] == "Calculator.add")
    assert add_sym["kind"] == "method"
    assert "def add(self, a: int, b: int) -> int" in add_sym["signature"]


def test_generate_repository_symbol_outline():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "calc.py"
        p.write_text(SAMPLE_PYTHON_CODE, encoding="utf-8")

        outline = generate_repository_symbol_outline(tmpdir)
        assert "calc.py" in outline["file_tree"]
        symbols = outline["symbol_catalog"]["calc.py"]
        assert len(symbols) >= 3


def test_extract_code_snippet():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "calc.py"
        p.write_text(SAMPLE_PYTHON_CODE, encoding="utf-8")

        snippet_info = extract_code_snippet(tmpdir, "calc.py", start_line=13, end_line=16, context_lines=1)
        assert snippet_info["start_line"] <= 13
        assert snippet_info["end_line"] >= 16
        assert "ZeroDivisionError" in snippet_info["snippet"]


def test_search_codebase():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "calc.py"
        p.write_text(SAMPLE_PYTHON_CODE, encoding="utf-8")

        results = search_codebase(tmpdir, "ZeroDivisionError")
        assert len(results) >= 1
        assert results[0]["file"] == "calc.py"
        assert "ZeroDivisionError" in results[0]["line_content"]


def test_search_traceback_symbols():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "calc.py"
        p.write_text(SAMPLE_PYTHON_CODE, encoding="utf-8")

        tb = f'Traceback (most recent call last):\n  File "{p}", line 14, in divide\n    raise ZeroDivisionError'
        matches = search_traceback_symbols(tmpdir, tb)
        assert len(matches) == 1
        assert matches[0]["function"] == "divide"
        assert matches[0]["line"] == 14


def test_build_repository_call_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "calc.py"
        p.write_text(SAMPLE_PYTHON_CODE, encoding="utf-8")

        graph = build_repository_call_graph(tmpdir)
        assert "calc:compute" in graph
        assert any("add" in callee for callee in graph["calc:compute"])
