"""
Tests for Phase 8: Observability, Structured Audit Logging & Tracing.
"""

import json
import tempfile
import time
from pathlib import Path

from agentic_code.config import Settings
from agentic_code.observability import (
    AuditLogger,
    DiffStats,
    setup_observability,
    trace_span,
)


def test_diff_stats_calculation():
    # 1. Single file diff
    sample_diff = """--- a/requests/models.py
+++ b/requests/models.py
@@ -10,3 +10,4 @@
 def send():
-    return False
+    # Fixed redirect capitalization
+    return True
"""
    stats = DiffStats.from_unified_diff(sample_diff)
    assert stats.files_modified == ["requests/models.py"]
    assert stats.lines_added == 2
    assert stats.lines_deleted == 1
    assert stats.total_lines_changed == 3

    # 2. Multi-file diff with git prefix
    multi_diff = """diff --git a/flask/app.py b/flask/app.py
--- a/flask/app.py
+++ b/flask/app.py
@@ -1,2 +1,3 @@
+import os
diff --git a/flask/blueprints.py b/flask/blueprints.py
--- a/flask/blueprints.py
+++ b/flask/blueprints.py
@@ -5,2 +5,3 @@
-old_code()
+new_code()
"""
    multi_stats = DiffStats.from_unified_diff(multi_diff)
    assert set(multi_stats.files_modified) == {"flask/app.py", "flask/blueprints.py"}
    assert multi_stats.lines_added == 2
    assert multi_stats.lines_deleted == 1
    assert multi_stats.total_lines_changed == 3

    # 3. Empty diff
    empty_stats = DiffStats.from_unified_diff("")
    assert empty_stats.files_modified == []
    assert empty_stats.total_lines_changed == 0


def test_audit_logger_lifecycle():
    logger = AuditLogger()
    run_id = logger.start_run(issue_id="issue-42", repo="psf/requests")
    assert run_id.startswith("run_issue-42_")
    assert logger.get_active_trace() is not None

    # Record agent decision
    logger.record_agent_decision(
        node="node_code_reader",
        persona="CodeReaderAgent",
        inputs={"issue_id": 42},
        outputs={"target_files": ["models.py"]},
        duration_seconds=0.45,
        token_usage={"prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180},
    )

    # Record sandbox trace
    logger.record_sandbox_trace(
        command="pytest -v test_requests.py",
        exit_code=0,
        duration_seconds=1.2,
        passed=True,
        container_id="sandbox-test-1",
    )

    # Record container lifecycle
    logger.record_container_lifecycle(
        action="START",
        container_id="sandbox-test-1",
        details={"isolated": True},
    )

    # Record diff & tokens
    diff_stats = logger.record_diff("--- a/foo.py\n+++ b/foo.py\n+bar")
    assert diff_stats.lines_added == 1
    logger.record_tokens(prompt_tokens=50, completion_tokens=25)

    # End run
    time.sleep(0.01)
    trace = logger.end_run(success=True)
    assert trace is not None
    assert trace.success is True
    assert trace.duration_seconds >= 0.01
    assert len(trace.agent_decisions) == 1
    assert len(trace.sandbox_executions) == 1
    assert len(trace.container_events) == 1
    assert trace.token_usage["prompt_tokens"] == 170
    assert trace.token_usage["completion_tokens"] == 85
    assert trace.token_usage["total_tokens"] == 255


def test_audit_logger_export():
    logger = AuditLogger()
    logger.start_run("issue-99", "pallets/flask")
    logger.end_run(success=False)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        export_file = Path(tmpdir) / "audit_log.json"
        data = logger.export_audit_log(str(export_file))

        assert data["runs_count"] >= 1
        assert export_file.exists()
        loaded = json.loads(export_file.read_text(encoding="utf-8"))
        assert loaded["runs_count"] == data["runs_count"]


def test_trace_span_context():
    with trace_span("test_task", attributes={"stage": "analysis"}) as span:
        assert span.name == "test_task"
        assert span.attributes["stage"] == "analysis"
        span.set_attribute("custom_val", 100)
        time.sleep(0.01)

    assert span.status == "OK"
    assert span.duration_seconds >= 0.01
    assert span.attributes["custom_val"] == 100


def test_setup_observability():
    custom = Settings(
        LANGCHAIN_TRACING_V2=True,
        LANGCHAIN_PROJECT="TestBenchmarkObservability",
        LANGCHAIN_API_KEY="test_key",
        OTEL_EXPORTER_ENABLED=False,
    )
    status = setup_observability(custom)
    assert status["langsmith_enabled"] is True
