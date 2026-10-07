"""
Observability, Structured Audit Logging & Tracing Layer (Phase 8).
Tracks latency, token usage, agent decisions, sandbox execution traces,
container lifecycles, and unified diff statistics for auditing.
Supports LangSmith and OpenTelemetry integrations.
"""

import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from pydantic import BaseModel, Field

from agentic_code.config import Settings, settings

logger = logging.getLogger("agentic_code.observability")


# ---------------------------------------------------------------------------
# Diff Statistics Model
# ---------------------------------------------------------------------------

class DiffStats(BaseModel):
    """Unified diff statistics for auditing code changes."""
    files_modified: List[str] = Field(default_factory=list)
    lines_added: int = 0
    lines_deleted: int = 0
    total_lines_changed: int = 0

    @classmethod
    def from_unified_diff(cls, diff_text: str) -> "DiffStats":
        """
        Parses a standard unified diff to extract modified files,
        lines added (+), and lines deleted (-).
        """
        if not diff_text or not diff_text.strip():
            return cls()

        files: set[str] = set()
        added = 0
        deleted = 0

        for line in diff_text.splitlines():
            # File indicators in unified diff
            if line.startswith("--- ") or line.startswith("+++ "):
                # e.g. "--- a/requests/models.py" or "+++ b/requests/models.py"
                parts = line.split(maxsplit=1)
                if len(parts) > 1:
                    raw_path = parts[1].strip()
                    # Strip Git prefixes (a/, b/)
                    cleaned = re.sub(r"^[ab]/", "", raw_path)
                    if cleaned and cleaned != "/dev/null":
                        files.add(cleaned)
                continue
            elif line.startswith("diff --git "):
                # e.g. "diff --git a/file.py b/file.py"
                match = re.search(r"diff --git a/(.+) b/(.+)", line)
                if match:
                    files.add(match.group(2))
                continue

            # Chunk headers
            if line.startswith("@@"):
                continue

            # Code line changes
            if line.startswith("+"):
                added += 1
            elif line.startswith("-"):
                deleted += 1

        return cls(
            files_modified=sorted(list(files)),
            lines_added=added,
            lines_deleted=deleted,
            total_lines_changed=added + deleted,
        )


# ---------------------------------------------------------------------------
# Tracing & Audit Event Models
# ---------------------------------------------------------------------------

class AgentDecisionTrace(BaseModel):
    """Record of an individual agent decision in the LangGraph workflow."""
    node: str
    persona: str
    inputs_summary: str
    outputs_summary: str
    duration_seconds: float
    timestamp: float = Field(default_factory=time.time)
    token_usage: Optional[Dict[str, int]] = None


class SandboxExecutionTrace(BaseModel):
    """Record of a Docker/local sandbox command execution."""
    command: str
    exit_code: int
    passed: bool
    duration_seconds: float
    container_id: str = "sandbox-container"
    timestamp: float = Field(default_factory=time.time)


class ContainerLifecycleEvent(BaseModel):
    """Lifecycle event of a sandbox container (e.g. CREATE, START, STOP, DESTROY, ROLLBACK)."""
    action: str
    container_id: str
    timestamp: float = Field(default_factory=time.time)
    details: Dict[str, Any] = Field(default_factory=dict)


class RunAuditTrace(BaseModel):
    """Full execution trace of an issue resolution run."""
    run_id: str
    issue_id: str
    repo: str
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_seconds: float = 0.0
    success: bool = False
    agent_decisions: List[AgentDecisionTrace] = Field(default_factory=list)
    sandbox_executions: List[SandboxExecutionTrace] = Field(default_factory=list)
    container_events: List[ContainerLifecycleEvent] = Field(default_factory=list)
    diff_stats: Optional[DiffStats] = None
    token_usage: Dict[str, int] = Field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )


# ---------------------------------------------------------------------------
# Audit Logger Engine
# ---------------------------------------------------------------------------

class AuditLogger:
    """
    Centralized, thread-safe audit logger for tracing agent decisions,
    sandbox executions, container lifecycles, diff metrics, and token usage.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: Dict[str, RunAuditTrace] = {}
        self._active_run_id: Optional[str] = None

    def start_run(self, issue_id: str, repo: str = "") -> str:
        """Starts a new tracked issue resolution execution run."""
        with self._lock:
            run_id = f"run_{issue_id}_{int(time.time() * 1000)}"
            trace = RunAuditTrace(
                run_id=run_id,
                issue_id=str(issue_id),
                repo=repo,
                start_time=time.time(),
            )
            self._runs[run_id] = trace
            self._active_run_id = run_id
            logger.info("Started audit trace run %s for issue %s (%s)", run_id, issue_id, repo)
            return run_id

    def get_active_trace(self) -> Optional[RunAuditTrace]:
        """Returns the currently active run trace, if any."""
        with self._lock:
            if self._active_run_id and self._active_run_id in self._runs:
                return self._runs[self._active_run_id]
            return None

    def record_agent_decision(
        self,
        node: str,
        persona: str,
        inputs: Any,
        outputs: Any,
        duration_seconds: float,
        token_usage: Optional[Dict[str, int]] = None,
    ) -> None:
        """Records an agent decision step into the active audit trace."""
        with self._lock:
            trace = self._get_or_create_active_trace()
            decision = AgentDecisionTrace(
                node=node,
                persona=persona,
                inputs_summary=str(inputs)[:200],
                outputs_summary=str(outputs)[:200],
                duration_seconds=duration_seconds,
                token_usage=token_usage,
            )
            trace.agent_decisions.append(decision)
            if token_usage:
                trace.token_usage["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
                trace.token_usage["completion_tokens"] += token_usage.get("completion_tokens", 0)
                trace.token_usage["total_tokens"] += token_usage.get("total_tokens", 0)

    def record_sandbox_trace(
        self,
        command: str,
        exit_code: int,
        duration_seconds: float,
        passed: bool,
        container_id: str = "sandbox-container",
    ) -> None:
        """Logs a sandbox command execution into the active trace."""
        with self._lock:
            trace = self._get_or_create_active_trace()
            sb_trace = SandboxExecutionTrace(
                command=command,
                exit_code=exit_code,
                passed=passed,
                duration_seconds=duration_seconds,
                container_id=container_id,
            )
            trace.sandbox_executions.append(sb_trace)

    def record_container_lifecycle(
        self,
        action: str,
        container_id: str = "sandbox-container",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Logs container lifecycle actions (e.g., CREATE, START, STOP, ROLLBACK)."""
        with self._lock:
            trace = self._get_or_create_active_trace()
            event = ContainerLifecycleEvent(
                action=action,
                container_id=container_id,
                details=details or {},
            )
            trace.container_events.append(event)

    def record_diff(self, patch: str) -> DiffStats:
        """Parses and logs diff statistics for the generated patch."""
        diff_stats = DiffStats.from_unified_diff(patch)
        with self._lock:
            trace = self._get_or_create_active_trace()
            trace.diff_stats = diff_stats
        return diff_stats

    def record_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Accumulates token usage into the active trace."""
        with self._lock:
            trace = self._get_or_create_active_trace()
            trace.token_usage["prompt_tokens"] += prompt_tokens
            trace.token_usage["completion_tokens"] += completion_tokens
            trace.token_usage["total_tokens"] += prompt_tokens + completion_tokens

    def end_run(self, success: bool) -> Optional[RunAuditTrace]:
        """Finalizes the active run, computes duration, and returns the trace."""
        with self._lock:
            if not self._active_run_id or self._active_run_id not in self._runs:
                return None
            trace = self._runs[self._active_run_id]
            trace.end_time = time.time()
            trace.duration_seconds = round(trace.end_time - trace.start_time, 3)
            trace.success = success
            self._active_run_id = None
            logger.info(
                "Completed audit trace run %s: success=%s, duration=%.2fs",
                trace.run_id,
                success,
                trace.duration_seconds,
            )
            return trace

    def export_audit_log(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Exports all recorded run audit traces to a structured dictionary or JSON file."""
        with self._lock:
            data = {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "runs_count": len(self._runs),
                "runs": [trace.model_dump() for trace in self._runs.values()],
            }

        if filepath:
            out_path = Path(filepath)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        return data

    def clear(self) -> None:
        """Resets all runs and active state."""
        with self._lock:
            self._runs.clear()
            self._active_run_id = None

    def _get_or_create_active_trace(self) -> RunAuditTrace:
        """Internal helper to ensure a valid active trace is always present."""
        if not self._active_run_id or self._active_run_id not in self._runs:
            self._active_run_id = f"run_auto_{int(time.time() * 1000)}"
            self._runs[self._active_run_id] = RunAuditTrace(
                run_id=self._active_run_id,
                issue_id="auto",
                repo="default",
                start_time=time.time(),
            )
        return self._runs[self._active_run_id]


# Global singleton audit logger
audit_logger = AuditLogger()


# ---------------------------------------------------------------------------
# Tracing Integration (LangSmith & OpenTelemetry Contexts)
# ---------------------------------------------------------------------------

class SpanContext:
    """Lightweight context for a span tracking latency and attributes."""

    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.attributes = attributes or {}
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.duration_seconds: float = 0.0
        self.status: str = "OK"

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def finish(self, status: str = "OK") -> None:
        self.end_time = time.time()
        self.duration_seconds = round(self.end_time - self.start_time, 4)
        self.status = status


@contextmanager
def trace_span(
    name: str, attributes: Optional[Dict[str, Any]] = None
) -> Generator[SpanContext, None, None]:
    """
    Context manager for tracing execution spans with OpenTelemetry / LangSmith hooks.
    Tracks latency, status, and custom attributes.
    """
    span = SpanContext(name, attributes)
    try:
        yield span
        span.finish(status="OK")
    except Exception as exc:
        span.finish(status="ERROR")
        span.set_attribute("error", str(exc))
        raise


def setup_observability(custom_settings: Optional[Settings] = None) -> Dict[str, Any]:
    """
    Initializes tracing hooks for LangSmith and OpenTelemetry based on settings.
    """
    cfg = custom_settings or settings
    status: Dict[str, Any] = {
        "langsmith_enabled": False,
        "otel_enabled": False,
    }

    # 1. LangSmith Tracing Configuration
    if cfg.LANGCHAIN_TRACING_V2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = cfg.LANGCHAIN_PROJECT
        if cfg.LANGCHAIN_API_KEY:
            os.environ["LANGCHAIN_API_KEY"] = cfg.LANGCHAIN_API_KEY
        status["langsmith_enabled"] = True
        logger.info("LangSmith tracing enabled for project '%s'", cfg.LANGCHAIN_PROJECT)

    # 2. OpenTelemetry Tracing Configuration
    if cfg.OTEL_EXPORTER_ENABLED:
        try:
            from opentelemetry import trace  # type: ignore
            # Basic OpenTelemetry setup if library is present
            tracer = trace.get_tracer("agentic_code")
            status["otel_enabled"] = True
            logger.info("OpenTelemetry tracer '%s' initialized", tracer)
        except ImportError:
            logger.warning(
                "OTEL_EXPORTER_ENABLED is True but 'opentelemetry' package is not installed."
            )
            status["otel_enabled"] = False

    return status
