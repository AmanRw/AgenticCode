"""
Tests for Docker Sandbox Engine and Fallback Sandbox.
"""

import sys
import tempfile

from agentic_code.sandbox.fallback import LocalSubprocessSandbox
from agentic_code.sandbox.sandbox import DockerSandbox


def test_sandbox_config():
    sandbox = DockerSandbox(
        repo_path=".",
        user="1000:1000",
        timeout_seconds=30,
        mem_limit="2048m",
        cpu_quota=100000,
        network_mode="none",
    )
    assert sandbox.user == "1000:1000"
    assert sandbox.timeout_seconds == 30
    assert sandbox.mem_limit == "2048m"
    assert sandbox.cpu_quota == 100000
    assert sandbox.network_mode == "none"


def test_fallback_sandbox_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = LocalSubprocessSandbox(tmpdir, timeout_seconds=10)
        # Run a simple python one-liner using current interpreter
        cmd = f'"{sys.executable}" -c "print(1 + 1)"'
        res = sandbox.execute(cmd)
        assert res.passed is True
        assert res.exit_code == 0
        assert "2" in res.stdout


def test_fallback_sandbox_failure():
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = LocalSubprocessSandbox(tmpdir, timeout_seconds=10)
        cmd = f'"{sys.executable}" -c "raise ValueError(\'Test error\')"'
        res = sandbox.execute(cmd)
        assert res.passed is False
        assert res.exit_code != 0
        assert "ValueError" in res.stderr or "ValueError" in res.stdout


def test_fallback_sandbox_timeout():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1-second timeout on a 5-second sleep
        sandbox = LocalSubprocessSandbox(tmpdir, timeout_seconds=1)
        cmd = f'"{sys.executable}" -c "import time; time.sleep(5)"'
        res = sandbox.execute(cmd, timeout=1)
        assert res.passed is False
        assert res.exit_code == -1
        assert "TIMEOUT" in res.stderr or "Timeout" in (res.error_summary or "")


def test_docker_sandbox_fallback_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Tests that DockerSandbox routes through fallback when Docker is unavailable
        sandbox = DockerSandbox(repo_path=tmpdir, timeout_seconds=10)
        cmd = f'"{sys.executable}" -c "print(\'AgenticCode Sandbox OK\')"'
        res = sandbox.execute_command(cmd)
        assert res.passed is True
        assert "AgenticCode Sandbox OK" in res.stdout
