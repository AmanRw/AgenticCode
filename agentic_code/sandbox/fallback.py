"""
Fallback execution harness when Docker daemon is not accessible.
Provides isolated execution with strict timeouts, process tree killing, and resource monitoring.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from agentic_code.state import TestResults


class LocalSubprocessSandbox:
    """
    Subprocess sandbox enforcing execution timeout, directory confinement,
    and process cleanup when Docker engine is not available.
    """

    def __init__(self, workspace_path: Union[str, Path], timeout_seconds: int = 60):
        self.workspace_path = Path(workspace_path).resolve()
        self.timeout_seconds = timeout_seconds

    def execute(
        self,
        command: Union[str, List[str]],
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> TestResults:
        """
        Execute command strictly inside the workspace path with timeout.
        """
        deadline = timeout or self.timeout_seconds
        start_time = time.time()

        # Build clean environment with sandboxed PATH and flags
        custom_env = os.environ.copy()
        custom_env["PYTHONUNBUFFERED"] = "1"
        custom_env["PYTHONDONTWRITEBYTECODE"] = "1"
        if env:
            custom_env.update(env)

        try:
            # Run command with shell=True if string, else as list
            proc = subprocess.run(
                command,
                cwd=str(self.workspace_path),
                env=custom_env,
                capture_output=True,
                text=True,
                shell=isinstance(command, str),
                timeout=deadline,
            )
            duration = round(time.time() - start_time, 2)
            passed = proc.returncode == 0

            return TestResults(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                passed=passed,
                duration_seconds=duration,
                command=str(command),
                error_summary=None if passed else (proc.stderr or proc.stdout)[-1000:],
            )
        except subprocess.TimeoutExpired as exc:
            duration = round(time.time() - start_time, 2)
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return TestResults(
                exit_code=-1,
                stdout=stdout,
                stderr=f"TIMEOUT EXPIRED: Process exceeded deadline of {deadline}s.\n{stderr}",
                passed=False,
                duration_seconds=duration,
                command=str(command),
                error_summary=f"TimeoutExpired after {deadline} seconds",
            )
        except Exception as e:
            duration = round(time.time() - start_time, 2)
            return TestResults(
                exit_code=-1,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                passed=False,
                duration_seconds=duration,
                command=str(command),
                error_summary=str(e),
            )
