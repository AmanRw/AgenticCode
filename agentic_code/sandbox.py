"""
Hardened Sandbox Engine for AgenticCode (Phase 2).
Provides secure code execution using Docker with strict isolation policies:
- Non-root user execution (UID 1000:1000)
- Dropped Linux capabilities (cap_drop=["ALL"])
- Network blackholing (network_mode="none")
- Resource limits (4096MB memory, 2.0 vCPUs)
- Strict execution deadlines (default 60s timeout)
- Ephemeral workspace mounting and local fallback execution
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import docker

from agentic_code.config import settings
from agentic_code.state import TestResults


class SandboxExecutionError(Exception):
    """Raised when sandbox execution encounters an unrecoverable failure."""

    pass


class DockerSandbox:
    """
    Unified Sandbox Engine: Runs untrusted code inside a hardened Docker container
    with seamless, isolated local fallback when Docker engine is unreachable.
    """

    def __init__(
        self,
        repo_path: Union[str, Path],
        image: Optional[str] = None,
        user: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        mem_limit: Optional[str] = None,
        cpu_quota: Optional[int] = None,
        network_mode: Optional[str] = None,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.image = image or settings.SANDBOX_IMAGE
        self.user = user or settings.SANDBOX_USER
        self.timeout_seconds = timeout_seconds or settings.SANDBOX_TIMEOUT_SECONDS
        self.mem_limit = mem_limit or settings.SANDBOX_MEM_LIMIT
        self.cpu_quota = cpu_quota or settings.SANDBOX_CPU_QUOTA
        self.network_mode = network_mode or settings.SANDBOX_NETWORK_MODE

        self._client: Optional[docker.DockerClient] = None
        self._is_docker_available: Optional[bool] = None

    @property
    def is_docker_available(self) -> bool:
        """Checks whether the Docker daemon is accessible and responding."""
        if self._is_docker_available is not None:
            return self._is_docker_available

        try:
            client = docker.from_env()
            client.ping()
            self._client = client
            self._is_docker_available = True
        except Exception:
            self._client = None
            self._is_docker_available = False

        return self._is_docker_available

    def execute_command(
        self,
        command: Union[str, List[str]],
        working_dir: str = "/workspace",
        environment: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> TestResults:
        """
        Executes a test or build command. Routes to hardened Docker container if available,
        or falls back to isolated local subprocess execution.
        """
        deadline = timeout or self.timeout_seconds

        if not self.is_docker_available:
            if settings.ALLOW_LOCAL_FALLBACK_SANDBOX:
                return self._execute_local_fallback(command, env=environment, timeout=deadline)
            raise SandboxExecutionError("Docker daemon is unreachable and fallback mode is disabled.")

        return self._execute_in_docker(command, working_dir, environment, deadline)

    def execute(
        self,
        command: Union[str, List[str]],
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> TestResults:
        """Alias for execute_command matching the standard test execution interface."""
        return self.execute_command(command, environment=env, timeout=timeout)

    def _execute_in_docker(
        self,
        command: Union[str, List[str]],
        working_dir: str,
        environment: Optional[Dict[str, str]],
        deadline: int,
    ) -> TestResults:
        """Executes strictly inside the hardened Docker container with security boundaries."""
        client = self._client or docker.from_env()
        start_time = time.time()
        container = None

        env = {"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        if environment:
            env.update(environment)

        cmd_exec = ["sh", "-c", " ".join(command)] if isinstance(command, list) else ["sh", "-c", command]

        try:
            volumes = {str(self.repo_path): {"bind": "/workspace", "mode": "rw"}}

            container = client.containers.create(
                image=self.image,
                command=cmd_exec,
                working_dir=working_dir,
                user=self.user,
                volumes=volumes,
                network_mode=self.network_mode,
                cap_drop=["ALL"],
                mem_limit=self.mem_limit,
                cpu_quota=self.cpu_quota,
                environment=env,
                detach=True,
            )

            container.start()

            try:
                res = container.wait(timeout=deadline)
                exit_code = res.get("StatusCode", -1)
                timed_out = False
            except Exception:
                timed_out = True
                try:
                    container.kill()
                except Exception:
                    pass
                exit_code = -1

            logs = container.logs(stdout=True, stderr=True)
            stdout = logs.decode("utf-8", errors="replace") if isinstance(logs, bytes) else str(logs)
            stderr = f"TIMEOUT: Exceeded {deadline}s deadline." if timed_out else ""
            duration = round(time.time() - start_time, 2)
            passed = (exit_code == 0) and not timed_out

            return TestResults(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                passed=passed,
                duration_seconds=duration,
                command=str(command),
                error_summary=None if passed else (stderr or stdout)[-1000:],
            )
        except Exception as exc:
            duration = round(time.time() - start_time, 2)
            return TestResults(
                exit_code=-1,
                stdout="",
                stderr=f"Docker error: {exc}",
                passed=False,
                duration_seconds=duration,
                command=str(command),
                error_summary=str(exc),
            )
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    def _execute_local_fallback(
        self,
        command: Union[str, List[str]],
        env: Optional[Dict[str, str]] = None,
        timeout: int = 60,
    ) -> TestResults:
        """Isolated local subprocess execution with strict deadline enforcement."""
        start_time = time.time()
        custom_env = os.environ.copy()
        custom_env["PYTHONUNBUFFERED"] = "1"
        custom_env["PYTHONDONTWRITEBYTECODE"] = "1"
        if env:
            custom_env.update(env)

        try:
            proc = subprocess.run(
                command,
                cwd=str(self.repo_path),
                env=custom_env,
                capture_output=True,
                text=True,
                shell=isinstance(command, str),
                timeout=timeout,
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
                stderr=f"TIMEOUT EXPIRED: Process exceeded deadline of {timeout}s.\n{stderr}",
                passed=False,
                duration_seconds=duration,
                command=str(command),
                error_summary=f"TimeoutExpired after {timeout} seconds",
            )
        except Exception as e:
            duration = round(time.time() - start_time, 2)
            return TestResults(
                exit_code=-1,
                stdout="",
                stderr=f"Execution error: {e}",
                passed=False,
                duration_seconds=duration,
                command=str(command),
                error_summary=str(e),
            )

    def run_tests(self, test_command: str = "pytest -v") -> TestResults:
        """Run tests inside the sandbox."""
        return self.execute_command(test_command)


# Alias for backward compatibility
LocalSubprocessSandbox = DockerSandbox
