"""
Hardened Docker Sandbox Engine for AgenticCode.
Enforces non-root execution, dropped capabilities, network isolation, strict resource quotas,
ephemeral volume mounts, and 60-second timeouts.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import docker
from docker.errors import DockerException

from agentic_code.config import settings
from agentic_code.sandbox.fallback import LocalSubprocessSandbox
from agentic_code.state import TestResults


class SandboxExecutionError(Exception):
    """Raised when sandbox execution fails critically."""
    pass


class DockerSandbox:
    """
    Hardened container harness responsible for securely running untrusted, AI-generated code.
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
        """Check if Docker daemon is accessible."""
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

    def get_client(self) -> docker.DockerClient:
        """Get or initialize Docker client."""
        if not self.is_docker_available or not self._client:
            raise DockerException("Docker daemon is not available or ping failed.")
        return self._client

    def ensure_image_exists(self) -> None:
        """Ensure sandbox image exists, or build from local Dockerfile if needed."""
        if not self.is_docker_available:
            return

        client = self.get_client()
        try:
            client.images.get(self.image)
        except docker.errors.ImageNotFound:
            dockerfile_dir = Path(__file__).parent
            client.images.build(
                path=str(dockerfile_dir),
                tag=self.image,
                rm=True,
            )

    def execute_command(
        self,
        command: Union[str, List[str]],
        working_dir: str = "/workspace",
        environment: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> TestResults:
        """
        Run command inside the hardened Docker sandbox with capability dropping,
        network blackholing, resource constraints, and ephemeral volume mounting.
        """
        deadline = timeout or self.timeout_seconds

        # Fallback to local subprocess sandbox if Docker is unavailable
        if not self.is_docker_available:
            if settings.ALLOW_LOCAL_FALLBACK_SANDBOX:
                fallback = LocalSubprocessSandbox(self.repo_path, timeout_seconds=deadline)
                return fallback.execute(command, env=environment, timeout=deadline)
            raise SandboxExecutionError("Docker daemon is unreachable and fallback mode is disabled.")

        client = self.get_client()
        start_time = time.time()
        container = None

        env = {
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if environment:
            env.update(environment)

        # Format command string / list for sh -c
        if isinstance(command, list):
            cmd_exec = ["sh", "-c", " ".join(command)]
        else:
            cmd_exec = ["sh", "-c", command]

        try:
            # Mount host repo_path into container /workspace as rw
            volumes = {
                str(self.repo_path): {
                    "bind": "/workspace",
                    "mode": "rw",
                }
            }

            # Security configurations as per specification:
            # 1. Capability Stripping: cap_drop=["ALL"]
            # 2. Network Blackholing: network_mode="none"
            # 3. Non-Root Execution: user="1000:1000"
            # 4. Resource limits: mem_limit="4096m", cpu_quota=200000
            # 5. Ephemeral container: auto_remove=False (to inspect logs), then removed in finally block
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

            # Wait with strict timeout deadline
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
            stderr = ""
            if timed_out:
                stderr = f"TIMEOUT EXPIRED: Execution exceeded deadline of {deadline} seconds."

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
                stderr=f"Docker sandbox error: {str(exc)}",
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

    def run_tests(self, test_command: str = "pytest -v") -> TestResults:
        """Run tests inside the hardened sandbox."""
        return self.execute_command(test_command)
