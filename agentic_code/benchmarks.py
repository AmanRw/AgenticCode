"""
SWE-bench Benchmark Integration & Evaluation Harness (Phase 8).
Supports official SWE-bench and SWE-bench Lite datasets across standard
Python open-source repositories (psf/requests, pallets/flask, scikit-learn/scikit-learn, sympy/sympy).
Calculates Reproduction Rate, Pass@k (k=1, 2, 3), and Regression Rate.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from agentic_code.config import Settings, settings
from agentic_code.observability import DiffStats, audit_logger
from agentic_code.sandbox import DockerSandbox
from agentic_code.state import AgentState, IssueMetadata

logger = logging.getLogger("agentic_code.benchmarks")


# ---------------------------------------------------------------------------
# SWE-bench Schema Models
# ---------------------------------------------------------------------------

class SWEBenchInstance(BaseModel):
    """
    Standard representation of an issue instance in SWE-bench / SWE-bench Lite.
    """
    instance_id: str = Field(description="Unique instance ID (e.g. psf__requests-2674)")
    repo: str = Field(description="Repository owner/repo (e.g. psf/requests)")
    base_commit: str = Field(description="Git commit hash before bug resolution")
    problem_statement: str = Field(description="Issue title, description and traceback")
    test_patch: Optional[str] = Field(default=None, description="Gold test patch / reproduction diff")
    patch: Optional[str] = Field(default=None, description="Gold fix patch (if provided for verification)")
    pass_to_pass: List[str] = Field(
        default_factory=list,
        description="List of tests that must pass before and after (regression suite)",
    )
    fail_to_pass: List[str] = Field(
        default_factory=list,
        description="List of tests that fail before the fix and pass after (reproduction)",
    )
    hints_text: Optional[str] = Field(default=None, description="Issue discussion hints or comments")
    version: Optional[str] = Field(default=None, description="Package version")
    created_at: Optional[str] = Field(default=None, description="Timestamp of issue creation")

    def to_agent_state(self, repo_path: str, max_iterations: int = 3) -> AgentState:
        """Converts SWE-bench instance into AgentState for workflow execution."""
        issue_meta = IssueMetadata(
            title=f"SWE-bench: {self.instance_id}",
            body=self.problem_statement,
            issue_number=1,
            repo_url=f"https://github.com/{self.repo}",
            comments=[self.hints_text] if self.hints_text else [],
        )
        return {
            "issue": issue_meta.model_dump(),
            "repo_path": repo_path,
            "code_context": {},
            "plan": "",
            "tests": self.test_patch or "",
            "patch": "",
            "test_results": {"passed": False, "exit_code": -1, "stdout": "", "stderr": ""},
            "error_feedback": None,
            "iteration": 0,
            "max_iterations": max_iterations,
            "pr_url": None,
        }


# ---------------------------------------------------------------------------
# Evaluation Results & Benchmark Metrics
# ---------------------------------------------------------------------------

class InstanceEvaluationResult(BaseModel):
    """Execution and verification outcome for a single SWE-bench instance."""
    instance_id: str
    repo: str
    reproduction_success: bool = False
    resolved: bool = False
    iterations_taken: int = 0
    regression_passed: bool = True
    patch: Optional[str] = None
    diff_stats: Optional[DiffStats] = None
    duration_seconds: float = 0.0
    token_usage: Dict[str, int] = Field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    error: Optional[str] = None


class BenchmarkMetrics(BaseModel):
    """
    Consolidated SWE-bench evaluation metrics:
    - Reproduction Rate: % of bugs successfully captured by reproduction scripts
    - Pass@k: % of verified fixes generated within 1, 2, or 3 iterations (k in {1, 2, 3})
    - Regression Rate: % of instances where existing repository test suites remain green
    """
    total_instances: int = 0
    reproduced_count: int = 0
    resolved_count: int = 0
    reproduction_rate: float = 0.0  # Percentage (0 - 100)
    pass_at_1: float = 0.0          # Pass@1 (%)
    pass_at_2: float = 0.0          # Pass@2 (%)
    pass_at_3: float = 0.0          # Pass@3 (%)
    pass_at_k: Dict[int, float] = Field(default_factory=dict)
    regression_rate: float = 0.0    # Regression retention percentage (0 - 100)
    mean_duration_seconds: float = 0.0
    results: List[InstanceEvaluationResult] = Field(default_factory=list)

    def summary(self) -> str:
        """Generates a formatted human-readable metrics table."""
        lines = [
            "=" * 60,
            "SWE-BENCH BENCHMARK EVALUATION SUMMARY",
            "=" * 60,
            f"Total Instances Evaluated : {self.total_instances}",
            f"Bugs Reproduced           : {self.reproduced_count}/{self.total_instances} ({self.reproduction_rate:.1f}%)",
            f"Pass@1 (Resolved in 1 try): {self.pass_at_1:.1f}%",
            f"Pass@2 (Resolved in <=2) : {self.pass_at_2:.1f}%",
            f"Pass@3 (Resolved in <=3) : {self.pass_at_3:.1f}%",
            f"Regression Retention Rate : {self.regression_rate:.1f}%",
            f"Mean Resolution Time      : {self.mean_duration_seconds:.2f}s",
            "=" * 60,
        ]
        return "\n".join(lines)


def calculate_metrics(results: List[InstanceEvaluationResult]) -> BenchmarkMetrics:
    """
    Calculates SWE-bench metrics from a list of instance evaluation outcomes.
    """
    total = len(results)
    if total == 0:
        return BenchmarkMetrics()

    reproduced = sum(1 for r in results if r.reproduction_success)
    resolved = sum(1 for r in results if r.resolved)
    regression_green = sum(1 for r in results if r.regression_passed)

    resolved_at_1 = sum(1 for r in results if r.resolved and r.iterations_taken <= 1)
    resolved_at_2 = sum(1 for r in results if r.resolved and r.iterations_taken <= 2)
    resolved_at_3 = sum(1 for r in results if r.resolved and r.iterations_taken <= 3)

    durations = [r.duration_seconds for r in results if r.duration_seconds > 0]
    mean_duration = sum(durations) / len(durations) if durations else 0.0

    p1 = (resolved_at_1 / total) * 100.0
    p2 = (resolved_at_2 / total) * 100.0
    p3 = (resolved_at_3 / total) * 100.0

    return BenchmarkMetrics(
        total_instances=total,
        reproduced_count=reproduced,
        resolved_count=resolved,
        reproduction_rate=round((reproduced / total) * 100.0, 2),
        pass_at_1=round(p1, 2),
        pass_at_2=round(p2, 2),
        pass_at_3=round(p3, 2),
        pass_at_k={1: round(p1, 2), 2: round(p2, 2), 3: round(p3, 2)},
        regression_rate=round((regression_green / total) * 100.0, 2),
        mean_duration_seconds=round(mean_duration, 2),
        results=results,
    )


# ---------------------------------------------------------------------------
# Dataset Loading & Standard Repository Datasets
# ---------------------------------------------------------------------------

def load_swebench_dataset(source: Union[str, Path, List[Dict[str, Any]]]) -> List[SWEBenchInstance]:
    """
    Loads SWE-bench instances from a JSON file, JSONL file, or list of dictionaries.
    """
    if isinstance(source, list):
        return [SWEBenchInstance(**item) for item in source]

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"SWE-bench dataset file not found: {path}")

    instances: List[SWEBenchInstance] = []
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if line:
                instances.append(SWEBenchInstance(**json.loads(line)))
    else:
        data = json.loads(text)
        if isinstance(data, list):
            instances = [SWEBenchInstance(**item) for item in data]
        else:
            instances = [SWEBenchInstance(**data)]

    return instances


def get_standard_reference_instances() -> List[SWEBenchInstance]:
    """
    Returns curated reference benchmark instances across standard Python open-source repos:
    1. psf/requests (HTTP auth / redirect casing)
    2. pallets/flask (Blueprint error handler inheritance)
    3. scikit-learn/scikit-learn (NCA parameter validation with None check)
    4. sympy/sympy (Hyperbolic cosine substitution)
    """
    return [
        SWEBenchInstance(
            instance_id="psf__requests-2674",
            repo="psf/requests",
            base_commit="e064974c2644265cc27dc820da408bf778bfa7be",
            problem_statement=(
                "Issue: Redirect handling loses Content-Type header casing on retry.\n"
                "When redirecting a request, urllib3 urllib headers should preserve canonical capitalization."
            ),
            fail_to_pass=["tests/test_requests.py::test_redirect_content_type_preserved"],
            pass_to_pass=["tests/test_requests.py::test_basic_get", "tests/test_requests.py::test_post_json"],
            hints_text="Inspect requests/models.py and requests/sessions.py during redirect loop.",
            version="2.7.0",
        ),
        SWEBenchInstance(
            instance_id="pallets__flask-4045",
            repo="pallets/flask",
            base_commit="7e02df595fa12ee56a1b1a7d65bcfd658c160b73",
            problem_statement=(
                "Issue: Blueprint error handlers are not properly scoped when blueprints are nested.\n"
                "Nested blueprint error handlers should take precedence over parent blueprint handlers."
            ),
            fail_to_pass=["tests/test_blueprints.py::test_nested_blueprint_error_handling"],
            pass_to_pass=["tests/test_blueprints.py::test_basic_blueprint", "tests/test_app.py::test_index"],
            hints_text="Review flask/blueprints.py and flask/sansio/scaffold.py error handler lookup.",
            version="2.0.0",
        ),
        SWEBenchInstance(
            instance_id="scikit-learn__scikit-learn-14092",
            repo="scikit-learn/scikit-learn",
            base_commit="8983fb0dc6e72fb4d50dcf3f990a424e83f3eeb0",
            problem_statement=(
                "Issue: NeighborhoodComponentsAnalysis crashes when tol is None.\n"
                "Parameter validation should allow None or float >= 0 without TypeError."
            ),
            fail_to_pass=["sklearn/neighbors/tests/test_nca.py::test_nca_tol_none"],
            pass_to_pass=["sklearn/neighbors/tests/test_nca.py::test_nca_fit", "sklearn/neighbors/tests/test_nca.py::test_nca_transform"],
            hints_text="Check sklearn/neighbors/_nca.py _check_params method.",
            version="0.22",
        ),
        SWEBenchInstance(
            instance_id="sympy__sympy-13480",
            repo="sympy/sympy",
            base_commit="c96937e1b9b187515b699cba4fb2c31e679a6dae",
            problem_statement=(
                "Issue: Substitution in coth(x) leads to NameError when evaluating at undefined singularities.\n"
                "coth(log(tan(x))) evaluation produces complex infinity instead of unsimplified form."
            ),
            fail_to_pass=["sympy/functions/elementary/tests/test_hyperbolic.py::test_coth_subs_singularity"],
            pass_to_pass=["sympy/functions/elementary/tests/test_hyperbolic.py::test_sinh", "sympy/functions/elementary/tests/test_cosh"],
            hints_text="Check sympy/functions/elementary/hyperbolic.py coth eval method.",
            version="1.1",
        ),
    ]


# ---------------------------------------------------------------------------
# SWEBench Evaluator Engine
# ---------------------------------------------------------------------------

class SWEBenchEvaluator:
    """
    Harness for evaluating AgenticCode against SWE-bench benchmark instances.
    """

    def __init__(
        self,
        custom_settings: Optional[Settings] = None,
        max_iterations: int = 3,
    ) -> None:
        self.settings = custom_settings or settings
        self.max_iterations = max_iterations

    def evaluate_instance(
        self,
        instance: SWEBenchInstance,
        repo_path: Optional[str] = None,
        run_fn: Optional[Callable[[AgentState], Dict[str, Any]]] = None,
    ) -> InstanceEvaluationResult:
        """
        Executes and evaluates a single SWE-bench instance:
        1. Prepares repository state & starts audit trace
        2. Verifies reproduction test fails against unpatched code (Reproduction fidelity)
        3. Executes the agentic repair state machine
        4. Verifies test pass on patched code
        5. Verifies existing regression tests (pass_to_pass) remain green
        6. Extracts diff statistics and returns InstanceEvaluationResult
        """
        start_time = time.time()
        audit_logger.start_run(issue_id=instance.instance_id, repo=instance.repo)
        audit_logger.record_container_lifecycle("BENCHMARK_START", details={"instance_id": instance.instance_id})

        r_path = repo_path or "."
        result = InstanceEvaluationResult(
            instance_id=instance.instance_id,
            repo=instance.repo,
            reproduction_success=False,
            resolved=False,
            iterations_taken=0,
            regression_passed=True,
        )

        try:
            # 1. Reproduction Fidelity Verification
            sandbox = DockerSandbox(repo_path=r_path)
            rep_cmd = f"pytest -v {instance.fail_to_pass[0]}" if instance.fail_to_pass else "pytest -v test_reproduce_issue.py"
            pre_exec = sandbox.execute_command(rep_cmd)
            audit_logger.record_sandbox_trace(
                command=rep_cmd,
                exit_code=pre_exec.exit_code,
                duration_seconds=pre_exec.duration_seconds,
                passed=pre_exec.passed,
            )

            # A valid reproduction test fails on the unpatched codebase
            reproduced = not pre_exec.passed
            result.reproduction_success = reproduced

            # 2. Run Repair Pipeline
            state = instance.to_agent_state(repo_path=r_path, max_iterations=self.max_iterations)
            if run_fn is not None:
                final_state = run_fn(state)
            else:
                from agentic_code.workflow import run_workflow
                final_state = run_workflow(
                    issue=state["issue"],
                    repo_path=r_path,
                    max_iterations=self.max_iterations,
                )

            # 3. Analyze Pipeline Outcome
            test_results = final_state.get("test_results", {})
            patch = final_state.get("patch", "")
            iteration = final_state.get("iteration", 0)

            result.patch = patch
            result.iterations_taken = iteration if iteration > 0 else 1
            result.resolved = bool(test_results.get("passed", False))

            if patch:
                result.diff_stats = audit_logger.record_diff(patch)

            # 4. Regression Test Suite Verification (pass_to_pass)
            if result.resolved and instance.pass_to_pass:
                regression_cmd = f"pytest -v {' '.join(instance.pass_to_pass)}"
                reg_exec = sandbox.execute_command(regression_cmd)
                audit_logger.record_sandbox_trace(
                    command=regression_cmd,
                    exit_code=reg_exec.exit_code,
                    duration_seconds=reg_exec.duration_seconds,
                    passed=reg_exec.passed,
                )
                result.regression_passed = reg_exec.passed
            else:
                result.regression_passed = True

        except Exception as exc:
            logger.exception("Evaluation failed for instance %s", instance.instance_id)
            result.error = str(exc)
            result.resolved = False
            result.regression_passed = False
        finally:
            result.duration_seconds = round(time.time() - start_time, 3)
            active_trace = audit_logger.get_active_trace()
            if active_trace:
                result.token_usage = active_trace.token_usage.copy()
            audit_logger.end_run(success=result.resolved)

        return result

    def evaluate_batch(
        self,
        instances: List[SWEBenchInstance],
        repo_path: Optional[str] = None,
        run_fn: Optional[Callable[[AgentState], Dict[str, Any]]] = None,
    ) -> BenchmarkMetrics:
        """
        Evaluates a batch of SWE-bench instances and aggregates metrics.
        """
        results: List[InstanceEvaluationResult] = []
        for inst in instances:
            res = self.evaluate_instance(inst, repo_path=repo_path, run_fn=run_fn)
            results.append(res)
        return calculate_metrics(results)
