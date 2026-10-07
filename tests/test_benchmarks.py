"""
Tests for Phase 8: SWE-bench Benchmark Integration & Evaluation Harness.
"""

import json
import tempfile
from pathlib import Path

from agentic_code.benchmarks import (
    BenchmarkMetrics,
    InstanceEvaluationResult,
    SWEBenchEvaluator,
    SWEBenchInstance,
    calculate_metrics,
    get_standard_reference_instances,
    load_swebench_dataset,
)
from agentic_code.state import AgentState


def test_swebench_instance_schema():
    instance = SWEBenchInstance(
        instance_id="psf__requests-2674",
        repo="psf/requests",
        base_commit="e064974c2644265cc27dc820da408bf778bfa7be",
        problem_statement="Case-insensitive redirect headers bug",
        fail_to_pass=["tests/test_requests.py::test_redirect_casing"],
        pass_to_pass=["tests/test_requests.py::test_get"],
    )
    assert instance.instance_id == "psf__requests-2674"
    assert len(instance.fail_to_pass) == 1
    assert len(instance.pass_to_pass) == 1

    state = instance.to_agent_state(repo_path="/tmp/requests", max_iterations=3)
    assert state["repo_path"] == "/tmp/requests"
    assert state["issue"]["title"] == "SWE-bench: psf__requests-2674"
    assert state["max_iterations"] == 3


def test_standard_reference_instances():
    instances = get_standard_reference_instances()
    assert len(instances) == 4
    repos = [inst.repo for inst in instances]
    assert "psf/requests" in repos
    assert "pallets/flask" in repos
    assert "scikit-learn/scikit-learn" in repos
    assert "sympy/sympy" in repos

    for inst in instances:
        assert len(inst.fail_to_pass) > 0
        assert len(inst.pass_to_pass) > 0
        assert inst.base_commit


def test_load_swebench_dataset_json_and_jsonl():
    raw_data = [
        {
            "instance_id": "test__repo-1",
            "repo": "test/repo",
            "base_commit": "abc123",
            "problem_statement": "Issue 1",
            "fail_to_pass": ["test_fail"],
            "pass_to_pass": ["test_pass"],
        },
        {
            "instance_id": "test__repo-2",
            "repo": "test/repo",
            "base_commit": "def456",
            "problem_statement": "Issue 2",
            "fail_to_pass": ["test_fail_2"],
            "pass_to_pass": ["test_pass_2"],
        },
    ]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        # 1. Test JSON loading
        json_file = Path(tmpdir) / "swebench.json"
        json_file.write_text(json.dumps(raw_data), encoding="utf-8")
        loaded_json = load_swebench_dataset(json_file)
        assert len(loaded_json) == 2
        assert loaded_json[0].instance_id == "test__repo-1"

        # 2. Test JSONL loading (official SWE-bench format)
        jsonl_file = Path(tmpdir) / "swebench.jsonl"
        jsonl_lines = "\n".join(json.dumps(item) for item in raw_data)
        jsonl_file.write_text(jsonl_lines, encoding="utf-8")
        loaded_jsonl = load_swebench_dataset(jsonl_file)
        assert len(loaded_jsonl) == 2
        assert loaded_jsonl[1].instance_id == "test__repo-2"


def test_metric_calculations():
    results = [
        # Instance 1: Resolved at k=1
        InstanceEvaluationResult(
            instance_id="inst-1",
            repo="psf/requests",
            reproduction_success=True,
            resolved=True,
            iterations_taken=1,
            regression_passed=True,
            duration_seconds=1.5,
        ),
        # Instance 2: Resolved at k=2
        InstanceEvaluationResult(
            instance_id="inst-2",
            repo="pallets/flask",
            reproduction_success=True,
            resolved=True,
            iterations_taken=2,
            regression_passed=True,
            duration_seconds=2.5,
        ),
        # Instance 3: Unresolved after k=3
        InstanceEvaluationResult(
            instance_id="inst-3",
            repo="scikit-learn/scikit-learn",
            reproduction_success=True,
            resolved=False,
            iterations_taken=3,
            regression_passed=True,
            duration_seconds=3.0,
        ),
        # Instance 4: Failed reproduction & regression break
        InstanceEvaluationResult(
            instance_id="inst-4",
            repo="sympy/sympy",
            reproduction_success=False,
            resolved=False,
            iterations_taken=3,
            regression_passed=False,
            duration_seconds=1.0,
        ),
    ]

    metrics = calculate_metrics(results)
    assert metrics.total_instances == 4
    assert metrics.reproduced_count == 3
    assert metrics.reproduction_rate == 75.0  # 3/4
    assert metrics.resolved_count == 2
    assert metrics.pass_at_1 == 25.0          # 1/4
    assert metrics.pass_at_2 == 50.0          # 2/4
    assert metrics.pass_at_3 == 50.0          # 2/4
    assert metrics.regression_rate == 75.0    # 3/4
    assert metrics.mean_duration_seconds == 2.0

    summary_text = metrics.summary()
    assert "Pass@1" in summary_text
    assert "Pass@2" in summary_text
    assert "Pass@3" in summary_text
    assert "Regression Retention Rate" in summary_text


def test_evaluator_batch_with_mock_runner():
    instances = [
        SWEBenchInstance(
            instance_id="mock-1",
            repo="mock/repo",
            base_commit="hash1",
            problem_statement="Mock problem 1",
            fail_to_pass=["test_mock"],
            pass_to_pass=["test_pass"],
        )
    ]

    def mock_workflow_runner(state: AgentState):
        state["test_results"] = {"passed": True, "exit_code": 0, "stdout": "All tests passed", "stderr": ""}
        state["patch"] = "--- a/mock.py\n+++ b/mock.py\n+mock_fixed = True\n"
        state["iteration"] = 1
        return state

    evaluator = SWEBenchEvaluator(max_iterations=3)
    metrics: BenchmarkMetrics = evaluator.evaluate_batch(
        instances,
        run_fn=mock_workflow_runner,
    )

    assert metrics.total_instances == 1
    assert len(metrics.results) == 1
    res = metrics.results[0]
    assert res.instance_id == "mock-1"
    assert res.resolved is True
    assert res.diff_stats is not None
    assert res.diff_stats.lines_added == 1
