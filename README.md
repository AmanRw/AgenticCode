# AgenticCode: End-to-End Autonomous GitHub Issue Resolution

An industrial-grade, multi-agent AI framework engineered to autonomously resolve GitHub issues from bug ingestion to verified Pull Request synthesis.

---

## Architecture Overview (Phases 1 – 8)

```
AgenticCode/
├── .env.example                     # Environment secret templates
├── pyproject.toml                   # uv/pip dependencies & tooling config
├── Dockerfile                       # Phase 2: Unprivileged UID 1000:1000, python:3.11-slim
├── action.yml                       # Phase 7: GitHub Action workflow wrapper
├── main.py                          # Full system verification script (Phases 1-8)
├── agentic_code/
│   ├── __init__.py                  # Package exports across all 8 phases
│   ├── config.py                    # Phase 1: Pydantic Settings & Gemini LLM Factory
│   ├── state.py                     # Phase 1: Shared Orchestrator AgentState (TypedDict)
│   ├── sandbox.py                   # Phase 2: Hardened Docker Sandbox & Subprocess Fallback
│   ├── tools.py                     # Phase 3: Code Retrieval & AST Analysis Tools
│   ├── agents.py                    # Phase 4: 5 Specialized Gemini Personas
│   ├── vcs.py                       # Phase 5: GitManager & PyGithub Client
│   ├── workflow.py                  # Phase 6: LangGraph State Machine & Cyclic Retry Loop
│   ├── webhook.py                   # Phase 7: FastAPI GitHub Webhook & Queue Engine
│   ├── observability.py             # Phase 8: Tracing, Diff Stats & Audit Logger
│   ├── benchmarks.py                # Phase 8: SWE-bench Benchmark Integration & Evaluator
│   └── cli.py                       # CLI application (resolve, evaluate, audit, server)
└── tests/                           # Complete test suite (46/46 passing)
    ├── test_state.py                # State & settings tests
    ├── test_sandbox.py              # Docker & fallback sandbox tests
    ├── test_ast_tools.py            # AST & search tools tests
    ├── test_agents.py               # 5 specialized agent personas tests
    ├── test_vcs.py                  # GitManager & GitHubClient tests
    ├── test_workflow.py             # LangGraph state machine & routing tests
    ├── test_webhook.py              # FastAPI webhook & HMAC security tests
    ├── test_observability.py        # Tracing, diff stats & audit logger tests
    └── test_benchmarks.py           # SWE-bench metrics & evaluation tests
```

---

## Phase Details

### Phase 1: Environment Setup & Project Foundation
- **Modern Packaging**: Configured using `uv` with Python 3.11+ support.
- **Core Dependencies**: Installed `langchain-google-genai`, `langgraph`, `langchain-core`, `docker`, `GitPython`, `PyGithub`, `tree-sitter`, `pytest`, `ruff`, `mypy`. Default LLM provider configured to **Google Gemini** (`gemini-2.5-pro`, `gemini-2.5-flash`).
- **Typed Orchestrator State (`AgentState`)**:
  ```python
  class AgentState(TypedDict):
      issue: dict                    # title, body, issue_number, repo_url
      repo_path: str                 # local clone path
      code_context: dict             # relevant files, ast symbols, call graphs
      plan: str                      # root-cause analysis and repair steps
      tests: str                     # reproduction & regression test code
      patch: str                     # unified diff or file updates
      test_results: dict             # exit_code, stdout, stderr, passed
      error_feedback: Optional[str]  # traceback/failure logs for iterative repair
      iteration: int                 # current retry count k
      max_iterations: int            # k_max threshold (e.g., 3)
      pr_url: Optional[str]          # created PR link
  ```
- **Configuration & Secrets**: Centralized in [agentic_code/config.py](agentic_code/config.py) using `pydantic-settings` with `.env` loading and multi-provider LLM factory.

---

### Phase 2: Hardened Docker Sandbox Engine
Built to securely run untrusted, AI-generated code without privilege escalation or exfiltration risks:
- **Base Dockerfile**: Based on `python:3.11-slim`, creates unprivileged user `sandboxuser` (`UID 1000:GID 1000`), pre-installs `pytest`, `ruff`, and `mypy`.
- **Capability Stripping**: `cap_drop=["ALL"]` blocks host privilege escalation.
- **Network Blackholing**: `network_mode="none"` blocks outbound network requests and exfiltration.
- **Non-Root Execution**: Strict `user="1000:1000"`.
- **Resource Quotas**: Enforces `mem_limit="4096m"` and `cpu_quota=200000` (2.0 vCPUs).
- **Execution Deadlines**: Strict 60-second timeouts with process tree termination.
- **Ephemeral Mounts & Fallback**: Mounts repository workspaces temporarily and includes `LocalSubprocessSandbox` for environments without active Docker engines.

---

### Phase 3: Code Retrieval & AST Analysis Tools
- **Symbol & File Outlines** ([agentic_code/tools.py](agentic_code/tools.py)): Parses Python Abstract Syntax Trees to extract top-level classes, method signatures, docstrings, and imports across the entire repository without exhausting LLM context windows.
- **Precise Snippet Extraction**: Slices target lines with surrounding context for focused LLM evaluation.
- **Lexical & Structural Search**: Fast keyword, regex, and error traceback parser mapping crash locations directly to repository files.
- **Call Graph Generation**: Analyzes function calls to construct caller/callee dependency graphs.

---

### Phase 4: Specialized Agent Personas
Five modular agent personas built with structured inputs, outputs, and system prompts:
1. **Agent 01: Code Reader** ([agentic_code/agents.py](agentic_code/agents.py)): Pinpoints target files and extracts relevant code snippets without overflowing context.
2. **Agent 02: Planner** ([agentic_code/agents.py](agentic_code/agents.py)): Diagnoses root cause and produces a 4-part Markdown repair plan (Identified Bug, Invariants, Logic Modifications, Test Requirements).
3. **Agent 04: Test Writer** ([agentic_code/agents.py](agentic_code/agents.py)): Generates standalone reproduction and regression tests that fail on unpatched code and pass once resolved.
4. **Agent 03: Code Writer** ([agentic_code/agents.py](agentic_code/agents.py)): Synthesizes minimal, unified diffs or file updates incorporating iterative failure feedback.
5. **Agent 05: PR Opener** ([agentic_code/agents.py](agentic_code/agents.py)): Synthesizes Git branch names, conventional commit messages, and Markdown PR bodies linking issues (`Fixes #<id>`).

---

### Phase 5: Git & GitHub VCS Integration Layer
- **Local Git Operations** ([agentic_code/vcs.py](agentic_code/vcs.py)):
  - Clones target repositories into ephemeral workspaces.
  - Creates and manages isolated fix branches (`agentic-fix/issue-<id>`).
  - Writes synthesized test files and applies unified diff patches.
  - Implements clean working tree rollbacks (`git checkout -f HEAD` and `git clean -fd`) ensuring failed repair iterations leave no dirty state.
  - Commits verified patches with automated bot authorship.
- **GitHub REST/GraphQL Client** ([agentic_code/vcs.py](agentic_code/vcs.py)):
  - Authenticates via GitHub App token or Personal Access Token (PAT).
  - Fetches complete issue metadata (title, body, author, labels, comments).
  - Opens linked Pull Requests with Markdown bodies (`Fixes #<id>`) detailing root causes, changes, and test proofs.
  - Posts diagnostic explanation comments directly to issue threads if repair loops exceed $k_{max}$ iterations.

---

### Phase 6: LangGraph State Machine & Cyclic Orchestration
- **Compiled StateGraph Architecture** ([agentic_code/workflow.py](agentic_code/workflow.py)):
  - Eight specialized nodes: `node_code_reader`, `node_planner`, `node_test_writer`, `node_reproduction_verifier`, `node_code_writer`, `node_sandbox_verifier`, `node_pr_opener`, and `node_escalation`.
  - **Reproduction Fidelity Verification**: Validates that newly synthesized test suites fail against the unpatched codebase prior to patch generation.
  - **Cyclic Repair Routing**: Conditional edge evaluates `state["test_results"]["passed"]`.
    - If `passed == True` $\rightarrow$ transitions to `node_pr_opener` $\rightarrow$ `END`.
    - If `passed == False` and `iteration < max_iterations`:
      - Appends failure tracebacks to `state["error_feedback"]`.
      - Increments `state["iteration"]`.
      - Rolls back the working tree cleanly using `GitManager.rollback_changes()`.
      - Loops back to `node_code_writer`.
    - If `passed == False` and `iteration >= max_iterations` $\rightarrow$ transitions to `node_escalation` $\rightarrow$ `END`.

---

### Phase 7: Webhook Ingestion & CI/CD Triggering
- **FastAPI Webhook Service** ([agentic_code/webhook.py](agentic_code/webhook.py)):
  - Endpoints: `GET /health`, `POST /webhook`, and `GET /queue`.
  - Validates GitHub `X-Hub-Signature-256` HMAC-SHA256 signatures with constant-time equality checks.
  - Automatically ingests `issues` (`opened`, `labeled: agentic-fix`) and `issue_comment` (`/fix` command trigger) events.
  - Dispatches automated background pipeline execution jobs to clone, branch, and run the state machine.
- **GitHub Action Packaging**:
  - [action.yml](action.yml) composite action enabling single-line inclusion in any repository.
  - [.github/workflows/agentic_code.yml](.github/workflows/agentic_code.yml) workflow triggering automatically on issue labels (`agentic-fix`) or comments (`/fix`).

---

### Phase 8: Benchmarking, Evaluation & Observability
- **SWE-bench Benchmark Integration** ([agentic_code/benchmarks.py](agentic_code/benchmarks.py)):
  - Official SWE-bench and SWE-bench Lite dataset schema representation (`SWEBenchInstance`).
  - Standard Python open-source reference datasets:
    - `psf/requests` (Redirect header casing)
    - `pallets/flask` (Blueprint nested error handling)
    - `scikit-learn/scikit-learn` (NCA parameter validation with None check)
    - `sympy/sympy` (Hyperbolic cosine singularity evaluation)
  - Key Evaluation Metrics:
    - **Reproduction Rate**: % of target bugs successfully captured by reproduction scripts failing on unpatched code.
    - **Pass@k**: % of verified fixes generated within 1, 2, or 3 iterations ($k \in \{1, 2, 3\}$).
    - **Regression Rate**: Retention verification confirming existing repository test suites (`pass_to_pass`) remain green.
- **Observability & Logging** ([agentic_code/observability.py](agentic_code/observability.py)):
  - **Tracing Integration**: LangSmith configuration (`LANGCHAIN_TRACING_V2`) and OpenTelemetry span wrappers (`trace_span`).
  - **Structured Audit Logger**: Thread-safe `AuditLogger` tracking agent decisions, duration, token usage, and container lifecycle events.
  - **Diff Statistics Extraction**: Analyzes unified diffs to compute modified files, lines added, and lines deleted.
- **CLI Commands** ([agentic_code/cli.py](agentic_code/cli.py)):
  - `agentic-code resolve <title> --repo <path>`: Run automated resolution.
  - `agentic-code evaluate [--dataset <path>]`: Run SWE-bench evaluation harness.
  - `agentic-code audit [--export <path>]`: View and export structured audit traces.
  - `agentic-code server`: Start FastAPI webhook ingestion daemon.
  - `agentic-code info`: Inspect active system and model configuration.

---

## Quickstart & Verification

### 1. Install Dependencies
```bash
uv sync
uv pip install -e .
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your GEMINI_API_KEY (and GITHUB_TOKEN)
```

### 3. Run Test Suite
```bash
uv run pytest
```
Result: **46 passed in ~48s (100% passing)**

### 4. Code Quality & Linting
```bash
uv run ruff check .
```
Result: **All checks passed! (0 errors)**

### 5. Run Complete Demonstration
```bash
uv run python main.py
```
Outputs visual status and architecture verification across all 8 phases.
