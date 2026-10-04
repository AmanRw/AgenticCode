# AgenticCode: End-to-End Autonomous GitHub Issue Resolution

An industrial-grade, multi-agent AI framework engineered to autonomously resolve GitHub issues from bug ingestion to verified Pull Request synthesis.

---

## Architecture Overview (Phases 1 – 4)

```
AgenticCode/
├── .env.example                     # Environment secret templates
├── pyproject.toml                   # uv/pip dependencies & tooling config
├── agentic_code/
│   ├── config.py                    # Phase 1: Pydantic Settings & LLM Factory
│   ├── state.py                     # Phase 1: Shared Orchestrator AgentState (TypedDict)
│   ├── sandbox/                     # Phase 2: Hardened Docker Sandbox Engine
│   │   ├── Dockerfile               # Unprivileged UID 1000:1000, python:3.11-slim
│   │   ├── sandbox.py               # Docker-py wrapper, security policies & quotas
│   │   └── fallback.py              # Isolated subprocess fallback sandbox
│   ├── tools/                       # Phase 3: Code Retrieval & AST Analysis Tools
│   │   ├── ast_tools.py             # AST parser, symbol outlines & snippet extractor
│   │   ├── search_tools.py          # Fast lexical & traceback pattern matching
│   │   └── graph_tools.py           # AST caller/callee call graph generator
│   └── agents/                      # Phase 4: Specialized Agent Personas
│       ├── base.py                  # Base persona, prompt orchestration & parsers
│       ├── code_reader.py           # Agent 01: Code Reader (localization)
│       ├── planner.py               # Agent 02: Planner (root-cause diagnosis)
│       ├── test_writer.py           # Agent 04: Test Writer (reproduction & regression)
│       ├── code_writer.py           # Agent 03: Code Writer (patch synthesis)
│       └── pr_opener.py             # Agent 05: PR Opener (commit & PR generation)
└── tests/                           # Complete test suite (20/20 passing)
    ├── test_state.py                # State & settings tests
    ├── test_sandbox.py              # Docker & fallback sandbox tests
    ├── test_ast_tools.py            # AST & search tools tests
    └── test_agents.py               # 5 specialized agent personas tests
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
- **Ephemeral Mounts & Fallback**: Mounts repository workspaces temporarily and includes [LocalSubprocessSandbox](agentic_code/sandbox/fallback.py) for environments without active Docker engines.

---

### Phase 3: Code Retrieval & AST Analysis Tools
- **Symbol & File Outlines** ([agentic_code/tools/ast_tools.py](agentic_code/tools/ast_tools.py)): Parses Python Abstract Syntax Trees to extract top-level classes, method signatures, docstrings, and imports across the entire repository without exhausting LLM context windows.
- **Precise Snippet Extraction**: Slices target lines with surrounding context for focused LLM evaluation.
- **Lexical & Structural Search** ([agentic_code/tools/search_tools.py](agentic_code/tools/search_tools.py)): Fast keyword, regex, and error traceback parser mapping crash locations directly to repository files.
- **Call Graph Generation** ([agentic_code/tools/graph_tools.py](agentic_code/tools/graph_tools.py)): Analyzes function calls to construct caller/callee dependency graphs.

---

### Phase 4: Specialized Agent Personas
Five modular agent personas built with structured inputs, outputs, and system prompts:
1. **Agent 01: Code Reader** ([agentic_code/agents/code_reader.py](agentic_code/agents/code_reader.py)): Pinpoints target files and extracts relevant code snippets without overflowing context.
2. **Agent 02: Planner** ([agentic_code/agents/planner.py](agentic_code/agents/planner.py)): Diagnoses root cause and produces a 4-part Markdown repair plan (Identified Bug, Invariants, Logic Modifications, Test Requirements).
3. **Agent 03: Test Writer** ([agentic_code/agents/test_writer.py](agentic_code/agents/test_writer.py)): Generates standalone reproduction and regression tests that fail on unpatched code and pass once resolved.
4. **Agent 04: Code Writer** ([agentic_code/agents/code_writer.py](agentic_code/agents/code_writer.py)): Synthesizes minimal, unified diffs or file updates incorporating iterative failure feedback.
5. **Agent 05: PR Opener** ([agentic_code/agents/pr_opener.py](agentic_code/agents/pr_opener.py)): Synthesizes Git branch names, conventional commit messages, and Markdown PR bodies linking issues (`Fixes #<id>`).

---

### Phase 5: Git & GitHub VCS Integration Layer
- **Local Git Operations** ([agentic_code/vcs.py](agentic_code/vcs.py)):
  - Clones target repositories into ephemeral workspaces.
  - Creates and manages isolated fix branches (`agentic-fix/issue-<id>`).
  - Writes synthesized test files and applies unified diff patches.
  - Implements clean working tree rollbacks (`git checkout -f HEAD` and `git clean -fd`) ensuring failed repair iterations leave no dirty state.
  - Commits verified patches with automated bot authorship.
- **GitHub REST/GraphQL Client** ([agentic_code/agents.py](agentic_code/agents.py)):
  - Authenticates via GitHub App token or Personal Access Token (PAT).
  - Fetches complete issue metadata (title, body, author, labels, comments).
  - Opens linked Pull Requests with Markdown bodies (`Fixes #<id>`) detailing root causes, changes, and test proofs.
  - Posts diagnostic explanation comments directly to issue threads if repair loops exceed $k_{max}$ iterations.

---

## Quickstart & Verification

### 1. Install Dependencies
```bash
uv sync
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
Result: `25 passed in ~14s`

### 4. Code Quality & Linting
```bash
uv run ruff check .
```
Result: `All checks passed!`

---

## Upcoming Phases (6 – 8)
- **Phase 6**: LangGraph State Machine & Cyclic Orchestration (parallel dispatch, reproduction verification, retry loops).
- **Phase 7**: Webhook Ingestion & CI/CD Triggering (FastAPI HMAC service & GitHub Action wrapper).
- **Phase 8**: Benchmarking & Observability (SWE-bench evaluation harness & OpenTelemetry/LangSmith tracing).

