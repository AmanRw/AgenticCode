"""
FastAPI Webhook Ingestion Service for AgenticCode (Phase 7).
Listens for GitHub Webhook events (issues.opened, issue_comment.created),
validates HMAC signatures (X-Hub-Signature-256), and queues jobs for execution.
"""

import hashlib
import hmac
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from agentic_code.config import settings
from agentic_code.vcs import GitManager
from agentic_code.workflow import run_workflow

logger = logging.getLogger("agentic_code.webhook")

app = FastAPI(
    title="AgenticCode Webhook Service",
    description="Listens for GitHub webhook events and triggers autonomous issue resolution pipelines.",
    version="0.1.0",
)

# In-memory execution job tracking for inspection and telemetry
JOB_QUEUE: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# HMAC Signature Verification
# ---------------------------------------------------------------------------

def verify_github_signature(
    payload_bytes: bytes,
    signature_header: Optional[str],
    secret: str,
) -> bool:
    """
    Validates GitHub HMAC-SHA256 signature (X-Hub-Signature-256).
    Uses constant-time comparison to protect against timing attacks.
    """
    if not signature_header:
        return False

    prefix = "sha256="
    if signature_header.startswith(prefix):
        signature_hash = signature_header[len(prefix):]
    else:
        signature_hash = signature_header

    mac = hmac.new(secret.encode("utf-8"), msg=payload_bytes, digestmod=hashlib.sha256)
    expected_hash = mac.hexdigest()

    return hmac.compare_digest(expected_hash, signature_hash)


# ---------------------------------------------------------------------------
# Background Pipeline Execution
# ---------------------------------------------------------------------------

def execute_pipeline_task(job_id: str, issue_dict: Dict[str, Any], repo_url: str) -> None:
    """
    Asynchronous background worker: clones repository, creates feature branch,
    and runs the LangGraph state machine.
    """
    logger.info(f"Starting pipeline execution for job {job_id} on issue #{issue_dict.get('issue_number')}")
    try:
        # Clone repo into ephemeral directory
        with GitManager.clone(repo_url=repo_url, token=settings.GITHUB_TOKEN) as git_mgr:
            branch_name = f"agentic-fix/issue-{issue_dict.get('issue_number')}"
            git_mgr.create_and_checkout_branch(branch_name)

            final_state = run_workflow(
                issue=issue_dict,
                repo_path=str(git_mgr.repo_path),
                max_iterations=settings.MAX_ITERATIONS,
            )

            # Update job record
            for job in JOB_QUEUE:
                if job["job_id"] == job_id:
                    job["status"] = "completed"
                    job["pr_url"] = final_state.get("pr_url")
                    job["iterations"] = final_state.get("iteration")
                    job["passed"] = final_state.get("test_results", {}).get("passed", False)
                    break
    except Exception as exc:
        logger.error(f"Pipeline job {job_id} failed: {exc}")
        for job in JOB_QUEUE:
            if job["job_id"] == job_id:
                job["status"] = "failed"
                job["error"] = str(exc)
                break


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint exposing runtime metadata."""
    return {
        "status": "healthy",
        "service": "AgenticCode Webhook",
        "version": "0.1.0",
        "provider": settings.DEFAULT_LLM_PROVIDER,
        "default_model": settings.DEFAULT_MODEL,
        "sandbox_image": settings.SANDBOX_IMAGE,
        "active_jobs_count": len(JOB_QUEUE),
    }


@app.get("/queue")
def get_job_queue() -> List[Dict[str, Any]]:
    """Returns recent jobs in the pipeline queue."""
    return JOB_QUEUE[-50:]


@app.post("/webhook")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: Optional[str] = Header(None, alias="X-GitHub-Event"),
    x_hub_signature_256: Optional[str] = Header(None, alias="X-Hub-Signature-256"),
) -> JSONResponse:
    """
    Main webhook ingestion endpoint.
    Validates HMAC signature and queues valid issues or comment triggers.
    """
    body_bytes = await request.body()

    # Validate signature if secret is configured
    webhook_secret = settings.GITHUB_WEBHOOK_SECRET
    if webhook_secret and webhook_secret != "disabled":
        if not verify_github_signature(body_bytes, x_hub_signature_256, webhook_secret):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid HMAC signature (X-Hub-Signature-256 mismatch).",
            )

    try:
        payload = json.loads(body_bytes.decode("utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Malformed JSON payload.")

    # 1. Handle ping event
    if x_github_event == "ping":
        return JSONResponse({"status": "pong", "zen": payload.get("zen")})

    # 2. Handle issue events (opened, labeled)
    if x_github_event == "issues":
        action = payload.get("action")
        issue_data = payload.get("issue", {})
        repo_data = payload.get("repository", {})

        if action in ["opened", "labeled"]:
            issue_num = issue_data.get("number")
            repo_url = repo_data.get("clone_url") or repo_data.get("html_url", "")

            # If triggered by label, ensure label is 'agentic-fix' or 'bug'
            if action == "labeled":
                label_name = payload.get("label", {}).get("name", "")
                if label_name not in ["agentic-fix", "bug", "auto-repair"]:
                    return JSONResponse({"status": "ignored", "reason": f"Label '{label_name}' does not trigger pipeline."})

            job_id = f"job_issue_{issue_num}_{len(JOB_QUEUE) + 1}"
            normalized_issue = {
                "title": issue_data.get("title", ""),
                "body": issue_data.get("body", ""),
                "issue_number": issue_num,
                "repo_url": repo_url,
            }

            JOB_QUEUE.append({
                "job_id": job_id,
                "issue_number": issue_num,
                "repo_url": repo_url,
                "status": "queued",
                "action": action,
            })

            background_tasks.add_task(execute_pipeline_task, job_id, normalized_issue, repo_url)

            return JSONResponse({
                "status": "queued",
                "job_id": job_id,
                "issue_number": issue_num,
                "action": action,
            })

    # 3. Handle comment events (e.g., /fix command)
    if x_github_event == "issue_comment":
        action = payload.get("action")
        comment_body = payload.get("comment", {}).get("body", "").strip().lower()
        issue_data = payload.get("issue", {})
        repo_data = payload.get("repository", {})

        if action == "created" and any(cmd in comment_body for cmd in ["/fix", "/agentic-fix", "@agentic-code fix"]):
            issue_num = issue_data.get("number")
            repo_url = repo_data.get("clone_url") or repo_data.get("html_url", "")
            job_id = f"job_comment_{issue_num}_{len(JOB_QUEUE) + 1}"

            normalized_issue = {
                "title": issue_data.get("title", ""),
                "body": issue_data.get("body", ""),
                "issue_number": issue_num,
                "repo_url": repo_url,
            }

            JOB_QUEUE.append({
                "job_id": job_id,
                "issue_number": issue_num,
                "repo_url": repo_url,
                "status": "queued",
                "trigger": "comment_command",
            })

            background_tasks.add_task(execute_pipeline_task, job_id, normalized_issue, repo_url)

            return JSONResponse({
                "status": "queued",
                "job_id": job_id,
                "issue_number": issue_num,
                "trigger": "comment_command",
            })

    return JSONResponse({"status": "ignored", "event": x_github_event, "action": payload.get("action")})
