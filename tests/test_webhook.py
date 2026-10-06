"""
Tests for Phase 7: Webhook Ingestion & CI/CD Triggering.
"""

import hashlib
import hmac
import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from agentic_code.config import settings
from agentic_code.webhook import app, verify_github_signature

client = TestClient(app)


def compute_signature(payload_bytes: bytes, secret: str) -> str:
    """Helper to compute valid GitHub X-Hub-Signature-256."""
    digest = hmac.new(secret.encode("utf-8"), msg=payload_bytes, digestmod=hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def test_hmac_signature_verification():
    secret = "my-secret-token"
    payload = b'{"action": "opened"}'

    # Valid signature
    valid_sig = compute_signature(payload, secret)
    assert verify_github_signature(payload, valid_sig, secret) is True

    # Tampered payload
    tampered_payload = b'{"action": "closed"}'
    assert verify_github_signature(tampered_payload, valid_sig, secret) is False

    # Invalid signature
    assert verify_github_signature(payload, "sha256=invalidhash", secret) is False

    # Missing signature
    assert verify_github_signature(payload, None, secret) is False


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "AgenticCode Webhook"
    assert "provider" in data


def test_webhook_unauthorized_when_signature_invalid():
    payload = json.dumps({"action": "opened"}).encode("utf-8")

    response = client.post(
        "/webhook",
        content=payload,
        headers={
            "X-GitHub-Event": "issues",
            "X-Hub-Signature-256": "sha256=bogus_signature",
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 401
    assert "Invalid HMAC signature" in response.json()["detail"]


def test_webhook_ping_event():
    secret = settings.GITHUB_WEBHOOK_SECRET or "dev-webhook-secret"
    payload = json.dumps({"zen": "Keep it logically awesome."}).encode("utf-8")
    sig = compute_signature(payload, secret)

    response = client.post(
        "/webhook",
        content=payload,
        headers={
            "X-GitHub-Event": "ping",
            "X-Hub-Signature-256": sig,
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "pong"


def test_webhook_issue_opened_trigger():
    secret = settings.GITHUB_WEBHOOK_SECRET or "dev-webhook-secret"
    payload_data = {
        "action": "opened",
        "issue": {
            "number": 42,
            "title": "Division by zero bug",
            "body": "Crash in divide()",
        },
        "repository": {
            "clone_url": "https://github.com/example/repo.git",
        },
    }
    payload_bytes = json.dumps(payload_data).encode("utf-8")
    sig = compute_signature(payload_bytes, secret)

    with patch("agentic_code.webhook.execute_pipeline_task"):
        response = client.post(
            "/webhook",
            content=payload_bytes,
            headers={
                "X-GitHub-Event": "issues",
                "X-Hub-Signature-256": sig,
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 200
        res_data = response.json()
        assert res_data["status"] == "queued"
        assert res_data["issue_number"] == 42


def test_webhook_issue_labeled_trigger():
    secret = settings.GITHUB_WEBHOOK_SECRET or "dev-webhook-secret"

    # Labeled with 'agentic-fix' -> queued
    payload_data = {
        "action": "labeled",
        "label": {"name": "agentic-fix"},
        "issue": {"number": 101, "title": "Memory leak", "body": "Fix required"},
        "repository": {"clone_url": "https://github.com/example/repo.git"},
    }
    payload_bytes = json.dumps(payload_data).encode("utf-8")
    sig = compute_signature(payload_bytes, secret)

    with patch("agentic_code.webhook.execute_pipeline_task"):
        response = client.post(
            "/webhook",
            content=payload_bytes,
            headers={
                "X-GitHub-Event": "issues",
                "X-Hub-Signature-256": sig,
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "queued"

    # Labeled with irrelevant label -> ignored
    payload_data["label"]["name"] = "documentation"
    payload_bytes = json.dumps(payload_data).encode("utf-8")
    sig = compute_signature(payload_bytes, secret)

    response = client.post(
        "/webhook",
        content=payload_bytes,
        headers={
            "X-GitHub-Event": "issues",
            "X-Hub-Signature-256": sig,
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ignored"


def test_webhook_comment_fix_command():
    secret = settings.GITHUB_WEBHOOK_SECRET or "dev-webhook-secret"

    # Comment with /fix command -> queued
    payload_data = {
        "action": "created",
        "comment": {"body": "/fix please investigate this crash"},
        "issue": {"number": 88, "title": "Bug", "body": "Details"},
        "repository": {"clone_url": "https://github.com/example/repo.git"},
    }
    payload_bytes = json.dumps(payload_data).encode("utf-8")
    sig = compute_signature(payload_bytes, secret)

    with patch("agentic_code.webhook.execute_pipeline_task"):
        response = client.post(
            "/webhook",
            content=payload_bytes,
            headers={
                "X-GitHub-Event": "issue_comment",
                "X-Hub-Signature-256": sig,
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        assert response.json()["trigger"] == "comment_command"


def test_queue_inspection_endpoint():
    response = client.get("/queue")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
