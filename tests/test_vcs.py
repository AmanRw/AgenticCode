"""
Tests for Phase 5 Git & GitHub VCS Integration Layer.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import git

from agentic_code.vcs import GitHubClient, GitManager


def test_git_manager_lifecycle():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        repo_path = Path(tmpdir)

        # Initialize a fresh git repo
        repo = git.Repo.init(repo_path)
        # Configure git identity for commits
        repo.config_writer().set_value("user", "name", "Test Runner").release()
        repo.config_writer().set_value("user", "email", "test@example.com").release()

        # Create an initial file and commit
        init_file = repo_path / "hello.py"
        init_file.write_text("def hello():\n    return 'world'\n", encoding="utf-8")
        repo.git.add("-A")
        repo.git.commit("-m", "Initial commit")
        repo.close()

        # Instantiate GitManager
        with GitManager(repo_path) as manager:
            assert manager.repo is not None

            # 1. Branch Creation & Checkout
            branch_name = manager.create_and_checkout_branch("agentic-fix/issue-42")
            assert branch_name == "agentic-fix/issue-42"
            assert manager.repo.active_branch.name == "agentic-fix/issue-42"

            # 2. Write Test File
            test_path = manager.write_test_file("test_reproduce.py", "def test_hello(): assert True")
            assert test_path.exists()
            assert "test_reproduce.py" in test_path.name

            # 3. Commit Changes
            commit_sha = manager.commit_changes("Add reproduction test")
            assert len(commit_sha) == 40
            assert not manager.repo.is_dirty()

            # 4. Dirty Modifications & Rollback
            # Modify hello.py and create an untracked scratch file
            init_file.write_text("def hello():\n    return 'broken'\n", encoding="utf-8")
            scratch = repo_path / "scratch.py"
            scratch.write_text("junk", encoding="utf-8")
            assert manager.repo.is_dirty() or len(manager.repo.untracked_files) > 0

            # Perform rollback
            manager.rollback_changes()
            assert not manager.repo.is_dirty()
            assert not scratch.exists()
            assert "return 'world'" in init_file.read_text(encoding="utf-8")


def test_git_manager_apply_patch():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        repo_path = Path(tmpdir)
        repo = git.Repo.init(repo_path)
        repo.config_writer().set_value("user", "name", "Test Runner").release()
        repo.config_writer().set_value("user", "email", "test@example.com").release()

        math_file = repo_path / "math_mod.py"
        math_file.write_text("def divide(a, b):\n    return a / b\n", encoding="utf-8")
        repo.git.add("-A")
        repo.git.commit("-m", "Initial commit")
        repo.close()

        with GitManager(repo_path) as manager:
            # Unified diff patch
            unified_diff = """--- a/math_mod.py
+++ b/math_mod.py
@@ -1,2 +1,4 @@
 def divide(a, b):
+    if b == 0:
+        raise ValueError("Cannot divide by zero")
     return a / b
"""
            manager.apply_patch(unified_diff)
            content = math_file.read_text(encoding="utf-8")
            assert "raise ValueError" in content


def test_github_client_url_extraction():
    client = GitHubClient(token="dummy_token")
    assert client.extract_repo_name("https://github.com/psf/requests.git") == "psf/requests"
    assert client.extract_repo_name("psf/requests") == "psf/requests"
    assert client.extract_repo_name("git@github.com:pallets/flask.git") == "pallets/flask"


def test_github_client_operations_mocked():
    client = GitHubClient(token="dummy_token")

    mock_issue = MagicMock()
    mock_issue.title = "ZeroDivisionError in divide()"
    mock_issue.body = "When b=0, code crashes."
    mock_issue.number = 42
    mock_issue.html_url = "https://github.com/owner/repo/issues/42"
    mock_issue.user.login = "octocat"
    lbl = MagicMock()
    lbl.name = "bug"
    mock_issue.labels = [lbl]
    comment_mock = MagicMock()
    comment_mock.body = "Reproduced on Python 3.11"
    mock_issue.get_comments.return_value = [comment_mock]

    mock_pr = MagicMock()
    mock_pr.html_url = "https://github.com/owner/repo/pull/43"

    mock_comment = MagicMock()
    mock_comment.html_url = "https://github.com/owner/repo/issues/42#issuecomment-99"

    mock_repo = MagicMock()
    mock_repo.get_issue.return_value = mock_issue
    mock_repo.create_pull.return_value = mock_pr
    mock_repo.default_branch = "main"

    with patch.object(GitHubClient, "gh", new_callable=MagicMock) as mock_gh:
        mock_gh.get_repo.return_value = mock_repo

        # Test fetch issue
        issue_meta = client.fetch_issue("owner/repo", 42)
        assert issue_meta.issue_number == 42
        assert issue_meta.title == "ZeroDivisionError in divide()"
        assert issue_meta.labels == ["bug"]
        assert len(issue_meta.comments) == 1

        # Test create PR
        pr_url = client.create_pull_request(
            repo_name="owner/repo",
            title="fix: resolve ZeroDivisionError",
            body="Fixes #42",
            head_branch="agentic-fix/issue-42",
            base_branch="main",
        )
        assert pr_url == "https://github.com/owner/repo/pull/43"

        # Test post comment
        mock_issue.create_comment.return_value = mock_comment
        c_url = client.post_issue_comment("owner/repo", 42, "Fixed in branch.")
        assert "issuecomment-99" in c_url

        # Test post diagnostic failure
        fail_url = client.post_diagnostic_failure(
            repo_name="owner/repo",
            issue_number=42,
            plan="Root cause: Zero division",
            iterations=3,
            error_feedback="AssertionError: Expected ValueError",
        )
        assert "issuecomment-99" in fail_url


def test_git_manager_rejects_dummy_patch():
    import pytest
    from agentic_code.vcs import GitOperationError

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        repo_path = Path(tmpdir)
        repo = git.Repo.init(repo_path)
        repo.config_writer().set_value("user", "name", "Test").release()
        repo.config_writer().set_value("user", "email", "test@example.com").release()
        file_p = repo_path / "simpleCalc.py"
        file_p.write_text("def op(a, b): return a + b\n", encoding="utf-8")
        repo.git.add("-A")
        repo.git.commit("-m", "Initial")
        repo.close()

        with GitManager(repo_path) as mgr:
            # 1. Reject dummy unified diff
            dummy_diff = "--- a/simpleCalc.py\n+++ b/simpleCalc.py\n@@ -1,1 +1,2 @@\n+# Fix applied\n"
            with pytest.raises(GitOperationError):
                mgr.apply_patch(dummy_diff, target_file="simpleCalc.py")

            # 2. Reject dummy comment code
            with pytest.raises(GitOperationError):
                mgr.apply_patch("# Fix applied", target_file="simpleCalc.py")

