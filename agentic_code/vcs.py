"""
Unified Git & GitHub VCS Integration Layer (Phase 5).
Manages repository lifecycle, clean rollbacks on failed repair attempts,
and GitHub API communication via PyGithub.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union

from git import GitCommandError, InvalidGitRepositoryError, Repo
from github import Auth, Github, GithubException

from agentic_code.config import settings
from agentic_code.state import IssueMetadata

logger = logging.getLogger("agentic_code.vcs")


class GitOperationError(Exception):
    """Raised when local Git operations fail."""

    pass


class GitHubClientError(Exception):
    """Raised when remote GitHub API calls fail."""

    pass


class GitManager:
    """Manages local Git repository lifecycle, branching, patch application, and clean rollbacks."""

    def __init__(self, repo_path: Optional[Union[str, Path]] = None):
        self.repo_path = Path(repo_path).resolve() if repo_path else None
        self._repo: Optional[Repo] = None

        if self.repo_path and (self.repo_path / ".git").exists():
            try:
                self._repo = Repo(self.repo_path)
            except (InvalidGitRepositoryError, GitCommandError) as e:
                raise GitOperationError(f"Failed to load repo at {self.repo_path}: {e}")

    @property
    def repo(self) -> Repo:
        if self._repo is None:
            raise GitOperationError("Repository has not been initialized.")
        return self._repo

    def close(self) -> None:
        """Closes repository file handles on Windows."""
        if self._repo is not None:
            self._repo.close()

    def __enter__(self) -> "GitManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @classmethod
    def clone(
        cls,
        repo_url: str,
        target_dir: Optional[Union[str, Path]] = None,
        depth: Optional[int] = None,
        token: Optional[str] = None,
    ) -> "GitManager":
        """Clones a target repository into an ephemeral directory."""
        if target_dir is None:
            base_work = Path(settings.WORK_DIR)
            base_work.mkdir(parents=True, exist_ok=True)
            destination = Path(tempfile.mkdtemp(prefix="agentic_clone_", dir=str(base_work)))
        else:
            destination = Path(target_dir).resolve()
            destination.mkdir(parents=True, exist_ok=True)

        clone_url = repo_url
        if token and "github.com" in repo_url and not repo_url.startswith("git@"):
            clean = repo_url.replace("https://", "").replace("http://", "")
            clone_url = f"https://x-access-token:{token}@{clean}"

        kwargs = {"depth": depth} if depth else {}
        try:
            repo = Repo.clone_from(clone_url, destination, **kwargs)
            mgr = cls(destination)
            mgr._repo = repo
            return mgr
        except GitCommandError as e:
            if destination.exists():
                shutil.rmtree(destination, ignore_errors=True)
            raise GitOperationError(f"Failed to clone {repo_url}: {e}")

    def create_and_checkout_branch(self, branch_name: str) -> str:
        """Creates and checks out an isolated feature branch."""
        try:
            existing = [b.name for b in self.repo.branches]
            if branch_name in existing:
                self.repo.git.checkout(branch_name)
            else:
                self.repo.git.checkout("-b", branch_name)
            return branch_name
        except GitCommandError as e:
            raise GitOperationError(f"Failed to checkout branch '{branch_name}': {e}")

    def apply_patch(self, patch_content: str) -> None:
        """Applies a unified diff patch with whitespace fix and 3-way merge fallback."""
        if not patch_content or not patch_content.strip():
            return
        text = patch_content.strip()

        if ("--- " in text and "+++ " in text) or "@@" in text:
            with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False, encoding="utf-8") as f:
                f.write(text + "\n")
                tmp_patch = f.name
            try:
                self.repo.git.apply("--whitespace=fix", tmp_patch)
            except GitCommandError:
                try:
                    self.repo.git.apply("--3way", "--whitespace=fix", tmp_patch)
                except GitCommandError as e:
                    raise GitOperationError(f"Failed to apply unified diff patch: {e}")
            finally:
                if os.path.exists(tmp_patch):
                    os.remove(tmp_patch)
        else:
            raise GitOperationError("Patch content is not in unified diff format.")

    def write_test_file(self, filename: str, content: str) -> Path:
        """Writes reproduction or regression test file to the workspace."""
        if self.repo_path is None:
            raise GitOperationError("No repository path configured.")
        tests_dir = self.repo_path / "tests"
        target_path = (tests_dir / filename) if tests_dir.is_dir() else (self.repo_path / filename)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")
        return target_path

    def rollback_changes(self) -> None:
        """Rolls back failed repair attempts cleanly (reverts modified files & removes untracked)."""
        try:
            # Safeguard: Do not destructively wipe the user's active development workspace on master/main
            try:
                if self.repo_path.resolve() == Path(".").resolve() and self.repo.active_branch.name in ("master", "main"):
                    logger.warning("Safeguard: Skipping destructive checkout rollback on active workspace root.")
                    return
            except Exception:
                pass
            self.repo.git.checkout("-f", "HEAD")
            self.repo.git.clean("-fd")
        except GitCommandError as e:
            raise GitOperationError(f"Failed to rollback working tree: {e}")

    def commit_changes(
        self,
        message: str,
        author_name: str = "AgenticCode Bot",
        author_email: str = "bot@agenticcode.ai",
    ) -> str:
        """Stages all changes and creates a commit with designated bot authorship."""
        try:
            self.repo.git.add("-A")
            env = {
                "GIT_AUTHOR_NAME": author_name,
                "GIT_AUTHOR_EMAIL": author_email,
                "GIT_COMMITTER_NAME": author_name,
                "GIT_COMMITTER_EMAIL": author_email,
            }
            self.repo.git.commit("-m", message, env=env)
            return self.repo.head.commit.hexsha
        except GitCommandError as e:
            raise GitOperationError(f"Failed to commit changes: {e}")

    def get_diff(self, base_ref: str = "HEAD~1") -> str:
        """Returns unified diff of recent changes."""
        try:
            return self.repo.git.diff("HEAD") or self.repo.git.diff(base_ref, "HEAD")
        except GitCommandError:
            try:
                return self.repo.git.diff()
            except GitCommandError:
                return ""

    def push_branch(self, branch_name: str, remote_name: str = "origin", force: bool = False) -> None:
        """Pushes branch to remote repository."""
        try:
            args = ["-u", remote_name, branch_name]
            if force:
                args.insert(0, "--force")
            self.repo.git.push(*args)
        except GitCommandError as e:
            raise GitOperationError(f"Failed to push branch '{branch_name}': {e}")


class GitHubClient:
    """Client for GitHub API operations: fetching issues, creating PRs, and posting diagnostics."""

    def __init__(self, token: Optional[str] = None, base_url: Optional[str] = None):
        self.token = token or settings.GITHUB_TOKEN
        self.base_url = base_url
        if self.token:
            auth = Auth.Token(self.token)
            self._gh = Github(auth=auth, base_url=base_url) if base_url else Github(auth=auth)
        else:
            self._gh = None

    @property
    def gh(self) -> Github:
        if self._gh is None:
            raise GitHubClientError("GitHubClient is unauthenticated. Provide GITHUB_TOKEN.")
        return self._gh

    def extract_repo_name(self, repo_str: str) -> str:
        """Normalizes URLs or owner/repo strings to 'owner/repo'."""
        clean = repo_str.strip().rstrip("/")
        if "github.com/" in clean:
            clean = clean.split("github.com/")[-1]
        elif "github.com:" in clean:
            clean = clean.split("github.com:")[-1]
        return clean[:-4] if clean.endswith(".git") else clean

    def fetch_issue(self, repo_name: str, issue_number: int) -> IssueMetadata:
        """Fetches issue details into typed IssueMetadata."""
        name = self.extract_repo_name(repo_name)
        try:
            repo = self.gh.get_repo(name)
            issue = repo.get_issue(number=issue_number)
            return IssueMetadata(
                title=issue.title or "",
                body=issue.body or "",
                issue_number=issue.number,
                repo_url=issue.html_url or f"https://github.com/{name}",
                author=issue.user.login if issue.user else None,
                labels=[lbl.name for lbl in issue.labels],
                comments=[c.body for c in issue.get_comments()],
            )
        except GithubException as e:
            raise GitHubClientError(f"Failed to fetch #{issue_number} from {name}: {e}")

    def create_pull_request(
        self,
        repo_name: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
    ) -> str:
        """Opens a Pull Request with Markdown body linking the issue."""
        name = self.extract_repo_name(repo_name)
        try:
            repo = self.gh.get_repo(name)
            base = base_branch
            try:
                repo.get_branch(base)
            except GithubException:
                base = repo.default_branch
            pr = repo.create_pull(title=title, body=body, head=head_branch, base=base)
            return pr.html_url
        except GithubException as e:
            raise GitHubClientError(f"Failed to create PR for {head_branch}: {e}")

    def post_issue_comment(self, repo_name: str, issue_number: int, comment_body: str) -> str:
        """Posts a comment to an issue thread."""
        name = self.extract_repo_name(repo_name)
        try:
            repo = self.gh.get_repo(name)
            issue = repo.get_issue(number=issue_number)
            comment = issue.create_comment(comment_body)
            return comment.html_url or str(comment.id)
        except GithubException as e:
            raise GitHubClientError(f"Failed to post comment to #{issue_number}: {e}")

    def post_diagnostic_failure(
        self,
        repo_name: str,
        issue_number: int,
        plan: str,
        iterations: int,
        error_feedback: Optional[str] = None,
    ) -> str:
        """Posts a structured diagnostic explanation when repair fails after k_max attempts."""
        body = f"""### ⚠️ AgenticCode Diagnostic Report: Automated Repair Unsuccessful

AgenticCode attempted to autonomously resolve this issue over **{iterations} repair iterations**, but could not verify a passing solution in the hardened sandbox.

#### Root-Cause Diagnosis
{plan}

#### Sandbox Execution Traces
```
{(error_feedback or 'Sandbox tests failed to pass assertions.')[-2000:]}
```

---
*Generated autonomously by **AgenticCode Escalation Node**.*"""
        return self.post_issue_comment(repo_name, issue_number, body)
