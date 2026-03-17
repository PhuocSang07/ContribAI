"""Pull Request lifecycle manager.

Handles the full PR workflow: fork → branch → commit → PR.
Generates detailed PR descriptions with context and testing info.
"""

from __future__ import annotations

import logging

from contribai.core.exceptions import PRCreationError
from contribai.core.models import Contribution, ContributionType, PRResult, PRStatus, Repository
from contribai.github.client import GitHubClient

logger = logging.getLogger(__name__)


class PRManager:
    """Manage the full pull request lifecycle."""

    def __init__(self, github: GitHubClient):
        self._github = github
        self._user: dict | None = None

    async def _get_user(self) -> dict:
        """Get and cache the authenticated user."""
        if not self._user:
            self._user = await self._github.get_authenticated_user()
        return self._user

    async def create_pr(self, contribution: Contribution, target_repo: Repository) -> PRResult:
        """Create a PR from a generated contribution.

        Full workflow:
        1. Fork the target repo (if not already forked)
        2. Create a feature branch on the fork
        3. Commit all file changes
        4. Create the pull request
        """
        user = await self._get_user()
        username = user["login"]

        try:
            # 1. Fork
            fork = await self._fork_if_needed(username, target_repo)
            fork_owner = fork.owner
            fork_name = fork.name

            # 2. Create branch
            branch = contribution.branch_name or f"contribai/fix-{contribution.finding.id}"
            await self._github.create_branch(fork_owner, fork_name, branch)

            # 3. Commit changes
            for change in contribution.changes + contribution.tests_added:
                # Get existing file SHA for updates
                sha = None
                if not change.is_new_file:
                    try:
                        await self._github.get_file_content(fork_owner, fork_name, change.path)
                        # We need the SHA to update - fetch via contents API

                        resp = await self._github._get(
                            f"/repos/{fork_owner}/{fork_name}/contents/{change.path}",
                            params={"ref": branch},
                        )
                        sha = resp.get("sha")
                    except Exception:
                        pass  # New file or couldn't get SHA

                await self._github.create_or_update_file(
                    fork_owner,
                    fork_name,
                    change.path,
                    change.new_content,
                    contribution.commit_message,
                    branch,
                    sha=sha,
                )

            # 4. Create PR
            pr_body = self._generate_pr_body(contribution)
            head = f"{fork_owner}:{branch}"

            pr_data = await self._github.create_pull_request(
                target_repo.owner,
                target_repo.name,
                title=contribution.title,
                body=pr_body,
                head=head,
                base=target_repo.default_branch,
            )

            result = PRResult(
                repo=target_repo,
                contribution=contribution,
                pr_number=pr_data["number"],
                pr_url=pr_data["html_url"],
                status=PRStatus.OPEN,
                branch_name=branch,
                fork_full_name=f"{fork_owner}/{fork_name}",
            )

            logger.info("✅ PR #%d created: %s", result.pr_number, result.pr_url)
            return result

        except Exception as e:
            raise PRCreationError(f"Failed to create PR: {e}") from e

    async def _fork_if_needed(self, username: str, repo: Repository) -> Repository:
        """Fork the repo if not already forked."""
        try:
            # Check if fork exists
            existing = await self._github.get_repo_details(username, repo.name)
            if existing.owner == username:
                logger.info("Fork already exists: %s/%s", username, repo.name)
                return existing
        except Exception:
            pass

        # Create fork
        return await self._github.fork_repository(repo.owner, repo.name)

    def _generate_pr_body(self, contribution: Contribution) -> str:
        """Generate a detailed PR description."""
        finding = contribution.finding

        # Type-specific emoji and label
        type_info = {
            ContributionType.SECURITY_FIX: ("🔒", "Security Fix"),
            ContributionType.CODE_QUALITY: ("✨", "Code Quality"),
            ContributionType.DOCS_IMPROVE: ("📝", "Documentation"),
            ContributionType.UI_UX_FIX: ("🎨", "UI/UX Improvement"),
            ContributionType.PERFORMANCE_OPT: ("⚡", "Performance"),
            ContributionType.FEATURE_ADD: ("🚀", "New Feature"),
            ContributionType.REFACTOR: ("♻️", "Refactoring"),
        }
        emoji, label = type_info.get(finding.type, ("🔧", "Fix"))

        # Files changed summary
        files_list = "\n".join(
            f"- `{c.path}` {'(new)' if c.is_new_file else '(modified)'}"
            for c in contribution.changes
        )

        body = f"""## {emoji} {label}

### Problem
{finding.description}

**Severity**: `{finding.severity.value}`
**File**: `{finding.file_path}`

### Solution
{contribution.description}

### Changes
{files_list}

### Testing
- [ ] Existing tests pass
- [ ] Manual review completed
- [ ] No new warnings/errors introduced

---

<details>
<summary>🤖 About this PR</summary>

This pull request was generated by [ContribAI](https://github.com/tang-vu/ContribAI), an AI agent
that helps improve open source projects. The change was:

1. **Discovered** by automated code analysis
2. **Generated** by AI with context-aware code generation
3. **Self-reviewed** by AI quality checks

If you have questions or feedback about this PR, please comment below.
We appreciate your time reviewing this contribution!

</details>
"""
        return body

    async def get_pr_status(self, owner: str, repo: str, pr_number: int) -> PRStatus:
        """Check the current status of a PR."""
        try:
            data = await self._github._get(f"/repos/{owner}/{repo}/pulls/{pr_number}")
            state = data.get("state", "open")
            merged = data.get("merged", False)

            if merged:
                return PRStatus.MERGED
            elif state == "closed":
                return PRStatus.CLOSED
            elif data.get("requested_reviewers"):
                return PRStatus.REVIEW_REQUESTED
            else:
                return PRStatus.OPEN
        except Exception:
            return PRStatus.PENDING
