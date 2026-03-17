"""Async GitHub API client.

Handles all GitHub REST API interactions: repo metadata,
file content, forking, branching, committing, and PR creation.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from contribai.core.exceptions import GitHubAPIError, RateLimitError
from contribai.core.models import FileNode, Issue, Repository

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


class GitHubClient:
    """Async GitHub REST API client."""

    def __init__(self, token: str, rate_limit_buffer: int = 100):
        self._token = token
        self._rate_limit_buffer = rate_limit_buffer
        self._client = httpx.AsyncClient(
            base_url=GITHUB_API,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
        )

    async def close(self):
        await self._client.aclose()

    # ── Core HTTP ──────────────────────────────────────────────────────────

    async def _request(self, method: str, url: str, **kwargs) -> Any:
        """Make an authenticated GitHub API request with error handling."""
        try:
            response = await self._client.request(method, url, **kwargs)
        except httpx.HTTPError as e:
            raise GitHubAPIError(f"HTTP error: {e}") from e

        if response.status_code == 403:
            remaining = response.headers.get("x-ratelimit-remaining", "?")
            reset = response.headers.get("x-ratelimit-reset")
            if remaining == "0":
                raise RateLimitError(reset_at=int(reset) if reset else None)
            raise GitHubAPIError(f"Forbidden: {response.text}", status_code=403)

        if response.status_code == 404:
            raise GitHubAPIError(f"Not found: {url}", status_code=404)

        if response.status_code >= 400:
            raise GitHubAPIError(
                f"GitHub API error {response.status_code}: {response.text}",
                status_code=response.status_code,
            )

        return response.json() if response.content else None

    async def _get(self, url: str, **kwargs) -> Any:
        return await self._request("GET", url, **kwargs)

    async def _post(self, url: str, **kwargs) -> Any:
        return await self._request("POST", url, **kwargs)

    async def _put(self, url: str, **kwargs) -> Any:
        return await self._request("PUT", url, **kwargs)

    # ── Rate Limit ─────────────────────────────────────────────────────────

    async def check_rate_limit(self) -> dict:
        """Check current rate limit status."""
        data = await self._get("/rate_limit")
        core = data["resources"]["core"]
        logger.info(
            "Rate limit: %d/%d remaining (resets at %s)",
            core["remaining"],
            core["limit"],
            core["reset"],
        )
        return core

    async def _ensure_rate_limit(self):
        """Ensure we have enough API calls remaining."""
        core = await self.check_rate_limit()
        if core["remaining"] < self._rate_limit_buffer:
            raise RateLimitError(
                reset_at=core["reset"],
                details={"remaining": core["remaining"], "buffer": self._rate_limit_buffer},
            )

    # ── Repository Operations ──────────────────────────────────────────────

    async def search_repositories(
        self,
        query: str,
        sort: str = "stars",
        order: str = "desc",
        per_page: int = 30,
    ) -> list[Repository]:
        """Search GitHub repositories."""
        data = await self._get(
            "/search/repositories",
            params={"q": query, "sort": sort, "order": order, "per_page": per_page},
        )
        return [self._parse_repo(item) for item in data.get("items", [])]

    async def get_repo_details(self, owner: str, repo: str) -> Repository:
        """Get detailed repository information."""
        data = await self._get(f"/repos/{owner}/{repo}")
        return self._parse_repo(data)

    async def get_file_tree(
        self, owner: str, repo: str, branch: str | None = None
    ) -> list[FileNode]:
        """Get the full file tree of a repository."""
        if not branch:
            details = await self.get_repo_details(owner, repo)
            branch = details.default_branch

        data = await self._get(
            f"/repos/{owner}/{repo}/git/trees/{branch}",
            params={"recursive": "1"},
        )
        return [
            FileNode(
                path=item["path"],
                type=item["type"],
                size=item.get("size", 0),
                sha=item["sha"],
            )
            for item in data.get("tree", [])
        ]

    async def get_file_content(self, owner: str, repo: str, path: str) -> str:
        """Get the content of a file from the repository."""
        data = await self._get(f"/repos/{owner}/{repo}/contents/{path}")
        if data.get("encoding") == "base64":
            return base64.b64decode(data["content"]).decode("utf-8")
        return data.get("content", "")

    async def get_open_issues(
        self, owner: str, repo: str, per_page: int = 30, labels: str | None = None
    ) -> list[Issue]:
        """Get open issues for a repository."""
        params: dict[str, Any] = {"state": "open", "per_page": per_page}
        if labels:
            params["labels"] = labels

        data = await self._get(f"/repos/{owner}/{repo}/issues", params=params)
        return [
            Issue(
                number=item["number"],
                title=item["title"],
                body=item.get("body"),
                labels=[lbl["name"] for lbl in item.get("labels", [])],
                state=item["state"],
                html_url=item["html_url"],
            )
            for item in data
            if "pull_request" not in item  # exclude PRs from issues
        ]

    async def get_contributing_guide(self, owner: str, repo: str) -> str | None:
        """Try to fetch CONTRIBUTING.md."""
        for path in ["CONTRIBUTING.md", "contributing.md", ".github/CONTRIBUTING.md"]:
            try:
                return await self.get_file_content(owner, repo, path)
            except GitHubAPIError:
                continue
        return None

    # ── Fork & Branch ──────────────────────────────────────────────────────

    async def fork_repository(self, owner: str, repo: str) -> Repository:
        """Fork a repository to the authenticated user's account."""
        data = await self._post(f"/repos/{owner}/{repo}/forks")
        logger.info("Forked %s/%s → %s", owner, repo, data["full_name"])
        return self._parse_repo(data)

    async def create_branch(
        self, owner: str, repo: str, branch_name: str, from_branch: str | None = None
    ) -> dict:
        """Create a new branch from the default or specified branch."""
        if not from_branch:
            details = await self.get_repo_details(owner, repo)
            from_branch = details.default_branch

        # Get the SHA of the source branch
        ref_data = await self._get(f"/repos/{owner}/{repo}/git/ref/heads/{from_branch}")
        sha = ref_data["object"]["sha"]

        data = await self._post(
            f"/repos/{owner}/{repo}/git/refs",
            json={"ref": f"refs/heads/{branch_name}", "sha": sha},
        )
        logger.info("Created branch %s on %s/%s", branch_name, owner, repo)
        return data

    # ── Commit & PR ────────────────────────────────────────────────────────

    async def create_or_update_file(
        self,
        owner: str,
        repo: str,
        path: str,
        content: str,
        message: str,
        branch: str,
        sha: str | None = None,
    ) -> dict:
        """Create or update a file in the repository."""
        encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")
        payload: dict[str, Any] = {
            "message": message,
            "content": encoded,
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha

        return await self._put(f"/repos/{owner}/{repo}/contents/{path}", json=payload)

    async def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str | None = None,
    ) -> dict:
        """Create a pull request."""
        if not base:
            details = await self.get_repo_details(owner, repo)
            base = details.default_branch

        data = await self._post(
            f"/repos/{owner}/{repo}/pulls",
            json={"title": title, "body": body, "head": head, "base": base},
        )
        logger.info("Created PR #%d on %s/%s: %s", data["number"], owner, repo, title)
        return data

    async def get_authenticated_user(self) -> dict:
        """Get the authenticated user's profile."""
        return await self._get("/user")

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_repo(data: dict) -> Repository:
        """Parse raw API response into Repository model."""
        owner = data.get("owner", {})
        return Repository(
            owner=owner.get("login", ""),
            name=data.get("name", ""),
            full_name=data.get("full_name", ""),
            description=data.get("description"),
            language=data.get("language"),
            stars=data.get("stargazers_count", 0),
            forks=data.get("forks_count", 0),
            open_issues=data.get("open_issues_count", 0),
            topics=data.get("topics", []),
            default_branch=data.get("default_branch", "main"),
            html_url=data.get("html_url", ""),
            clone_url=data.get("clone_url", ""),
            has_license=data.get("license") is not None,
        )
