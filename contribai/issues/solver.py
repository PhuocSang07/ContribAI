"""Issue-driven contribution engine.

Reads open GitHub issues, classifies them, and generates
targeted contributions that solve specific issues.
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum

from contribai.core.models import (
    ContributionType,
    Finding,
    Issue,
    RepoContext,
    Repository,
    Severity,
)

logger = logging.getLogger(__name__)


class IssueCategory(StrEnum):
    """Classification categories for GitHub issues."""

    BUG = "bug"
    FEATURE = "feature"
    DOCS = "docs"
    SECURITY = "security"
    PERFORMANCE = "performance"
    UI_UX = "ui_ux"
    GOOD_FIRST_ISSUE = "good_first_issue"
    UNSOLVABLE = "unsolvable"


# Labels that map to categories
LABEL_MAP: dict[str, IssueCategory] = {
    "bug": IssueCategory.BUG,
    "fix": IssueCategory.BUG,
    "defect": IssueCategory.BUG,
    "feature": IssueCategory.FEATURE,
    "enhancement": IssueCategory.FEATURE,
    "feature-request": IssueCategory.FEATURE,
    "documentation": IssueCategory.DOCS,
    "docs": IssueCategory.DOCS,
    "security": IssueCategory.SECURITY,
    "vulnerability": IssueCategory.SECURITY,
    "performance": IssueCategory.PERFORMANCE,
    "ui": IssueCategory.UI_UX,
    "ux": IssueCategory.UI_UX,
    "accessibility": IssueCategory.UI_UX,
    "good first issue": IssueCategory.GOOD_FIRST_ISSUE,
    "good-first-issue": IssueCategory.GOOD_FIRST_ISSUE,
    "beginner": IssueCategory.GOOD_FIRST_ISSUE,
    "help wanted": IssueCategory.GOOD_FIRST_ISSUE,
}

CATEGORY_TO_CONTRIB = {
    IssueCategory.BUG: ContributionType.CODE_QUALITY,
    IssueCategory.FEATURE: ContributionType.FEATURE_ADD,
    IssueCategory.DOCS: ContributionType.DOCS_IMPROVE,
    IssueCategory.SECURITY: ContributionType.SECURITY_FIX,
    IssueCategory.PERFORMANCE: ContributionType.PERFORMANCE_OPT,
    IssueCategory.UI_UX: ContributionType.UI_UX_FIX,
    IssueCategory.GOOD_FIRST_ISSUE: ContributionType.CODE_QUALITY,
}


class IssueSolver:
    """Analyzes and solves GitHub issues using LLM."""

    def __init__(self, llm, github):
        self._llm = llm
        self._github = github

    def classify_issue(self, issue: Issue) -> IssueCategory:
        """Classify an issue based on labels and title keywords.

        Args:
            issue: GitHub issue to classify.

        Returns:
            IssueCategory enum value.
        """
        # Check labels first (most reliable)
        for label in issue.labels:
            label_lower = label.lower().strip()
            if label_lower in LABEL_MAP:
                return LABEL_MAP[label_lower]

        # Fall back to keyword matching on title
        title_lower = issue.title.lower()
        keyword_map = {
            IssueCategory.BUG: ["bug", "fix", "error", "crash", "broken", "fail", "issue"],
            IssueCategory.FEATURE: ["add", "feature", "implement", "support", "new"],
            IssueCategory.DOCS: ["doc", "readme", "typo", "documentation", "example"],
            IssueCategory.SECURITY: ["security", "vulnerability", "cve", "xss", "injection"],
            IssueCategory.PERFORMANCE: ["slow", "performance", "optimize", "speed", "memory"],
            IssueCategory.UI_UX: ["ui", "ux", "responsive", "accessibility", "design"],
        }

        for category, keywords in keyword_map.items():
            if any(kw in title_lower for kw in keywords):
                return category

        return IssueCategory.BUG  # Default to bug

    def _estimate_complexity(self, issue: Issue) -> int:
        """Estimate issue complexity (1-5). Lower is easier.

        Args:
            issue: GitHub issue to assess.

        Returns:
            Complexity score from 1 (trivial) to 5 (very complex).
        """
        score = 2  # baseline

        # Good first issues are simple
        if any("first" in label.lower() or "beginner" in label.lower() for label in issue.labels):
            return 1

        # Body length hints at complexity
        body_len = len(issue.body or "")
        if body_len > 2000:
            score += 1
        if body_len > 5000:
            score += 1

        # Multiple file references = complex
        if issue.body:
            file_refs = re.findall(r"[\w/]+\.\w{1,4}", issue.body)
            if len(file_refs) > 3:
                score += 1

        return min(score, 5)

    def filter_solvable(self, issues: list[Issue], max_complexity: int = 3) -> list[Issue]:
        """Filter issues to only those the agent can likely solve.

        Args:
            issues: List of GitHub issues.
            max_complexity: Maximum complexity score to attempt.

        Returns:
            Filtered list of solvable issues.
        """
        solvable = []
        for issue in issues:
            category = self.classify_issue(issue)
            if category == IssueCategory.UNSOLVABLE:
                continue

            complexity = self._estimate_complexity(issue)
            if complexity > max_complexity:
                continue

            solvable.append(issue)

        return solvable

    async def solve_issue(
        self,
        issue: Issue,
        repo: Repository,
        context: RepoContext,
    ) -> Finding | None:
        """Convert a GitHub issue into a Finding for the generator.

        Uses LLM to understand the issue and identify the relevant
        files and changes needed.

        Args:
            issue: GitHub issue to solve.
            repo: Repository containing the issue.
            context: Repository context for LLM prompting.

        Returns:
            Finding object that can be fed to the ContributionGenerator.
        """
        category = self.classify_issue(issue)
        contrib_type = CATEGORY_TO_CONTRIB.get(category, ContributionType.CODE_QUALITY)

        # Build context prompt
        file_tree_str = "\n".join(f"  {f.path}" for f in context.file_tree[:50] if f.type == "blob")

        relevant_code = ""
        for path, content in list(context.relevant_files.items())[:3]:
            relevant_code += f"\n### {path}\n```\n{content[:2000]}\n```\n"

        prompt = f"""Analyze this GitHub issue and determine:
1. Which file(s) need changes
2. What changes are needed
3. The severity of the issue

## Repository: {repo.full_name} ({repo.language})

## Issue #{issue.number}: {issue.title}
{issue.body or "No description provided."}

## Labels: {", ".join(issue.labels) if issue.labels else "none"}

## File Tree:
{file_tree_str}

{relevant_code}

Respond in this exact format:
FILE_PATH: <main file to change>
SEVERITY: <low|medium|high|critical>
TITLE: <short descriptive title>
DESCRIPTION: <what needs to be changed and why>
SUGGESTION: <specific implementation suggestion>
"""

        try:
            response = await self._llm.complete(
                prompt,
                system="You are a senior developer analyzing GitHub issues. "
                "Identify the root cause and suggest a specific fix.",
            )

            # Parse structured response
            lines = response.strip().split("\n")
            parsed = {}
            for line in lines:
                if ":" in line:
                    key, _, value = line.partition(":")
                    parsed[key.strip().upper()] = value.strip()

            file_path = parsed.get("FILE_PATH", "unknown")
            severity_str = parsed.get("SEVERITY", "medium").lower()
            severity_map = {
                "low": Severity.LOW,
                "medium": Severity.MEDIUM,
                "high": Severity.HIGH,
                "critical": Severity.CRITICAL,
            }

            return Finding(
                id=f"issue-{issue.number}",
                type=contrib_type,
                severity=severity_map.get(severity_str, Severity.MEDIUM),
                title=parsed.get("TITLE", issue.title),
                description=parsed.get("DESCRIPTION", issue.body or issue.title),
                file_path=file_path,
                suggestion=parsed.get("SUGGESTION"),
                confidence=0.85,
            )

        except Exception as e:
            logger.error("Failed to analyze issue #%d: %s", issue.number, e)
            return None
