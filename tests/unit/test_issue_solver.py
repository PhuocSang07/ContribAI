"""Tests for the issue solver engine."""

import pytest
from unittest.mock import AsyncMock

from contribai.core.models import Issue, Repository, RepoContext, FileNode
from contribai.issues.solver import IssueSolver, IssueCategory


@pytest.fixture
def solver(mock_llm, mock_github):
    return IssueSolver(llm=mock_llm, github=mock_github)


@pytest.fixture
def bug_issue():
    return Issue(
        number=1, title="App crashes on login", body="TypeError when user logs in", labels=["bug"]
    )


@pytest.fixture
def feature_issue():
    return Issue(
        number=2,
        title="Add dark mode support",
        body="Please add dark theme toggle",
        labels=["enhancement"],
    )


@pytest.fixture
def docs_issue():
    return Issue(
        number=3, title="Fix typo in README", body="Line 42 has a typo", labels=["documentation"]
    )


@pytest.fixture
def unlabeled_issue():
    return Issue(number=4, title="Fix the broken login page", body="Login doesn't work")


class TestClassifyIssue:
    def test_bug_by_label(self, solver, bug_issue):
        assert solver.classify_issue(bug_issue) == IssueCategory.BUG

    def test_feature_by_label(self, solver, feature_issue):
        assert solver.classify_issue(feature_issue) == IssueCategory.FEATURE

    def test_docs_by_label(self, solver, docs_issue):
        assert solver.classify_issue(docs_issue) == IssueCategory.DOCS

    def test_security_by_label(self, solver):
        issue = Issue(number=5, title="Something", labels=["security"])
        assert solver.classify_issue(issue) == IssueCategory.SECURITY

    def test_good_first_issue(self, solver):
        issue = Issue(number=6, title="Something", labels=["good first issue"])
        assert solver.classify_issue(issue) == IssueCategory.GOOD_FIRST_ISSUE

    def test_keyword_fallback_bug(self, solver, unlabeled_issue):
        assert solver.classify_issue(unlabeled_issue) == IssueCategory.BUG

    def test_keyword_fallback_feature(self, solver):
        issue = Issue(number=7, title="Add support for webhooks", labels=[])
        assert solver.classify_issue(issue) == IssueCategory.FEATURE

    def test_keyword_fallback_docs(self, solver):
        issue = Issue(number=8, title="Update documentation for API", labels=[])
        assert solver.classify_issue(issue) == IssueCategory.DOCS


class TestEstimateComplexity:
    def test_good_first_issue_simple(self, solver):
        issue = Issue(number=1, title="Easy fix", labels=["good first issue"])
        assert solver._estimate_complexity(issue) == 1

    def test_short_issue_moderate(self, solver, bug_issue):
        assert solver._estimate_complexity(bug_issue) <= 3

    def test_long_body_complex(self, solver):
        issue = Issue(number=2, title="Complex bug", body="x" * 3000, labels=[])
        assert solver._estimate_complexity(issue) >= 3

    def test_many_file_references(self, solver):
        issue = Issue(
            number=3,
            title="Fix stuff",
            body="Change src/a.py src/b.py src/c.py src/d.py src/e.py",
            labels=[],
        )
        assert solver._estimate_complexity(issue) >= 3


class TestFilterSolvable:
    def test_filters_basic(self, solver, bug_issue, docs_issue):
        issues = [bug_issue, docs_issue]
        solvable = solver.filter_solvable(issues, max_complexity=3)
        assert len(solvable) == 2

    def test_filters_complex(self, solver):
        complex_issue = Issue(
            number=99,
            title="Redesign everything",
            body="x" * 10000 + " file1.py file2.py file3.py file4.py",
            labels=[],
        )
        solvable = solver.filter_solvable([complex_issue], max_complexity=2)
        assert len(solvable) == 0


class TestSolveIssue:
    @pytest.mark.asyncio
    async def test_solve_returns_finding(self, solver, bug_issue, sample_repo):
        solver._llm.complete = AsyncMock(
            return_value="""FILE_PATH: src/auth.py
SEVERITY: high
TITLE: Fix TypeError in login handler
DESCRIPTION: The login handler raises TypeError when user object is None
SUGGESTION: Add null check before accessing user properties"""
        )

        context = RepoContext(
            repo=sample_repo,
            file_tree=[FileNode(path="src/auth.py", type="blob", size=500, sha="x")],
        )
        finding = await solver.solve_issue(bug_issue, sample_repo, context)
        assert finding is not None
        assert finding.file_path == "src/auth.py"
        assert "login" in finding.title.lower() or "TypeError" in finding.title

    @pytest.mark.asyncio
    async def test_solve_handles_llm_failure(self, solver, bug_issue, sample_repo):
        solver._llm.complete = AsyncMock(side_effect=Exception("LLM down"))
        context = RepoContext(repo=sample_repo)
        result = await solver.solve_issue(bug_issue, sample_repo, context)
        assert result is None
