"""Context compressor for token-efficient LLM interactions.

Inspired by SWE-agent's HistoryProcessor — compresses analysis context
to stay within token limits without losing critical information.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Rough estimate: 1 token ≈ 4 chars for English code
CHARS_PER_TOKEN = 4


class ContextCompressor:
    """Compresses file content and analysis context to fit within token budgets.

    Strategies:
    1. Truncate large files keeping head + tail
    2. Strip comments and blank lines from non-critical files
    3. Extract key patterns (imports, class/function signatures)
    """

    def __init__(self, max_context_tokens: int = 30_000):
        self._max_tokens = max_context_tokens
        self._max_chars = max_context_tokens * CHARS_PER_TOKEN

    def compress_files(
        self,
        files: dict[str, str],
        *,
        max_per_file_tokens: int = 3_000,
    ) -> dict[str, str]:
        """Compress a dict of {path: content} to fit within budget.

        Args:
            files: Mapping of file paths to their full content.
            max_per_file_tokens: Max tokens per individual file.

        Returns:
            Compressed mapping with same keys, shorter values.
        """
        max_per_file_chars = max_per_file_tokens * CHARS_PER_TOKEN
        total_budget = self._max_chars
        compressed: dict[str, str] = {}
        total_chars = 0

        # Sort files by size — process smallest first to keep more files
        sorted_files = sorted(files.items(), key=lambda x: len(x[1]))

        for path, content in sorted_files:
            remaining = total_budget - total_chars
            if remaining <= 0:
                logger.debug("Budget exhausted, skipping %s", path)
                break

            per_file_limit = min(max_per_file_chars, remaining)
            if len(content) <= per_file_limit:
                compressed[path] = content
            else:
                compressed[path] = self._truncate_middle(content, per_file_limit)

            total_chars += len(compressed[path])

        skipped = len(files) - len(compressed)
        if skipped > 0:
            logger.info(
                "Compressed %d files, skipped %d (budget: %d tokens)",
                len(compressed),
                skipped,
                self._max_tokens,
            )
        return compressed

    def compress_text(self, text: str, max_tokens: int | None = None) -> str:
        """Compress arbitrary text to fit within token budget.

        Args:
            text: Text to compress.
            max_tokens: Token limit (defaults to instance max).

        Returns:
            Compressed text.
        """
        limit_chars = (max_tokens or self._max_tokens) * CHARS_PER_TOKEN
        if len(text) <= limit_chars:
            return text
        return self._truncate_middle(text, limit_chars)

    def extract_signatures(self, content: str, language: str = "python") -> str:
        """Extract key structural elements from source code.

        Keeps imports, class definitions, and function signatures while
        stripping implementation bodies. Useful for overview context.

        Args:
            content: Full source code.
            language: Programming language hint.

        Returns:
            Skeleton of the source file.
        """
        if language in ("python", "py"):
            return self._extract_python_signatures(content)
        # Fallback: keep first 50 and last 20 lines
        lines = content.splitlines()
        if len(lines) <= 70:
            return content
        head = lines[:50]
        tail = lines[-20:]
        return "\n".join([*head, f"... ({len(lines) - 70} lines omitted) ...", *tail])

    @staticmethod
    def summarize_findings_compact(findings: list) -> str:
        """Ultra-compact finding summary for prompt injection.

        Args:
            findings: List of Finding objects.

        Returns:
            One-line-per-finding compact summary.
        """
        if not findings:
            return "No issues."
        parts: list[str] = []
        for f in findings[:10]:  # cap at 10 to save tokens
            sev = f.severity.value if hasattr(f.severity, "value") else str(f.severity)
            path = getattr(f, "file_path", "") or ""
            parts.append(f"[{sev}] {f.title} ({path})")
        suffix = f" +{len(findings) - 10} more" if len(findings) > 10 else ""
        return "\n".join(parts) + suffix

    # ── Internal methods ───────────────────────────────────────────────────

    @staticmethod
    def _truncate_middle(text: str, max_chars: int) -> str:
        """Keep first and last portions, replace middle with marker."""
        if len(text) <= max_chars:
            return text
        # 60% head, 40% tail
        head_size = int(max_chars * 0.6)
        tail_size = max_chars - head_size - 60  # 60 chars for marker
        if tail_size < 0:
            return text[:max_chars]
        omitted = len(text) - head_size - tail_size
        marker = f"\n\n... ({omitted} chars / ~{omitted // CHARS_PER_TOKEN} tokens omitted) ...\n\n"
        return text[:head_size] + marker + text[-tail_size:]

    @staticmethod
    def _extract_python_signatures(content: str) -> str:
        """Extract imports, class defs, and function signatures from Python."""
        lines = content.splitlines()
        result: list[str] = []
        in_docstring = False

        for line in lines:
            stripped = line.strip()

            # Track triple-quote docstrings
            if '"""' in stripped or "'''" in stripped:
                count = stripped.count('"""') + stripped.count("'''")
                if count % 2 == 1:
                    in_docstring = not in_docstring
                if not in_docstring:
                    continue

            if in_docstring:
                continue

            # Keep imports, definitions, decorators, constants
            is_import = stripped.startswith(("import ", "from "))
            is_def = re.match(r"^(class |def |async def )", stripped)
            is_decorator = stripped.startswith("@")
            is_constant = (
                re.match(r"^[A-Z_][A-Z_0-9]+ =", stripped)
                and not line.startswith((" ", "\t"))
            )
            if is_import or is_def or is_decorator or is_constant:
                result.append(line)

        return "\n".join(result)
