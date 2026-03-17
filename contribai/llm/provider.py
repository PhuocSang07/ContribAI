"""LLM Provider abstraction with multi-provider support.

Gemini is the primary/default provider. All providers implement
the same async interface for easy swapping.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from contribai.core.exceptions import LLMError, LLMRateLimitError

if TYPE_CHECKING:
    from contribai.core.config import LLMConfig

logger = logging.getLogger(__name__)


# ── Abstract base ──────────────────────────────────────────────────────────────


class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Single-turn completion."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Multi-turn chat completion."""

    async def close(self):  # noqa: B027
        """Clean up any resources."""


# ── Gemini (primary) ──────────────────────────────────────────────────────────


class GeminiProvider(LLMProvider):
    """Google Gemini provider - primary/default."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from google import genai

            self._client = genai.Client(api_key=config.api_key)
        except ImportError as e:
            raise LLMError("google-genai package not installed") from e

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        from google.genai import types

        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        try:
            config = types.GenerateContentConfig(
                system_instruction=system,
                temperature=temp,
                max_output_tokens=max_tok,
            )
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            return response.text or ""
        except Exception as e:
            error_msg = str(e).lower()
            if "rate" in error_msg or "quota" in error_msg or "429" in error_msg:
                raise LLMRateLimitError(f"Gemini rate limit: {e}") from e
            raise LLMError(f"Gemini error: {e}") from e

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        from google.genai import types

        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        try:
            contents = []
            for msg in messages:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

            config = types.GenerateContentConfig(
                system_instruction=system,
                temperature=temp,
                max_output_tokens=max_tok,
            )
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            return response.text or ""
        except Exception as e:
            error_msg = str(e).lower()
            if "rate" in error_msg or "quota" in error_msg or "429" in error_msg:
                raise LLMRateLimitError(f"Gemini rate limit: {e}") from e
            raise LLMError(f"Gemini chat error: {e}") from e


# ── OpenAI ─────────────────────────────────────────────────────────────────────


class OpenAIProvider(LLMProvider):
    """OpenAI provider (GPT-4o, etc.)."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from openai import AsyncOpenAI

            kwargs = {"api_key": config.api_key}
            if config.base_url:
                kwargs["base_url"] = config.base_url
            self._client = AsyncOpenAI(**kwargs)
        except ImportError as e:
            raise LLMError("openai package not installed") from e

    async def complete(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.chat(messages, **kwargs)

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        all_messages = list(messages)
        if system and not any(m["role"] == "system" for m in all_messages):
            all_messages.insert(0, {"role": "system", "content": system})

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=all_messages,
                temperature=temp,
                max_tokens=max_tok,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            error_msg = str(e).lower()
            if "rate" in error_msg or "429" in error_msg:
                raise LLMRateLimitError(f"OpenAI rate limit: {e}") from e
            raise LLMError(f"OpenAI error: {e}") from e

    async def close(self):
        await self._client.close()


# ── Anthropic ──────────────────────────────────────────────────────────────────


class AnthropicProvider(LLMProvider):
    """Anthropic provider (Claude)."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=config.api_key)
        except ImportError as e:
            raise LLMError("anthropic package not installed") from e

    async def complete(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, system=system, **kwargs)

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": max_tok,
            }
            if system:
                kwargs["system"] = system

            response = await self._client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as e:
            error_msg = str(e).lower()
            if "rate" in error_msg or "429" in error_msg:
                raise LLMRateLimitError(f"Anthropic rate limit: {e}") from e
            raise LLMError(f"Anthropic error: {e}") from e

    async def close(self):
        await self._client.close()


# ── Ollama (local) ─────────────────────────────────────────────────────────────


class OllamaProvider(LLMProvider):
    """Ollama local model provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._base_url = config.base_url or "http://localhost:11434"
        try:
            import httpx

            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=120.0)
        except ImportError as e:
            raise LLMError("httpx package not installed") from e

    async def complete(self, prompt: str, *, system: str | None = None, **kwargs) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.chat(messages, **kwargs)

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        temp = temperature if temperature is not None else self.temperature

        all_messages = list(messages)
        if system and not any(m["role"] == "system" for m in all_messages):
            all_messages.insert(0, {"role": "system", "content": system})

        try:
            payload = {
                "model": self.model,
                "messages": all_messages,
                "stream": False,
                "options": {"temperature": temp},
            }
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            raise LLMError(f"Ollama error: {e}") from e

    async def close(self):
        await self._client.aclose()


# ── Factory ────────────────────────────────────────────────────────────────────


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}


def create_llm_provider(config: LLMConfig) -> LLMProvider:
    """Create an LLM provider instance from config."""
    provider_cls = _PROVIDERS.get(config.provider)
    if not provider_cls:
        raise LLMError(
            f"Unknown LLM provider: {config.provider}. Available: {', '.join(_PROVIDERS.keys())}"
        )
    logger.info("Using LLM provider: %s (model: %s)", config.provider, config.model)
    return provider_cls(config)
