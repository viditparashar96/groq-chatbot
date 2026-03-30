"""
OpenAI provider implementation.

Supports streaming (Actor) and Structured Outputs with constrained decoding (Director)
for GPT-4.1, GPT-4.1-mini, and GPT-4.1-nano.
"""

from __future__ import annotations

import json
import os
from typing import Iterator

from providers.base import (
    LLMProvider,
    ModelInfo,
    StreamChunk,
    JsonResponse,
    UsageStats,
    ProviderError,
    TimingTracker,
)


# ── OpenAI Model Registry ───────────────────────────────────────────────────

OPENAI_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gpt-4.1",
        name="GPT-4.1",
        provider="openai",
        roles=["actor", "director"],
        streaming=True,
        supports_json_schema=True,
        supports_json_mode=True,
        supports_reasoning=False,
        supports_caching=True,
        reasoning_type="",
        reasoning_efforts=[],
        default_effort="",
        speed="~80 t/s",
        context_window=1047576,
        max_completion=32768,
        price_input=2.00,
        price_output=8.00,
        tier="Production",
        multilingual="95+ languages",
        multilingual_score="★★★★☆",
        notes="Flagship model. Best for complex tasks. Structured Outputs with constrained decoding.",
    ),
    ModelInfo(
        id="gpt-4.1-mini",
        name="GPT-4.1 Mini",
        provider="openai",
        roles=["actor", "director"],
        streaming=True,
        supports_json_schema=True,
        supports_json_mode=True,
        supports_reasoning=False,
        supports_caching=True,
        reasoning_type="",
        reasoning_efforts=[],
        default_effort="",
        speed="~110 t/s",
        context_window=1047576,
        max_completion=32768,
        price_input=0.40,
        price_output=1.60,
        tier="Production",
        multilingual="95+ languages",
        multilingual_score="★★★★☆",
        notes="Fast and cheap. Great Director model. Structured Outputs with constrained decoding.",
    ),
    ModelInfo(
        id="gpt-4.1-nano",
        name="GPT-4.1 Nano",
        provider="openai",
        roles=["director"],
        streaming=True,
        supports_json_schema=True,
        supports_json_mode=True,
        supports_reasoning=False,
        supports_caching=True,
        reasoning_type="",
        reasoning_efforts=[],
        default_effort="",
        speed="~200 t/s",
        context_window=1047576,
        max_completion=32768,
        price_input=0.10,
        price_output=0.40,
        tier="Production",
        multilingual="95+ languages",
        multilingual_score="★★★☆☆",
        notes="Cheapest, fastest. Default Director model. 100% schema compliance via constrained decoding.",
    ),
]

_OPENAI_MODEL_MAP: dict[str, ModelInfo] = {m.id: m for m in OPENAI_MODELS}


# ── Provider ─────────────────────────────────────────────────────────────────


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-4.1 family)."""

    name = "openai"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ProviderError(
                    "openai package not installed. Run: pip install openai",
                    provider="openai",
                )
            if not self.api_key:
                raise ProviderError(
                    "OPENAI_API_KEY not set.",
                    provider="openai",
                )
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError:
            return False
        return bool(self.api_key)

    def list_models(self) -> list[ModelInfo]:
        return list(OPENAI_MODELS)

    def get_model(self, model_id: str) -> ModelInfo | None:
        return _OPENAI_MODEL_MAP.get(model_id)

    # ── Streaming (Actor) ────────────────────────────────────────────────

    def stream_chat(
        self,
        model_id: str,
        messages: list[dict],
        *,
        reasoning_effort: str = "",
        show_reasoning: bool = True,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        system_prompt: str | None = None,
    ) -> Iterator[StreamChunk]:
        model_info = self._require_model(model_id)
        msgs = self._prepare_messages(messages, system_prompt)

        params = {
            "model": model_id,
            "messages": msgs,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        timer = TimingTracker()
        usage_data = None

        try:
            stream = self.client.chat.completions.create(**params)

            for chunk in stream:
                # Usage comes in the final chunk
                if chunk.usage:
                    usage_data = chunk.usage

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                content_text = getattr(delta, "content", None) or ""

                if content_text:
                    timer.mark_first_token()
                    timer.mark_first_content()
                    yield StreamChunk(text=content_text, is_content=True)

        except Exception as e:
            raise ProviderError(str(e), provider="openai", model=model_id)

        timer.mark_done()
        stats = self._extract_usage(usage_data, timer)
        yield StreamChunk(is_done=True, usage=stats)

    # ── Streaming + JSON (Unified) ──────────────────────────────────────

    def stream_json_chat(
        self,
        model_id: str,
        messages: list[dict],
        *,
        schema: dict | None = None,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        system_prompt: str | None = None,
    ) -> Iterator[StreamChunk]:
        """
        Stream a structured JSON response — TTFT + schema compliance in one call.

        OpenAI supports stream=True + response_format=json_schema together.
        Groq does NOT — this is OpenAI-exclusive.

        Yields StreamChunks with JSON tokens. The final chunk has usage stats.
        Caller should concatenate all content chunks to get the full JSON.
        """
        model_info = self._require_model(model_id)
        msgs = self._prepare_messages(messages, system_prompt)

        params = {
            "model": model_id,
            "messages": msgs,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # Structured Outputs + streaming
        if schema and model_info.supports_json_schema:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.get("title", "response"),
                    "schema": schema,
                    "strict": True,
                },
            }
        else:
            params["response_format"] = {"type": "json_object"}

        timer = TimingTracker()
        usage_data = None

        try:
            stream = self.client.chat.completions.create(**params)

            for chunk in stream:
                if chunk.usage:
                    usage_data = chunk.usage

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                content_text = getattr(delta, "content", None) or ""

                if content_text:
                    timer.mark_first_token()
                    timer.mark_first_content()
                    yield StreamChunk(text=content_text, is_content=True)

        except Exception as e:
            raise ProviderError(str(e), provider="openai", model=model_id)

        timer.mark_done()
        stats = self._extract_usage(usage_data, timer)
        yield StreamChunk(is_done=True, usage=stats)

    # ── Structured JSON (Director) ───────────────────────────────────────

    def json_chat(
        self,
        model_id: str,
        messages: list[dict],
        *,
        schema: dict | None = None,
        reasoning_effort: str = "",
        temperature: float = 0.6,
        max_tokens: int = 4096,
        system_prompt: str | None = None,
    ) -> JsonResponse:
        model_info = self._require_model(model_id)
        msgs = self._prepare_messages(messages, system_prompt)

        params = {
            "model": model_id,
            "messages": msgs,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }

        # Structured Outputs (constrained decoding) — the key advantage
        if schema and model_info.supports_json_schema:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.get("title", "response"),
                    "schema": schema,
                    "strict": True,
                },
            }
        else:
            params["response_format"] = {"type": "json_object"}

        timer = TimingTracker()
        try:
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            raise ProviderError(str(e), provider="openai", model=model_id)

        timer.mark_done()
        content = response.choices[0].message.content
        stats = self._extract_usage(getattr(response, "usage", None), timer)

        parsed = None
        schema_valid = False
        try:
            parsed = json.loads(content)
            schema_valid = True
        except (json.JSONDecodeError, TypeError):
            pass

        return JsonResponse(
            content=content,
            parsed=parsed,
            schema_valid=schema_valid,
            usage=stats,
        )

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _require_model(self, model_id: str) -> ModelInfo:
        model = self.get_model(model_id)
        if not model:
            raise ProviderError(f"Unknown OpenAI model: {model_id}", provider="openai", model=model_id)
        return model

    def _prepare_messages(
        self, messages: list[dict], system_prompt: str | None
    ) -> list[dict]:
        msgs = list(messages)
        if system_prompt:
            if not msgs or msgs[0].get("role") != "system":
                msgs.insert(0, {"role": "system", "content": system_prompt})
        return msgs

    def _extract_usage(self, usage, timer: TimingTracker) -> UsageStats:
        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0

        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            # OpenAI prompt caching info
            details = getattr(usage, "prompt_tokens_details", None)
            if details:
                cached_tokens = getattr(details, "cached_tokens", 0) or 0

        return UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            total_ms=round(timer.total_ms),
            ttft_ms=round(timer.ttft_ms),
            ttfc_ms=round(timer.ttfc_ms),
        )
