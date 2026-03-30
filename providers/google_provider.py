"""
Google Gemini provider implementation.

Supports streaming (Actor) and controlled generation with response_schema (Director)
for Gemini 2.0 Flash and Gemini 2.5 Pro.
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


# ── Google Model Registry ────────────────────────────────────────────────────

GOOGLE_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gemini-2.0-flash",
        name="Gemini 2.0 Flash",
        provider="google",
        roles=["actor", "director"],
        streaming=True,
        supports_json_schema=True,
        supports_json_mode=True,
        supports_reasoning=False,
        supports_caching=False,
        reasoning_type="",
        reasoning_efforts=[],
        default_effort="",
        speed="~200 t/s",
        context_window=1048576,
        max_completion=8192,
        price_input=0.10,
        price_output=0.40,
        tier="Production",
        multilingual="100+ languages",
        multilingual_score="★★★★☆",
        notes="Extremely fast and cheap. Good Director candidate. Schema-constrained JSON.",
    ),
    ModelInfo(
        id="gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        provider="google",
        roles=["actor"],
        streaming=True,
        supports_json_schema=True,
        supports_json_mode=True,
        supports_reasoning=True,
        supports_caching=False,
        reasoning_type="",
        reasoning_efforts=[],
        default_effort="",
        speed="~60 t/s",
        context_window=1048576,
        max_completion=8192,
        price_input=1.25,
        price_output=10.00,
        tier="Production",
        multilingual="100+ languages",
        multilingual_score="★★★★★",
        notes="Best quality Gemini. Deep thinking capability.",
    ),
]

_GOOGLE_MODEL_MAP: dict[str, ModelInfo] = {m.id: m for m in GOOGLE_MODELS}


# ── Provider ─────────────────────────────────────────────────────────────────


class GoogleProvider(LLMProvider):
    """Google Gemini API provider."""

    name = "google"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError:
                raise ProviderError(
                    "google-genai package not installed. Run: pip install google-genai",
                    provider="google",
                )
            if not self.api_key:
                raise ProviderError(
                    "GOOGLE_API_KEY not set.",
                    provider="google",
                )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        try:
            from google import genai  # noqa: F401
        except ImportError:
            return False
        return bool(self.api_key)

    def list_models(self) -> list[ModelInfo]:
        return list(GOOGLE_MODELS)

    def get_model(self, model_id: str) -> ModelInfo | None:
        return _GOOGLE_MODEL_MAP.get(model_id)

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
        from google.genai import types

        self._require_model(model_id)
        system_text, contents = self._convert_messages(messages, system_prompt)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_text,
        )

        timer = TimingTracker()
        try:
            response_stream = self.client.models.generate_content_stream(
                model=model_id,
                contents=contents,
                config=config,
            )

            for chunk in response_stream:
                if chunk.text:
                    timer.mark_first_token()
                    timer.mark_first_content()
                    yield StreamChunk(text=chunk.text, is_content=True)

        except Exception as e:
            raise ProviderError(str(e), provider="google", model=model_id)

        timer.mark_done()
        # Gemini streaming doesn't easily expose token counts per-chunk
        stats = UsageStats(
            total_ms=round(timer.total_ms),
            ttft_ms=round(timer.ttft_ms),
            ttfc_ms=round(timer.ttfc_ms),
        )
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
        from google.genai import types

        self._require_model(model_id)
        system_text, contents = self._convert_messages(messages, system_prompt)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_text,
            response_mime_type="application/json",
        )

        # Add schema if provided
        if schema:
            config.response_schema = schema

        timer = TimingTracker()
        try:
            response = self.client.models.generate_content(
                model=model_id,
                contents=contents,
                config=config,
            )
        except Exception as e:
            raise ProviderError(str(e), provider="google", model=model_id)

        timer.mark_done()
        content = response.text or ""

        # Extract usage
        prompt_tokens = 0
        completion_tokens = 0
        if response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

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
            usage=UsageStats(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_ms=round(timer.total_ms),
            ),
        )

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _require_model(self, model_id: str) -> ModelInfo:
        model = self.get_model(model_id)
        if not model:
            raise ProviderError(f"Unknown Google model: {model_id}", provider="google", model=model_id)
        return model

    def _convert_messages(
        self, messages: list[dict], system_prompt: str | None
    ) -> tuple[str | None, list]:
        """Convert OpenAI-style messages to Gemini format."""
        from google.genai import types

        system_parts = []
        contents = []

        if system_prompt:
            system_parts.append(system_prompt)

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_parts.append(content)
            elif role == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=content)],
                ))
            elif role == "assistant":
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(text=content)],
                ))

        system_text = "\n\n".join(system_parts) if system_parts else None
        return system_text, contents
