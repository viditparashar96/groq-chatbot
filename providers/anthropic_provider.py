"""
Anthropic provider implementation.

Supports streaming (Actor) and tool-use based structured JSON (Director)
for Claude Sonnet 4.6 and Claude Haiku 4.5.
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


# ── Anthropic Model Registry ────────────────────────────────────────────────

ANTHROPIC_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="claude-sonnet-4-6",
        name="Claude Sonnet 4.6",
        provider="anthropic",
        roles=["actor"],
        streaming=True,
        supports_json_schema=False,
        supports_json_mode=False,
        supports_reasoning=False,
        supports_caching=True,
        reasoning_type="",
        reasoning_efforts=[],
        default_effort="",
        speed="~80 t/s",
        context_window=200000,
        max_completion=8192,
        price_input=3.00,
        price_output=15.00,
        tier="Production",
        multilingual="95+ languages",
        multilingual_score="★★★★★",
        notes="Best overall quality. Extended thinking available. Tool use for structured output.",
    ),
    ModelInfo(
        id="claude-haiku-4-5",
        name="Claude Haiku 4.5",
        provider="anthropic",
        roles=["actor", "director"],
        streaming=True,
        supports_json_schema=False,
        supports_json_mode=False,
        supports_reasoning=False,
        supports_caching=True,
        reasoning_type="",
        reasoning_efforts=[],
        default_effort="",
        speed="~150 t/s",
        context_window=200000,
        max_completion=8192,
        price_input=0.80,
        price_output=4.00,
        tier="Production",
        multilingual="95+ languages",
        multilingual_score="★★★★☆",
        notes="Fast and capable. Good Director candidate via tool use.",
    ),
]

_ANTHROPIC_MODEL_MAP: dict[str, ModelInfo] = {m.id: m for m in ANTHROPIC_MODELS}


# ── Provider ─────────────────────────────────────────────────────────────────


class AnthropicProvider(LLMProvider):
    """Anthropic API provider (Claude family)."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ProviderError(
                    "anthropic package not installed. Run: pip install anthropic",
                    provider="anthropic",
                )
            if not self.api_key:
                raise ProviderError(
                    "ANTHROPIC_API_KEY not set.",
                    provider="anthropic",
                )
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        try:
            from anthropic import Anthropic  # noqa: F401
        except ImportError:
            return False
        return bool(self.api_key)

    def list_models(self) -> list[ModelInfo]:
        return list(ANTHROPIC_MODELS)

    def get_model(self, model_id: str) -> ModelInfo | None:
        return _ANTHROPIC_MODEL_MAP.get(model_id)

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
        self._require_model(model_id)

        # Anthropic uses a separate system parameter, not a system message
        system_text, api_messages = self._split_system(messages, system_prompt)

        timer = TimingTracker()
        input_tokens = 0
        output_tokens = 0

        try:
            with self.client.messages.stream(
                model=model_id,
                messages=api_messages,
                system=system_text or "",
                max_tokens=max_tokens,
                temperature=temperature,
            ) as stream:
                for text in stream.text_stream:
                    timer.mark_first_token()
                    timer.mark_first_content()
                    yield StreamChunk(text=text, is_content=True)

                # Get final message for usage
                response = stream.get_final_message()
                if response.usage:
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens

        except Exception as e:
            raise ProviderError(str(e), provider="anthropic", model=model_id)

        timer.mark_done()
        stats = UsageStats(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_ms=round(timer.total_ms),
            ttft_ms=round(timer.ttft_ms),
            ttfc_ms=round(timer.ttfc_ms),
        )
        yield StreamChunk(is_done=True, usage=stats)

    # ── Structured JSON (Director) — via tool use ────────────────────────

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
        self._require_model(model_id)

        system_text, api_messages = self._split_system(messages, system_prompt)

        # Use tool_use to get structured output
        tool_name = schema.get("title", "analyze") if schema else "analyze"
        tools = [{
            "name": tool_name,
            "description": "Produce structured JSON analysis of the conversation.",
            "input_schema": schema or {"type": "object"},
        }]

        # Force tool use
        tool_choice = {"type": "tool", "name": tool_name}

        timer = TimingTracker()
        try:
            response = self.client.messages.create(
                model=model_id,
                messages=api_messages,
                system=system_text or "",
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
            )
        except Exception as e:
            raise ProviderError(str(e), provider="anthropic", model=model_id)

        timer.mark_done()

        # Extract tool use result
        parsed = None
        content = ""
        schema_valid = False
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                parsed = block.input
                content = json.dumps(parsed, ensure_ascii=False)
                schema_valid = True
                break

        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0

        return JsonResponse(
            content=content,
            parsed=parsed,
            schema_valid=schema_valid,
            usage=UsageStats(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_ms=round(timer.total_ms),
            ),
        )

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _require_model(self, model_id: str) -> ModelInfo:
        model = self.get_model(model_id)
        if not model:
            raise ProviderError(f"Unknown Anthropic model: {model_id}", provider="anthropic", model=model_id)
        return model

    def _split_system(
        self, messages: list[dict], system_prompt: str | None
    ) -> tuple[str | None, list[dict]]:
        """
        Anthropic API takes system as a separate param, not a message.
        Extract system messages and return (system_text, remaining_messages).
        """
        system_parts = []
        api_messages = []

        if system_prompt:
            system_parts.append(system_prompt)

        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                api_messages.append(msg)

        system_text = "\n\n".join(system_parts) if system_parts else None
        return system_text, api_messages
