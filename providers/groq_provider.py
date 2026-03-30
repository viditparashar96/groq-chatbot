"""
Groq provider implementation.

Supports streaming (Actor) and structured JSON (Director) for all Groq-hosted
models: Qwen3-32B, GPT-OSS 20B, GPT-OSS 120B.
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


# ── Groq Model Registry ─────────────────────────────────────────────────────

GROQ_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="openai/gpt-oss-20b",
        name="GPT-OSS 20B",
        provider="groq",
        roles=["actor"],
        streaming=True,
        supports_json_schema=True,
        supports_json_mode=True,
        supports_reasoning=True,
        supports_caching=True,
        reasoning_type="include_reasoning",
        reasoning_efforts=["low", "medium", "high"],
        default_effort="medium",
        speed="~1000 t/s",
        context_window=131072,
        max_completion=65536,
        price_input=0.075,
        price_output=0.30,
        tier="Production",
        multilingual="14+ languages (primarily English-optimized, STEM focus)",
        multilingual_score="★★☆☆☆",
        notes="Fastest TTFT. MoE with 3.6B active params. Best for English speed-critical.",
    ),
    ModelInfo(
        id="openai/gpt-oss-120b",
        name="GPT-OSS 120B",
        provider="groq",
        roles=["actor"],
        streaming=True,
        supports_json_schema=True,
        supports_json_mode=True,
        supports_reasoning=True,
        supports_caching=True,
        reasoning_type="include_reasoning",
        reasoning_efforts=["low", "medium", "high"],
        default_effort="medium",
        speed="~500 t/s",
        context_window=131072,
        max_completion=65536,
        price_input=0.15,
        price_output=0.60,
        tier="Production",
        multilingual="81+ languages (English-primary, competitive on 14-lang benchmarks)",
        multilingual_score="★★★☆☆",
        notes="Best reasoning quality. MoE with 5.1B active params. Near o4-mini parity.",
    ),
]

# Quick lookup
_GROQ_MODEL_MAP: dict[str, ModelInfo] = {m.id: m for m in GROQ_MODELS}


# ── Provider ─────────────────────────────────────────────────────────────────


class GroqProvider(LLMProvider):
    """Groq LPU inference provider."""

    name = "groq"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from groq import Groq
            except ImportError:
                raise ProviderError(
                    "groq package not installed. Run: pip install groq",
                    provider="groq",
                )
            if not self.api_key:
                raise ProviderError(
                    "GROQ_API_KEY not set. Get one at https://console.groq.com/keys",
                    provider="groq",
                )
            self._client = Groq(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        try:
            from groq import Groq  # noqa: F401
        except ImportError:
            return False
        return bool(self.api_key)

    def list_models(self) -> list[ModelInfo]:
        return list(GROQ_MODELS)

    def get_model(self, model_id: str) -> ModelInfo | None:
        return _GROQ_MODEL_MAP.get(model_id)

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
        params = self._build_params(
            model_info, msgs,
            stream=True,
            reasoning_effort=reasoning_effort,
            show_reasoning=show_reasoning,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        timer = TimingTracker()
        content_tokens = 0
        reasoning_tokens = 0
        usage = None

        try:
            stream = self.client.chat.completions.create(**params)

            for chunk in stream:
                delta = chunk.choices[0].delta

                # Reasoning tokens (GPT-OSS delta.reasoning field)
                reasoning_text = getattr(delta, "reasoning", None)
                if reasoning_text:
                    timer.mark_first_token()
                    reasoning_tokens += 1
                    yield StreamChunk(
                        text=reasoning_text,
                        is_reasoning=True,
                    )

                # Content tokens
                content_text = getattr(delta, "content", None) or ""
                if content_text:
                    timer.mark_first_token()
                    timer.mark_first_content()
                    content_tokens += 1
                    yield StreamChunk(
                        text=content_text,
                        is_content=True,
                    )

                # Usage from final chunk (Groq-specific)
                x_groq = getattr(chunk, "x_groq", None)
                if x_groq:
                    usage = getattr(x_groq, "usage", None)

        except Exception as e:
            raise ProviderError(str(e), provider="groq", model=model_id)

        timer.mark_done()
        stats = self._extract_usage(usage, timer)
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

        # If model doesn't support strict schema, inject schema as text
        if schema and not model_info.supports_json_schema:
            msgs = self._inject_schema_as_text(msgs, schema)

        params = self._build_params(
            model_info, msgs,
            stream=False,
            reasoning_effort=reasoning_effort,
            show_reasoning=False,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=schema if model_info.supports_json_schema else None,
            json_mode=True,
        )

        timer = TimingTracker()
        try:
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            raise ProviderError(str(e), provider="groq", model=model_id)

        timer.mark_done()
        content = response.choices[0].message.content
        usage_obj = getattr(response, "usage", None)
        stats = self._extract_usage(usage_obj, timer)

        # Parse JSON
        parsed = None
        schema_valid = False
        try:
            parsed = json.loads(content)
            schema_valid = True  # basic validity; full schema validation deferred
        except (json.JSONDecodeError, TypeError):
            pass

        return JsonResponse(
            content=content,
            parsed=parsed,
            schema_valid=schema_valid,
            usage=stats,
        )

    # ── Cache Warmup ─────────────────────────────────────────────────────

    def warm_cache(self, model_id: str, system_prompt: str) -> dict:
        """Prime the system prompt cache with a lightweight request."""
        model_info = self._require_model(model_id)
        if not model_info.supports_caching:
            return {"status": "skipped", "reason": f"{model_info.name} does not support caching"}

        timer = TimingTracker()
        try:
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "hi"},
                ],
                max_completion_tokens=1,
                temperature=0.0,
            )
            timer.mark_done()
            return {"status": "ok", "elapsed_ms": round(timer.total_ms)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _require_model(self, model_id: str) -> ModelInfo:
        model = self.get_model(model_id)
        if not model:
            raise ProviderError(f"Unknown Groq model: {model_id}", provider="groq", model=model_id)
        return model

    def _prepare_messages(
        self, messages: list[dict], system_prompt: str | None
    ) -> list[dict]:
        msgs = list(messages)
        if system_prompt:
            if not msgs or msgs[0].get("role") != "system":
                msgs.insert(0, {"role": "system", "content": system_prompt})
        return msgs

    def _inject_schema_as_text(self, messages: list[dict], schema: dict) -> list[dict]:
        """For models without strict schema support, inject schema as text guidance."""
        msgs = list(messages)
        instruction = (
            f"You must respond with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```"
        )
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {
                "role": "system",
                "content": msgs[0]["content"] + "\n\n" + instruction,
            }
        else:
            msgs.insert(0, {"role": "system", "content": instruction})
        return msgs

    def _build_params(
        self,
        model_info: ModelInfo,
        messages: list[dict],
        *,
        stream: bool,
        reasoning_effort: str,
        show_reasoning: bool,
        temperature: float,
        max_tokens: int,
        json_schema: dict | None = None,
        json_mode: bool = False,
    ) -> dict:
        params = {
            "model": model_info.id,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "top_p": 0.95,
            "stream": stream,
        }

        # Reasoning params (model-specific)
        effort = reasoning_effort or model_info.default_effort
        if model_info.reasoning_type == "reasoning_format":
            params["reasoning_format"] = "parsed" if show_reasoning else "hidden"
            if effort != "default":
                params["reasoning_effort"] = effort
        elif model_info.reasoning_type == "include_reasoning":
            params["include_reasoning"] = show_reasoning
            params["reasoning_effort"] = effort

        # JSON output (non-streaming only)
        if not stream and (json_schema or json_mode):
            if json_schema and model_info.supports_json_schema:
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": json_schema.get("title", "response"),
                        "schema": json_schema,
                        "strict": True,
                    },
                }
            else:
                params["response_format"] = {"type": "json_object"}

        return params

    def _extract_usage(self, usage, timer: TimingTracker) -> UsageStats:
        """Extract usage stats from Groq API response."""
        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0

        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
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
