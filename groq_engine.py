"""
Groq Engine — Shared inference engine for CLI and UI.

Now delegates to providers.groq_provider.GroqProvider internally.
Public interface is unchanged: GroqEngine, MODELS, estimate_tokens.
"""

import os
import json

from providers.base import estimate_tokens, ProviderError
from providers.groq_provider import GroqProvider, GROQ_MODELS, _GROQ_MODEL_MAP


# ── Backward-compatible MODELS dict ─────────────────────────────────────────
# app.py accesses MODELS[model_id]["name"], MODELS[model_id]["speed"], etc.
# We build this from the new ModelInfo objects.

def _model_info_to_dict(m) -> dict:
    """Convert ModelInfo to the legacy dict format."""
    return {
        "name": m.name,
        "speed": m.speed,
        "tier": m.tier,
        "reasoning_type": m.reasoning_type,
        "reasoning_efforts": list(m.reasoning_efforts),
        "default_effort": m.default_effort,
        "context": f"{m.context_window:,}",
        "max_completion": m.max_completion,
        "multilingual": m.multilingual,
        "multilingual_score": m.multilingual_score,
        "price_in": f"${m.price_input}",
        "price_out": f"${m.price_output}",
        "streaming": m.streaming,
        "notes": m.notes,
        "supports_strict_schema": m.supports_json_schema,
        "supports_caching": m.supports_caching,
    }

MODELS = {m.id: _model_info_to_dict(m) for m in GROQ_MODELS}


# ── GroqEngine (delegates to GroqProvider) ───────────────────────────────────

class GroqEngine:
    """
    Core inference engine for Groq models — UI and CLI agnostic.

    Wraps GroqProvider with conversation history management and
    cumulative stats tracking. Public interface is unchanged.
    """

    def __init__(self, model_id: str, api_key: str = None,
                 reasoning_effort: str = None, show_reasoning: bool = True,
                 system_prompt: str = None,
                 json_schema: dict = None, json_mode: bool = False,
                 max_completion_tokens: int = 4096):

        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set")

        # Create the provider
        self.provider = GroqProvider(api_key=self.api_key)
        self.model_id = model_id
        self.model_info = MODELS[model_id]
        self._model = _GROQ_MODEL_MAP[model_id]
        self.show_reasoning = show_reasoning
        self.messages = []

        # Cumulative stats
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.total_prompt_tokens = 0
        self.cache_hits = 0
        self.total_requests = 0

        # Output config
        self.max_completion_tokens = max_completion_tokens
        self.json_schema = json_schema
        self.json_mode = json_mode
        self.output_mode = "json" if (json_schema or json_mode) else "stream"

        # System prompt
        self.system_prompt = system_prompt
        self.system_prompt_tokens = estimate_tokens(system_prompt) if system_prompt else 0
        if system_prompt:
            self.messages.insert(0, {"role": "system", "content": system_prompt})

        # Reasoning effort
        if reasoning_effort and reasoning_effort in self.model_info["reasoning_efforts"]:
            self.reasoning_effort = reasoning_effort
        else:
            self.reasoning_effort = self.model_info["default_effort"]

    def _update_stats(self, usage) -> dict:
        """Update cumulative stats from a UsageStats object."""
        stats = {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}
        if not usage:
            return stats

        stats["prompt_tokens"] = usage.prompt_tokens
        stats["completion_tokens"] = usage.completion_tokens
        stats["cached_tokens"] = usage.cached_tokens

        self.total_requests += 1
        self.total_prompt_tokens += stats["prompt_tokens"]
        self.total_cached_tokens += stats["cached_tokens"]
        self.total_input_tokens += stats["prompt_tokens"]
        self.total_output_tokens += stats["completion_tokens"]
        if stats["cached_tokens"] > 0:
            self.cache_hits += 1

        return stats

    def get_cache_stats(self) -> dict:
        """Return current cache statistics."""
        cache_rate = (self.total_cached_tokens * 100 // max(1, self.total_prompt_tokens)) if self.total_prompt_tokens else 0
        price_per_m = self._model.price_input
        savings = (self.total_cached_tokens / 1_000_000) * price_per_m * 0.5
        return {
            "model": self.model_id,
            "supports_caching": self._model.supports_caching,
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "cache_rate": cache_rate,
            "estimated_savings": savings,
        }

    def stream_response(self, user_message: str):
        """
        Generator that yields dicts as the response streams.
        Yields:
            {"type": "reasoning", "text": "..."}
            {"type": "content", "text": "..."}
            {"type": "done", "stats": {...}}
            {"type": "error", "error": "..."}
        """
        self.messages.append({"role": "user", "content": user_message})

        content_buffer = ""
        reasoning_buffer = ""

        try:
            for chunk in self.provider.stream_chat(
                model_id=self.model_id,
                messages=self.messages,
                reasoning_effort=self.reasoning_effort,
                show_reasoning=self.show_reasoning,
                max_tokens=self.max_completion_tokens,
            ):
                if chunk.is_done:
                    # Final chunk — extract stats
                    cache_stats = self._update_stats(chunk.usage)
                    self.messages.append({"role": "assistant", "content": content_buffer})

                    yield {
                        "type": "done",
                        "stats": {
                            "content": content_buffer,
                            "reasoning": reasoning_buffer,
                            "ttft_ms": chunk.usage.ttft_ms if chunk.usage else 0,
                            "ttfc_ms": chunk.usage.ttfc_ms if chunk.usage else 0,
                            "total_ms": chunk.usage.total_ms if chunk.usage else 0,
                            "cached_tokens": cache_stats["cached_tokens"],
                            "prompt_tokens": cache_stats["prompt_tokens"],
                            "completion_tokens": cache_stats["completion_tokens"],
                        },
                    }
                elif chunk.is_reasoning:
                    reasoning_buffer += chunk.text
                    if self.show_reasoning:
                        yield {"type": "reasoning", "text": chunk.text}
                elif chunk.is_content:
                    content_buffer += chunk.text
                    yield {"type": "content", "text": chunk.text}

        except ProviderError as e:
            self.messages.pop()
            yield {"type": "error", "error": str(e)}

    def structured_response(self, user_message: str) -> dict:
        """Send message and get structured JSON response (non-streaming)."""
        self.messages.append({"role": "user", "content": user_message})

        try:
            result = self.provider.json_chat(
                model_id=self.model_id,
                messages=self.messages,
                schema=self.json_schema,
                reasoning_effort=self.reasoning_effort,
                max_tokens=self.max_completion_tokens,
            )
        except ProviderError as e:
            self.messages.pop()
            return {"type": "error", "error": str(e)}

        cache_stats = self._update_stats(result.usage)
        self.messages.append({"role": "assistant", "content": result.content})

        return {
            "type": "done",
            "content": result.content,
            "parsed": result.parsed,
            "stats": {
                "total_ms": result.usage.total_ms if result.usage else 0,
                "cached_tokens": cache_stats["cached_tokens"],
                "prompt_tokens": cache_stats["prompt_tokens"],
                "completion_tokens": cache_stats["completion_tokens"],
            },
        }

    def warm_cache(self) -> dict:
        """Prime the system prompt cache. Returns result dict."""
        if not self.system_prompt:
            return {"status": "skipped", "reason": "No system prompt"}
        if not self._model.supports_caching:
            return {"status": "skipped", "reason": f"{self._model.name} does not support caching"}

        result = self.provider.warm_cache(self.model_id, self.system_prompt)
        return result

    def clear_history(self):
        """Clear conversation history, preserving system prompt."""
        if self.system_prompt:
            self.messages = [{"role": "system", "content": self.system_prompt}]
        else:
            self.messages = []

    def get_conversation_tokens(self) -> int:
        """Estimate total tokens in current conversation."""
        return sum(estimate_tokens(m["content"]) for m in self.messages)
