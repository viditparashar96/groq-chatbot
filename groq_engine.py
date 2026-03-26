"""
Groq Engine — Shared inference engine for CLI and UI.

Provides GroqEngine class with streaming/structured response support,
prompt caching, and model comparison capabilities.
"""

import os
import time
import json
from groq import Groq


# ── Model Registry ──────────────────────────────────────────────────────────

MODELS = {
    "qwen/qwen3-32b": {
        "name": "Qwen3-32B",
        "speed": "~400 t/s",
        "tier": "Preview",
        "reasoning_type": "reasoning_format",
        "reasoning_efforts": ["none", "default"],
        "default_effort": "default",
        "context": "131,072",
        "max_completion": 40960,
        "multilingual": "119 languages (Hindi, Arabic dialects, Afrikaans, Swahili, CJK, European, Dravidian, etc.)",
        "multilingual_score": "★★★★★",
        "price_in": "$0.29",
        "price_out": "$0.59",
        "streaming": True,
        "notes": "Best multilingual. Uses <think> tags (raw) or separate reasoning field (parsed).",
        "supports_strict_schema": False,
        "supports_caching": False,
    },
    "openai/gpt-oss-20b": {
        "name": "GPT-OSS 20B",
        "speed": "~1000 t/s",
        "tier": "Production",
        "reasoning_type": "include_reasoning",
        "reasoning_efforts": ["low", "medium", "high"],
        "default_effort": "medium",
        "context": "131,072",
        "max_completion": 65536,
        "multilingual": "14+ languages (primarily English-optimized, STEM focus)",
        "multilingual_score": "★★☆☆☆",
        "price_in": "$0.075",
        "price_out": "$0.30",
        "streaming": True,
        "notes": "Fastest TTFT. MoE with 3.6B active params. Best for English speed-critical.",
        "supports_strict_schema": True,
        "supports_caching": True,
    },
    "openai/gpt-oss-120b": {
        "name": "GPT-OSS 120B",
        "speed": "~500 t/s",
        "tier": "Production",
        "reasoning_type": "include_reasoning",
        "reasoning_efforts": ["low", "medium", "high"],
        "default_effort": "medium",
        "context": "131,072",
        "max_completion": 65536,
        "multilingual": "81+ languages (English-primary, competitive on 14-lang benchmarks)",
        "multilingual_score": "★★★☆☆",
        "price_in": "$0.15",
        "price_out": "$0.60",
        "streaming": True,
        "notes": "Best reasoning quality. MoE with 5.1B active params. Near o4-mini parity.",
        "supports_strict_schema": True,
        "supports_caching": True,
    },
}


def estimate_tokens(text: str) -> int:
    """Rough estimate: ~1 token per 4 characters."""
    return max(1, len(text) // 4)


class GroqEngine:
    """Core inference engine for Groq models — UI and CLI agnostic."""

    def __init__(self, model_id: str, api_key: str = None,
                 reasoning_effort: str = None, show_reasoning: bool = True,
                 system_prompt: str = None,
                 json_schema: dict = None, json_mode: bool = False,
                 max_completion_tokens: int = 4096):

        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set")

        self.client = Groq(api_key=self.api_key)
        self.model_id = model_id
        self.model_info = MODELS[model_id]
        self.show_reasoning = show_reasoning
        self.messages = []

        # Token & cache stats
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

    def _build_request_params(self, messages: list, force_stream: bool = None) -> dict:
        """Build API request parameters."""
        is_json = self.output_mode == "json"
        stream = not is_json if force_stream is None else force_stream

        params = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.6,
            "max_completion_tokens": self.max_completion_tokens,
            "top_p": 0.95,
            "stream": stream,
        }

        if self.model_info["reasoning_type"] == "reasoning_format":
            params["reasoning_format"] = "parsed" if self.show_reasoning else "hidden"
            if self.reasoning_effort != "default":
                params["reasoning_effort"] = self.reasoning_effort
        else:
            params["include_reasoning"] = self.show_reasoning
            params["reasoning_effort"] = self.reasoning_effort

        if is_json and not stream:
            if self.json_schema and self.model_info.get("supports_strict_schema"):
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.json_schema.get("title", "response"),
                        "schema": self.json_schema,
                        "strict": True,
                    }
                }
            else:
                params["response_format"] = {"type": "json_object"}

        return params

    def _update_cache_stats(self, usage) -> dict:
        """Extract and update cache statistics from API usage."""
        stats = {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}
        if not usage:
            return stats

        stats["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
        stats["completion_tokens"] = getattr(usage, "completion_tokens", 0) or 0

        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            stats["cached_tokens"] = getattr(details, "cached_tokens", 0) or 0

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
        price_per_m = float(self.model_info["price_in"].replace("$", ""))
        savings = (self.total_cached_tokens / 1_000_000) * price_per_m * 0.5
        return {
            "model": self.model_id,
            "supports_caching": self.model_info.get("supports_caching", False),
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
        params = self._build_request_params(self.messages)

        start_time = time.perf_counter()
        first_token_time = None
        first_content_time = None
        content_buffer = ""
        reasoning_buffer = ""
        usage = None

        try:
            stream = self.client.chat.completions.create(**params)

            for chunk in stream:
                delta = chunk.choices[0].delta
                now = time.perf_counter()

                if first_token_time is None:
                    first_token_time = now

                # Reasoning (GPT-OSS)
                reasoning_text = getattr(delta, "reasoning", None)
                if reasoning_text:
                    reasoning_buffer += reasoning_text
                    if self.show_reasoning:
                        yield {"type": "reasoning", "text": reasoning_text}

                # Content
                content_text = getattr(delta, "content", None) or ""
                if content_text:
                    if first_content_time is None:
                        first_content_time = now
                    content_buffer += content_text
                    yield {"type": "content", "text": content_text}

                # Usage from final chunk
                x_groq = getattr(chunk, "x_groq", None)
                if x_groq:
                    usage = getattr(x_groq, "usage", None)

        except Exception as e:
            self.messages.pop()
            yield {"type": "error", "error": str(e)}
            return

        end_time = time.perf_counter()
        cache_stats = self._update_cache_stats(usage)

        self.messages.append({"role": "assistant", "content": content_buffer})

        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
        ttfc = (first_content_time - start_time) * 1000 if first_content_time else 0
        total_time = (end_time - start_time) * 1000

        yield {
            "type": "done",
            "stats": {
                "content": content_buffer,
                "reasoning": reasoning_buffer,
                "ttft_ms": round(ttft),
                "ttfc_ms": round(ttfc),
                "total_ms": round(total_time),
                "cached_tokens": cache_stats["cached_tokens"],
                "prompt_tokens": cache_stats["prompt_tokens"],
                "completion_tokens": cache_stats["completion_tokens"],
            }
        }

    def structured_response(self, user_message: str) -> dict:
        """Send message and get structured JSON response (non-streaming)."""
        self.messages.append({"role": "user", "content": user_message})

        messages = list(self.messages)
        if self.json_schema and not self.model_info.get("supports_strict_schema"):
            schema_instruction = (
                f"You must respond with valid JSON matching this schema:\n"
                f"```json\n{json.dumps(self.json_schema, indent=2)}\n```"
            )
            if messages and messages[0]["role"] == "system":
                messages[0] = {
                    "role": "system",
                    "content": messages[0]["content"] + "\n\n" + schema_instruction
                }
            else:
                messages.insert(0, {"role": "system", "content": schema_instruction})

        params = self._build_request_params(messages)
        start_time = time.perf_counter()

        try:
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            self.messages.pop()
            return {"type": "error", "error": str(e)}

        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000
        content = response.choices[0].message.content
        cache_stats = self._update_cache_stats(getattr(response, "usage", None))

        self.messages.append({"role": "assistant", "content": content})

        parsed = None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            pass

        return {
            "type": "done",
            "content": content,
            "parsed": parsed,
            "stats": {
                "total_ms": round(total_time),
                "cached_tokens": cache_stats["cached_tokens"],
                "prompt_tokens": cache_stats["prompt_tokens"],
                "completion_tokens": cache_stats["completion_tokens"],
            }
        }

    def warm_cache(self) -> dict:
        """Prime the system prompt cache. Returns result dict."""
        if not self.system_prompt:
            return {"status": "skipped", "reason": "No system prompt"}
        if not self.model_info.get("supports_caching"):
            return {"status": "skipped", "reason": f"{self.model_info['name']} does not support caching"}

        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "hi"}
                ],
                max_completion_tokens=1,
                temperature=0.0,
            )
            elapsed = (time.perf_counter() - start) * 1000
            if response.usage:
                self._update_cache_stats(response.usage)
            return {"status": "ok", "elapsed_ms": round(elapsed)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def clear_history(self):
        """Clear conversation history, preserving system prompt."""
        if self.system_prompt:
            self.messages = [{"role": "system", "content": self.system_prompt}]
        else:
            self.messages = []

    def get_conversation_tokens(self) -> int:
        """Estimate total tokens in current conversation."""
        return sum(estimate_tokens(m["content"]) for m in self.messages)
