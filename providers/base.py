"""
Base classes for the LLM provider abstraction.

All providers (Groq, OpenAI, Anthropic, Google) implement LLMProvider.
The Actor uses stream_chat(), the Director uses json_chat().
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


# ── Data Models ──────────────────────────────────────────────────────────────


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""

    text: str = ""
    is_content: bool = False
    is_reasoning: bool = False
    is_done: bool = False

    # Populated only on the final chunk (is_done=True)
    usage: UsageStats | None = None


@dataclass
class UsageStats:
    """Token usage and timing stats from a single API call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_ms: float = 0
    ttft_ms: float = 0       # time to first token
    ttfc_ms: float = 0       # time to first content token


@dataclass
class JsonResponse:
    """Result from a structured JSON (Director) call."""

    content: str = ""                # raw JSON string
    parsed: dict | None = None       # parsed JSON object
    schema_valid: bool = False       # did it match the schema?
    usage: UsageStats | None = None


@dataclass
class ModelInfo:
    """Metadata about a model available from a provider."""

    id: str                          # e.g. "qwen/qwen3-32b", "gpt-4.1-nano"
    name: str                        # display name e.g. "Qwen3-32B"
    provider: str                    # e.g. "groq", "openai"

    # Capabilities
    roles: list[str] = field(default_factory=lambda: ["actor"])  # "actor", "director"
    streaming: bool = True
    supports_json_schema: bool = False   # strict constrained decoding
    supports_json_mode: bool = False     # basic json_object mode
    supports_reasoning: bool = False
    supports_caching: bool = False

    # Reasoning config
    reasoning_type: str = ""             # "reasoning_format", "include_reasoning", etc.
    reasoning_efforts: list[str] = field(default_factory=list)
    default_effort: str = ""

    # Specs
    speed: str = ""                      # e.g. "~1000 t/s"
    context_window: int = 0
    max_completion: int = 0

    # Pricing (per 1M tokens)
    price_input: float = 0.0
    price_output: float = 0.0

    # Display
    tier: str = ""                       # "Production", "Preview"
    multilingual: str = ""
    multilingual_score: str = ""
    notes: str = ""

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request."""
        return (input_tokens * self.price_input + output_tokens * self.price_output) / 1_000_000


class ProviderError(Exception):
    """Raised when a provider encounters an error."""

    def __init__(self, message: str, provider: str = "", model: str = ""):
        self.provider = provider
        self.model = model
        super().__init__(message)


# ── Abstract Provider ────────────────────────────────────────────────────────


class LLMProvider(ABC):
    """
    Abstract base for all LLM providers.

    Each provider implements:
    - stream_chat()  → used by the Actor for streaming responses
    - json_chat()    → used by the Director for structured JSON output
    - list_models()  → available models for this provider
    """

    name: str = ""  # e.g. "groq", "openai"

    @abstractmethod
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
        """
        Stream a chat response token-by-token. Used by the Actor.

        Yields StreamChunk objects. The final chunk has is_done=True
        and carries UsageStats.
        """
        ...

    @abstractmethod
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
        """
        Get a structured JSON response. Used by the Director.

        If schema is provided and the model supports it, uses constrained
        decoding for 100% schema compliance. Otherwise falls back to
        json_object mode.
        """
        ...

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        """Return all models available from this provider."""
        ...

    def get_model(self, model_id: str) -> ModelInfo | None:
        """Look up a specific model by ID."""
        for m in self.list_models():
            if m.id == model_id:
                return m
        return None

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured (API key set, SDK installed)."""
        ...


# ── Helpers ──────────────────────────────────────────────────────────────────


def estimate_tokens(text: str) -> int:
    """Rough estimate: ~1 token per 4 characters."""
    return max(1, len(text) // 4)


class TimingTracker:
    """Helper to track TTFT/TTFC/total timing during streaming."""

    def __init__(self):
        self.start_time = time.perf_counter()
        self.first_token_time: float | None = None
        self.first_content_time: float | None = None
        self.end_time: float | None = None

    def mark_first_token(self):
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    def mark_first_content(self):
        if self.first_content_time is None:
            self.first_content_time = time.perf_counter()

    def mark_done(self):
        self.end_time = time.perf_counter()

    @property
    def ttft_ms(self) -> float:
        if self.first_token_time is None:
            return 0
        return (self.first_token_time - self.start_time) * 1000

    @property
    def ttfc_ms(self) -> float:
        if self.first_content_time is None:
            return 0
        return (self.first_content_time - self.start_time) * 1000

    @property
    def total_ms(self) -> float:
        end = self.end_time or time.perf_counter()
        return (end - self.start_time) * 1000
