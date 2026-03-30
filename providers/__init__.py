"""
LLM Provider abstraction layer.

Supports multiple providers (Groq, OpenAI, Anthropic, Google) with a unified
interface for streaming (Actor) and structured JSON (Director) responses.
"""

from providers.base import (
    LLMProvider,
    StreamChunk,
    ModelInfo,
    JsonResponse,
    ProviderError,
    UsageStats,
    TimingTracker,
    estimate_tokens,
)
from providers.registry import get_registry, ProviderRegistry

__all__ = [
    "LLMProvider",
    "StreamChunk",
    "ModelInfo",
    "JsonResponse",
    "ProviderError",
    "UsageStats",
    "TimingTracker",
    "estimate_tokens",
    "get_registry",
    "ProviderRegistry",
]
