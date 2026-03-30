"""
Provider registry — discovers and manages available LLM providers.

Auto-detects which providers are available based on installed packages
and API keys. Provides unified access to all models across providers.
"""

from __future__ import annotations

import os
from typing import Optional

from providers.base import LLMProvider, ModelInfo


# ── Registry ─────────────────────────────────────────────────────────────────


class ProviderRegistry:
    """
    Discovers and manages all available LLM providers.

    Lazily initializes providers — no API calls until you actually use one.
    """

    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}
        self._discovered = False

    def discover(self, force: bool = False) -> None:
        """Discover all available providers based on installed packages + API keys."""
        if self._discovered and not force:
            return
        self._providers = {}

        # Groq
        try:
            from providers.groq_provider import GroqProvider
            p = GroqProvider()
            if p.is_available():
                self._providers["groq"] = p
        except Exception:
            pass

        # OpenAI
        try:
            from providers.openai_provider import OpenAIProvider
            p = OpenAIProvider()
            if p.is_available():
                self._providers["openai"] = p
        except Exception:
            pass

        # Anthropic
        try:
            from providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider()
            if p.is_available():
                self._providers["anthropic"] = p
        except Exception:
            pass

        # Google
        try:
            from providers.google_provider import GoogleProvider
            p = GoogleProvider()
            if p.is_available():
                self._providers["google"] = p
        except Exception:
            pass

        self._discovered = True

    def get_provider(self, name: str) -> Optional[LLMProvider]:
        """Get a specific provider by name."""
        self.discover()
        return self._providers.get(name)

    def list_providers(self) -> dict[str, LLMProvider]:
        """Return all available providers."""
        self.discover()
        return dict(self._providers)

    def list_all_models(self) -> list[ModelInfo]:
        """Return all models from all available providers."""
        self.discover()
        models = []
        for provider in self._providers.values():
            models.extend(provider.list_models())
        return models

    def list_actor_models(self) -> list[ModelInfo]:
        """Return all models that can serve as Actor (streaming)."""
        return [m for m in self.list_all_models() if "actor" in m.roles]

    def list_director_models(self) -> list[ModelInfo]:
        """Return all models that can serve as Director (JSON)."""
        return [m for m in self.list_all_models() if "director" in m.roles]

    def find_model(self, model_id: str) -> tuple[LLMProvider, ModelInfo] | None:
        """
        Find a model across all providers.

        Supports both plain IDs ("gpt-4.1-nano") and prefixed IDs ("openai/gpt-4.1-nano").
        Returns (provider, model_info) or None.
        """
        self.discover()

        # Check if it's a prefixed ID like "openai/gpt-4.1-nano"
        if "/" in model_id:
            # Could be "provider/model" or "org/model" (like Groq's "qwen/qwen3-32b")
            parts = model_id.split("/", 1)
            provider_name = parts[0]

            # Try as provider prefix first
            if provider_name in self._providers:
                provider = self._providers[provider_name]
                model = provider.get_model(parts[1])
                if model:
                    return (provider, model)

            # Try full ID across all providers (handles "qwen/qwen3-32b" on Groq)
            for provider in self._providers.values():
                model = provider.get_model(model_id)
                if model:
                    return (provider, model)
        else:
            # Plain ID — search all providers
            for provider in self._providers.values():
                model = provider.get_model(model_id)
                if model:
                    return (provider, model)

        return None

    def find_default_director(self) -> tuple[LLMProvider, ModelInfo] | None:
        """
        Find the best default Director model.

        Priority:
        1. gpt-4.1-nano (cheapest, fastest, 100% schema compliance)
        2. gpt-4.1-mini (same guarantees, better quality)
        3. Any Groq model with json_schema support
        """
        self.discover()

        # Priority list
        preferences = [
            ("openai", "gpt-4.1-nano"),
            ("openai", "gpt-4.1-mini"),
            ("openai", "gpt-4.1"),
        ]

        for provider_name, model_id in preferences:
            provider = self._providers.get(provider_name)
            if provider:
                model = provider.get_model(model_id)
                if model and "director" in model.roles:
                    return (provider, model)

        # Fallback: any director-capable model
        for provider in self._providers.values():
            for model in provider.list_models():
                if "director" in model.roles:
                    return (provider, model)

        return None

    def provider_status(self) -> list[dict]:
        """Return status of all known providers (for UI display)."""
        self.discover()
        known = [
            ("groq", "GROQ_API_KEY", "groq"),
            ("openai", "OPENAI_API_KEY", "openai"),
            ("anthropic", "ANTHROPIC_API_KEY", "anthropic"),
            ("google", "GOOGLE_API_KEY", "google-genai"),
        ]
        result = []
        for name, env_var, pkg in known:
            has_key = bool(os.environ.get(env_var))
            has_pkg = self._check_package(pkg)
            available = name in self._providers
            model_count = len(self._providers[name].list_models()) if available else 0
            result.append({
                "name": name,
                "env_var": env_var,
                "has_key": has_key,
                "has_package": has_pkg,
                "available": available,
                "model_count": model_count,
            })
        return result

    @staticmethod
    def _check_package(package: str) -> bool:
        try:
            __import__(package.replace("-", "_"))
            return True
        except ImportError:
            return False


# ── Module-level singleton ───────────────────────────────────────────────────

def get_registry() -> ProviderRegistry:
    """Get a fresh provider registry (re-discovers on every call)."""
    reg = ProviderRegistry()
    reg.discover()
    return reg
