"""
Director — Structured JSON analysis engine.

Runs in parallel with the Actor. Takes the conversation context and produces
structured metadata (intent, entities, follow-ups, guardrails) via a Director
model with constrained JSON output.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from providers.base import LLMProvider, JsonResponse, ProviderError


# ── Default Director System Prompt ───────────────────────────────────────────

DIRECTOR_SYSTEM_PROMPT = """You are the Director — a background analysis engine that runs in parallel with a chat assistant (the Actor). Your job is to analyze the user's message and conversation context, then produce structured JSON metadata.

You do NOT generate the user-facing response. The Actor does that. You produce analysis that enriches the Actor's response with metadata.

Analyze the conversation and produce JSON matching the provided schema. Be concise and accurate.

Guidelines:
- intent.type: classify what the user is doing (question, instruction, creative, debug, conversation)
- intent.complexity: how complex is this request? (low = simple lookup, medium = requires reasoning, high = multi-step or expert-level)
- intent.topic: 2-5 word topic label
- response_analysis.confidence: your confidence that the Actor will handle this well (0.0-1.0)
- response_analysis.key_entities: extract the most important nouns/concepts (max 5)
- suggestions.follow_ups: 1-3 natural follow-up questions the user might ask next
- suggestions.related_topics: 1-3 related topics for exploration
- guardrails.flags: flag if the request is off_topic, sensitive, or has factual_risk; otherwise ["none"]
- guardrails.note: explain any flags, or empty string if none"""


# ── Director Class ───────────────────────────────────────────────────────────


class Director:
    """
    Structured JSON analysis engine.

    Takes a provider + model_id + schema, sends conversation to the model,
    returns parsed JSON metadata.
    """

    def __init__(
        self,
        provider: LLMProvider,
        model_id: str,
        schema: dict | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
    ):
        self.provider = provider
        self.model_id = model_id
        self.schema = schema or self._load_default_schema()
        self.system_prompt = system_prompt or DIRECTOR_SYSTEM_PROMPT
        self.max_tokens = max_tokens

        # Cumulative stats
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.schema_passes = 0
        self.total_ms = 0.0

    def analyze(self, messages: list[dict]) -> DirectorResult:
        """
        Analyze the conversation and return structured metadata.

        Args:
            messages: The conversation history (same as Actor sees).

        Returns:
            DirectorResult with parsed data and stats.
        """
        # Build Director-specific messages: system prompt + conversation
        director_messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        # Include conversation context (last N messages to stay within budget)
        # Sanitize: only pass role + content — APIs reject extra fields
        context_messages = messages[-20:]
        for msg in context_messages:
            if msg.get("role") == "system":
                continue  # skip Actor's system prompt
            director_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

        try:
            result: JsonResponse = self.provider.json_chat(
                model_id=self.model_id,
                messages=director_messages,
                schema=self.schema,
                max_tokens=self.max_tokens,
                temperature=0.3,  # lower temp for more consistent analysis
            )
        except ProviderError as e:
            return DirectorResult(
                success=False,
                error=str(e),
            )

        # Update stats
        self.total_requests += 1
        if result.usage:
            self.total_tokens += result.usage.prompt_tokens + result.usage.completion_tokens
            self.total_ms += result.usage.total_ms
            model = self.provider.get_model(self.model_id)
            if model and result.usage:
                self.total_cost += model.estimate_cost(
                    result.usage.prompt_tokens,
                    result.usage.completion_tokens,
                )
        if result.schema_valid:
            self.schema_passes += 1

        return DirectorResult(
            success=True,
            data=result.parsed,
            raw_json=result.content,
            schema_valid=result.schema_valid,
            total_ms=result.usage.total_ms if result.usage else 0,
            prompt_tokens=result.usage.prompt_tokens if result.usage else 0,
            completion_tokens=result.usage.completion_tokens if result.usage else 0,
            cached_tokens=result.usage.cached_tokens if result.usage else 0,
        )

    def get_stats(self) -> dict:
        """Return cumulative Director stats."""
        return {
            "model": self.model_id,
            "provider": self.provider.name,
            "total_requests": self.total_requests,
            "schema_passes": self.schema_passes,
            "schema_pass_rate": (
                f"{self.schema_passes}/{self.total_requests}"
                if self.total_requests > 0
                else "0/0"
            ),
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "avg_ms": round(self.total_ms / max(1, self.total_requests)),
        }

    @staticmethod
    def _load_default_schema() -> dict:
        """Load the default Director schema from schemas/director_default.json."""
        schema_path = Path(__file__).parent / "schemas" / "director_default.json"
        if schema_path.exists():
            with open(schema_path) as f:
                return json.load(f)
        # Fallback minimal schema
        return {
            "title": "director_analysis",
            "type": "object",
            "properties": {
                "intent": {"type": "object"},
                "suggestions": {"type": "object"},
            },
        }


# ── Director Result ──────────────────────────────────────────────────────────


class DirectorResult:
    """Result from a Director analysis call."""

    def __init__(
        self,
        success: bool = False,
        data: dict | None = None,
        raw_json: str = "",
        schema_valid: bool = False,
        error: str = "",
        total_ms: float = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
    ):
        self.success = success
        self.data = data or {}
        self.raw_json = raw_json
        self.schema_valid = schema_valid
        self.error = error
        self.total_ms = total_ms
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cached_tokens = cached_tokens

    # ── Convenience accessors ────────────────────────────────────────────

    @property
    def intent(self) -> dict:
        return self.data.get("intent", {})

    @property
    def response_analysis(self) -> dict:
        return self.data.get("response_analysis", {})

    @property
    def suggestions(self) -> dict:
        return self.data.get("suggestions", {})

    @property
    def guardrails(self) -> dict:
        return self.data.get("guardrails", {})

    @property
    def follow_ups(self) -> list[str]:
        return self.suggestions.get("follow_ups", [])

    @property
    def key_entities(self) -> list[str]:
        return self.response_analysis.get("key_entities", [])

    @property
    def confidence(self) -> float:
        return self.response_analysis.get("confidence", 0.0)

    @property
    def flags(self) -> list[str]:
        return self.guardrails.get("flags", ["none"])
