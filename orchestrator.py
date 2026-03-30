"""
Orchestrator — Runs Actor (streaming) and Director (JSON) in parallel.

The Actor streams in the caller's thread (needs stdout/yield access).
The Director runs in a background thread via ThreadPoolExecutor.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Iterator

from providers.base import LLMProvider, StreamChunk, ProviderError
from director import Director, DirectorResult


# ── Turn Result ──────────────────────────────────────────────────────────────


@dataclass
class TurnStats:
    """Combined stats for a single Actor + Director turn."""

    # Actor
    actor_ttft_ms: float = 0
    actor_ttfc_ms: float = 0
    actor_total_ms: float = 0
    actor_tokens: int = 0
    actor_prompt_tokens: int = 0
    actor_cached_tokens: int = 0
    actor_cost: float = 0.0

    # Director
    director_total_ms: float = 0
    director_tokens: int = 0
    director_prompt_tokens: int = 0
    director_cached_tokens: int = 0
    director_schema_valid: bool = False
    director_cost: float = 0.0

    # Combined
    wall_ms: float = 0  # max(actor, director) since they run in parallel

    @property
    def total_cost(self) -> float:
        return self.actor_cost + self.director_cost

    @property
    def director_finished_before_actor(self) -> bool:
        return self.director_total_ms < self.actor_total_ms

    @property
    def director_delta_ms(self) -> float:
        """How much earlier/later the Director finished vs Actor. Negative = earlier."""
        return self.director_total_ms - self.actor_total_ms


# ── Orchestrator ─────────────────────────────────────────────────────────────


class Orchestrator:
    """
    Runs Actor and Director in parallel for each conversation turn.

    Usage:
        orch = Orchestrator(actor_provider, actor_model_id, director)

        # Stream actor response while director runs in background
        for chunk in orch.stream_turn(messages, user_message):
            if chunk.is_content:
                print(chunk.text, end="")

        # After streaming, get director result
        result = orch.last_director_result
        stats = orch.last_turn_stats
    """

    def __init__(
        self,
        actor_provider: LLMProvider,
        actor_model_id: str,
        director: Director | None = None,
        *,
        reasoning_effort: str = "",
        show_reasoning: bool = True,
        max_tokens: int = 4096,
    ):
        self.actor_provider = actor_provider
        self.actor_model_id = actor_model_id
        self.director = director

        self.reasoning_effort = reasoning_effort
        self.show_reasoning = show_reasoning
        self.max_tokens = max_tokens

        # Results from last turn
        self.last_director_result: DirectorResult | None = None
        self.last_turn_stats: TurnStats | None = None

        # Session-level stats
        self.session_stats = SessionStats()

    def stream_turn(
        self,
        messages: list[dict],
    ) -> Iterator[StreamChunk]:
        """
        Run a single conversation turn: Actor streams + Director analyzes in parallel.

        Yields StreamChunk objects from the Actor. After the generator is exhausted,
        access self.last_director_result and self.last_turn_stats.
        """
        turn_start = time.perf_counter()
        turn_stats = TurnStats()
        director_future: Future | None = None

        # Launch Director in background thread
        if self.director:
            executor = ThreadPoolExecutor(max_workers=1)
            director_future = executor.submit(self.director.analyze, messages)

        # Stream Actor in current thread
        actor_usage = None
        try:
            for chunk in self.actor_provider.stream_chat(
                model_id=self.actor_model_id,
                messages=messages,
                reasoning_effort=self.reasoning_effort,
                show_reasoning=self.show_reasoning,
                max_tokens=self.max_tokens,
            ):
                if chunk.is_done:
                    actor_usage = chunk.usage
                else:
                    yield chunk

        except ProviderError as e:
            # Yield a synthetic error chunk
            yield StreamChunk(text=str(e), is_done=True)
            if director_future:
                director_future.cancel()
                executor.shutdown(wait=False)
            return

        # Collect Actor stats
        if actor_usage:
            turn_stats.actor_ttft_ms = actor_usage.ttft_ms
            turn_stats.actor_ttfc_ms = actor_usage.ttfc_ms
            turn_stats.actor_total_ms = actor_usage.total_ms
            turn_stats.actor_tokens = actor_usage.completion_tokens
            turn_stats.actor_prompt_tokens = actor_usage.prompt_tokens
            turn_stats.actor_cached_tokens = actor_usage.cached_tokens
            actor_model = self.actor_provider.get_model(self.actor_model_id)
            if actor_model:
                turn_stats.actor_cost = actor_model.estimate_cost(
                    actor_usage.prompt_tokens, actor_usage.completion_tokens
                )

        # Collect Director result
        if director_future:
            try:
                self.last_director_result = director_future.result(timeout=30)
                dr = self.last_director_result
                turn_stats.director_total_ms = dr.total_ms
                turn_stats.director_tokens = dr.completion_tokens
                turn_stats.director_prompt_tokens = dr.prompt_tokens
                turn_stats.director_cached_tokens = dr.cached_tokens
                turn_stats.director_schema_valid = dr.schema_valid
                director_model = self.director.provider.get_model(self.director.model_id)
                if director_model:
                    turn_stats.director_cost = director_model.estimate_cost(
                        dr.prompt_tokens, dr.completion_tokens
                    )
            except Exception as e:
                self.last_director_result = DirectorResult(
                    success=False, error=str(e)
                )
            finally:
                executor.shutdown(wait=False)

        # Combined wall time
        turn_stats.wall_ms = (time.perf_counter() - turn_start) * 1000
        self.last_turn_stats = turn_stats

        # Update session stats
        self.session_stats.record_turn(turn_stats, self.last_director_result)

        # Yield final done chunk with actor usage
        yield StreamChunk(is_done=True, usage=actor_usage)


# ── Session Stats ────────────────────────────────────────────────────────────


@dataclass
class SessionStats:
    """Cumulative stats across all turns in a session."""

    turn_count: int = 0
    start_time: float = field(default_factory=time.time)

    # Actor timings (for history/sparklines)
    ttft_history: list[float] = field(default_factory=list)
    ttfc_history: list[float] = field(default_factory=list)
    total_history: list[float] = field(default_factory=list)
    throughput_history: list[float] = field(default_factory=list)

    # Director timings
    director_total_history: list[float] = field(default_factory=list)
    director_passes: int = 0
    director_enabled: bool = False

    # Cost
    total_cost: float = 0.0
    actor_cost: float = 0.0
    director_cost: float = 0.0

    # Tokens
    total_actor_tokens: int = 0
    total_director_tokens: int = 0
    total_cached_tokens: int = 0

    def record_turn(self, stats: TurnStats, director_result: DirectorResult | None):
        self.turn_count += 1

        # Actor
        self.ttft_history.append(stats.actor_ttft_ms)
        self.ttfc_history.append(stats.actor_ttfc_ms)
        self.total_history.append(stats.actor_total_ms)
        if stats.actor_total_ms > 0:
            tps = stats.actor_tokens / (stats.actor_total_ms / 1000)
            self.throughput_history.append(round(tps))
        self.total_actor_tokens += stats.actor_tokens
        self.total_cached_tokens += stats.actor_cached_tokens
        self.actor_cost += stats.actor_cost

        # Director
        if director_result and director_result.success:
            self.director_enabled = True
            self.director_total_history.append(stats.director_total_ms)
            if stats.director_schema_valid:
                self.director_passes += 1
            self.total_director_tokens += stats.director_tokens
            self.total_cached_tokens += stats.director_cached_tokens
            self.director_cost += stats.director_cost

        self.total_cost = self.actor_cost + self.director_cost

    @property
    def actor_cost_pct(self) -> float:
        if self.total_cost == 0:
            return 0
        return (self.actor_cost / self.total_cost) * 100

    @property
    def director_cost_pct(self) -> float:
        if self.total_cost == 0:
            return 0
        return (self.director_cost / self.total_cost) * 100

    @property
    def avg_ttft(self) -> float:
        if not self.ttft_history:
            return 0
        return sum(self.ttft_history) / len(self.ttft_history)

    @property
    def avg_ttfc(self) -> float:
        if not self.ttfc_history:
            return 0
        return sum(self.ttfc_history) / len(self.ttfc_history)

    @property
    def avg_total(self) -> float:
        if not self.total_history:
            return 0
        return sum(self.total_history) / len(self.total_history)

    @property
    def avg_throughput(self) -> float:
        if not self.throughput_history:
            return 0
        return sum(self.throughput_history) / len(self.throughput_history)

    @property
    def director_pass_rate(self) -> str:
        if not self.director_enabled:
            return "N/A"
        return f"{self.director_passes}/{len(self.director_total_history)}"

    def to_dict(self) -> dict:
        """Serialize for UI display."""
        return {
            "turn_count": self.turn_count,
            "actor": {
                "avg_ttft_ms": round(self.avg_ttft),
                "avg_ttfc_ms": round(self.avg_ttfc),
                "avg_total_ms": round(self.avg_total),
                "avg_throughput": round(self.avg_throughput),
                "total_tokens": self.total_actor_tokens,
                "total_cached": self.total_cached_tokens,
                "cost": round(self.actor_cost, 6),
            },
            "director": {
                "enabled": self.director_enabled,
                "pass_rate": self.director_pass_rate,
                "total_tokens": self.total_director_tokens,
                "cost": round(self.director_cost, 6),
            },
            "total_cost": round(self.total_cost, 6),
            "ttft_history": self.ttft_history,
            "throughput_history": self.throughput_history,
        }
