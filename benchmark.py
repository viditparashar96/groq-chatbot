"""
Benchmark — Full model comparison across all providers.

Runs a standardized prompt against every available model in both
Actor (streaming) and Director (JSON) modes, collecting TTFT, TTFC,
total time, throughput, schema compliance, and cost metrics.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from providers.base import LLMProvider, ModelInfo, StreamChunk, ProviderError
from providers.registry import get_registry


# ── Benchmark Result Types ───────────────────────────────────────────────────


@dataclass
class ActorResult:
    """Result from benchmarking a model as Actor (streaming)."""

    provider: str = ""
    model: str = ""
    model_name: str = ""
    ttft_ms: float = 0
    ttfc_ms: float = 0
    total_ms: float = 0
    tokens_per_sec: float = 0
    content_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0
    error: str = ""
    content_preview: str = ""


@dataclass
class DirectorResult:
    """Result from benchmarking a model as Director (JSON)."""

    provider: str = ""
    model: str = ""
    model_name: str = ""
    total_ms: float = 0
    tokens_per_sec: float = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    schema_pass: bool = False
    cost: float = 0.0
    error: str = ""


@dataclass
class CombinedResult:
    """Result from an Actor+Director parallel pair."""

    actor_provider: str = ""
    actor_model: str = ""
    actor_name: str = ""
    director_provider: str = ""
    director_model: str = ""
    director_name: str = ""
    wall_ms: float = 0          # max(actor, director)
    user_ttft_ms: float = 0     # actor's TTFT (user sees this)
    actor_total_ms: float = 0
    director_total_ms: float = 0
    combined_cost: float = 0.0


@dataclass
class BenchmarkResults:
    """Full benchmark results."""

    prompt: str = ""
    actor: list[ActorResult] = field(default_factory=list)
    director: list[DirectorResult] = field(default_factory=list)
    combined: list[CombinedResult] = field(default_factory=list)
    total_time_sec: float = 0
    models_tested: int = 0
    providers_tested: int = 0
    api_calls: int = 0


# ── ANSI Colors (standalone) ────────────────────────────────────────────────

class _C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

C = _C()


# ── Benchmark Engine ─────────────────────────────────────────────────────────


class Benchmark:
    """
    Runs a standard prompt against all available models.

    Produces three tables:
    1. Actor (Streaming) — TTFT, TTFC, total, tok/s, cost
    2. Director (JSON) — total, schema compliance, cost
    3. Combined pairs — wall time, user TTFT, combined cost
    """

    def __init__(self, director_schema: dict | None = None):
        self.registry = get_registry()
        self.director_schema = director_schema or self._load_default_schema()

    def run(
        self,
        prompt: str = "Explain quantum entanglement in simple terms. Keep it under 100 words.",
        providers: list[str] | None = None,
    ) -> BenchmarkResults:
        """Run the full benchmark."""
        start = time.perf_counter()
        results = BenchmarkResults(prompt=prompt)

        available = self.registry.list_providers()
        if providers:
            available = {k: v for k, v in available.items() if k in providers}

        if not available:
            print(f"{C.RED}No providers available. Set API keys.{C.RESET}")
            return results

        results.providers_tested = len(available)
        provider_names = list(available.keys())

        # Phase 1: Actor benchmarks (sequential — each needs to stream)
        print(f"\n{C.BOLD}{C.CYAN}{'═' * 90}{C.RESET}")
        print(f"{C.BOLD}{C.CYAN}  ACTOR-DIRECTOR BENCHMARK{C.RESET}")
        print(f"{C.BOLD}{C.CYAN}{'═' * 90}{C.RESET}")
        print(f"  {C.DIM}Prompt: \"{prompt}\"{C.RESET}")
        print(f"  {C.DIM}Providers: {', '.join(provider_names)}{C.RESET}\n")

        print(f"  {C.BOLD}Phase 1: Actor (Streaming){C.RESET}")
        for provider_name, provider in available.items():
            for model in provider.list_models():
                if "actor" in model.roles:
                    result = self._benchmark_actor(provider, model, prompt)
                    results.actor.append(result)
                    results.api_calls += 1
                    results.models_tested += 1
                    status = f"{C.GREEN}OK{C.RESET}" if not result.error else f"{C.RED}FAIL{C.RESET}"
                    print(f"    [{status}] {model.provider}/{model.id} — "
                          f"TTFT: {result.ttft_ms:.0f}ms, Total: {result.total_ms:.0f}ms, "
                          f"{result.tokens_per_sec:.0f} tok/s, ${result.cost:.6f}")

        # Phase 2: Director benchmarks (parallel)
        print(f"\n  {C.BOLD}Phase 2: Director (JSON){C.RESET}")
        director_futures = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for provider_name, provider in available.items():
                for model in provider.list_models():
                    if "director" in model.roles:
                        f = pool.submit(self._benchmark_director, provider, model, prompt)
                        director_futures[f] = model

            for f in as_completed(director_futures):
                model = director_futures[f]
                result = f.result()
                results.director.append(result)
                results.api_calls += 1
                status = f"{C.GREEN}OK{C.RESET}" if not result.error else f"{C.RED}FAIL{C.RESET}"
                schema = "✅" if result.schema_pass else "⚠️"
                print(f"    [{status}] {model.provider}/{model.id} — "
                      f"Total: {result.total_ms:.0f}ms, Schema: {schema}, ${result.cost:.6f}")

        # Phase 3: Combined pairs
        print(f"\n  {C.BOLD}Phase 3: Best Pairs (Actor + Director){C.RESET}")
        best_director = min(results.director, key=lambda d: d.total_ms) if results.director else None
        if best_director:
            for ar in results.actor:
                if not ar.error:
                    wall = max(ar.total_ms, best_director.total_ms)
                    combined = CombinedResult(
                        actor_provider=ar.provider,
                        actor_model=ar.model,
                        actor_name=ar.model_name,
                        director_provider=best_director.provider,
                        director_model=best_director.model,
                        director_name=best_director.model_name,
                        wall_ms=wall,
                        user_ttft_ms=ar.ttft_ms,
                        actor_total_ms=ar.total_ms,
                        director_total_ms=best_director.total_ms,
                        combined_cost=ar.cost + best_director.cost,
                    )
                    results.combined.append(combined)

        results.total_time_sec = time.perf_counter() - start

        # Print tables
        self._print_actor_table(results.actor)
        self._print_director_table(results.director)
        self._print_combined_table(results.combined)
        self._print_summary(results)

        return results

    # ── Benchmark Individual Models ──────────────────────────────────────

    def _benchmark_actor(self, provider: LLMProvider, model: ModelInfo, prompt: str) -> ActorResult:
        """Stream a response, capture all timing metrics."""
        messages = [{"role": "user", "content": prompt}]
        result = ActorResult(provider=model.provider, model=model.id, model_name=model.name)

        try:
            content_buf = ""
            content_tokens = 0
            reasoning_tokens = 0

            for chunk in provider.stream_chat(
                model_id=model.id,
                messages=messages,
                max_tokens=512,
                temperature=0.6,
            ):
                if chunk.is_done and chunk.usage:
                    result.ttft_ms = chunk.usage.ttft_ms
                    result.ttfc_ms = chunk.usage.ttfc_ms
                    result.total_ms = chunk.usage.total_ms
                    result.prompt_tokens = chunk.usage.prompt_tokens
                    result.cached_tokens = chunk.usage.cached_tokens
                    total_completion = chunk.usage.completion_tokens
                    result.content_tokens = total_completion
                    result.total_tokens = result.prompt_tokens + total_completion
                elif chunk.is_content:
                    content_buf += chunk.text
                    content_tokens += 1
                elif chunk.is_reasoning:
                    reasoning_tokens += 1

            result.reasoning_tokens = reasoning_tokens
            if result.content_tokens == 0:
                result.content_tokens = content_tokens
                result.total_tokens = result.prompt_tokens + content_tokens + reasoning_tokens

            if result.total_ms > 0:
                result.tokens_per_sec = (result.content_tokens + reasoning_tokens) / (result.total_ms / 1000)
            result.cost = model.estimate_cost(result.prompt_tokens, result.content_tokens)
            result.content_preview = content_buf[:100]

        except ProviderError as e:
            result.error = str(e)

        return result

    def _benchmark_director(self, provider: LLMProvider, model: ModelInfo, prompt: str) -> DirectorResult:
        """JSON request, check schema compliance."""
        messages = [{"role": "user", "content": prompt}]
        result = DirectorResult(provider=model.provider, model=model.id, model_name=model.name)

        try:
            resp = provider.json_chat(
                model_id=model.id,
                messages=messages,
                schema=self.director_schema,
                max_tokens=1024,
                temperature=0.3,
            )
            if resp.usage:
                result.total_ms = resp.usage.total_ms
                result.prompt_tokens = resp.usage.prompt_tokens
                result.completion_tokens = resp.usage.completion_tokens
                result.total_tokens = result.prompt_tokens + result.completion_tokens
            result.schema_pass = resp.schema_valid
            if result.total_ms > 0 and result.completion_tokens > 0:
                result.tokens_per_sec = result.completion_tokens / (result.total_ms / 1000)
            result.cost = model.estimate_cost(result.prompt_tokens, result.completion_tokens)

        except ProviderError as e:
            result.error = str(e)

        return result

    # ── Table Printing ───────────────────────────────────────────────────

    def _print_actor_table(self, results: list[ActorResult]):
        if not results:
            return

        print(f"\n  {C.BOLD}{'─' * 88}{C.RESET}")
        print(f"  {C.BOLD}  ACTOR MODE (Streaming){C.RESET}")
        print(f"  {'─' * 88}")
        print(f"  {C.BOLD}  {'Model':<30} {'TTFT':>7} {'TTFC':>7} {'Total':>7} {'Tok/s':>7} {'Tokens':>7} {'Cost':>10}{C.RESET}")
        print(f"  {'─' * 88}")

        for r in sorted(results, key=lambda x: x.ttft_ms if not x.error else 99999):
            if r.error:
                print(f"  {C.RED}  {r.provider}/{r.model:<25} ERROR: {r.error[:40]}{C.RESET}")
            else:
                print(f"    {r.provider}/{r.model:<28} "
                      f"{r.ttft_ms:>6.0f}ms {r.ttfc_ms:>6.0f}ms {r.total_ms:>6.0f}ms "
                      f"{r.tokens_per_sec:>6.0f} {r.total_tokens:>7} ${r.cost:>9.6f}")

        print(f"  {'─' * 88}")

    def _print_director_table(self, results: list[DirectorResult]):
        if not results:
            return

        print(f"\n  {C.BOLD}  DIRECTOR MODE (JSON, strict schema){C.RESET}")
        print(f"  {'─' * 88}")
        print(f"  {C.BOLD}  {'Model':<30} {'Total':>7} {'Tok/s':>7} {'Tokens':>7} {'Schema':>8} {'Cost':>10}{C.RESET}")
        print(f"  {'─' * 88}")

        for r in sorted(results, key=lambda x: x.total_ms if not x.error else 99999):
            if r.error:
                print(f"  {C.RED}  {r.provider}/{r.model:<25} ERROR: {r.error[:40]}{C.RESET}")
            else:
                schema = f"{C.GREEN}pass{C.RESET}" if r.schema_pass else f"{C.YELLOW}fail{C.RESET}"
                print(f"    {r.provider}/{r.model:<28} "
                      f"{r.total_ms:>6.0f}ms {r.tokens_per_sec:>6.0f} {r.total_tokens:>7} "
                      f"   {schema}    ${r.cost:>9.6f}")

        print(f"  {'─' * 88}")

    def _print_combined_table(self, results: list[CombinedResult]):
        if not results:
            return

        print(f"\n  {C.BOLD}  ACTOR + DIRECTOR PARALLEL (combined){C.RESET}")
        print(f"  {'─' * 88}")
        print(f"  {C.BOLD}  {'Actor':<22} + {'Director':<20} {'Wall':>7} {'TTFT':>7} {'Cost':>10}{C.RESET}")
        print(f"  {'─' * 88}")

        for r in sorted(results, key=lambda x: x.wall_ms):
            print(f"    {r.actor_provider}/{r.actor_model:<19} + {r.director_provider}/{r.director_model:<17} "
                  f"{r.wall_ms:>6.0f}ms {r.user_ttft_ms:>6.0f}ms ${r.combined_cost:>9.6f}")

        print(f"  {'─' * 88}")

        # Best picks
        if results:
            fastest = min(results, key=lambda r: r.user_ttft_ms)
            cheapest = min(results, key=lambda r: r.combined_cost)
            best_wall = min(results, key=lambda r: r.wall_ms)
            print(f"\n  {C.GREEN}  ★ Best TTFT:  {fastest.actor_provider}/{fastest.actor_model} ({fastest.user_ttft_ms:.0f}ms){C.RESET}")
            print(f"  {C.GREEN}  ★ Best Wall:  {best_wall.actor_provider}/{best_wall.actor_model} ({best_wall.wall_ms:.0f}ms, ${best_wall.combined_cost:.6f}){C.RESET}")
            print(f"  {C.GREEN}  ★ Cheapest:   {cheapest.actor_provider}/{cheapest.actor_model} ({cheapest.wall_ms:.0f}ms, ${cheapest.combined_cost:.6f}){C.RESET}")

    def _print_summary(self, results: BenchmarkResults):
        print(f"\n{C.BOLD}{C.CYAN}{'═' * 90}{C.RESET}")
        print(f"  Benchmark completed in {results.total_time_sec:.1f}s │ "
              f"{results.models_tested} models │ "
              f"{results.providers_tested} providers │ "
              f"{results.api_calls} API calls")
        print(f"{C.BOLD}{C.CYAN}{'═' * 90}{C.RESET}\n")

    def to_json(self, results: BenchmarkResults) -> str:
        """Serialize benchmark results to JSON."""
        data = {
            "prompt": results.prompt,
            "total_time_sec": round(results.total_time_sec, 2),
            "models_tested": results.models_tested,
            "providers_tested": results.providers_tested,
            "api_calls": results.api_calls,
            "actor": [
                {
                    "provider": r.provider, "model": r.model, "name": r.model_name,
                    "ttft_ms": round(r.ttft_ms), "ttfc_ms": round(r.ttfc_ms),
                    "total_ms": round(r.total_ms), "tokens_per_sec": round(r.tokens_per_sec),
                    "total_tokens": r.total_tokens, "cost": round(r.cost, 6),
                    "error": r.error,
                }
                for r in results.actor
            ],
            "director": [
                {
                    "provider": r.provider, "model": r.model, "name": r.model_name,
                    "total_ms": round(r.total_ms), "tokens_per_sec": round(r.tokens_per_sec),
                    "total_tokens": r.total_tokens, "schema_pass": r.schema_pass,
                    "cost": round(r.cost, 6), "error": r.error,
                }
                for r in results.director
            ],
            "combined": [
                {
                    "actor": f"{r.actor_provider}/{r.actor_model}",
                    "director": f"{r.director_provider}/{r.director_model}",
                    "wall_ms": round(r.wall_ms), "user_ttft_ms": round(r.user_ttft_ms),
                    "combined_cost": round(r.combined_cost, 6),
                }
                for r in results.combined
            ],
        }
        return json.dumps(data, indent=2)

    @staticmethod
    def _load_default_schema() -> dict:
        """Load the default Director schema."""
        from pathlib import Path
        schema_path = Path(__file__).parent / "schemas" / "director_default.json"
        if schema_path.exists():
            with open(schema_path) as f:
                return json.load(f)
        return {"type": "object", "properties": {"intent": {"type": "object"}}}


# ── CLI Entry Point ──────────────────────────────────────────────────────────


def run_benchmark_cli():
    """Run benchmark from command line."""
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark all available LLM models")
    parser.add_argument("--prompt", "-p", default="Explain quantum entanglement in simple terms. Keep it under 100 words.")
    parser.add_argument("--providers", help="Comma-separated list of providers (e.g., groq,openai)")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    args = parser.parse_args()

    providers = args.providers.split(",") if args.providers else None
    bench = Benchmark()
    results = bench.run(prompt=args.prompt, providers=providers)

    if args.output:
        with open(args.output, "w") as f:
            f.write(bench.to_json(results))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    run_benchmark_cli()
