# Actor-Director Architecture Plan

## The Core Idea

Two LLMs run **in parallel** on every user message:

| Role | What it does | Output | Why |
|------|-------------|--------|-----|
| **Actor** | Streams the user-facing response token-by-token | Streaming text (current behavior) | Fast TTFT, great UX — user sees tokens immediately |
| **Director** | Analyzes the conversation and produces structured metadata | Strict JSON (non-streaming) | Structured data for UI enrichment, guardrails, follow-ups |

The Actor gives the user their answer fast. The Director gives the **system** structured intelligence about that answer — which can be displayed alongside, after, or used to influence future turns.

---

## Why This Pattern?

Streaming and structured JSON are mutually exclusive in most APIs. You either get fast token-by-token output OR you get schema-guaranteed JSON — not both. The Actor-Director pattern solves this by running both simultaneously:

```
User message
    ├──→ [Actor]    streaming response  ──→  user sees tokens flowing
    └──→ [Director] JSON analysis       ──→  metadata arrives when complete
```

The Director's JSON enriches the conversation without blocking the Actor's stream.

---

## Director Model Selection: GPT-4.1-nano

### Why GPT-4.1-nano?

| Criterion | GPT-4.1-nano | GPT-4.1-mini | Gemini 2.0 Flash | Haiku 4.5 | Groq Llama |
|-----------|-------------|-------------|-------------------|-----------|------------|
| **Strict JSON schema** | ✅ Constrained decoding (100%) | ✅ Same | ✅ Good (~98%) | ❌ No native schema mode | ❌ No schema enforcement |
| **Latency** | Ultra-fast | Fast | Very fast | Fast | Ultra-fast |
| **Cost (input/1M)** | ~$0.10 | ~$0.40 | ~$0.10 | ~$0.25 | Varies |
| **Cost (output/1M)** | ~$0.40 | ~$1.60 | ~$0.40 | ~$1.25 | Varies |
| **Schema compliance** | 100% (token-level) | 100% (token-level) | ~98% | ~95% (tool use) | ~90% |

**Winner: GPT-4.1-nano** — cheapest, fastest, and **0% schema failure rate** via OpenAI's constrained decoding. The Director's job is analytical, not creative — nano's quality is more than sufficient.

**Runner-up: Gemini 2.0 Flash** — if the user wants to avoid OpenAI dependency.

### Backup Strategy
Allow any provider's model to serve as Director. The user can swap the Director model at runtime, just like they swap the Actor model today.

---

## What the Director Produces

The Director outputs a fixed JSON schema on every turn. This is the **Director Directive**:

```json
{
  "intent": {
    "type": "question | instruction | creative | debug | conversation",
    "complexity": "low | medium | high",
    "topic": "string — brief topic label"
  },
  "response_analysis": {
    "sentiment": "neutral | positive | negative | cautious",
    "confidence": 0.0-1.0,
    "contains_code": true/false,
    "language_detected": "en | hi | ar | ...",
    "key_entities": ["entity1", "entity2"]
  },
  "suggestions": {
    "follow_ups": ["suggested follow-up question 1", "..."],
    "related_topics": ["topic1", "topic2"]
  },
  "guardrails": {
    "flags": ["none | off_topic | sensitive | factual_risk"],
    "note": "optional explanation"
  }
}
```

This schema is **extensible** — users can provide their own Director schema via `--director-schema schema.json`, just like the existing `--json-schema` flag.

---

## Multi-Provider Architecture

The current code is Groq-only. To support GPT-4.1, Claude, Gemini, etc., we need a provider abstraction layer.

### Provider Interface

```python
class LLMProvider(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
    def stream_chat(self, messages, **kwargs) -> Iterator[StreamChunk]:
        """Streaming response — used by Actor."""
        ...

    @abstractmethod
    def json_chat(self, messages, schema: dict, **kwargs) -> dict:
        """Structured JSON response — used by Director."""
        ...

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        """Available models for this provider."""
        ...
```

### Providers to Implement

| Provider | SDK | Actor (streaming) | Director (JSON) | API Key Env Var |
|----------|-----|-------------------|-----------------|-----------------|
| **Groq** | `groq` | ✅ Current behavior | ✅ `json_schema` mode | `GROQ_API_KEY` |
| **OpenAI** | `openai` | ✅ Streaming API | ✅ Structured Outputs (constrained decoding) | `OPENAI_API_KEY` |
| **Anthropic** | `anthropic` | ✅ Streaming API | ✅ Tool use for structured output | `ANTHROPIC_API_KEY` |
| **Google** | `google-genai` | ✅ Streaming API | ✅ `response_schema` controlled generation | `GOOGLE_API_KEY` |

### Model Registry (expanded)

```python
PROVIDERS = {
    "groq": {
        "sdk": "groq",
        "models": {
            "qwen/qwen3-32b": { ... },       # existing
            "openai/gpt-oss-20b": { ... },    # existing
            "openai/gpt-oss-120b": { ... },   # existing
        }
    },
    "openai": {
        "sdk": "openai",
        "models": {
            "gpt-4.1": { "roles": ["actor", "director"], ... },
            "gpt-4.1-mini": { "roles": ["actor", "director"], ... },
            "gpt-4.1-nano": { "roles": ["director"], "default_director": True, ... },
        }
    },
    "anthropic": {
        "sdk": "anthropic",
        "models": {
            "claude-sonnet-4-6": { "roles": ["actor"], ... },
            "claude-haiku-4-5": { "roles": ["actor", "director"], ... },
        }
    },
    "google": {
        "sdk": "google-genai",
        "models": {
            "gemini-2.0-flash": { "roles": ["actor", "director"], ... },
            "gemini-2.5-pro": { "roles": ["actor"], ... },
        }
    },
}
```

---

## Parallel Execution

Both calls happen concurrently using `concurrent.futures.ThreadPoolExecutor`:

```python
import concurrent.futures

def handle_message(user_message):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Actor streams in main thread (needs stdout access)
        # Director runs in background thread
        director_future = executor.submit(director.analyze, messages, user_message)

        # Stream actor response (blocking, prints to terminal)
        actor_result = actor.stream_response(messages, user_message)

        # Collect director result (may already be done)
        director_result = director_future.result(timeout=30)

    # Display director metadata after stream completes
    display_director_insights(director_result)
```

The Actor streams to the terminal in real-time. The Director runs silently in a background thread. Once the Actor finishes streaming, the Director's metadata is displayed below the response.

---

## File Structure

```
groq-chatbot/
├── chatbot.py                  # CLI entry point (refactored from monolith)
├── providers/
│   ├── __init__.py
│   ├── base.py                 # LLMProvider ABC, StreamChunk, ModelInfo
│   ├── groq_provider.py        # Groq implementation (extracted from current code)
│   ├── openai_provider.py      # OpenAI implementation
│   ├── anthropic_provider.py   # Anthropic implementation
│   └── google_provider.py      # Google Gemini implementation
├── actor.py                    # Actor: streaming response handler
├── director.py                 # Director: JSON analysis engine
├── orchestrator.py             # Parallel execution coordinator
├── schemas/
│   └── director_default.json   # Default Director output schema
├── ui/
│   └── terminal.py             # ANSI color output, stats display
├── requirements.txt            # Updated with all provider SDKs
└── CLAUDE.md                   # Updated project docs
```

---

## Implementation Phases

### Phase 1: Provider Abstraction (refactor, no new features)
1. Extract `LLMProvider` base class with `stream_chat()` and `json_chat()`
2. Move current Groq logic into `GroqProvider`
3. Refactor `GroqChatbot` to use the provider interface
4. **Verify**: Existing Groq functionality works identically

### Phase 2: Add OpenAI Provider + Director
1. Implement `OpenAIProvider` with streaming + Structured Outputs
2. Build `Director` class that takes a provider + schema and returns JSON
3. Build `Orchestrator` that runs Actor + Director in parallel via ThreadPoolExecutor
4. Wire up GPT-4.1-nano as the default Director model
5. **Verify**: Actor streams while Director runs silently, metadata shows after

### Phase 3: Add Anthropic + Google Providers
1. Implement `AnthropicProvider` (streaming + tool-use JSON)
2. Implement `GoogleProvider` (streaming + controlled generation)
3. Add provider auto-detection based on available API keys
4. **Verify**: Mix-and-match (e.g., Claude Actor + GPT-4.1-nano Director)

### Phase 4: CLI + UX Polish
1. New CLI flags: `--actor`, `--director`, `--director-schema`
2. In-chat commands: `/actor`, `/director`, `/schema`
3. Director metadata display (follow-ups, entities, confidence)
4. Provider/model discovery: `/providers`, `/models`

### Phase 5: Benchmark System
1. Build `Benchmark` class — runs a standard prompt against all available models
2. Measure Actor metrics (TTFT, TTFR, TTFC, total, tok/s, cost) per model
3. Measure Director metrics (total time, schema compliance, quality, cost) per model
4. Measure combined Actor+Director parallel wall time for all viable pairs
5. CLI flag: `--benchmark` with `--benchmark-prompt` and `--benchmark-output`
6. In-chat: `/benchmark` command, `/stats` for session summary
7. Per-turn stats line showing both Actor and Director timings
8. Session summary on `/quit` with averages, min/max, totals, cost breakdown

---

## CLI Interface (Target)

```bash
# Current behavior preserved (Groq-only, no Director)
python chatbot.py --model qwen/qwen3-32b

# Actor-Director mode: Groq Actor + OpenAI Director
python chatbot.py --actor groq/qwen/qwen3-32b --director openai/gpt-4.1-nano

# Custom Director schema
python chatbot.py --actor groq/openai/gpt-oss-120b --director openai/gpt-4.1-nano --director-schema my_schema.json

# Cross-provider: Claude Actor + GPT Director
python chatbot.py --actor anthropic/claude-sonnet-4-6 --director openai/gpt-4.1-nano

# Auto-detect: picks best available Actor and Director from available API keys
python chatbot.py --auto

# Benchmark all available models
python chatbot.py --benchmark
python chatbot.py --benchmark --providers groq,openai --benchmark-prompt "Explain recursion"
python chatbot.py --benchmark --benchmark-output results.json
```

---

## In-Chat Commands (Target)

```
/actor [provider/model]     — Switch Actor model
/director [provider/model]  — Switch Director model
/director off               — Disable Director (stream-only mode)
/schema                     — Show current Director schema
/insights                   — Toggle Director insights display
/providers                  — List available providers and API key status
/models                     — List all available models across providers
/benchmark                  — Run full model comparison benchmark
/stats                      — Show session timing/cost summary
```

---

## Terminal Output (Target)

```
  You ❯ Explain quantum entanglement in simple terms

  💭 Reasoning:
  The user wants a simple explanation of quantum entanglement...

  🤖 Response:
  Imagine you have two coins that are magically linked...
  [streaming tokens appear here in real-time]

  ─────────────────────────────────────────────────────────────
  ⏱  TTFT: 45ms │ Total: 2.1s │ Actor: groq/qwen3-32b
  💭 Reasoning: ~89 words │ Content: 342 chars
  ─────────────────────────────────────────────────────────────
  📋 Director (gpt-4.1-nano, 380ms):
     Intent: question (medium) │ Topic: quantum physics
     Confidence: 0.85 │ Language: en
     Entities: quantum entanglement, superposition, Bell's theorem
     Follow-ups:
       → What is quantum superposition?
       → How is entanglement used in quantum computing?
  ─────────────────────────────────────────────────────────────
```

---

## Dependencies (Updated requirements.txt)

```
groq>=0.28.0          # Groq (existing)
openai>=1.30.0        # OpenAI (GPT-4.1 family)
anthropic>=0.30.0     # Anthropic (Claude family)
google-genai>=1.0.0   # Google (Gemini family)
streamlit>=1.35.0     # Streamlit UI (existing)
```

Only `groq` remains required. Other SDKs are optional — the system auto-detects which providers are available based on installed packages + API keys.

---

## Key Design Decisions

1. **GPT-4.1-nano as default Director** — cheapest, fastest, 100% schema compliance via constrained decoding. The Director's job is analytical, not creative.

2. **ThreadPoolExecutor, not asyncio** — the Actor needs synchronous stdout access for streaming. Threading is simpler and works well for 2 concurrent HTTP calls.

3. **Provider abstraction, not SDK abstraction** — each provider has quirks (Groq's `x_groq.usage`, OpenAI's `structured_outputs`, Anthropic's tool use). The abstraction is at the chat-completion level, not the HTTP level.

4. **Director is optional** — `--director off` or simply omitting the flag runs in legacy stream-only mode. Zero breaking changes.

5. **Director schema is customizable** — users can provide their own schema for domain-specific analysis (e.g., medical entity extraction, code review metadata).

6. **Single conversation history** — both Actor and Director share the same conversation history. The Director gets a system prompt explaining its analysis role, but sees the same user/assistant messages.

---

## Phase 5: Benchmark — Full Model Comparison Table

A built-in benchmarking system that runs a standardized prompt against **every available model** across **both modes** (streaming Actor + JSON Director) and produces a side-by-side comparison table with real measured timings.

### How It Works

```bash
# Run the benchmark
python chatbot.py --benchmark

# Benchmark specific providers only
python chatbot.py --benchmark --providers groq,openai

# Benchmark with a custom prompt
python chatbot.py --benchmark --benchmark-prompt "Explain recursion in 3 sentences"

# Export results to JSON
python chatbot.py --benchmark --benchmark-output results.json
```

In-chat: `/benchmark` command triggers it mid-session.

### What Gets Measured

For **every model** that has a valid API key configured:

| Metric | Streaming (Actor) | JSON (Director) | Description |
|--------|-------------------|-----------------|-------------|
| **TTFT** | ✅ | N/A | Time to first token — how fast the user sees something |
| **TTFR** | ✅ | N/A | Time to first reasoning token (if reasoning enabled) |
| **TTFC** | ✅ | N/A | Time to first content token (after reasoning completes) |
| **Total Time** | ✅ | ✅ | Wall-clock time from request to completion |
| **Tokens/sec** | ✅ | ✅ | Output throughput (completion tokens / total time) |
| **Total Tokens** | ✅ | ✅ | Prompt + completion + reasoning token counts |
| **Reasoning Tokens** | ✅ | ✅ | Tokens spent on chain-of-thought |
| **Content Tokens** | ✅ | ✅ | Tokens in the actual answer |
| **Schema Compliance** | N/A | ✅ | Did the JSON match the Director schema? (pass/fail + error) |
| **Cost** | ✅ | ✅ | Estimated cost based on model pricing |
| **Cache Hit** | ✅ | ✅ | Whether prompt caching was used |

### The Benchmark Table Output

```
══════════════════════════════════════════════════════════════════════════════════════════════════
  ACTOR-DIRECTOR BENCHMARK — "Explain quantum entanglement simply"
══════════════════════════════════════════════════════════════════════════════════════════════════

  ┌─ ACTOR MODE (Streaming) ──────────────────────────────────────────────────────────────────┐
  │                                                                                           │
  │  Model                      │ TTFT    │ TTFC    │ Total   │ Tok/s  │ Tokens │ Cost       │
  │  ───────────────────────────┼─────────┼─────────┼─────────┼────────┼────────┼────────────│
  │  groq/qwen3-32b             │  42ms   │  310ms  │  1.8s   │  412   │  743   │ $0.00065   │
  │  groq/gpt-oss-20b           │  18ms   │   18ms  │  0.7s   │ 1021   │  715   │ $0.00027   │
  │  groq/gpt-oss-120b          │  31ms   │  580ms  │  2.1s   │  498   │ 1046   │ $0.00078   │
  │  openai/gpt-4.1             │ 320ms   │  320ms  │  4.2s   │   78   │  328   │ $0.00328   │
  │  openai/gpt-4.1-mini        │ 180ms   │  180ms  │  2.8s   │  112   │  314   │ $0.00063   │
  │  anthropic/claude-sonnet-4-6│ 410ms   │  410ms  │  5.1s   │   62   │  316   │ $0.00570   │
  │  anthropic/claude-haiku-4-5 │ 190ms   │  190ms  │  2.2s   │  143   │  315   │ $0.00047   │
  │  google/gemini-2.0-flash    │  95ms   │   95ms  │  1.5s   │  210   │  315   │ $0.00016   │
  │                                                                                           │
  └───────────────────────────────────────────────────────────────────────────────────────────┘

  ┌─ DIRECTOR MODE (JSON, strict schema) ─────────────────────────────────────────────────────┐
  │                                                                                           │
  │  Model                      │ Total   │ Tok/s  │ Tokens │ Schema  │ Cost       │ Quality  │
  │  ───────────────────────────┼─────────┼────────┼────────┼─────────┼────────────┼──────────│
  │  openai/gpt-4.1-nano        │  0.4s   │  312   │  125   │ ✅ pass │ $0.00006   │ ★★★☆☆   │
  │  openai/gpt-4.1-mini        │  1.2s   │  104   │  125   │ ✅ pass │ $0.00025   │ ★★★★☆   │
  │  openai/gpt-4.1             │  2.1s   │   60   │  126   │ ✅ pass │ $0.00126   │ ★★★★★   │
  │  google/gemini-2.0-flash    │  0.6s   │  208   │  125   │ ✅ pass │ $0.00006   │ ★★★☆☆   │
  │  groq/gpt-oss-20b           │  0.3s   │  417   │  125   │ ⚠️ partial│ $0.00005 │ ★★★☆☆   │
  │  groq/gpt-oss-120b          │  0.8s   │  156   │  126   │ ✅ pass │ $0.00010   │ ★★★★☆   │
  │  anthropic/claude-haiku-4-5 │  0.9s   │  139   │  125   │ ⚠️ partial│ $0.00019 │ ★★★★☆   │
  │                                                                                           │
  └───────────────────────────────────────────────────────────────────────────────────────────┘

  ┌─ ACTOR + DIRECTOR PARALLEL (combined) ────────────────────────────────────────────────────┐
  │                                                                                           │
  │  Actor                   + Director              │ Wall Time │ User TTFT │ Combined Cost  │
  │  ────────────────────────┼───────────────────────┼───────────┼───────────┼────────────────│
  │  groq/qwen3-32b          │ openai/gpt-4.1-nano   │   1.8s    │   42ms    │ $0.00071      │
  │  groq/gpt-oss-20b        │ openai/gpt-4.1-nano   │   0.7s    │   18ms    │ $0.00033      │
  │  groq/gpt-oss-120b       │ openai/gpt-4.1-nano   │   2.1s    │   31ms    │ $0.00084      │
  │  openai/gpt-4.1-mini     │ openai/gpt-4.1-nano   │   2.8s    │  180ms    │ $0.00069      │
  │  anthropic/claude-sonnet  │ openai/gpt-4.1-nano   │   5.1s    │  410ms    │ $0.00576      │
  │  google/gemini-2.0-flash  │ openai/gpt-4.1-nano   │   1.5s    │   95ms    │ $0.00022      │
  │                                                                                           │
  │  ★ Best TTFT:      groq/gpt-oss-20b (18ms)                                               │
  │  ★ Best Total:     groq/gpt-oss-20b + gpt-4.1-nano (0.7s, $0.00033)                      │
  │  ★ Best Value:     google/gemini-2.0-flash + gpt-4.1-nano (1.5s, $0.00022)               │
  │  ★ Best Quality:   groq/gpt-oss-120b + gpt-4.1-nano (2.1s, $0.00084)                     │
  │                                                                                           │
  └───────────────────────────────────────────────────────────────────────────────────────────┘

  ══════════════════════════════════════════════════════════════════════════════════════════════
  Benchmark completed in 28.3s │ 8 models tested │ 3 providers │ 22 API calls
  ══════════════════════════════════════════════════════════════════════════════════════════════
```

### Benchmark Implementation

```python
class Benchmark:
    """Runs a standard prompt against all available models and collects timing data."""

    def __init__(self, providers: dict[str, LLMProvider], director_schema: dict):
        self.providers = providers
        self.director_schema = director_schema
        self.results = {"actor": [], "director": [], "combined": []}

    def run(self, prompt: str = "Explain quantum entanglement in simple terms",
            system_prompt: str = None) -> dict:
        """
        For each model:
        1. Run as Actor (streaming) — measure TTFT, TTFC, total, tokens/sec
        2. Run as Director (JSON) — measure total, schema compliance, quality
        3. Run best Actor + Director pairs in parallel — measure wall time
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Phase 1: Test each model as Actor (sequential, each needs stdout)
        for provider_name, provider in self.providers.items():
            for model in provider.list_models():
                if "actor" in model.roles:
                    result = self._benchmark_actor(provider, model, messages)
                    self.results["actor"].append(result)

        # Phase 2: Test each model as Director (can run in parallel)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = []
            for provider_name, provider in self.providers.items():
                for model in provider.list_models():
                    if "director" in model.roles:
                        f = pool.submit(self._benchmark_director, provider, model, messages)
                        futures.append(f)
            for f in concurrent.futures.as_completed(futures):
                self.results["director"].append(f.result())

        # Phase 3: Test Actor+Director pairs in parallel
        self._benchmark_combined(messages)

        return self.results

    def _benchmark_actor(self, provider, model, messages) -> dict:
        """Stream a response, capture TTFT/TTFC/total/throughput."""
        start = time.perf_counter()
        ttft = ttfc = None
        token_count = 0
        reasoning_tokens = 0

        for chunk in provider.stream_chat(model.id, messages):
            now = time.perf_counter()
            if ttft is None:
                ttft = now - start
            if chunk.is_content and ttfc is None:
                ttfc = now - start
            if chunk.is_content:
                token_count += 1
            if chunk.is_reasoning:
                reasoning_tokens += 1

        total = time.perf_counter() - start
        return {
            "provider": provider.name,
            "model": model.id,
            "ttft_ms": ttft * 1000 if ttft else None,
            "ttfc_ms": ttfc * 1000 if ttfc else None,
            "total_ms": total * 1000,
            "tokens_per_sec": token_count / total if total > 0 else 0,
            "total_tokens": token_count + reasoning_tokens,
            "reasoning_tokens": reasoning_tokens,
            "content_tokens": token_count,
            "cost": model.estimate_cost(token_count + reasoning_tokens, token_count),
        }

    def _benchmark_director(self, provider, model, messages) -> dict:
        """JSON request, check schema compliance."""
        start = time.perf_counter()
        result = provider.json_chat(model.id, messages, schema=self.director_schema)
        total = time.perf_counter() - start

        schema_ok = self._validate_schema(result.content, self.director_schema)

        return {
            "provider": provider.name,
            "model": model.id,
            "total_ms": total * 1000,
            "tokens_per_sec": result.completion_tokens / total if total > 0 else 0,
            "total_tokens": result.total_tokens,
            "schema_pass": schema_ok,
            "cost": model.estimate_cost(result.prompt_tokens, result.completion_tokens),
        }
```

### Benchmark Metrics Glossary

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| **TTFT** (Time to First Token) | Latency from request sent → first token received | UX feel — how long the user stares at a blank screen |
| **TTFR** (Time to First Reasoning) | Latency from request → first reasoning token | How fast the model starts "thinking" |
| **TTFC** (Time to First Content) | Latency from request → first non-reasoning content token | The real "wait" for the user (reasoning is dimmed/hidden) |
| **Total Time** | Wall-clock start to finish | End-to-end latency |
| **Tokens/sec** | Completion tokens / total time | Throughput — how fast tokens flow |
| **Schema Compliance** | Does Director JSON match the schema exactly? | Reliability — failed schema = useless Director turn |
| **Cost** | (input_tokens × input_price + output_tokens × output_price) / 1M | Budget planning |
| **Wall Time (Combined)** | max(Actor time, Director time) — since they run in parallel | The *actual* user-perceived latency in Actor-Director mode |
| **Quality Score** | Richness and accuracy of Director's JSON analysis | A nano model might comply with schema but give shallow analysis |

### Per-Turn Stats (Always Visible)

Every turn in the chat shows a compact stats line. When Actor-Director mode is active, it shows both:

```
──────────────────────────────────────────────────────────────────────
⏱  Actor:    TTFT 42ms │ TTFC 310ms │ Total 1.8s │ 412 tok/s │ $0.0007
📋 Director: Total 380ms │ Schema ✅ │ $0.0001
⚡ Parallel:  Wall 1.8s (Director finished 1.4s before Actor)
💰 Turn cost: $0.0008 │ Session: $0.0142 (23 turns)
──────────────────────────────────────────────────────────────────────
```

### Session Summary (on `/quit` or `/stats`)

```
══════════════════════════════════════════════════════════════════
  SESSION SUMMARY — 23 turns, 12m 34s
══════════════════════════════════════════════════════════════════
  Actor:     groq/qwen3-32b
  Director:  openai/gpt-4.1-nano

  Actor Stats:
    Avg TTFT:      38ms (min 18ms, max 89ms)
    Avg TTFC:      295ms (min 180ms, max 620ms)
    Avg Total:     1.9s (min 0.4s, max 4.2s)
    Avg Tok/s:     398 (min 210, max 520)
    Total tokens:  17,842 (reasoning: 4,210, content: 13,632)
    Cache hit rate: 87% (20/23 requests)

  Director Stats:
    Avg Total:     420ms (min 280ms, max 810ms)
    Schema pass:   23/23 (100%)
    Total tokens:  2,875

  Combined:
    Total cost:    $0.0142
    Actor cost:    $0.0128 (90%)
    Director cost: $0.0014 (10%)
══════════════════════════════════════════════════════════════════
```

---

## Phase 6: Streamlit UI — Actor-Director + Benchmark Dashboard

The existing Streamlit app (`app.py`) has two modes: **Chat** and **Compare Models**. We expand this to a full Actor-Director UI with live benchmark visualization.

### Updated Streamlit Modes

| Mode | Current | After |
|------|---------|-------|
| **Chat** | Single Groq model, streaming or JSON | Actor-Director: streaming Actor + Director insights panel |
| **Compare** | 3 Groq models side-by-side | All available models across all providers, Actor + Director modes |
| **Benchmark** | ❌ Does not exist | Full benchmark dashboard with interactive charts and tables |
| **Dashboard** | ❌ Does not exist | Live session stats, cost tracking, timing history |

### Mode 1: Actor-Director Chat

The chat view gets a **split layout** — the streaming response on the left, Director insights on the right. The Director panel populates as soon as the JSON arrives (often before the Actor finishes streaming).

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  ⚡ Actor-Director Chat                                                 [Settings] │
├───────────────────────────────────────────┬─────────────────────────────────────────┤
│                                           │                                         │
│  You: Explain quantum entanglement        │  📋 Director Insights                  │
│                                           │  ─────────────────────                  │
│  💭 Reasoning                             │                                         │
│  ▸ The user wants a simple explanation... │  Intent: question (medium)              │
│                                           │  Topic: quantum physics                 │
│  🤖 Response                              │  Confidence: 0.85                       │
│  Imagine you have two coins that are      │  Language: en                           │
│  magically linked. When you flip one      │                                         │
│  and it lands on heads, the other         │  Key Entities:                          │
│  instantly becomes tails — no matter      │  • quantum entanglement                 │
│  how far apart they are. ▌                │  • superposition                        │
│                                           │  • Bell's theorem                       │
│                                           │                                         │
│                                           │  Suggested Follow-ups:                  │
│                                           │  ❯ What is quantum superposition?       │
│                                           │  ❯ How is entanglement used in QC?      │
│                                           │                                         │
│                                           │  ⚠️ Guardrails: none                    │
│                                           │                                         │
├───────────────────────────────────────────┴─────────────────────────────────────────┤
│ ⏱ Actor: TTFT 42ms │ TTFC 310ms │ 1.8s │ 412 tok/s │ 📋 Director: 380ms │ ✅     │
│ 💰 Turn: $0.0008 │ Session: $0.0142 (23 turns)                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  Type your message...                                                    [Send]    │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

#### Streamlit Implementation

```python
# Chat layout: 70/30 split
actor_col, director_col = st.columns([7, 3])

with actor_col:
    # Streaming response (existing pattern, enhanced)
    reasoning_ph = st.empty()
    content_ph = st.empty()

    for chunk in actor_engine.stream_response(prompt):
        # ... same streaming logic as current app.py ...

with director_col:
    st.subheader("📋 Director Insights")
    director_ph = st.empty()

    # Director result arrives from the background thread
    director_result = director_future.result(timeout=30)
    with director_ph.container():
        render_director_insights(director_result)
```

#### Director Insights Component

The Director panel is a reusable Streamlit component:

```python
def render_director_insights(data: dict):
    """Render Director JSON as a rich Streamlit panel."""

    # Intent badge
    intent = data.get("intent", {})
    col1, col2 = st.columns(2)
    col1.metric("Intent", intent.get("type", "-"))
    col2.metric("Complexity", intent.get("complexity", "-"))
    st.caption(f"Topic: {intent.get('topic', '-')}")

    st.divider()

    # Response analysis
    analysis = data.get("response_analysis", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence", f"{analysis.get('confidence', 0):.0%}")
    col2.metric("Sentiment", analysis.get("sentiment", "-"))
    col3.metric("Language", analysis.get("language_detected", "-"))

    if analysis.get("key_entities"):
        st.markdown("**Key Entities:**")
        for entity in analysis["key_entities"]:
            st.markdown(f"• {entity}")

    st.divider()

    # Suggested follow-ups (clickable!)
    suggestions = data.get("suggestions", {})
    if suggestions.get("follow_ups"):
        st.markdown("**Suggested Follow-ups:**")
        for q in suggestions["follow_ups"]:
            # Clicking sets the chat input
            if st.button(f"❯ {q}", key=f"followup_{hash(q)}"):
                st.session_state.next_prompt = q
                st.rerun()

    # Guardrails
    guardrails = data.get("guardrails", {})
    flags = guardrails.get("flags", [])
    if flags and flags != ["none"]:
        st.warning(f"⚠️ Flags: {', '.join(flags)}")
        if guardrails.get("note"):
            st.caption(guardrails["note"])
    else:
        st.success("✅ No guardrail flags")
```

#### Clickable Follow-ups

The Director's `follow_ups` render as **clickable buttons**. When clicked, they auto-fill the chat input and trigger a new turn — creating a guided conversation loop:

```
User asks → Actor streams → Director suggests follow-ups → User clicks one → loop
```

This is the core UX advantage of the Director in the Streamlit UI — it transforms a blank chat input into a guided exploration.

### Mode 2: Enhanced Compare Mode

The current Compare mode sends the same prompt to 3 Groq models. The enhanced version:
- Supports **all providers** (not just Groq)
- Tests both **Actor mode** (streaming) and **Director mode** (JSON)
- Shows a **unified comparison table** with all metrics

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  Model Comparison — All Providers                                                    │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Prompt: "Explain quantum entanglement in simple terms"                             │
│                                                                                      │
│  [Tab: Actor (Streaming)]  [Tab: Director (JSON)]  [Tab: Combined]                  │
│                                                                                      │
│  ┌─ Actor Results ──────────────────────────────────────────────────────────────┐    │
│  │                                                                              │    │
│  │  ┌─ groq/qwen3-32b ──┐  ┌─ groq/gpt-oss-20b ─┐  ┌─ openai/gpt-4.1 ────┐  │    │
│  │  │  TTFT: 42ms        │  │  TTFT: 18ms         │  │  TTFT: 320ms        │  │    │
│  │  │  Total: 1.8s       │  │  Total: 0.7s        │  │  Total: 4.2s        │  │    │
│  │  │  Tok/s: 412        │  │  Tok/s: 1021        │  │  Tok/s: 78          │  │    │
│  │  │  Cost: $0.0007     │  │  Cost: $0.0003      │  │  Cost: $0.0033      │  │    │
│  │  │                    │  │                     │  │                     │  │    │
│  │  │  💭 Reasoning...   │  │  💭 Reasoning...    │  │  Response text...   │  │    │
│  │  │  Response text...  │  │  Response text...   │  │                     │  │    │
│  │  └────────────────────┘  └─────────────────────┘  └─────────────────────┘  │    │
│  └──────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  Comparison Summary                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────────┐    │
│  │ Model              │ TTFT  │ TTFC  │ Total │ Tok/s │ Tokens │ Cached │ Cost │    │
│  │ groq/qwen3-32b     │ 42ms  │ 310ms │ 1.8s  │  412  │  743   │ 87%    │$0.07│    │
│  │ groq/gpt-oss-20b   │ 18ms  │  18ms │ 0.7s  │ 1021  │  715   │ 92%    │$0.03│    │
│  │ groq/gpt-oss-120b  │ 31ms  │ 580ms │ 2.1s  │  498  │ 1046   │ 89%    │$0.08│    │
│  │ openai/gpt-4.1     │320ms  │ 320ms │ 4.2s  │   78  │  328   │  0%    │$0.33│    │
│  │ openai/gpt-4.1-mini│180ms  │ 180ms │ 2.8s  │  112  │  314   │  0%    │$0.06│    │
│  │ anthropic/sonnet    │410ms  │ 410ms │ 5.1s  │   62  │  316   │  0%    │$0.57│    │
│  │ google/gemini-flash │ 95ms  │  95ms │ 1.5s  │  210  │  315   │  0%    │$0.02│    │
│  └──────────────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### Mode 3: Benchmark Dashboard (New Page)

A dedicated benchmark page with **interactive Plotly charts** and **exportable data**.

#### Layout

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  📊 Benchmark Dashboard                                                              │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Benchmark Prompt: [Explain quantum entanglement in simple terms    ] [▶ Run]        │
│  Providers: [✅ Groq] [✅ OpenAI] [☐ Anthropic] [☐ Google]                           │
│                                                                                      │
│  ┌─ TTFT Comparison (bar chart) ────────────────────────────────────────────────┐    │
│  │                                                                              │    │
│  │  groq/gpt-oss-20b    ██ 18ms                                                │    │
│  │  groq/gpt-oss-120b   ███ 31ms                                               │    │
│  │  groq/qwen3-32b      ████ 42ms                                              │    │
│  │  google/gemini-flash  █████████ 95ms                                         │    │
│  │  openai/gpt-4.1-mini ██████████████████ 180ms                               │    │
│  │  openai/gpt-4.1      ████████████████████████████████ 320ms                  │    │
│  │  anthropic/sonnet     █████████████████████████████████████████ 410ms         │    │
│  │                                                                              │    │
│  └──────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  ┌─ Cost vs Speed (scatter) ───────┐  ┌─ Throughput (tok/s bar) ────────────┐       │
│  │                                 │  │                                      │       │
│  │  Cost ↑                         │  │  groq/gpt-oss-20b     ████████ 1021 │       │
│  │  │  ● sonnet                    │  │  groq/gpt-oss-120b    ████ 498      │       │
│  │  │                              │  │  groq/qwen3-32b       ███ 412       │       │
│  │  │     ● gpt-4.1                │  │  google/gemini-flash   ██ 210       │       │
│  │  │                              │  │  anthropic/haiku       █ 143        │       │
│  │  │ ● qwen  ● 120b              │  │  openai/gpt-4.1-mini  █ 112        │       │
│  │  │  ● mini  ● flash             │  │  openai/gpt-4.1        78          │       │
│  │  │    ● 20b                     │  │  anthropic/sonnet       62          │       │
│  │  └──────────────── Speed →      │  └──────────────────────────────────────┘       │
│  │                                 │                                                 │
│  └─────────────────────────────────┘                                                 │
│                                                                                      │
│  ┌─ Director Schema Compliance ─────────────────────────────────────────────────┐    │
│  │                                                                              │    │
│  │  openai/gpt-4.1-nano  ✅ 100%  │ 0.4s  │ $0.00006  │ ★★★☆☆               │    │
│  │  openai/gpt-4.1-mini  ✅ 100%  │ 1.2s  │ $0.00025  │ ★★★★☆               │    │
│  │  openai/gpt-4.1       ✅ 100%  │ 2.1s  │ $0.00126  │ ★★★★★               │    │
│  │  google/gemini-flash   ✅ pass  │ 0.6s  │ $0.00006  │ ★★★☆☆               │    │
│  │  groq/gpt-oss-20b     ⚠️  92%  │ 0.3s  │ $0.00005  │ ★★★☆☆               │    │
│  │  anthropic/haiku       ⚠️  95%  │ 0.9s  │ $0.00019  │ ★★★★☆               │    │
│  └──────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  ┌─ Best Pairs (Actor + Director) ──────────────────────────────────────────────┐    │
│  │                                                                              │    │
│  │  ★ Fastest:   groq/gpt-oss-20b + gpt-4.1-nano     │ 0.7s  │ $0.0003       │    │
│  │  ★ Cheapest:  google/gemini-flash + gpt-4.1-nano   │ 1.5s  │ $0.0002       │    │
│  │  ★ Quality:   groq/gpt-oss-120b + gpt-4.1-nano     │ 2.1s  │ $0.0008       │    │
│  └──────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  [📥 Export JSON]  [📥 Export CSV]                                                   │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

#### Charts (Plotly)

| Chart | Type | X-axis | Y-axis | Color |
|-------|------|--------|--------|-------|
| **TTFT Comparison** | Horizontal bar | Model name | TTFT (ms) | Provider |
| **Cost vs Speed** | Scatter | Total time (ms) | Cost ($) | Provider, size = tokens |
| **Throughput** | Horizontal bar | Model name | Tokens/sec | Provider |
| **Timing Breakdown** | Stacked bar | Model name | TTFT + reasoning + content time | Phase |
| **Director Compliance** | Table with progress bars | Model name | Schema pass rate | Pass/Fail color |
| **Session Timeline** | Line chart | Turn number | TTFT/Total per turn | Actor vs Director |

#### Streamlit Implementation

```python
import plotly.express as px
import plotly.graph_objects as go

def render_benchmark_dashboard(results: dict):
    """Full benchmark page with interactive charts."""

    # TTFT bar chart
    st.subheader("Time to First Token (TTFT)")
    actor_df = pd.DataFrame(results["actor"])
    actor_df = actor_df.sort_values("ttft_ms")
    fig = px.bar(actor_df, x="ttft_ms", y="model", orientation="h",
                 color="provider", title="TTFT by Model",
                 labels={"ttft_ms": "TTFT (ms)", "model": ""})
    st.plotly_chart(fig, use_container_width=True)

    # Cost vs Speed scatter
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cost vs Speed")
        fig = px.scatter(actor_df, x="total_ms", y="cost",
                         size="total_tokens", color="provider",
                         hover_name="model", title="Cost vs Latency",
                         labels={"total_ms": "Total Time (ms)", "cost": "Cost ($)"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Throughput (tok/s)")
        fig = px.bar(actor_df.sort_values("tokens_per_sec"),
                     x="tokens_per_sec", y="model", orientation="h",
                     color="provider", title="Output Throughput")
        st.plotly_chart(fig, use_container_width=True)

    # Director compliance table
    st.subheader("Director: Schema Compliance")
    director_df = pd.DataFrame(results["director"])
    st.dataframe(director_df[["model", "total_ms", "schema_pass", "cost"]],
                 use_container_width=True, hide_index=True)

    # Best pairs
    st.subheader("Best Actor + Director Pairs")
    combined_df = pd.DataFrame(results["combined"])
    col1, col2, col3 = st.columns(3)
    fastest = combined_df.loc[combined_df["wall_ms"].idxmin()]
    cheapest = combined_df.loc[combined_df["combined_cost"].idxmin()]
    col1.metric("⚡ Fastest", fastest["actor_model"],
                delta=f"{fastest['wall_ms']:.0f}ms, ${fastest['combined_cost']:.4f}")
    col2.metric("💰 Cheapest", cheapest["actor_model"],
                delta=f"{cheapest['wall_ms']:.0f}ms, ${cheapest['combined_cost']:.4f}")

    # Export
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Export JSON",
                           json.dumps(results, indent=2),
                           "benchmark_results.json", "application/json")
    with col2:
        csv = actor_df.to_csv(index=False)
        st.download_button("📥 Export CSV", csv,
                           "benchmark_results.csv", "text/csv")
```

### Mode 4: Session Dashboard (Sidebar + Page)

A **live session stats panel** in the sidebar that updates every turn, plus a full **dashboard page** accessible via `/stats`.

#### Sidebar Stats (always visible during chat)

```python
with st.sidebar:
    st.divider()
    st.subheader("📊 Session Stats")

    session = st.session_state.session_stats

    # Compact metrics
    col1, col2 = st.columns(2)
    col1.metric("Turns", session["turn_count"])
    col2.metric("Session Cost", f"${session['total_cost']:.4f}")

    # Actor avg TTFT sparkline
    if session["ttft_history"]:
        st.caption("TTFT trend:")
        st.line_chart(session["ttft_history"], height=60)

    # Director pass rate
    if session["director_enabled"]:
        pass_rate = session["director_passes"] / max(1, session["turn_count"])
        st.progress(pass_rate, text=f"Director schema: {pass_rate:.0%}")

    # Cost breakdown
    st.caption(f"Actor: ${session['actor_cost']:.4f} ({session['actor_cost_pct']:.0f}%)")
    st.caption(f"Director: ${session['director_cost']:.4f} ({session['director_cost_pct']:.0f}%)")
```

#### Session Stats Data Model

```python
# Stored in st.session_state.session_stats
session_stats = {
    "turn_count": 0,
    "start_time": time.time(),

    # Actor metrics (lists for history/sparklines)
    "ttft_history": [],          # [42, 38, 55, 31, ...]
    "ttfc_history": [],
    "total_history": [],
    "throughput_history": [],

    # Director metrics
    "director_enabled": True,
    "director_total_history": [],
    "director_passes": 0,

    # Cost tracking
    "total_cost": 0.0,
    "actor_cost": 0.0,
    "director_cost": 0.0,
    "actor_cost_pct": 0,
    "director_cost_pct": 0,

    # Token tracking
    "total_actor_tokens": 0,
    "total_director_tokens": 0,
    "total_cached_tokens": 0,
    "cache_hit_rate": 0.0,
}
```

### Updated Sidebar Layout

```
┌─────────────────────────────────┐
│  ⚡ Actor-Director Chatbot      │
│                                 │
│  Mode: [Chat ▾]                 │
│                                 │
│  ── Actor ──────────────────    │
│  Provider: [Groq ▾]            │
│  Model: [qwen3-32b ▾]          │
│  Reasoning: [default ▾]        │
│                                 │
│  ── Director ───────────────    │
│  [✅ Enable Director]           │
│  Provider: [OpenAI ▾]          │
│  Model: [gpt-4.1-nano ▾]      │
│  Schema: director_default.json  │
│  [Upload Custom Schema]        │
│                                 │
│  ── Settings ───────────────    │
│  Show Reasoning [✅]            │
│  Show Director Panel [✅]       │
│  Max Tokens: [4096]            │
│                                 │
│  ── Providers ──────────────    │
│  Groq: ✅ API key set           │
│  OpenAI: ✅ API key set         │
│  Anthropic: ❌ No key           │
│  Google: ❌ No key              │
│                                 │
│  ── Session Stats ──────────    │
│  Turns: 23 │ Cost: $0.0142     │
│  TTFT trend: ▁▂▁▃▁▂▁           │
│  Director: 23/23 schema ✅     │
│  Cache rate: 87%               │
│                                 │
│  [Clear Chat] [Run Benchmark]  │
└─────────────────────────────────┘
```

### File Structure Update

```
groq-chatbot/
├── app.py                     # Streamlit entry point (refactored)
├── ui/
│   ├── __init__.py
│   ├── chat_page.py           # Actor-Director chat mode
│   ├── compare_page.py        # Multi-provider compare mode
│   ├── benchmark_page.py      # Benchmark dashboard with charts
│   ├── components.py          # Shared: director panel, stats bar, follow-up buttons
│   └── sidebar.py             # Provider config, session stats, settings
├── ...
```

### Dependencies (UI additions)

```
plotly>=5.18.0         # Interactive charts in benchmark dashboard
pandas>=2.0.0          # DataFrames for comparison tables & exports
```

### Implementation Order within Phase 6

1. **Sidebar refactor** — Actor/Director model pickers, provider status indicators
2. **Director insights component** — `render_director_insights()` with clickable follow-ups
3. **Chat page** — 70/30 split layout, parallel Actor stream + Director thread
4. **Compare page** — Extend to all providers, add Actor/Director tabs
5. **Benchmark page** — Run benchmark, render Plotly charts, export buttons
6. **Session stats** — Live sidebar stats, TTFT sparkline, cost breakdown
7. **Polish** — Loading states, error handling, responsive layout

---

## Open Questions

1. **Should the Director see the Actor's response?** Two options:
   - **Pre-response**: Director analyzes the user message + history (current plan — both run in parallel)
   - **Post-response**: Director analyzes after the Actor finishes (adds latency but can evaluate the actual response)
   - **Recommendation**: Start with pre-response (parallel). Add post-response as an optional second pass later.

2. **Director system prompt**: Should it be hardcoded or user-configurable?
   - **Recommendation**: Ship a good default, allow override via `--director-prompt`.

3. **Cost tracking**: Show combined Actor + Director costs per turn?
   - **Recommendation**: Yes, break down by role in the stats line.

4. **Streamlit UI**: The current project has a Streamlit frontend. Should the Director metadata render in the Streamlit UI too?
   - **Recommendation**: Yes, in Phase 4. Director insights as an expandable sidebar panel.
