# Groq Chatbot Optimization Plan

> Goal: Transform from simple chatbot into an optimized LLM inference layer with structured output, large system prompts, caching, and minimal latency — all on the free tier.

---

## Free Tier Reality Check

**The models you're using (`qwen/qwen3-32b`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b`) may not be available on the free plan.** The free tier docs only list Llama and a few other models. Check your actual limits at `console.groq.com/settings/limits`.

If these models ARE available to you, here are the known free-tier constraints:

| Constraint | Typical Free Limit |
|---|---|
| Requests/min | 30 RPM |
| Requests/day | 1,000–14,400 RPD (model-dependent) |
| Tokens/min | 6,000–30,000 TPM |
| Tokens/day | 100,000–500,000 TPD |

**Maximizing free tier usage:**
- Prompt caching works on the free tier (GPT-OSS models) — no tier restriction
- Cached tokens do NOT count toward rate limits (huge win for large system prompts)
- Keep `max_completion_tokens` tight — don't request 65K if you need 2K
- Batch related questions into single requests where possible
- Prefer `gpt-oss-20b` ($0.075/M in) over `qwen3-32b` ($0.29/M in) when multilingual isn't needed

---

## Phase 1: JSON Structured Output

**What:** Add `response_format` support so the chatbot can return structured JSON responses on demand.

**Key constraint: Streaming and Structured Outputs are mutually exclusive on Groq.** You must choose per-request:
- Streaming mode (current) — for interactive chat, tokens appear in real-time
- Structured mode (new) — for JSON output, waits for full response then parses

**Implementation:**

1. Add a `--json-schema` CLI flag that accepts a path to a JSON schema file
2. Add an `/output [json|stream]` in-chat command to toggle modes
3. Modify `_build_request_params()` to conditionally add `response_format`:

```python
# When JSON mode is active:
params["stream"] = False  # required — streaming not supported with structured output
params["response_format"] = {
    "type": "json_schema",
    "json_schema": {
        "name": "response",
        "strict": True,  # only on GPT-OSS models
        "schema": self.json_schema
    }
}
```

4. Add a non-streaming response handler alongside `stream_response()`:

```python
def structured_response(self, user_message: str) -> dict:
    """Send message and get structured JSON back (no streaming)."""
    # ... build params with response_format ...
    response = self.client.chat.completions.create(**params)
    return json.loads(response.choices[0].message.content)
```

**Model compatibility:**
| Feature | gpt-oss-20b | gpt-oss-120b | qwen3-32b |
|---|---|---|---|
| `json_schema` + `strict: true` | Yes | Yes | No |
| `json_object` (basic) | Yes | Yes | Needs testing |

**Files to change:** `chatbot.py` — add `structured_response()`, modify `_build_request_params()`, add CLI arg, add `/output` command.

---

## Phase 2: Large System Prompt Support (40-50K tokens)

**What:** Support system prompts up to 40-50K tokens loaded from external files.

**Context budget (131K window):**
```
System prompt:    ~50,000 tokens
Conversation:    ~15,000 tokens (history)
User message:     ~1,000 tokens
Max completion:   ~65,000 tokens (gpt-oss) / ~40,000 tokens (qwen3)
                 ─────────────
Total:           ~131,000 tokens
```

**Implementation:**

1. Add `--system-prompt` flag to load from file:

```python
parser.add_argument("--system-prompt", "-s", type=str,
    help="Path to system prompt file (supports .txt, .md)")
```

2. Add system prompt to message history:

```python
def __init__(self, ..., system_prompt_path: str = None):
    self.system_prompt = ""
    if system_prompt_path:
        with open(system_prompt_path) as f:
            self.system_prompt = f.read()
    # Always prepend system message
    if self.system_prompt:
        self.messages = [{"role": "system", "content": self.system_prompt}]
```

3. Add token counting/estimation to warn when approaching limits:

```python
def estimate_tokens(self, text: str) -> int:
    """Rough estimate: 1 token ≈ 4 chars for English."""
    return len(text) // 4

def check_context_budget(self):
    total = sum(self.estimate_tokens(m["content"]) for m in self.messages)
    remaining = 131072 - total - self.model_info["max_completion"]
    if remaining < 2000:
        print(f"⚠ Context nearly full: ~{total} tokens used")
```

4. Add `/system` command to view/reload system prompt info at runtime.

**Files to change:** `chatbot.py` — modify `__init__`, `_build_request_params()`, add token estimation, add CLI arg.

---

## Phase 3: Prompt Caching Optimization

**What:** Leverage Groq's automatic prompt caching for large system prompts.

**How caching works on Groq:**
- **Automatic** — no code changes needed for basic caching
- **Supported models:** `openai/gpt-oss-20b`, `openai/gpt-oss-120b` only (NOT qwen3)
- **50% cost reduction** on cached input tokens
- **Cached tokens don't count toward rate limits** (critical for free tier)
- **Requires exact prefix match** from the start of the prompt
- **Cache TTL:** 2 hours without use
- **Min cacheable prefix:** 128–1,024 tokens

**Optimization strategy:**

1. **Keep system prompt as the first message and never change it** — this ensures the prefix match works across all requests in a conversation.

2. **Structure messages for maximum cache hits:**
```
[system prompt]          ← cached after first request (50K tokens free on subsequent calls)
[user msg 1]
[assistant msg 1]
[user msg 2]             ← only this + completion tokens count toward rate limits
```

3. **Add cache monitoring** — read `usage` from API response:

```python
# In response handling:
if hasattr(chunk, 'x_groq') and chunk.x_groq:
    usage = chunk.x_groq.usage
    if usage:
        cached = getattr(usage, 'prompt_tokens_details', {})
        # Log: cached_tokens vs total prompt_tokens
```

4. **Add `/cache` command** to show cache stats (tokens saved, cost saved).

5. **Warm-up request** — optionally send a minimal first request on startup to prime the cache:

```python
def warm_cache(self):
    """Send a lightweight request to prime the system prompt cache."""
    self.client.chat.completions.create(
        model=self.model_id,
        messages=[
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "hi"}
        ],
        max_completion_tokens=1
    )
```

**Impact on free tier:**
- A 50K-token system prompt normally costs ~50K tokens per request against your TPM/TPD limits
- With caching: first request costs 50K, subsequent requests cost ~0 for the system prompt portion
- This effectively multiplies your free tier capacity by 10-50x for repeated conversations

**Files to change:** `chatbot.py` — add cache stats tracking, warm-up logic, `/cache` command.

---

## Phase 4: Latency Optimization

**What:** Minimize TTFT and total response time.

**Changes (ordered by impact):**

1. **Reduce input tokens** — TTFT scales linearly with input size
   - Implement conversation history trimming (keep last N turns, or last M tokens)
   - Summarize old conversation history instead of sending raw messages

2. **Tighten `max_completion_tokens`** — currently hardcoded to 4096
   - Add `--max-tokens` CLI flag
   - Default to smaller values for chat (1024-2048), larger for JSON output

3. **Use async client for non-blocking operations:**

```python
from groq import AsyncGroq

class GroqChatbot:
    def __init__(self, ...):
        self.async_client = AsyncGroq(api_key=api_key)
```

4. **Connection reuse** — the Groq SDK already handles this via `httpx`, but ensure you're reusing the client instance (already done).

5. **Model selection guidance in code:**
   - `gpt-oss-20b`: lowest TTFT, use for English speed-critical
   - `gpt-oss-120b`: best reasoning, use for complex tasks
   - `qwen3-32b`: multilingual, use for non-English

**Files to change:** `chatbot.py` — add history trimming, configurable max tokens, async support.

---

## Phase 5: Architecture Refactor (LLM Service Layer)

**What:** Refactor from single-file chatbot into a reusable LLM inference service.

```
groq-chatbot/
├── groq_service/
│   ├── __init__.py
│   ├── client.py          # GroqClient — core API wrapper with caching, structured output
│   ├── models.py          # Model registry and capabilities
│   ├── streaming.py       # Stream handler (reasoning parsing, token display)
│   ├── structured.py      # JSON schema validation and structured output
│   ├── cache.py           # Cache monitoring and warm-up
│   └── config.py          # Configuration and defaults
├── chatbot.py             # CLI chatbot (thin layer over groq_service)
├── requirements.txt
└── .env.example
```

**Key design:**
- `GroqClient` exposes two methods: `stream()` and `structured()`
- Other applications import `groq_service` directly, bypassing the chatbot CLI
- System prompts loaded from files, cached automatically
- JSON schemas validated before sending to API

**Do this phase only after phases 1-4 are validated and working.**

---

## Implementation Order

| Phase | Effort | Impact | Dependency |
|---|---|---|---|
| Phase 1: JSON Structured Output | Medium | High | None |
| Phase 2: Large System Prompts | Low | High | None |
| Phase 3: Prompt Caching | Low | Very High (free tier) | Phase 2 |
| Phase 4: Latency Optimization | Medium | Medium | Phases 2+3 |
| Phase 5: Refactor | High | Maintainability | Phases 1-4 |

**Recommended start:** Phase 2 → Phase 3 → Phase 1 → Phase 4 → Phase 5

Rationale: System prompt support + caching gives the biggest immediate win for free tier usage. JSON structured output is next since it changes the API calling pattern. Latency optimization and refactor build on top.

---

## Critical Constraints to Remember

1. **Streaming + Structured Output are mutually exclusive** — must toggle per-request
2. **`strict: true` only works on GPT-OSS models** — not on Qwen3
3. **Prompt caching only on GPT-OSS models** — Qwen3 gets no caching benefit
4. **Cache requires exact prefix match** — never modify the system prompt mid-conversation
5. **Free tier rate limits are per-organization**, not per-key
6. **Cache TTL is 2 hours** — long idle periods reset the cache
7. **Verify your free tier actually has access** to these models at `console.groq.com/settings/limits`
