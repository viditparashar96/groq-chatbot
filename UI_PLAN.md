# UI Implementation Plan — Streamlit Web Interface

## Goal
Build a Streamlit web UI that mirrors all CLI features and adds **side-by-side async model comparison** — send one prompt to all 3 models simultaneously and compare responses, timing, reasoning, and cache stats in real-time.

## Tech Stack
- **Streamlit** — Python-native UI, built-in chat components, streaming support
- **asyncio + threading** — Run 3 model calls in parallel for comparison mode
- **Existing `GroqChatbot` class** — Reuse the core engine from `chatbot.py` (no duplication)

## File Structure
```
groq-chatbot/
├── chatbot.py              # CLI chatbot (unchanged)
├── app.py                  # Streamlit UI (new)
├── groq_engine.py          # Shared engine extracted from chatbot.py (new)
├── requirements.txt        # Add streamlit
├── airtel_system_prompt.md # Test system prompt
└── test_schema.json        # Test JSON schema
```

---

## Phase 1: Extract Shared Engine (`groq_engine.py`)

**Why:** Both CLI and UI need the same core logic. Extract the reusable parts so we don't duplicate code.

**What to extract:**
- `MODELS` dict
- `estimate_tokens()` function
- New `GroqEngine` class (refactored from `GroqChatbot`) with:
  - `__init__()` — same params but no CLI-specific prints
  - `_build_request_params()` — unchanged
  - `stream_response_iter()` — yields chunks instead of printing (generator)
  - `structured_response()` — returns parsed result dict
  - `_update_cache_stats()` — unchanged
  - `warm_cache()` — returns result instead of printing
  - `get_cache_stats()` — returns stats dict
  - `clear_history()` — preserves system prompt
  - `switch_model()` — handles model switch logic

**What stays in `chatbot.py`:**
- CLI-specific `GroqChatbot` class wraps `GroqEngine` (or imports from it)
- ANSI color printing, `run()` loop, argparse — all CLI-only

**Key design:** `stream_response_iter()` yields dicts like:
```python
{"type": "reasoning", "text": "..."}
{"type": "content", "text": "..."}
{"type": "stats", "ttft_ms": 42, "total_ms": 350, "cached_tokens": 15000, ...}
```
This lets both CLI and UI consume the same stream differently.

---

## Phase 2: Basic Streamlit Chat UI (`app.py`)

**Single model chat with all CLI options exposed in sidebar.**

### Sidebar Controls
- **Model selector** — dropdown: qwen/qwen3-32b, openai/gpt-oss-20b, openai/gpt-oss-120b
- **Reasoning effort** — dropdown (options change based on model)
- **Show reasoning** — toggle checkbox
- **Output mode** — radio: Stream / JSON
- **Max tokens** — slider (256 to 8192, default 4096)
- **System prompt** — file uploader (.txt, .md) + text area for inline editing
- **JSON schema** — file uploader (.json)
- **Warm cache** — button
- **Clear chat** — button

### Main Area
- Chat message history (Streamlit `st.chat_message`)
- Streaming text display using `st.write_stream`
- Reasoning shown in a collapsible expander (`st.expander("Reasoning")`)
- JSON responses shown with `st.json()` (built-in syntax highlighting)
- Stats bar after each response: TTFT, total time, cache info, model name

### Session State
- `st.session_state.messages` — conversation history
- `st.session_state.engine` — GroqEngine instance
- `st.session_state.cache_stats` — running cache stats

---

## Phase 3: Side-by-Side Model Comparison Mode

**The killer feature: send one prompt to all 3 models simultaneously, see responses side-by-side.**

### UI Layout
- Toggle at top: **Chat Mode** / **Compare Mode**
- In Compare Mode:
  - 3 columns (`st.columns(3)`)
  - Each column header: model name + speed
  - User types one prompt → fires all 3 models in parallel
  - Each column streams its response independently
  - Stats row at bottom of each column: TTFT, total time, cache hit, tokens

### Async Implementation
- Use `concurrent.futures.ThreadPoolExecutor` (Streamlit doesn't support native asyncio well)
- Spawn 3 threads, one per model
- Each thread runs its own `GroqEngine` instance
- Use `st.empty()` containers to update each column as chunks arrive
- Collect results and display comparison summary

### Comparison Summary (after all 3 finish)
- Table: Model | TTFT | Total Time | Tokens | Cached | Cost
- Highlight the fastest (green) and slowest (red)
- Show reasoning side-by-side if enabled

---

## Phase 4: Cache Dashboard & Metrics

### Cache Stats Panel (in sidebar or dedicated tab)
- Per-model cache stats: requests, hits, cached tokens, savings
- Visual progress bars for cache hit rate
- Cost comparison: with cache vs without cache
- Token usage over session: bar chart

### Response Metrics
- TTFT comparison across requests (line chart over time)
- Total response time trend
- Tokens per request breakdown

---

## Phase 5: Polish & Advanced Features

- **Dark/Light theme** — Streamlit native theme support
- **Export conversation** — download chat as JSON/Markdown
- **Keyboard shortcuts** — Cmd+Enter to send
- **Mobile responsive** — Streamlit handles this mostly
- **Error handling** — graceful API error display with retry button
- **Rate limit indicator** — show estimated remaining tokens/requests for the day

---

## Implementation Order

| Phase | What | Effort | Dependency |
|---|---|---|---|
| Phase 1 | Extract `groq_engine.py` | Medium | None |
| Phase 2 | Basic Streamlit chat | Medium | Phase 1 |
| Phase 3 | Side-by-side comparison | High | Phase 2 |
| Phase 4 | Cache dashboard | Low | Phase 3 |
| Phase 5 | Polish | Low | Phase 4 |

**Start:** Phase 1 → Phase 2 → Phase 3 (the critical path)

---

## Run Command
```bash
# Install
pip install streamlit

# Run UI
streamlit run app.py

# CLI still works independently
python chatbot.py -m openai/gpt-oss-20b -s airtel_system_prompt.md
```
