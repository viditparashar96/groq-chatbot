# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An Actor-Director chatbot with multi-provider LLM support. Two LLMs run in parallel:
- **Actor**: Streams user-facing responses (fast TTFT) via Groq, OpenAI, etc.
- **Director**: Produces structured JSON metadata (intent, entities, follow-ups) via constrained decoding.

Supports Groq (Qwen3, GPT-OSS) and OpenAI (GPT-4.1 family) with provider auto-detection.

## Running

```bash
# Install
pip install -r requirements.txt
pip install openai  # optional: enables GPT-4.1 Director

# CLI chatbot
python chatbot.py
python chatbot.py --model qwen/qwen3-32b

# Streamlit UI (Actor-Director)
streamlit run app.py
```

Requires `GROQ_API_KEY`. Optional: `OPENAI_API_KEY` for Director (GPT-4.1-nano).

## Architecture

```
groq-chatbot/
├── chatbot.py               # CLI entry point (~920 lines, standalone)
├── app.py                   # Streamlit UI with Actor-Director split layout
├── groq_engine.py           # GroqEngine wrapper (backward compat, delegates to provider)
├── director.py              # Director: JSON analysis engine + DirectorResult
├── orchestrator.py          # Parallel Actor+Director via ThreadPoolExecutor
├── providers/
│   ├── __init__.py          # Package exports
│   ├── base.py              # LLMProvider ABC, StreamChunk, ModelInfo, UsageStats
│   ├── groq_provider.py     # Groq implementation (streaming + JSON)
│   ├── openai_provider.py   # OpenAI implementation (GPT-4.1 family)
│   └── registry.py          # Auto-discovers available providers by API key
├── schemas/
│   └── director_default.json # Default Director output schema
└── requirements.txt
```

### Key Modules

- **`providers/base.py`**: `LLMProvider` ABC with `stream_chat()` (Actor) and `json_chat()` (Director). All providers implement this.
- **`providers/registry.py`**: `ProviderRegistry` discovers providers from installed packages + env vars. `find_default_director()` picks the best Director model (prefers gpt-4.1-nano).
- **`groq_engine.py`**: `GroqEngine` class wraps `GroqProvider` with conversation history + cumulative stats. `MODELS` dict is backward-compatible. Both `app.py` and `chatbot.py` import from here.
- **`director.py`**: `Director` class takes a provider + schema, sends conversation to Director model, returns `DirectorResult` with `.data`, `.follow_ups`, `.key_entities`, `.confidence`.
- **`orchestrator.py`**: `Orchestrator` runs Actor (streaming, main thread) + Director (JSON, background thread) in parallel. `SessionStats` tracks TTFT history, cost breakdown.
- **`app.py`**: Streamlit UI. Chat mode has 70/30 split (Actor stream | Director insights panel). Director follow-ups are clickable buttons. Session stats in sidebar.

### Provider Model IDs

- Groq: `qwen/qwen3-32b`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b`
- OpenAI: `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`

## Key Details

- Python 3.10+ required
- Core dependency: `groq>=0.28.0`. Optional: `openai>=1.30.0`
- `chatbot.py` is still standalone (imports only `groq`, not the provider layer)
- No tests, no linter config, no build system
