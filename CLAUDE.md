# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A single-file Python CLI chatbot that streams responses with reasoning support from Groq's LLM API. Supports three models (Qwen3-32B, GPT-OSS 20B, GPT-OSS 120B) with hot-swapping, multilingual input, and configurable reasoning effort.

## Running

```bash
# Install
pip install -r requirements.txt

# Run (interactive model picker)
python chatbot.py

# Run with specific model
python chatbot.py --model qwen/qwen3-32b
python chatbot.py -m openai/gpt-oss-20b -e high
python chatbot.py --show-comparison
```

Requires `GROQ_API_KEY` environment variable (free from console.groq.com/keys).

## Architecture

Everything lives in `chatbot.py` (~550 lines):

- **`MODELS` dict** (top of file): Registry of supported models with their capabilities, reasoning types, and pricing. Each model has a `reasoning_type` that determines API parameter style.
- **`GroqChatbot` class**: Core chat engine. Manages conversation history, builds model-specific API params via `_build_request_params()`, and handles streaming in `stream_response()`.
- **Two reasoning paradigms**: Qwen3 uses `reasoning_format` param with `<think>` tag parsing in the stream. GPT-OSS models use `include_reasoning` param with a separate `delta.reasoning` field. The streaming logic in `stream_response()` handles both.
- **`C` class**: ANSI color constants for terminal output.
- **In-chat commands**: Handled in the `run()` method's main loop (`/model`, `/effort`, `/reasoning`, `/clear`, `/compare`, `/help`, `/quit`).

## Key Details

- Only dependency: `groq>=0.28.0`
- Python 3.10+ required
- No tests, no linter config, no build system — it's a standalone script
