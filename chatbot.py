#!/usr/bin/env python3
"""
Groq Reasoning Chatbot — CLI chatbot with streaming + reasoning support.

Supports:
  • qwen/qwen3-32b        — Best multilingual (119 langs), reasoning via <think> tags
  • openai/gpt-oss-20b    — Fastest TTFT (~1000 t/s), reasoning via dedicated field  
  • openai/gpt-oss-120b   — Best reasoning quality (~500 t/s), reasoning via dedicated field

Usage:
  python chatbot.py                          # Interactive model picker
  python chatbot.py --model qwen/qwen3-32b   # Direct model selection
  python chatbot.py --model openai/gpt-oss-20b --reasoning-effort high
  python chatbot.py --show-comparison         # Show model comparison table

Environment:
  GROQ_API_KEY=gsk_...   # Required — get yours at https://console.groq.com/keys
"""

import os
import sys
import time
import argparse
import re
import json
from concurrent.futures import ThreadPoolExecutor
from groq import Groq

# Optional: Director support (graceful fallback if not available)
try:
    from providers.registry import get_registry
    from director import Director, DirectorResult
    DIRECTOR_AVAILABLE = True
except ImportError:
    DIRECTOR_AVAILABLE = False


# ── Model Registry ──────────────────────────────────────────────────────────

MODELS = {
    "qwen/qwen3-32b": {
        "name": "Qwen3-32B",
        "speed": "~400 t/s",
        "tier": "Preview",
        "reasoning_type": "reasoning_format",        # uses reasoning_format param
        "reasoning_efforts": ["none", "default"],     # reasoning_effort options
        "default_effort": "default",
        "context": "131,072",
        "max_completion": 40960,
        "multilingual": "119 languages (Hindi, Arabic dialects, Afrikaans, Swahili, CJK, European, Dravidian, etc.)",
        "multilingual_score": "★★★★★",
        "price_in": "$0.29",
        "price_out": "$0.59",
        "streaming": True,
        "notes": "Best multilingual. Uses <think> tags (raw) or separate reasoning field (parsed).",
        "supports_strict_schema": False,
        "supports_caching": False,
    },
    "openai/gpt-oss-20b": {
        "name": "GPT-OSS 20B",
        "speed": "~1000 t/s",
        "tier": "Production",
        "reasoning_type": "include_reasoning",        # uses include_reasoning param
        "reasoning_efforts": ["low", "medium", "high"],
        "default_effort": "medium",
        "context": "131,072",
        "max_completion": 65536,
        "multilingual": "14+ languages (primarily English-optimized, STEM focus)",
        "multilingual_score": "★★☆☆☆",
        "price_in": "$0.075",
        "price_out": "$0.30",
        "streaming": True,
        "notes": "Fastest TTFT. MoE with 3.6B active params. Best for English speed-critical.",
        "supports_strict_schema": True,
        "supports_caching": True,
    },
    "openai/gpt-oss-120b": {
        "name": "GPT-OSS 120B",
        "speed": "~500 t/s",
        "tier": "Production",
        "reasoning_type": "include_reasoning",        # uses include_reasoning param
        "reasoning_efforts": ["low", "medium", "high"],
        "default_effort": "medium",
        "context": "131,072",
        "max_completion": 65536,
        "multilingual": "81+ languages (English-primary, competitive on 14-lang benchmarks)",
        "multilingual_score": "★★★☆☆",
        "price_in": "$0.15",
        "price_out": "$0.60",
        "streaming": True,
        "notes": "Best reasoning quality. MoE with 5.1B active params. Near o4-mini parity.",
        "supports_strict_schema": True,
        "supports_caching": True,
    },
}


# ── ANSI Colors ─────────────────────────────────────────────────────────────

class C:
    RESET    = "\033[0m"
    BOLD     = "\033[1m"
    DIM      = "\033[2m"
    RED      = "\033[91m"
    GREEN    = "\033[92m"
    YELLOW   = "\033[93m"
    BLUE     = "\033[94m"
    MAGENTA  = "\033[95m"
    CYAN     = "\033[96m"
    WHITE    = "\033[97m"
    BG_DARK  = "\033[48;5;236m"


# ── Comparison Table ────────────────────────────────────────────────────────

def print_comparison_table():
    """Print a detailed comparison table of all supported models."""
    print(f"\n{C.BOLD}{C.CYAN}{'═' * 100}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  GROQ REASONING MODELS — STREAMING + REASONING + MULTILINGUAL COMPARISON{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'═' * 100}{C.RESET}\n")

    headers = [
        "Feature",
        "qwen/qwen3-32b",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    ]

    rows = [
        ("Output Speed", "~400 t/s", "~1000 t/s ⚡", "~500 t/s"),
        ("TTFT", "Fast", "Ultra-fast ⚡", "Fast"),
        ("Tier", "Preview", "Production ✓", "Production ✓"),
        ("Streaming", "✅ Yes", "✅ Yes", "✅ Yes"),
        ("Reasoning", "✅ Yes", "✅ Yes", "✅ Yes"),
        ("Reasoning Efforts", "none / default", "low / medium / high", "low / medium / high"),
        ("Reasoning Format", "raw / parsed / hidden", "N/A (use include_reasoning)", "N/A (use include_reasoning)"),
        ("Context Window", "131K tokens", "131K tokens", "131K tokens"),
        ("Max Output", "40,960 tokens", "65,536 tokens", "65,536 tokens"),
        ("Architecture", "Dense 32.8B", "MoE 21B (3.6B active)", "MoE 117B (5.1B active)"),
        ("Multilingual", "119 langs ★★★★★", "14+ langs ★★☆☆☆", "81+ langs ★★★☆☆"),
        ("Hindi", "✅ Excellent", "⚠️ Basic", "⚠️ Moderate"),
        ("Arabic", "✅ 8 dialects", "❌ Poor", "⚠️ Moderate"),
        ("African langs", "✅ Afrikaans, Swahili+", "❌ Limited", "⚠️ Some"),
        ("Price (input/1M)", "$0.29", "$0.075 💰", "$0.15"),
        ("Price (output/1M)", "$0.59", "$0.30 💰", "$0.60"),
        ("Best For", "Multilingual + Reasoning", "Speed-critical English", "Complex reasoning"),
    ]

    # Calculate column widths
    col_w = [max(len(h), max(len(r[i]) for r in rows)) + 2 for i, h in enumerate(headers)]

    # Print header
    header_line = "│".join(f" {C.BOLD}{h:<{col_w[i]-1}}{C.RESET}" for i, h in enumerate(headers))
    sep_line = "┼".join("─" * col_w[i] for i in range(len(headers)))

    print(f"  {header_line}")
    print(f"  {sep_line}")

    # Print rows
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            if i == 0:
                cells.append(f" {C.YELLOW}{cell:<{col_w[i]-1}}{C.RESET}")
            else:
                cells.append(f" {cell:<{col_w[i]-1}}")
        print(f"  {'│'.join(cells)}")

    print(f"\n{C.DIM}  Note: Qwen3-32B is the ONLY model that checks all 3 boxes: Streaming + Reasoning + Strong Multilingual{C.RESET}")
    print(f"{C.DIM}  GPT-OSS models are trained on mostly English data — use Qwen3 for Hindi/Arabic/African languages.{C.RESET}\n")


# ── Token Estimation ───────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough estimate: ~1 token per 4 characters."""
    return max(1, len(text) // 4)


# ── Chat Engine ─────────────────────────────────────────────────────────────

class GroqChatbot:
    """CLI chatbot with streaming + reasoning for Groq models."""

    def __init__(self, model_id: str, reasoning_effort: str = None, show_reasoning: bool = True,
                 system_prompt: str = None, system_prompt_file: str = None,
                 json_schema: dict = None, json_mode: bool = False,
                 max_completion_tokens: int = 4096,
                 director_enabled: bool = False):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print(f"\n{C.RED}✗ GROQ_API_KEY not set!{C.RESET}")
            print(f"  Get your free key at: {C.CYAN}https://console.groq.com/keys{C.RESET}")
            print(f"  Then run: {C.DIM}export GROQ_API_KEY=gsk_...{C.RESET}\n")
            sys.exit(1)

        self.client = Groq(api_key=api_key)
        self.model_id = model_id
        self.model_info = MODELS[model_id]
        self.show_reasoning = show_reasoning
        self.messages = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_reasoning_tokens = 0

        # Cache stats
        self.total_cached_tokens = 0
        self.total_prompt_tokens = 0
        self.cache_hits = 0
        self.total_requests = 0

        # Output configuration
        self.max_completion_tokens = max_completion_tokens
        self.json_schema = json_schema
        self.json_mode = json_mode
        self.output_mode = "json" if (json_schema or json_mode) else "stream"

        # System prompt
        self.system_prompt = system_prompt
        self.system_prompt_file = system_prompt_file
        self.system_prompt_tokens = estimate_tokens(system_prompt) if system_prompt else 0
        if system_prompt:
            self.messages.insert(0, {"role": "system", "content": system_prompt})
            # Context budget warning
            remaining = 131072 - self.system_prompt_tokens - self.max_completion_tokens
            if remaining < 10000:
                print(f"{C.YELLOW}⚠ System prompt uses ~{self.system_prompt_tokens} tokens. "
                      f"Only ~{remaining} tokens remain for conversation.{C.RESET}")

        # Set reasoning effort
        if reasoning_effort:
            if reasoning_effort not in self.model_info["reasoning_efforts"]:
                valid = ", ".join(self.model_info["reasoning_efforts"])
                print(f"{C.YELLOW}⚠ '{reasoning_effort}' not valid for {model_id}. Valid: {valid}{C.RESET}")
                self.reasoning_effort = self.model_info["default_effort"]
            else:
                self.reasoning_effort = reasoning_effort
        else:
            self.reasoning_effort = self.model_info["default_effort"]

        # Director (optional)
        self.director = None
        self.director_stats = {"requests": 0, "passes": 0, "total_ms": 0, "total_cost": 0.0}
        if director_enabled and DIRECTOR_AVAILABLE:
            registry = get_registry()
            result = registry.find_default_director()
            if result:
                provider, model_info = result
                self.director = Director(provider=provider, model_id=model_info.id)
                print(f"  {C.GREEN}✓ Director: {model_info.name} ({provider.name}){C.RESET}")
            else:
                print(f"  {C.YELLOW}⚠ No Director model available. Set OPENAI_API_KEY for best results.{C.RESET}")

    def _build_request_params(self, messages: list) -> dict:
        """Build the API request parameters based on model type."""
        is_json = self.output_mode == "json"
        params = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0.6,
            "max_completion_tokens": self.max_completion_tokens,
            "top_p": 0.95,
            "stream": not is_json,  # streaming and structured output are mutually exclusive
        }

        if self.model_info["reasoning_type"] == "reasoning_format":
            # Qwen3 models — use reasoning_format
            params["reasoning_format"] = "parsed" if self.show_reasoning else "hidden"
            if self.reasoning_effort != "default":
                params["reasoning_effort"] = self.reasoning_effort
        else:
            # GPT-OSS models — use include_reasoning
            params["include_reasoning"] = self.show_reasoning
            params["reasoning_effort"] = self.reasoning_effort

        # JSON structured output (requires stream=False)
        if is_json:
            if self.json_schema and self.model_info.get("supports_strict_schema"):
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.json_schema.get("title", "response"),
                        "schema": self.json_schema,
                        "strict": True,
                    }
                }
            else:
                # Basic json_object mode (Qwen3, or no schema provided)
                params["response_format"] = {"type": "json_object"}

        return params

    def stream_response(self, user_message: str) -> dict:
        """
        Send a message and stream the response with reasoning.
        Returns dict with content, reasoning, and timing stats.
        """
        self.messages.append({"role": "user", "content": user_message})
        params = self._build_request_params(self.messages)

        start_time = time.perf_counter()
        first_token_time = None
        first_content_time = None

        content_buffer = ""
        reasoning_buffer = ""
        is_in_think_tag = False
        token_count = 0
        usage = None

        try:
            stream = self.client.chat.completions.create(**params)

            # ── Phase: Stream reasoning tokens ──
            reasoning_started = False
            content_started = False

            for chunk in stream:
                delta = chunk.choices[0].delta
                now = time.perf_counter()

                # Track first token
                if first_token_time is None:
                    first_token_time = now

                # ── Handle reasoning (GPT-OSS: delta.reasoning field) ──
                reasoning_text = getattr(delta, "reasoning", None)
                if reasoning_text:
                    if not reasoning_started and self.show_reasoning:
                        reasoning_started = True
                        print(f"\n  {C.DIM}{C.MAGENTA}💭 Reasoning:{C.RESET}")
                        print(f"  {C.DIM}{C.MAGENTA}", end="", flush=True)
                    reasoning_buffer += reasoning_text
                    if self.show_reasoning:
                        print(reasoning_text, end="", flush=True)

                # ── Handle content ──
                content_text = getattr(delta, "content", None) or ""
                if content_text:
                    # For Qwen3 raw mode: content may contain <think> tags
                    if self.model_info["reasoning_type"] == "reasoning_format":
                        # Handle <think> tag parsing in stream
                        for char in content_text:
                            if is_in_think_tag:
                                reasoning_buffer += char
                                if reasoning_buffer.endswith("</think>"):
                                    is_in_think_tag = False
                                    if self.show_reasoning:
                                        # Print reasoning minus the closing tag
                                        clean = reasoning_buffer[:-len("</think>")]
                                        if clean.startswith("<think>"):
                                            clean = clean[len("<think>"):]
                                        # Already printed character by character
                                        print(f"{C.RESET}", end="", flush=True)
                                    reasoning_buffer_clean = reasoning_buffer
                                    if reasoning_buffer_clean.startswith("<think>"):
                                        reasoning_buffer_clean = reasoning_buffer_clean[len("<think>"):]
                                    if reasoning_buffer_clean.endswith("</think>"):
                                        reasoning_buffer_clean = reasoning_buffer_clean[:-len("</think>")]
                                    reasoning_buffer = reasoning_buffer_clean
                                elif self.show_reasoning and not reasoning_buffer.startswith("<think>"[:len(reasoning_buffer)]):
                                    # We're past the opening tag, print chars
                                    if len(reasoning_buffer) > len("<think>"):
                                        pass  # will be handled below
                                continue

                            if content_text.strip().startswith("<think>") and not content_started and not is_in_think_tag:
                                is_in_think_tag = True
                                reasoning_buffer = "<think>"
                                if self.show_reasoning:
                                    print(f"\n  {C.DIM}{C.MAGENTA}💭 Reasoning:{C.RESET}")
                                    print(f"  {C.DIM}{C.MAGENTA}", end="", flush=True)
                                break
                            else:
                                if not content_started:
                                    content_started = True
                                    first_content_time = now
                                    if reasoning_started or reasoning_buffer:
                                        print(f"{C.RESET}")  # close reasoning color
                                    print(f"\n  {C.GREEN}{C.BOLD}🤖 Response:{C.RESET}")
                                    print(f"  {C.WHITE}", end="", flush=True)
                                content_buffer += char
                                print(char, end="", flush=True)
                                token_count += 1
                    else:
                        # GPT-OSS models: content is always the answer
                        if not content_started:
                            content_started = True
                            first_content_time = now
                            if reasoning_started:
                                print(f"{C.RESET}")  # close reasoning color
                            print(f"\n  {C.GREEN}{C.BOLD}🤖 Response:{C.RESET}")
                            print(f"  {C.WHITE}", end="", flush=True)
                        content_buffer += content_text
                        print(content_text, end="", flush=True)
                        token_count += 1

                # Check for usage in final chunk
                x_groq = getattr(chunk, "x_groq", None)
                if x_groq:
                    usage = getattr(x_groq, "usage", None)

            print(f"{C.RESET}")  # Reset colors

        except Exception as e:
            print(f"\n{C.RED}✗ API Error: {e}{C.RESET}")
            # Remove the failed user message
            self.messages.pop()
            return None

        end_time = time.perf_counter()

        # ── Cache & usage stats ──
        cache_stats = self._update_cache_stats(usage)

        # ── Timing stats ──
        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
        ttfc = (first_content_time - start_time) * 1000 if first_content_time else 0
        total_time = (end_time - start_time) * 1000

        # Add assistant message to history
        self.messages.append({"role": "assistant", "content": content_buffer})

        # Print stats
        print(f"\n  {C.DIM}{'─' * 60}{C.RESET}")
        print(f"  {C.DIM}⏱  TTFT: {ttft:.0f}ms │ Time to content: {ttfc:.0f}ms │ Total: {total_time:.0f}ms{C.RESET}")
        if reasoning_buffer:
            r_words = len(reasoning_buffer.split())
            print(f"  {C.DIM}💭 Reasoning: ~{r_words} words │ Content: {len(content_buffer)} chars{C.RESET}")
        if cache_stats["cached_tokens"] > 0:
            print(f"  {C.DIM}💾 Cache: {cache_stats['cached_tokens']}/{cache_stats['prompt_tokens']} prompt tokens cached ({cache_stats['cached_tokens']*100//max(1,cache_stats['prompt_tokens'])}%){C.RESET}")
        print(f"  {C.DIM}🔧 Model: {self.model_id} │ Effort: {self.reasoning_effort}{C.RESET}")
        print(f"  {C.DIM}{'─' * 60}{C.RESET}")

        return {
            "content": content_buffer,
            "reasoning": reasoning_buffer,
            "ttft_ms": ttft,
            "ttfc_ms": ttfc,
            "total_ms": total_time,
            "cached_tokens": cache_stats["cached_tokens"],
        }

    def _print_json_highlighted(self, json_str: str):
        """Print JSON with ANSI syntax highlighting."""
        print(f"\n  {C.GREEN}{C.BOLD}JSON Response:{C.RESET}")
        for line in json_str.split("\n"):
            highlighted = line
            # Color keys: "key":
            highlighted = re.sub(
                r'"([^"]+)"(\s*:)',
                f'{C.CYAN}"\\1"{C.RESET}\\2',
                highlighted
            )
            # Color string values: : "value"
            highlighted = re.sub(
                r'(:\s*)"([^"]*)"',
                f'\\1{C.GREEN}"\\2"{C.RESET}',
                highlighted
            )
            # Color numbers
            highlighted = re.sub(
                r'(?<=[\s:,\[])(-?\d+\.?\d*)',
                f'{C.YELLOW}\\1{C.RESET}',
                highlighted
            )
            # Color booleans and null
            highlighted = re.sub(
                r'\b(true|false|null)\b',
                f'{C.MAGENTA}\\1{C.RESET}',
                highlighted
            )
            print(f"  {highlighted}")

    def structured_response(self, user_message: str) -> dict:
        """Send a message and get a structured JSON response (non-streaming)."""
        self.messages.append({"role": "user", "content": user_message})

        # For Qwen3 with schema but no strict support, inject schema as text guidance
        messages = list(self.messages)
        if self.json_schema and not self.model_info.get("supports_strict_schema"):
            schema_instruction = (
                f"You must respond with valid JSON matching this schema:\n"
                f"```json\n{json.dumps(self.json_schema, indent=2)}\n```"
            )
            if messages and messages[0]["role"] == "system":
                messages[0] = {
                    "role": "system",
                    "content": messages[0]["content"] + "\n\n" + schema_instruction
                }
            else:
                messages.insert(0, {"role": "system", "content": schema_instruction})

        params = self._build_request_params(messages)
        start_time = time.perf_counter()

        try:
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            print(f"\n{C.RED}✗ API Error: {e}{C.RESET}")
            self.messages.pop()
            return None

        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000

        content = response.choices[0].message.content

        # Parse and pretty-print JSON
        try:
            parsed = json.loads(content)
            formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
            self._print_json_highlighted(formatted)
        except json.JSONDecodeError:
            print(f"\n  {C.YELLOW}⚠ Response is not valid JSON:{C.RESET}")
            print(f"  {C.WHITE}{content}{C.RESET}")

        # Cache & usage stats
        cache_stats = self._update_cache_stats(getattr(response, "usage", None))

        # Add to history
        self.messages.append({"role": "assistant", "content": content})

        # Print stats
        print(f"\n  {C.DIM}{'─' * 60}{C.RESET}")
        print(f"  {C.DIM}⏱  Total: {total_time:.0f}ms │ Mode: JSON (non-streaming){C.RESET}")
        if cache_stats["cached_tokens"] > 0:
            print(f"  {C.DIM}💾 Cache: {cache_stats['cached_tokens']}/{cache_stats['prompt_tokens']} prompt tokens cached ({cache_stats['cached_tokens']*100//max(1,cache_stats['prompt_tokens'])}%){C.RESET}")
        print(f"  {C.DIM}🔧 Model: {self.model_id} │ Effort: {self.reasoning_effort}{C.RESET}")
        print(f"  {C.DIM}{'─' * 60}{C.RESET}")

        return {"content": content, "total_ms": total_time}

    def _update_cache_stats(self, usage) -> dict:
        """Extract and update cache statistics from API usage response.
        Returns dict with prompt_tokens, cached_tokens, completion_tokens."""
        stats = {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}
        if not usage:
            return stats

        stats["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
        stats["completion_tokens"] = getattr(usage, "completion_tokens", 0) or 0

        # Extract cached tokens from prompt_tokens_details
        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            stats["cached_tokens"] = getattr(details, "cached_tokens", 0) or 0

        # Update running totals
        self.total_requests += 1
        self.total_prompt_tokens += stats["prompt_tokens"]
        self.total_cached_tokens += stats["cached_tokens"]
        self.total_input_tokens += stats["prompt_tokens"]
        self.total_output_tokens += stats["completion_tokens"]
        if stats["cached_tokens"] > 0:
            self.cache_hits += 1

        return stats

    def warm_cache(self):
        """Send a lightweight request to prime the system prompt cache."""
        if not self.system_prompt:
            return
        if not self.model_info.get("supports_caching"):
            print(f"  {C.DIM}Cache warm-up skipped: {self.model_info['name']} does not support prompt caching.{C.RESET}")
            return

        print(f"  {C.DIM}Warming cache for ~{self.system_prompt_tokens} token system prompt...{C.RESET}", end="", flush=True)
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "hi"}
                ],
                max_completion_tokens=1,
                temperature=0.0,
            )
            elapsed = (time.perf_counter() - start) * 1000
            # Track the warm-up usage
            if response.usage:
                self._update_cache_stats(response.usage)
            print(f" done ({elapsed:.0f}ms)")
            print(f"  {C.GREEN}✓ Cache primed. Subsequent requests will be faster and cheaper.{C.RESET}")
        except Exception as e:
            print(f" failed: {e}")

    def _print_director_insights(self, result):
        """Print Director analysis results in the terminal."""
        if not DIRECTOR_AVAILABLE:
            return
        if not result.success:
            print(f"  {C.DIM}📋 Director failed: {result.error}{C.RESET}")
            return

        self.director_stats["requests"] += 1
        self.director_stats["total_ms"] += result.total_ms
        if result.schema_valid:
            self.director_stats["passes"] += 1

        # Compute cost
        if self.director:
            model = self.director.provider.get_model(self.director.model_id)
            if model:
                cost = model.estimate_cost(result.prompt_tokens, result.completion_tokens)
                self.director_stats["total_cost"] += cost

        data = result.data
        if not data:
            return

        print(f"  {C.DIM}{'─' * 60}{C.RESET}")
        print(f"  {C.CYAN}📋 Director{C.RESET} ({self.director.model_id}, {result.total_ms:.0f}ms, schema: {'✅' if result.schema_valid else '⚠️'})")

        # Intent
        intent = data.get("intent", {})
        if intent:
            print(f"  {C.DIM}   Intent: {intent.get('type', '?')} ({intent.get('complexity', '?')}) │ Topic: {intent.get('topic', '?')}{C.RESET}")

        # Analysis
        analysis = data.get("response_analysis", {})
        if analysis:
            entities = analysis.get("key_entities", [])
            conf = analysis.get("confidence", 0)
            lang = analysis.get("language_detected", "?")
            print(f"  {C.DIM}   Confidence: {conf:.0%} │ Language: {lang} │ Sentiment: {analysis.get('sentiment', '?')}{C.RESET}")
            if entities:
                print(f"  {C.DIM}   Entities: {', '.join(entities)}{C.RESET}")

        # Follow-ups
        suggestions = data.get("suggestions", {})
        follow_ups = suggestions.get("follow_ups", [])
        if follow_ups:
            print(f"  {C.DIM}   Follow-ups:{C.RESET}")
            for q in follow_ups:
                print(f"  {C.DIM}     → {q}{C.RESET}")

        # Guardrails
        guardrails = data.get("guardrails", {})
        flags = guardrails.get("flags", ["none"])
        if flags and flags != ["none"]:
            print(f"  {C.YELLOW}   ⚠️ Flags: {', '.join(flags)}{C.RESET}")
            note = guardrails.get("note", "")
            if note:
                print(f"  {C.DIM}     {note}{C.RESET}")

        print(f"  {C.DIM}{'─' * 60}{C.RESET}")

    def run(self):
        """Run the interactive chat loop."""
        print(f"\n{C.BOLD}{C.CYAN}{'═' * 60}{C.RESET}")
        title = "  ACTOR-DIRECTOR CHATBOT" if self.director else "  GROQ REASONING CHATBOT"
        print(f"{C.BOLD}{C.CYAN}{title}{C.RESET}")
        print(f"{C.BOLD}{C.CYAN}{'═' * 60}{C.RESET}")
        print(f"  {C.WHITE}Model:     {C.YELLOW}{self.model_info['name']}{C.RESET} ({self.model_id})")
        print(f"  {C.WHITE}Speed:     {C.GREEN}{self.model_info['speed']}{C.RESET}")
        print(f"  {C.WHITE}Tier:      {self.model_info['tier']}")
        print(f"  {C.WHITE}Reasoning: {C.GREEN}ON{C.RESET} (effort: {self.reasoning_effort})")
        if self.output_mode == "json":
            print(f"  {C.WHITE}Output:    {C.YELLOW}JSON{C.RESET} (non-streaming)")
        else:
            print(f"  {C.WHITE}Output:    {C.GREEN}Streaming{C.RESET}")
        if self.system_prompt:
            print(f"  {C.WHITE}System:    {C.GREEN}Loaded{C.RESET} (~{self.system_prompt_tokens} tokens from {self.system_prompt_file})")
        if self.model_info.get("supports_caching") and self.system_prompt:
            print(f"  {C.WHITE}Caching:   {C.GREEN}Enabled{C.RESET} (auto, 50% cost savings on cached tokens)")
        elif self.system_prompt:
            print(f"  {C.WHITE}Caching:   {C.RED}Not supported{C.RESET} ({self.model_info['name']} does not support caching)")
        print(f"  {C.WHITE}Multilingual: {self.model_info['multilingual_score']}")
        if self.director:
            print(f"  {C.WHITE}Director:  {C.GREEN}{self.director.model_id}{C.RESET} ({self.director.provider.name})")
        print(f"{C.BOLD}{C.CYAN}{'═' * 60}{C.RESET}")
        print(f"  {C.DIM}Commands: /quit /clear /model /effort /reasoning /output /director /providers /stats /help{C.RESET}")
        print(f"  {C.DIM}Supports: English, Hindi, Arabic, and 100+ more languages{C.RESET}\n")

        while True:
            try:
                user_input = input(f"  {C.BLUE}{C.BOLD}You ❯ {C.RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n\n  {C.DIM}👋 Goodbye!{C.RESET}\n")
                break

            if not user_input:
                continue

            # ── Commands ──
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]

                if cmd in ("/quit", "/exit", "/q"):
                    print(f"\n  {C.DIM}👋 Goodbye!{C.RESET}\n")
                    break

                elif cmd == "/clear":
                    if self.system_prompt:
                        self.messages = [{"role": "system", "content": self.system_prompt}]
                    else:
                        self.messages = []
                    print(f"  {C.GREEN}✓ Chat history cleared.{C.RESET}\n")
                    continue

                elif cmd == "/model":
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        print(f"  {C.YELLOW}Available models:{C.RESET}")
                        for mid, info in MODELS.items():
                            marker = " ◀ current" if mid == self.model_id else ""
                            print(f"    {C.CYAN}{mid}{C.RESET} — {info['name']} ({info['speed']}){C.GREEN}{marker}{C.RESET}")
                        print(f"  {C.DIM}Usage: /model qwen/qwen3-32b{C.RESET}\n")
                    else:
                        new_model = parts[1].strip()
                        if new_model in MODELS:
                            self.model_id = new_model
                            self.model_info = MODELS[new_model]
                            self.reasoning_effort = self.model_info["default_effort"]
                            if self.system_prompt:
                                self.messages = [{"role": "system", "content": self.system_prompt}]
                            else:
                                self.messages = []
                            print(f"  {C.GREEN}✓ Switched to {self.model_info['name']} (history cleared){C.RESET}")
                            print(f"  {C.DIM}  Reasoning effort reset to: {self.reasoning_effort}{C.RESET}")
                            if self.json_schema and not self.model_info.get("supports_strict_schema"):
                                print(f"  {C.YELLOW}⚠ {self.model_info['name']} does not support strict JSON schema. "
                                      f"Falling back to json_object mode with schema guidance.{C.RESET}")
                            print()
                        else:
                            print(f"  {C.RED}✗ Unknown model: {new_model}{C.RESET}\n")
                    continue

                elif cmd == "/effort":
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        valid = ", ".join(self.model_info["reasoning_efforts"])
                        print(f"  {C.YELLOW}Current: {self.reasoning_effort} │ Valid: {valid}{C.RESET}")
                        print(f"  {C.DIM}Usage: /effort high{C.RESET}\n")
                    else:
                        new_effort = parts[1].strip().lower()
                        if new_effort in self.model_info["reasoning_efforts"]:
                            self.reasoning_effort = new_effort
                            print(f"  {C.GREEN}✓ Reasoning effort set to: {new_effort}{C.RESET}\n")
                        else:
                            valid = ", ".join(self.model_info["reasoning_efforts"])
                            print(f"  {C.RED}✗ Invalid effort. Valid for {self.model_id}: {valid}{C.RESET}\n")
                    continue

                elif cmd == "/reasoning":
                    self.show_reasoning = not self.show_reasoning
                    state = "ON" if self.show_reasoning else "OFF"
                    color = C.GREEN if self.show_reasoning else C.RED
                    print(f"  {color}✓ Reasoning display: {state}{C.RESET}\n")
                    continue

                elif cmd == "/output":
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        print(f"  {C.YELLOW}Current output mode: {self.output_mode}{C.RESET}")
                        print(f"  {C.DIM}Usage: /output json  or  /output stream{C.RESET}\n")
                    else:
                        mode = parts[1].strip().lower()
                        if mode == "json":
                            self.output_mode = "json"
                            print(f"  {C.GREEN}✓ Output mode: JSON (non-streaming){C.RESET}\n")
                        elif mode == "stream":
                            self.output_mode = "stream"
                            print(f"  {C.GREEN}✓ Output mode: Streaming{C.RESET}\n")
                        else:
                            print(f"  {C.RED}✗ Invalid mode. Use 'json' or 'stream'{C.RESET}\n")
                    continue

                elif cmd == "/system":
                    if self.system_prompt:
                        print(f"  {C.YELLOW}System Prompt Info:{C.RESET}")
                        print(f"    File:   {self.system_prompt_file}")
                        print(f"    Tokens: ~{self.system_prompt_tokens}")
                        print(f"    Chars:  {len(self.system_prompt)}")
                        preview = self.system_prompt[:200].replace("\n", " ")
                        print(f"    Preview: {C.DIM}{preview}...{C.RESET}\n")
                    else:
                        print(f"  {C.DIM}No system prompt loaded. Use --system-prompt flag.{C.RESET}\n")
                    continue

                elif cmd == "/cache":
                    print(f"  {C.YELLOW}Prompt Cache Stats:{C.RESET}")
                    print(f"    Model:           {self.model_id}")
                    supports = self.model_info.get("supports_caching", False)
                    print(f"    Caching support: {C.GREEN + 'Yes' + C.RESET if supports else C.RED + 'No' + C.RESET}")
                    print(f"    Total requests:  {self.total_requests}")
                    print(f"    Cache hits:      {self.cache_hits}/{self.total_requests}")
                    print(f"    Prompt tokens:   {self.total_prompt_tokens:,}")
                    print(f"    Cached tokens:   {self.total_cached_tokens:,}")
                    if self.total_prompt_tokens > 0:
                        pct = self.total_cached_tokens * 100 // self.total_prompt_tokens
                        print(f"    Cache rate:      {pct}%")
                        # Estimate cost savings (cached tokens are 50% cheaper)
                        price_per_m = float(self.model_info["price_in"].replace("$", ""))
                        saved = (self.total_cached_tokens / 1_000_000) * price_per_m * 0.5
                        print(f"    Est. savings:    ${saved:.6f}")
                    if self.system_prompt and supports:
                        print(f"    {C.DIM}System prompt (~{self.system_prompt_tokens} tokens) is auto-cached after first request.{C.RESET}")
                        print(f"    {C.DIM}Cache TTL: 2 hours. Cached tokens don't count toward rate limits.{C.RESET}")
                    elif self.system_prompt and not supports:
                        print(f"    {C.DIM}Switch to a GPT-OSS model to enable caching.{C.RESET}")
                    print()
                    continue

                elif cmd == "/compare":
                    print_comparison_table()
                    continue

                elif cmd == "/director":
                    if not DIRECTOR_AVAILABLE:
                        print(f"  {C.RED}✗ Director not available. Install provider packages.{C.RESET}\n")
                        continue
                    parts = user_input.split(maxsplit=1)
                    if len(parts) < 2:
                        if self.director:
                            print(f"  {C.GREEN}Director: ON{C.RESET}")
                            print(f"    Model: {self.director.model_id} ({self.director.provider.name})")
                            ds = self.director_stats
                            print(f"    Requests: {ds['requests']} │ Schema: {ds['passes']}/{ds['requests']}")
                            if ds['requests'] > 0:
                                print(f"    Avg: {ds['total_ms']/ds['requests']:.0f}ms │ Cost: ${ds['total_cost']:.6f}")
                        else:
                            print(f"  {C.RED}Director: OFF{C.RESET}")
                        print(f"  {C.DIM}Usage: /director on  or  /director off{C.RESET}\n")
                    else:
                        arg = parts[1].strip().lower()
                        if arg == "on":
                            if not self.director:
                                registry = get_registry()
                                result = registry.find_default_director()
                                if result:
                                    provider, model_info = result
                                    self.director = Director(provider=provider, model_id=model_info.id)
                                    print(f"  {C.GREEN}✓ Director enabled: {model_info.name} ({provider.name}){C.RESET}\n")
                                else:
                                    print(f"  {C.RED}✗ No Director model available.{C.RESET}\n")
                            else:
                                print(f"  {C.DIM}Director already enabled.{C.RESET}\n")
                        elif arg == "off":
                            self.director = None
                            print(f"  {C.GREEN}✓ Director disabled.{C.RESET}\n")
                        else:
                            print(f"  {C.RED}✗ Use: /director on  or  /director off{C.RESET}\n")
                    continue

                elif cmd == "/providers":
                    if not DIRECTOR_AVAILABLE:
                        print(f"  {C.RED}✗ Provider system not available.{C.RESET}\n")
                        continue
                    registry = get_registry()
                    print(f"\n  {C.BOLD}Available Providers:{C.RESET}")
                    for s in registry.provider_status():
                        icon = f"{C.GREEN}✅{C.RESET}" if s["available"] else f"{C.RED}❌{C.RESET}"
                        detail = f"{s['model_count']} models" if s["available"] else ("no key" if not s["has_key"] else "no package")
                        print(f"    {icon} {C.WHITE}{s['name']:<12}{C.RESET} {detail}  ({s['env_var']})")
                    print()

                    all_models = registry.list_all_models()
                    if all_models:
                        print(f"  {C.BOLD}All Models:{C.RESET}")
                        for m in all_models:
                            roles = ", ".join(m.roles)
                            print(f"    {C.CYAN}{m.provider}/{m.id}{C.RESET} — {m.name} [{roles}] {m.speed}")
                    print()
                    continue

                elif cmd == "/stats":
                    print(f"\n  {C.BOLD}Session Stats:{C.RESET}")
                    print(f"    Actor:     {self.model_id}")
                    print(f"    Requests:  {self.total_requests}")
                    print(f"    Tokens:    {self.total_input_tokens:,} in / {self.total_output_tokens:,} out")
                    if self.total_cached_tokens > 0:
                        rate = self.total_cached_tokens * 100 // max(1, self.total_prompt_tokens)
                        print(f"    Cache:     {self.total_cached_tokens:,}/{self.total_prompt_tokens:,} ({rate}%)")
                    if self.director:
                        ds = self.director_stats
                        print(f"    Director:  {self.director.model_id}")
                        print(f"    Dir Reqs:  {ds['requests']} │ Schema: {ds['passes']}/{ds['requests']}")
                        print(f"    Dir Cost:  ${ds['total_cost']:.6f}")
                        if ds['requests'] > 0:
                            print(f"    Dir Avg:   {ds['total_ms']/ds['requests']:.0f}ms")
                    print()
                    continue

                elif cmd == "/help":
                    print(f"""
  {C.BOLD}Available Commands:{C.RESET}
    {C.CYAN}/model [name]{C.RESET}     — List or switch models
    {C.CYAN}/effort [level]{C.RESET}   — Set reasoning effort (low/medium/high or none/default)
    {C.CYAN}/reasoning{C.RESET}        — Toggle reasoning display ON/OFF
    {C.CYAN}/output [mode]{C.RESET}    — Switch output mode: json or stream
    {C.CYAN}/director [on/off]{C.RESET}— Toggle Director (parallel JSON analysis)
    {C.CYAN}/providers{C.RESET}        — List available providers and models
    {C.CYAN}/stats{C.RESET}            — Show session statistics
    {C.CYAN}/system{C.RESET}           — Show system prompt info
    {C.CYAN}/cache{C.RESET}            — Show prompt cache statistics
    {C.CYAN}/clear{C.RESET}            — Clear conversation history
    {C.CYAN}/compare{C.RESET}          — Show model comparison table
    {C.CYAN}/quit{C.RESET}             — Exit the chatbot

  {C.BOLD}Multilingual Tips:{C.RESET}
    Just type in any language! Examples:
    • Hindi:  "मुझे भारत के बारे में बताओ"
    • Arabic: "اشرح لي الذكاء الاصطناعي"
    • Mix:    "Explain quantum physics in Hindi"
""")
                    continue
                else:
                    print(f"  {C.RED}✗ Unknown command: {cmd}. Type /help{C.RESET}\n")
                    continue

            # ── Context budget check ──
            conversation_tokens = sum(estimate_tokens(m["content"]) for m in self.messages)
            total_budget = conversation_tokens + self.max_completion_tokens
            if total_budget > 130000:
                print(f"  {C.RED}✗ Conversation likely exceeds context window (~{conversation_tokens} tokens). Use /clear.{C.RESET}\n")
                continue
            elif total_budget > 120000:
                print(f"  {C.YELLOW}⚠ Context nearly full: ~{conversation_tokens} tokens used. Consider /clear.{C.RESET}")

            # ── Send message (Actor + optional Director in parallel) ──
            director_future = None
            if self.director and self.output_mode != "json":
                # Launch Director in background before streaming
                self.messages.append({"role": "user", "content": user_input})
                executor = ThreadPoolExecutor(max_workers=1)
                director_future = executor.submit(self.director.analyze, self.messages)
                self.messages.pop()  # stream_response will re-add it

            if self.output_mode == "json":
                self.structured_response(user_input)
            else:
                self.stream_response(user_input)

            # ── Show Director insights ──
            if director_future:
                try:
                    dir_result = director_future.result(timeout=30)
                    self._print_director_insights(dir_result)
                except Exception as e:
                    print(f"  {C.DIM}📋 Director error: {e}{C.RESET}")
                finally:
                    executor.shutdown(wait=False)
            print()


# ── Interactive Model Picker ────────────────────────────────────────────────

def pick_model_interactive() -> str:
    """Let the user pick a model interactively."""
    print(f"\n{C.BOLD}{C.CYAN}  Select a model:{C.RESET}\n")

    model_list = list(MODELS.keys())
    for i, (mid, info) in enumerate(MODELS.items(), 1):
        ml = info["multilingual_score"]
        print(f"    {C.BOLD}{C.CYAN}[{i}]{C.RESET} {C.WHITE}{info['name']:<16}{C.RESET} "
              f"{info['speed']:<12} "
              f"Multilingual: {ml}  "
              f"{C.DIM}({mid}){C.RESET}")

    print(f"\n    {C.DIM}Tip: Choose [1] Qwen3 for Hindi/Arabic/multilingual{C.RESET}")
    print(f"    {C.DIM}     Choose [2] GPT-OSS 20B for fastest English{C.RESET}")
    print(f"    {C.DIM}     Choose [3] GPT-OSS 120B for best reasoning{C.RESET}")

    while True:
        try:
            choice = input(f"\n  {C.BLUE}Enter choice [1-3]: {C.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)

        if choice in ("1", "2", "3"):
            return model_list[int(choice) - 1]
        print(f"  {C.RED}Please enter 1, 2, or 3{C.RESET}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Groq Reasoning Chatbot — Streaming + Reasoning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chatbot.py                                    # Interactive picker
  python chatbot.py --model qwen/qwen3-32b             # Qwen3 (multilingual)
  python chatbot.py --model openai/gpt-oss-20b          # GPT-OSS 20B (fastest)
  python chatbot.py --model openai/gpt-oss-120b --reasoning-effort high
  python chatbot.py --show-comparison                   # Show comparison table
  python chatbot.py --no-reasoning                      # Hide reasoning output
  python chatbot.py -s system_prompt.md                 # Load system prompt
  python chatbot.py --json-mode                         # Basic JSON output
  python chatbot.py --json-schema schema.json           # Strict JSON schema output
  python chatbot.py -s prompt.md --json-schema out.json --max-tokens 8192
  python chatbot.py -s prompt.md -m openai/gpt-oss-20b --warm-cache  # Prime cache on startup
        """
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()),
        help="Model to use (default: interactive picker)"
    )
    parser.add_argument(
        "--reasoning-effort", "-e",
        help="Reasoning effort level (model-dependent: low/medium/high or none/default)"
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Hide reasoning output (still used internally)"
    )
    parser.add_argument(
        "--show-comparison", "-c",
        action="store_true",
        help="Show model comparison table and exit"
    )
    parser.add_argument(
        "--system-prompt", "-s",
        help="Path to system prompt file (.txt, .md)"
    )
    parser.add_argument(
        "--json-schema",
        help="Path to JSON schema file for structured output (strict mode on GPT-OSS)"
    )
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="Enable basic JSON output mode (valid JSON, no schema enforcement)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max completion tokens (default: 4096)"
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="Prime the prompt cache on startup (sends a lightweight request)"
    )
    parser.add_argument(
        "--director",
        action="store_true",
        help="Enable Director (parallel JSON analysis). Requires OPENAI_API_KEY or Groq GPT-OSS."
    )

    args = parser.parse_args()

    # Show comparison table
    if args.show_comparison:
        print_comparison_table()
        sys.exit(0)

    # Load system prompt
    system_prompt = None
    system_prompt_file = None
    if args.system_prompt:
        path = args.system_prompt
        if not os.path.isfile(path):
            print(f"{C.RED}✗ System prompt file not found: {path}{C.RESET}")
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
        system_prompt_file = path
        token_est = estimate_tokens(system_prompt)
        print(f"{C.DIM}Loaded system prompt: {path} (~{token_est} tokens, {len(system_prompt)} chars){C.RESET}")

    # Load JSON schema
    json_schema = None
    if args.json_schema:
        path = args.json_schema
        if not os.path.isfile(path):
            print(f"{C.RED}✗ JSON schema file not found: {path}{C.RESET}")
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            try:
                json_schema = json.load(f)
            except json.JSONDecodeError as e:
                print(f"{C.RED}✗ Invalid JSON schema: {e}{C.RESET}")
                sys.exit(1)
        print(f"{C.DIM}Loaded JSON schema: {path}{C.RESET}")

    if args.json_schema and args.json_mode:
        print(f"{C.YELLOW}Note: --json-schema takes precedence over --json-mode{C.RESET}")

    # Pick model
    model_id = args.model or pick_model_interactive()

    # Director flag validation
    director_enabled = args.director
    if director_enabled and not DIRECTOR_AVAILABLE:
        print(f"{C.YELLOW}⚠ --director requires the provider layer. Ensure providers/ package is available.{C.RESET}")
        director_enabled = False

    # Create and run chatbot
    bot = GroqChatbot(
        model_id=model_id,
        reasoning_effort=args.reasoning_effort,
        show_reasoning=not args.no_reasoning,
        system_prompt=system_prompt,
        system_prompt_file=system_prompt_file,
        json_schema=json_schema,
        json_mode=args.json_mode,
        max_completion_tokens=args.max_tokens,
        director_enabled=director_enabled,
    )

    # Warm cache if requested
    if args.warm_cache:
        bot.warm_cache()

    bot.run()


if __name__ == "__main__":
    main()
