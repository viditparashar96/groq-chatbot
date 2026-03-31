"""
Actor-Director Chatbot — Streamlit Web UI

Two LLMs run in parallel:
  - Actor: streams the user-facing response (fast TTFT)
  - Director: produces structured JSON metadata (intent, entities, follow-ups)

Run: streamlit run app.py
"""

from __future__ import annotations

import os
import time
import json
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

from groq_engine import GroqEngine, MODELS, estimate_tokens
from providers.registry import get_registry
from director import Director, DirectorResult


# ── Page Config ──���──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Actor-Director Chatbot",
    page_icon="⚡",
    layout="wide",
)


# ── Session State Init ──────────────────────────────────────────────────────

def init_state():
    defaults = {
        "messages": [],
        "engines": {},
        "compare_results": [],
        "system_prompt": None,
        "system_prompt_name": None,
        "json_schema": None,
        "json_schema_name": None,
        "director_enabled": False,
        "director_results": {},   # msg_index -> DirectorResult data
        "session_stats": {
            "turn_count": 0,
            "ttft_history": [],
            "total_cost": 0.0,
            "actor_cost": 0.0,
            "director_cost": 0.0,
            "director_passes": 0,
            "director_total": 0,
        },
        "cache_tracker": {
            "last_request_time": None,  # timestamp of last API call
            "cache_warm": False,
            "total_cached_tokens": 0,
            "total_prompt_tokens": 0,
            "estimated_savings": 0.0,
        },
        "next_prompt": None,      # for clickable follow-ups
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Info Dialog ─────────────────────────────────────────────────────────────

@st.dialog("Actor-Director Architecture", width="large")
def show_info_dialog():
    st.markdown("""
### How It Works

Two LLMs run **in parallel** on every message:

| Role | What it does | Output |
|------|-------------|--------|
| **Actor** | Streams the user-facing response token-by-token | Real-time text |
| **Director** | Analyzes the conversation and produces structured metadata | Strict JSON |

The Actor gives you the answer fast. The Director gives the system intelligence
about that answer — intent, entities, follow-ups, guardrails.

---

### Actor Models (Streaming)

| Model | Speed | Provider | Best For |
|-------|-------|----------|----------|
| Qwen3-32B | ~400 t/s | Groq | Multilingual (119 langs) |
| GPT-OSS 20B | ~1000 t/s | Groq | Speed-critical English |
| GPT-OSS 120B | ~500 t/s | Groq | Complex reasoning |
| GPT-4.1 | ~80 t/s | OpenAI | Best quality |
| GPT-4.1 Mini | ~110 t/s | OpenAI | Balance of speed & quality |

### Director Models (JSON)

| Model | Schema Compliance | Cost | Speed |
|-------|-------------------|------|-------|
| GPT-4.1 Nano | 100% (constrained) | $0.10/$0.40 | Ultra-fast |
| GPT-4.1 Mini | 100% (constrained) | $0.40/$1.60 | Fast |
| GPT-OSS 20B | ~95% (Groq) | $0.075/$0.30 | Ultra-fast |

---

### Key Trade-offs

- **Streaming + JSON are mutually exclusive** per API call — that's why we use two models
- **Director is optional** — disable it for pure streaming mode
- **Director adds cost** — but GPT-4.1 Nano is only ~$0.0001 per turn
- **Follow-up suggestions** from the Director are clickable — guided conversation

### Prompt Caching

| | Groq | OpenAI |
|---|---|---|
| **TTL** | 2 hours | 5-10 min |
| **Min tokens** | 128-1024 (varies) | 1,024 |
| **Discount** | 50% off cached input | 50% off cached input |
| **Models** | GPT-OSS 20B, 120B | All GPT-4.1 |

Cached tokens don't count toward rate limits on Groq.
""")


# ── Top Bar ─────────────────────────────────────────────────────────────────

top_left, top_right = st.columns([9, 1])
with top_left:
    st.caption("⚡ Actor-Director Chatbot")
with top_right:
    if st.button("ℹ️ Info"):
        show_info_dialog()


# ── Helper: Get or Create Engine ────────────────────────────────────────────

def get_engine(model_id: str, **kwargs) -> GroqEngine:
    """Get cached Groq engine or create new one."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("Set GROQ_API_KEY environment variable before running.")
        st.stop()

    cache_key = f"{model_id}_{kwargs.get('reasoning_effort', '')}_{kwargs.get('output_mode', '')}"
    if cache_key not in st.session_state.engines:
        engine = GroqEngine(
            model_id=model_id,
            api_key=api_key,
            system_prompt=st.session_state.system_prompt,
            **kwargs,
        )
        st.session_state.engines[cache_key] = engine
    return st.session_state.engines[cache_key]


def reset_engines():
    """Clear all cached engines (on settings change)."""
    st.session_state.engines = {}


def get_all_actor_models():
    """Get all actor models from all available providers."""
    registry = get_registry()
    actors = registry.list_actor_models()
    # Build a list of (display_label, provider_name, model_id, model_info)
    result = []
    for m in actors:
        label = f"{m.provider}/{m.id}"
        result.append({
            "label": label,
            "display": f"{m.name} ({m.speed}) [{m.provider}]",
            "provider": m.provider,
            "model_id": m.id,
            "model_info": m,
        })
    return result


def stream_from_provider(provider_name, model_id, prompt, history,
                         reasoning_effort="", show_reasoning=True, max_tokens=4096):
    """
    Stream a response from any provider. Yields same dict format as GroqEngine:
    {"type": "reasoning"/"content"/"done"/"error", ...}

    Args:
        history: Clean message list (role+content only), WITHOUT the new user prompt.
        prompt: The new user message to send.
    """
    from providers.base import ProviderError

    registry = get_registry()
    provider = registry.get_provider(provider_name)
    if not provider:
        yield {"type": "error", "error": f"Provider '{provider_name}' not available"}
        return

    msgs = list(history) + [{"role": "user", "content": prompt}]

    content_buf = ""
    reasoning_buf = ""

    try:
        for chunk in provider.stream_chat(
            model_id=model_id,
            messages=msgs,
            reasoning_effort=reasoning_effort,
            show_reasoning=show_reasoning,
            max_tokens=max_tokens,
        ):
            if chunk.is_done:
                usage = chunk.usage
                yield {
                    "type": "done",
                    "stats": {
                        "content": content_buf,
                        "reasoning": reasoning_buf,
                        "ttft_ms": usage.ttft_ms if usage else 0,
                        "ttfc_ms": usage.ttfc_ms if usage else 0,
                        "total_ms": usage.total_ms if usage else 0,
                        "cached_tokens": usage.cached_tokens if usage else 0,
                        "prompt_tokens": usage.prompt_tokens if usage else 0,
                        "completion_tokens": usage.completion_tokens if usage else 0,
                    },
                }
            elif chunk.is_reasoning:
                reasoning_buf += chunk.text
                if show_reasoning:
                    yield {"type": "reasoning", "text": chunk.text}
            elif chunk.is_content:
                content_buf += chunk.text
                yield {"type": "content", "text": chunk.text}
    except ProviderError as e:
        yield {"type": "error", "error": str(e)}
    except Exception as e:
        yield {"type": "error", "error": str(e)}


def get_director() -> Director | None:
    """Create a Director instance if enabled and a provider is available."""
    if not st.session_state.director_enabled:
        return None

    registry = get_registry()
    result = registry.find_default_director()
    if not result:
        return None

    provider, model_info = result
    return Director(
        provider=provider,
        model_id=model_info.id,
    )


# ── Director Insights Component ─────────────────────────────────────────────

def render_director_insights(data: dict, msg_index: int):
    """Render Director JSON as a rich panel."""
    if not data:
        st.caption("Director: no data")
        return

    # Intent
    intent = data.get("intent", {})
    col1, col2 = st.columns(2)
    col1.metric("Intent", intent.get("type", "-"))
    col2.metric("Complexity", intent.get("complexity", "-"))
    st.caption(f"Topic: {intent.get('topic', '-')}")

    st.divider()

    # Analysis
    analysis = data.get("response_analysis", {})
    col1, col2, col3 = st.columns(3)
    confidence = analysis.get("confidence", 0)
    col1.metric("Confidence", f"{confidence:.0%}")
    col2.metric("Sentiment", analysis.get("sentiment", "-"))
    col3.metric("Language", analysis.get("language_detected", "-"))

    entities = analysis.get("key_entities", [])
    if entities:
        st.markdown("**Key Entities:** " + " · ".join(entities))

    st.divider()

    # Follow-ups (clickable)
    suggestions = data.get("suggestions", {})
    follow_ups = suggestions.get("follow_ups", [])
    if follow_ups:
        st.markdown("**Suggested Follow-ups:**")
        for i, q in enumerate(follow_ups):
            if st.button(f"→ {q}", key=f"followup_{msg_index}_{i}"):
                st.session_state.next_prompt = q
                st.rerun()

    related = suggestions.get("related_topics", [])
    if related:
        st.caption("Related: " + " · ".join(related))

    # Guardrails
    guardrails = data.get("guardrails", {})
    flags = guardrails.get("flags", ["none"])
    if flags and flags != ["none"]:
        st.warning(f"Flags: {', '.join(flags)}")
        note = guardrails.get("note", "")
        if note:
            st.caption(note)


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚡ Actor-Director")

    mode = st.radio("Mode", ["Chat", "Compare Models"], horizontal=True)

    st.divider()

    # ── Actor Settings ──
    st.subheader("Actor (Streaming)")

    # Build actor model list from all providers
    all_actors = get_all_actor_models()
    actor_labels = [a["label"] for a in all_actors]
    actor_display = {a["label"]: a["display"] for a in all_actors}
    actor_lookup = {a["label"]: a for a in all_actors}

    if mode == "Chat":
        selected_actor = st.selectbox(
            "Actor Model",
            actor_labels,
            format_func=lambda x: actor_display.get(x, x),
        )
        actor_entry = actor_lookup[selected_actor]
        actor_provider_name = actor_entry["provider"]
        actor_model_id = actor_entry["model_id"]
        actor_model_info = actor_entry["model_info"]

        # For Groq models, keep backward compat with model_id used by GroqEngine
        is_groq_actor = actor_provider_name == "groq"
        # Legacy model_id for GroqEngine compat
        model_id = actor_model_id if is_groq_actor else None

        reasoning_effort = ""
    else:
        st.caption("Sends prompt to all actor models in parallel")
        reasoning_effort = ""
        is_groq_actor = True  # Compare mode uses all models

    show_reasoning = False
    max_tokens = st.slider("Max Tokens", 256, 8192, 4096, step=256)

    st.divider()

    # ── Director Settings (auto-detect, no model picker) ──
    st.subheader("Director (JSON)")

    director_enabled = st.checkbox("Enable Director", value=st.session_state.director_enabled)
    if director_enabled != st.session_state.director_enabled:
        st.session_state.director_enabled = director_enabled
        st.rerun()

    if st.session_state.director_enabled:
        registry = get_registry()
        default_dir = registry.find_default_director()
        if default_dir:
            provider, model_info = default_dir
            st.caption(f"Auto-selected: **{model_info.name}** ({provider.name})")
        else:
            st.warning("No Director model available. Set OPENAI_API_KEY for GPT-4.1-nano or use Groq GPT-OSS.")
            st.session_state.director_enabled = False

    # ── Provider Status ──
    st.divider()
    st.subheader("Providers")
    registry = get_registry()
    for s in registry.provider_status():
        icon = "✅" if s["available"] else "❌"
        st.caption(f"{icon} **{s['name']}** — {s['model_count']} models" if s["available"]
                   else f"{icon} **{s['name']}** — {'no key' if not s['has_key'] else 'no package'}")

    st.divider()

    # ── System Prompt ──
    st.subheader("System Prompt")
    uploaded_prompt = st.file_uploader("Upload (.txt, .md)", type=["txt", "md"])
    if uploaded_prompt:
        content = uploaded_prompt.read().decode("utf-8")
        if content != st.session_state.system_prompt:
            st.session_state.system_prompt = content
            st.session_state.system_prompt_name = uploaded_prompt.name
            reset_engines()
            st.rerun()

    if st.session_state.system_prompt:
        tokens = estimate_tokens(st.session_state.system_prompt)
        st.success(f"Loaded: {st.session_state.system_prompt_name} (~{tokens:,} tokens)")
        with st.expander("Preview"):
            st.text(st.session_state.system_prompt[:500] + "...")
        if st.button("Remove System Prompt"):
            st.session_state.system_prompt = None
            st.session_state.system_prompt_name = None
            reset_engines()
            st.rerun()

    st.divider()

    # ── Cache Status ──
    st.subheader("Prompt Cache")
    ct = st.session_state.cache_tracker
    if ct["last_request_time"]:
        import time as _time
        elapsed = _time.time() - ct["last_request_time"]
        # Groq TTL = 2 hours (7200s), OpenAI TTL = 5-10 min (~600s)
        groq_remaining = max(0, 7200 - elapsed)
        openai_remaining = max(0, 600 - elapsed)

        if groq_remaining > 0:
            groq_mins = int(groq_remaining // 60)
            st.caption(f"Groq cache: **{groq_mins}m** remaining (TTL 2hr)")
            st.progress(groq_remaining / 7200)
        else:
            st.caption("Groq cache: **expired**")

        if openai_remaining > 0:
            openai_mins = int(openai_remaining // 60)
            st.caption(f"OpenAI cache: **{openai_mins}m** remaining (TTL ~10m)")
            st.progress(openai_remaining / 600)
        else:
            st.caption("OpenAI cache: **expired**")

        if ct["total_prompt_tokens"] > 0:
            rate = ct["total_cached_tokens"] * 100 // ct["total_prompt_tokens"]
            st.caption(f"Hit rate: {rate}% ({ct['total_cached_tokens']:,}/{ct['total_prompt_tokens']:,} tokens)")
            st.caption(f"Savings: ~${ct['estimated_savings']:.4f}")
    else:
        st.caption("No requests yet")

    st.divider()

    # ── Session Stats ──
    st.subheader("Session Stats")
    ss = st.session_state.session_stats
    col1, col2 = st.columns(2)
    col1.metric("Turns", ss["turn_count"])
    col2.metric("Cost", f"${ss['total_cost']:.4f}")

    if ss["ttft_history"]:
        st.caption("TTFT trend:")
        st.line_chart(ss["ttft_history"], height=60)

    if ss["director_total"] > 0:
        rate = ss["director_passes"] / max(1, ss["director_total"])
        st.progress(rate, text=f"Director schema: {rate:.0%} ({ss['director_passes']}/{ss['director_total']})")

    st.divider()

    # ── Actions ──
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.compare_results = []
            st.session_state.director_results = {}
            for engine in st.session_state.engines.values():
                engine.clear_history()
            st.rerun()
    with col2:
        if st.button("Warm Cache", use_container_width=True):
            if mode == "Chat" and is_groq_actor and model_id:
                engine = get_engine(
                    model_id,
                    reasoning_effort=reasoning_effort,
                    show_reasoning=show_reasoning,
                    max_completion_tokens=max_tokens,
                )
                result = engine.warm_cache()
                if result["status"] == "ok":
                    st.toast(f"Cache primed in {result['elapsed_ms']}ms", icon="✅")
                else:
                    st.toast(result.get("reason", result.get("error", "Failed")), icon="⚠️")
            elif mode == "Chat":
                st.toast("Cache warmup only available for Groq models", icon="ℹ️")


# ── Chat Mode ───────────────────────────────────────────────────────────────

if mode == "Chat":
    # Header
    header_parts = [f"Chat — {actor_model_info.name} [{actor_provider_name}]"]
    if st.session_state.director_enabled:
        header_parts.append("+ Director")
    st.header(" ".join(header_parts))

    # Display chat history
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            # Check if we have Director insights for this message
            dir_data = st.session_state.director_results.get(idx)

            if dir_data:
                # 70/30 split: actor response + director insights
                actor_col, director_col = st.columns([7, 3])
                with actor_col:
                    with st.chat_message("assistant"):
                        if msg.get("reasoning") and show_reasoning:
                            with st.expander("💭 Reasoning", expanded=False):
                                st.markdown(msg["reasoning"])
                        st.markdown(msg["content"])
                        if msg.get("stats"):
                            s = msg["stats"]
                            cols = st.columns(4)
                            cols[0].caption(f"TTFT: {s.get('ttft_ms', '-')}ms")
                            cols[1].caption(f"Total: {s.get('total_ms', '-')}ms")
                            cols[2].caption(f"Cached: {s.get('cached_tokens', 0)}/{s.get('prompt_tokens', 0)}")
                            cols[3].caption(f"Output: {s.get('completion_tokens', '-')} tokens")
                with director_col:
                    with st.container(border=True):
                        st.markdown("**📋 Director Insights**")
                        if dir_data.get("_meta"):
                            st.caption(f"Model: {dir_data['_meta'].get('model', '?')} | {dir_data['_meta'].get('total_ms', '?')}ms | Schema: {'���' if dir_data['_meta'].get('schema_valid') else '⚠️'}")
                        render_director_insights(dir_data, idx)
            else:
                # No Director — full width
                with st.chat_message("assistant"):
                    if msg.get("reasoning") and show_reasoning:
                        with st.expander("���� Reasoning", expanded=False):
                            st.markdown(msg["reasoning"])
                    st.markdown(msg["content"])
                    if msg.get("stats"):
                        s = msg["stats"]
                        cols = st.columns(4)
                        cols[0].caption(f"TTFT: {s.get('ttft_ms', '-')}ms")
                        cols[1].caption(f"Total: {s.get('total_ms', '-')}ms")
                        cols[2].caption(f"Cached: {s.get('cached_tokens', 0)}/{s.get('prompt_tokens', 0)}")
                        cols[3].caption(f"Output: {s.get('completion_tokens', '-')} tokens")

    # Check for follow-up prompt from Director suggestion click
    prompt = st.session_state.next_prompt
    st.session_state.next_prompt = None

    # Chat input
    if prompt is None:
        prompt = st.chat_input("Type your message...")

    if prompt:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build the streaming source — Groq uses GroqEngine, others use provider layer
        if is_groq_actor and model_id:
            engine = get_engine(
                model_id,
                reasoning_effort=reasoning_effort,
                show_reasoning=show_reasoning,
                max_completion_tokens=max_tokens,
            )
            actor_stream = engine.stream_response(prompt)
        else:
            # Non-Groq provider: stream via provider layer
            # Build clean message history (role + content only), excluding the user msg we just added
            clean_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]  # exclude the user prompt we just appended
            ]
            actor_stream = stream_from_provider(
                actor_provider_name, actor_model_id, prompt, clean_history,
                reasoning_effort=reasoning_effort,
                show_reasoning=show_reasoning,
                max_tokens=max_tokens,
            )

        # Launch Director in background (if enabled)
        director_future = None
        director_instance = get_director()
        if director_instance:
            executor = ThreadPoolExecutor(max_workers=1)
            director_future = executor.submit(
                director_instance.analyze,
                st.session_state.messages,
            )

        # Determine layout
        if st.session_state.director_enabled:
            actor_col, director_col = st.columns([7, 3])
        else:
            actor_col = st.container()
            director_col = None

        # Stream Actor response
        with actor_col:
            with st.chat_message("assistant"):
                reasoning_placeholder = st.empty()
                content_placeholder = st.empty()
                stats_placeholder = st.empty()

                reasoning_buf = ""
                content_buf = ""
                final_stats = None

                for chunk in actor_stream:
                    if chunk["type"] == "reasoning":
                        reasoning_buf += chunk["text"]
                        with reasoning_placeholder.expander("💭 Reasoning", expanded=True):
                            st.markdown(reasoning_buf)
                    elif chunk["type"] == "content":
                        content_buf += chunk["text"]
                        content_placeholder.markdown(content_buf + "▌")
                    elif chunk["type"] == "done":
                        final_stats = chunk["stats"]
                        content_placeholder.markdown(content_buf)
                    elif chunk["type"] == "error":
                        st.error(f"API Error: {chunk['error']}")

                if final_stats:
                    cols = stats_placeholder.columns(4)
                    cols[0].caption(f"TTFT: {final_stats['ttft_ms']}ms")
                    cols[1].caption(f"Total: {final_stats['total_ms']}ms")
                    cols[2].caption(f"Cached: {final_stats['cached_tokens']}/{final_stats['prompt_tokens']}")
                    cols[3].caption(f"Output: {final_stats['completion_tokens']} tokens")

                    msg_index = len(st.session_state.messages)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": content_buf,
                        "reasoning": reasoning_buf if reasoning_buf else None,
                        "stats": final_stats,
                    })

                    # Update session stats
                    ss = st.session_state.session_stats
                    ss["turn_count"] += 1
                    ss["ttft_history"].append(final_stats.get("ttft_ms", 0))

                    # Update cache tracker
                    import time as _time
                    ct = st.session_state.cache_tracker
                    ct["last_request_time"] = _time.time()
                    cached = final_stats.get("cached_tokens", 0)
                    prompt_t = final_stats.get("prompt_tokens", 0)
                    ct["total_cached_tokens"] += cached
                    ct["total_prompt_tokens"] += prompt_t
                    if cached > 0:
                        ct["cache_warm"] = True
                        # 50% discount on cached input tokens
                        ct["estimated_savings"] += (cached / 1_000_000) * 0.15 * 0.5

        # Collect Director result
        if director_future and director_col:
            with director_col:
                with st.container(border=True):
                    st.markdown("**📋 Director Insights**")
                    with st.spinner("Analyzing..."):
                        try:
                            dir_result: DirectorResult = director_future.result(timeout=30)
                        except Exception as e:
                            dir_result = DirectorResult(success=False, error=str(e))
                        finally:
                            executor.shutdown(wait=False)

                    if dir_result.success and dir_result.data:
                        # Show Director model/timing
                        st.caption(
                            f"Model: {director_instance.model_id} | "
                            f"{dir_result.total_ms:.0f}ms | "
                            f"Schema: {'✅' if dir_result.schema_valid else '⚠️'}"
                        )
                        render_director_insights(dir_result.data, msg_index)

                        # Save for history replay
                        dir_save = dict(dir_result.data)
                        dir_save["_meta"] = {
                            "model": director_instance.model_id,
                            "total_ms": round(dir_result.total_ms),
                            "schema_valid": dir_result.schema_valid,
                        }
                        st.session_state.director_results[msg_index] = dir_save

                        # Update session stats
                        ss = st.session_state.session_stats
                        ss["director_total"] += 1
                        if dir_result.schema_valid:
                            ss["director_passes"] += 1
                        model = director_instance.provider.get_model(director_instance.model_id)
                        if model and dir_result.prompt_tokens:
                            cost = model.estimate_cost(dir_result.prompt_tokens, dir_result.completion_tokens)
                            ss["director_cost"] += cost
                            ss["total_cost"] = ss["actor_cost"] + ss["director_cost"]
                    else:
                        st.warning(f"Director failed: {dir_result.error or 'unknown error'}")

        # Combined stats line
        if final_stats and st.session_state.director_enabled:
            ss = st.session_state.session_stats
            st.caption(
                f"💰 Turn cost: ~${ss.get('total_cost', 0):.4f} total | "
                f"Actor: ${ss.get('actor_cost', 0):.4f} | "
                f"Director: ${ss.get('director_cost', 0):.4f}"
            )


# ── Compare Mode ────────────────────────────────────────────────────────────

elif mode == "Compare Models":
    st.header("Model Comparison")
    st.caption("Same prompt sent to all models in parallel")

    # ── Show previous comparisons (clean layout) ──
    for comp in st.session_state.compare_results:
        st.divider()
        st.markdown(f"**Prompt:** {comp['prompt']}")

        # Summary table first — the key data
        if comp.get("summary"):
            st.dataframe(comp["summary"], use_container_width=True, hide_index=True)

        # Director summary
        if comp.get("director_summary"):
            with st.expander("Director (JSON) Results"):
                st.dataframe(comp["director_summary"], use_container_width=True, hide_index=True)

        # Individual responses in tabs
        actor_items = list(comp["results"].items())
        if actor_items:
            tab_names = [k for k, _ in actor_items]
            tabs = st.tabs(tab_names)
            for tab, (key, result) in zip(tabs, actor_items):
                with tab:
                    if result.get("type") == "error":
                        st.error(result["error"])
                    else:
                        reasoning = result.get("reasoning", "")
                        if reasoning and show_reasoning:
                            with st.expander("💭 Reasoning"):
                                st.markdown(reasoning)
                        st.markdown(result.get("content", ""))

    # ── Compare input ──
    if prompt := st.chat_input("Type a prompt to compare across all models..."):
        st.divider()
        st.markdown(f"**Prompt:** {prompt}")

        # Get all actor models
        registry = get_registry()
        compare_actors = registry.list_actor_models()
        num_actors = len(compare_actors)

        # Progress bar
        progress_bar = st.progress(0, text="Running models...")
        status_text = st.empty()

        # Capture for threads
        _system_prompt = st.session_state.system_prompt

        def run_actor_compare(actor_model):
            key = f"{actor_model.provider}/{actor_model.id}"
            provider = registry.get_provider(actor_model.provider)
            if not provider:
                return key, {"type": "error", "error": f"Provider {actor_model.provider} not available"}

            msgs = []
            if _system_prompt:
                msgs.append({"role": "system", "content": _system_prompt})
            msgs.append({"role": "user", "content": prompt})

            content = ""
            reasoning = ""
            stats = {}
            try:
                for chunk in provider.stream_chat(
                    model_id=actor_model.id,
                    messages=msgs,
                    reasoning_effort=actor_model.default_effort or reasoning_effort,
                    show_reasoning=show_reasoning,
                    max_tokens=max_tokens,
                ):
                    if chunk.is_done and chunk.usage:
                        u = chunk.usage
                        stats = {
                            "ttft_ms": u.ttft_ms, "ttfc_ms": u.ttfc_ms,
                            "total_ms": u.total_ms, "prompt_tokens": u.prompt_tokens,
                            "completion_tokens": u.completion_tokens,
                            "cached_tokens": u.cached_tokens,
                        }
                    elif chunk.is_reasoning:
                        reasoning += chunk.text
                    elif chunk.is_content:
                        content += chunk.text
            except Exception as e:
                return key, {"type": "error", "error": str(e)}
            return key, {"type": "done", "content": content, "reasoning": reasoning, "stats": stats}

        def run_director(provider, model_info):
            try:
                d = Director(provider=provider, model_id=model_info.id)
                clean_msgs = []
                if _system_prompt:
                    clean_msgs.append({"role": "system", "content": _system_prompt})
                clean_msgs.append({"role": "user", "content": prompt})
                result = d.analyze(clean_msgs)
                return {
                    "success": result.success, "data": result.data,
                    "total_ms": result.total_ms, "schema_valid": result.schema_valid,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens, "error": result.error,
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        def run_unified(provider, model_info, schema):
            """Run OpenAI stream+JSON in a single call."""
            try:
                msgs = []
                if _system_prompt:
                    msgs.append({"role": "system", "content": _system_prompt})
                msgs.append({"role": "user", "content": prompt})

                content = ""
                stats = {}
                for chunk in provider.stream_json_chat(
                    model_id=model_info.id,
                    messages=msgs,
                    schema=schema,
                    temperature=0.6,
                    max_tokens=max_tokens,
                ):
                    if chunk.is_done and chunk.usage:
                        u = chunk.usage
                        stats = {
                            "ttft_ms": u.ttft_ms, "ttfc_ms": u.ttfc_ms,
                            "total_ms": u.total_ms, "prompt_tokens": u.prompt_tokens,
                            "completion_tokens": u.completion_tokens,
                            "cached_tokens": u.cached_tokens,
                        }
                    elif chunk.is_content:
                        content += chunk.text

                # Parse the streamed JSON
                import json as _json
                parsed = None
                schema_valid = False
                try:
                    parsed = _json.loads(content)
                    schema_valid = True
                except Exception:
                    pass

                return {
                    "success": True, "content": content, "data": parsed,
                    "schema_valid": schema_valid, "stats": stats, "error": "",
                }
            except Exception as e:
                return {"success": False, "error": str(e), "stats": {}, "data": None}

        # ── Run all actors + directors + unified in parallel ──
        actor_results = {}
        director_results_map = {}
        unified_results_map = {}
        actor_keys = [f"{am.provider}/{am.id}" for am in compare_actors]

        director_models = registry.list_director_models()

        # Find OpenAI models that support stream+JSON (unified)
        from providers.openai_provider import OpenAIProvider
        openai_provider = registry.get_provider("openai")
        unified_models = []
        if openai_provider and isinstance(openai_provider, OpenAIProvider):
            unified_models = [m for m in openai_provider.list_models() if m.supports_json_schema]

        # Load director schema for unified calls
        from director import Director as _Dir
        _default_schema = _Dir._load_default_schema()

        total_jobs = num_actors + len(director_models) + len(unified_models)
        completed = 0

        with ThreadPoolExecutor(max_workers=min(total_jobs, 10)) as executor:
            # Submit actors
            actor_futures = {executor.submit(run_actor_compare, am): am for am in compare_actors}
            # Submit directors
            dir_futures = {}
            for dm in director_models:
                prov = registry.get_provider(dm.provider)
                if prov:
                    label = f"{dm.provider}/{dm.id}"
                    dir_futures[executor.submit(run_director, prov, dm)] = (label, dm)
            # Submit unified (OpenAI stream+JSON)
            unified_futures = {}
            for um in unified_models:
                label = f"openai/{um.id}"
                unified_futures[executor.submit(run_unified, openai_provider, um, _default_schema)] = (label, um)

            all_futures = list(actor_futures.keys()) + list(dir_futures.keys()) + list(unified_futures.keys())
            for future in as_completed(all_futures):
                completed += 1
                progress_bar.progress(completed / total_jobs, text=f"Completed {completed}/{total_jobs}")

                if future in actor_futures:
                    am = actor_futures[future]
                    key = f"{am.provider}/{am.id}"
                    try:
                        _, result = future.result()
                        actor_results[key] = result
                    except Exception as e:
                        actor_results[key] = {"type": "error", "error": str(e)}
                    status_text.caption(f"Actor done: {key}")

                elif future in dir_futures:
                    label, dm = dir_futures[future]
                    try:
                        d_result = future.result()
                    except Exception as e:
                        d_result = {"success": False, "error": str(e)}
                    director_results_map[label] = d_result
                    status_text.caption(f"Director done: {label}")

                elif future in unified_futures:
                    label, um = unified_futures[future]
                    try:
                        u_result = future.result()
                    except Exception as e:
                        u_result = {"success": False, "error": str(e), "stats": {}, "data": None}
                    unified_results_map[label] = u_result
                    status_text.caption(f"Unified done: {label}")

        progress_bar.empty()
        status_text.empty()

        # ── Actor Summary Table ──
        st.subheader("Actor Results (Streaming)")
        actor_summary = []
        for key in actor_keys:
            r = actor_results.get(key, {})
            s = r.get("stats", {})
            actor_summary.append({
                "Model": key,
                "TTFT (ms)": round(s["ttft_ms"]) if isinstance(s.get("ttft_ms"), (int, float)) else "-",
                "Total (ms)": round(s["total_ms"]) if isinstance(s.get("total_ms"), (int, float)) else "-",
                "Tokens Out": s.get("completion_tokens", "-"),
                "Cached": s.get("cached_tokens", 0),
                "Status": "OK" if r.get("type") != "error" else "ERROR",
            })
        st.dataframe(actor_summary, use_container_width=True, hide_index=True)

        # ── Director Summary Table ──
        director_summary = []
        if director_results_map:
            st.subheader("Director Results (JSON)")
            for label, d_result in director_results_map.items():
                director_summary.append({
                    "Model": label,
                    "Total (ms)": round(d_result["total_ms"]) if d_result.get("success") else "-",
                    "Schema": "Pass" if d_result.get("schema_valid") else "Fail",
                    "Tokens Out": d_result.get("completion_tokens", "-"),
                    "Status": "OK" if d_result.get("success") else "ERROR",
                })
            st.dataframe(director_summary, use_container_width=True, hide_index=True)

            # Director JSON expandable
            for label, d_result in director_results_map.items():
                if d_result.get("success") and d_result.get("data"):
                    with st.expander(f"Director JSON — {label}"):
                        st.json(d_result["data"])

        # ── Unified (Stream + JSON) — OpenAI only ──
        unified_summary = []
        if unified_results_map:
            st.subheader("Unified (Stream + JSON) — Single Call")
            st.caption("OpenAI exclusive: streaming TTFT + strict JSON schema in one API call")

            for label, u_result in unified_results_map.items():
                s = u_result.get("stats", {})
                unified_summary.append({
                    "Model": label,
                    "TTFT (ms)": round(s["ttft_ms"]) if isinstance(s.get("ttft_ms"), (int, float)) else "-",
                    "Total (ms)": round(s["total_ms"]) if isinstance(s.get("total_ms"), (int, float)) else "-",
                    "Tokens Out": s.get("completion_tokens", "-"),
                    "Schema": "Pass" if u_result.get("schema_valid") else "Fail",
                    "Status": "OK" if u_result.get("success") else "ERROR",
                })
            st.dataframe(unified_summary, use_container_width=True, hide_index=True)

            # Unified JSON expandable
            for label, u_result in unified_results_map.items():
                if u_result.get("data"):
                    with st.expander(f"Unified JSON — {label}"):
                        st.json(u_result["data"])
                elif u_result.get("error"):
                    st.error(f"{label}: {u_result['error']}")

        # ── Head-to-Head: Actor+Director vs Unified ──
        if unified_results_map and director_results_map and actor_results:
            st.subheader("Head-to-Head: Actor+Director vs Unified")
            st.caption("Actor+Director = 2 parallel calls (stream + JSON). Unified = 1 call (stream JSON).")

            h2h_data = []

            # Best actor+director pair
            best_actor = None
            best_actor_ms = float("inf")
            for key, r in actor_results.items():
                if r.get("type") != "error":
                    ms = r.get("stats", {}).get("total_ms", float("inf"))
                    if ms < best_actor_ms:
                        best_actor_ms = ms
                        best_actor = (key, r)

            best_director = None
            best_dir_ms = float("inf")
            for label, d in director_results_map.items():
                if d.get("success"):
                    ms = d.get("total_ms", float("inf"))
                    if ms < best_dir_ms:
                        best_dir_ms = ms
                        best_director = (label, d)

            if best_actor and best_director:
                a_key, a_r = best_actor
                d_label, d_r = best_director
                a_stats = a_r.get("stats", {})
                wall_ms = max(a_stats.get("total_ms", 0), d_r.get("total_ms", 0))
                h2h_data.append({
                    "Approach": f"Actor+Director ({a_key} + {d_label})",
                    "API Calls": 2,
                    "User TTFT (ms)": round(a_stats.get("ttft_ms", 0)),
                    "Wall Time (ms)": round(wall_ms),
                    "Has Streaming": "Yes",
                    "Has JSON": "Yes",
                    "Schema": "Pass" if d_r.get("schema_valid") else "Fail",
                })

            for label, u_r in unified_results_map.items():
                if u_r.get("success"):
                    s = u_r.get("stats", {})
                    h2h_data.append({
                        "Approach": f"Unified ({label})",
                        "API Calls": 1,
                        "User TTFT (ms)": round(s.get("ttft_ms", 0)),
                        "Wall Time (ms)": round(s.get("total_ms", 0)),
                        "Has Streaming": "Yes",
                        "Has JSON": "Yes",
                        "Schema": "Pass" if u_r.get("schema_valid") else "Fail",
                    })

            if h2h_data:
                st.dataframe(h2h_data, use_container_width=True, hide_index=True)

        # ── Model Responses in Tabs (one tab per model, full width) ──
        st.subheader("Full Responses")
        tab_names = actor_keys
        tabs = st.tabs(tab_names)
        for tab, key in zip(tabs, actor_keys):
            with tab:
                r = actor_results.get(key, {})
                if r.get("type") == "error":
                    st.error(r["error"])
                else:
                    # Metrics row
                    s = r.get("stats", {})
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("TTFT", f"{s.get('ttft_ms', '-')}ms")
                    c2.metric("Total", f"{s.get('total_ms', '-')}ms")
                    c3.metric("Tokens Out", s.get("completion_tokens", "-"))
                    c4.metric("Cached", f"{s.get('cached_tokens', 0)}/{s.get('prompt_tokens', 0)}")

                    # Reasoning
                    reasoning = r.get("reasoning", "")
                    if reasoning and show_reasoning:
                        with st.expander("💭 Reasoning", expanded=False):
                            st.markdown(reasoning)

                    # Response
                    st.markdown(r.get("content", ""))

        # Save comparison
        st.session_state.compare_results.append({
            "prompt": prompt,
            "results": actor_results,
            "director_results": director_results_map,
            "unified_results": unified_results_map,
            "summary": actor_summary,
            "director_summary": director_summary,
            "unified_summary": unified_summary,
        })
