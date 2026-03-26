"""
Groq Chatbot — Streamlit Web UI

Run: streamlit run app.py
"""

import os
import time
import json
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from groq_engine import GroqEngine, MODELS, estimate_tokens


# ── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Groq Reasoning Chatbot",
    page_icon="⚡",
    layout="wide",
)


# ── Session State Init ──────────────────────────────────────────────────────

def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "engines" not in st.session_state:
        st.session_state.engines = {}
    if "compare_results" not in st.session_state:
        st.session_state.compare_results = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = None
    if "system_prompt_name" not in st.session_state:
        st.session_state.system_prompt_name = None

init_state()


# ── Info Dialog ─────────────────────────────────────────────────────────────

@st.dialog("Groq Platform — Capabilities & Trade-offs", width="large")
def show_info_dialog():
    st.markdown("""
### Model Capabilities

| Feature | Qwen3-32B | GPT-OSS 20B | GPT-OSS 120B |
|---|---|---|---|
| **Streaming** | Yes | Yes | Yes |
| **Reasoning** | Yes | Yes | Yes |
| **Tool Calling** | Yes | Yes | Yes |
| **JSON Schema (strict)** | No | Yes | Yes |
| **Prompt Caching** | No | Yes (Dev tier) | Yes (Dev tier) |
| **Multilingual** | 119 langs | 14+ langs | 81+ langs |
| **Speed** | ~400 t/s | ~1000 t/s | ~500 t/s |
| **Max Output** | 40,960 | 65,536 | 65,536 |

---

### Key Trade-offs

**Streaming + JSON are mutually exclusive on Groq.**
You must pick one per request — stream for real-time chat, or JSON for structured output (non-streaming). This is Groq-specific; OpenAI supports both together.

**Tool Calling + Structured Output are mutually exclusive.**
Cannot use `response_format` and `tools` in the same request.

**Prompt Caching — GPT-OSS models only, all tiers.**
Caching is automatic on GPT-OSS models and works on the free tier too — no code changes needed. Cached tokens are 50% cheaper and **don't count toward rate limits** (huge for free tier). Cache expires after ~2 hours of inactivity. First request primes the cache, subsequent requests benefit.

**`strict: true` JSON Schema — GPT-OSS only.**
Qwen3 does not support strict schema enforcement. Falls back to `json_object` mode with schema injected as text guidance.

---

### Free Tier Limits

| Constraint | Limit |
|---|---|
| Tokens per minute (TPM) | ~6,000–8,000 (model-dependent) |
| Tokens per day (TPD) | ~100,000–500,000 |
| Requests per minute | 30 |
| Prompt caching | Available (GPT-OSS models) |
| Max output per request | ~8,000 tokens |

**Impact:** A 15K-token system prompt may exceed the free tier TPM limit on the first request. However, once cached, subsequent requests won't count those tokens toward TPM. Use "Warm Cache" to prime the cache before chatting.

---

### Recommendations

- **English + speed critical →** GPT-OSS 20B (fastest TTFT, cheapest)
- **Multilingual (Hindi/Arabic) →** Qwen3-32B (only model with strong non-English support)
- **Complex reasoning →** GPT-OSS 120B (best quality, near o4-mini parity)
- **Free tier →** Keep system prompts under 3K tokens, prefer GPT-OSS 20B
- **Production →** Upgrade to Developer tier for caching + higher rate limits
""")


# ── Top Bar ─────────────────────────────────────────────────────────────────

top_left, top_right = st.columns([9, 1])
with top_left:
    st.caption("⚡ Groq Reasoning Chatbot")
with top_right:
    if st.button("ℹ️ Info"):
        show_info_dialog()


# ── Helper: Get or Create Engine ────────────────────────────────────────────

def get_engine(model_id: str, **kwargs) -> GroqEngine:
    """Get cached engine or create new one."""
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


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚡ Groq Chatbot")

    # Mode toggle
    mode = st.radio("Mode", ["Chat", "Compare Models"], horizontal=True)

    st.divider()

    # Model selection (chat mode only)
    if mode == "Chat":
        model_id = st.selectbox(
            "Model",
            list(MODELS.keys()),
            format_func=lambda x: f"{MODELS[x]['name']} ({MODELS[x]['speed']})",
        )
        model_info = MODELS[model_id]

        # Reasoning effort
        effort_options = model_info["reasoning_efforts"]
        reasoning_effort = st.selectbox(
            "Reasoning Effort",
            effort_options,
            index=effort_options.index(model_info["default_effort"]),
        )

    else:
        st.caption("Sends prompt to all 3 models in parallel")
        reasoning_effort = st.selectbox(
            "Reasoning Effort (GPT-OSS)",
            ["low", "medium", "high"],
            index=1,
        )

    # Common settings
    show_reasoning = st.checkbox("Show Reasoning", value=True)
    output_mode = st.radio("Output Mode", ["Stream", "JSON"], horizontal=True)
    max_tokens = st.slider("Max Tokens", 256, 8192, 4096, step=256)

    # JSON Schema — show prominently when JSON mode is selected
    json_schema = None
    if "json_schema" not in st.session_state:
        st.session_state.json_schema = None
        st.session_state.json_schema_name = None

    if output_mode == "JSON":
        st.divider()
        st.subheader("JSON Schema (Required)")
        uploaded_schema = st.file_uploader("Upload JSON schema (.json)", type=["json"], key="json_uploader")
        if uploaded_schema:
            try:
                schema_content = json.loads(uploaded_schema.read().decode("utf-8"))
                st.session_state.json_schema = schema_content
                st.session_state.json_schema_name = uploaded_schema.name
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON schema: {e}")

        if st.session_state.json_schema:
            json_schema = st.session_state.json_schema
            st.success(f"Schema: {st.session_state.json_schema_name}")
            with st.expander("Preview Schema"):
                st.json(json_schema)
            if st.button("Remove Schema"):
                st.session_state.json_schema = None
                st.session_state.json_schema_name = None
                reset_engines()
                st.rerun()
        else:
            st.warning("Upload a JSON schema file to use JSON output mode. Without a schema, responses will be basic JSON (no structure enforcement).")

    st.divider()

    # System prompt
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

    # Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", width="stretch"):
            st.session_state.messages = []
            st.session_state.compare_results = []
            for engine in st.session_state.engines.values():
                engine.clear_history()
            st.rerun()
    with col2:
        if st.button("Warm Cache", width="stretch"):
            if mode == "Chat":
                engine = get_engine(
                    model_id,
                    reasoning_effort=reasoning_effort,
                    show_reasoning=show_reasoning,
                    max_completion_tokens=max_tokens,
                    json_mode=(output_mode == "JSON"),
                    json_schema=json_schema,
                )
                result = engine.warm_cache()
                if result["status"] == "ok":
                    st.toast(f"Cache primed in {result['elapsed_ms']}ms", icon="✅")
                else:
                    st.toast(result.get("reason", result.get("error", "Failed")), icon="⚠️")
            else:
                for mid in MODELS:
                    e = get_engine(mid, reasoning_effort=reasoning_effort if "default" not in MODELS[mid]["reasoning_efforts"] else MODELS[mid]["default_effort"],
                                   show_reasoning=show_reasoning, max_completion_tokens=max_tokens)
                    r = e.warm_cache()
                    status = "✅" if r["status"] == "ok" else "⏭️"
                    st.toast(f"{status} {MODELS[mid]['name']}: {r.get('elapsed_ms', r.get('reason', r.get('error', '')))}")


# ── Chat Mode ───────────────────────────────────────────────────────────────

if mode == "Chat":
    st.header(f"Chat — {MODELS[model_id]['name']}")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
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

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get engine
        engine = get_engine(
            model_id,
            reasoning_effort=reasoning_effort,
            show_reasoning=show_reasoning,
            max_completion_tokens=max_tokens,
            json_mode=(output_mode == "JSON"),
            json_schema=json_schema,
        )

        # Generate response
        with st.chat_message("assistant"):
            if output_mode == "Stream":
                reasoning_placeholder = st.empty()
                content_placeholder = st.empty()
                stats_placeholder = st.empty()

                reasoning_buf = ""
                content_buf = ""
                final_stats = None

                for chunk in engine.stream_response(prompt):
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

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": content_buf,
                        "reasoning": reasoning_buf if reasoning_buf else None,
                        "stats": final_stats,
                    })
            else:
                # JSON mode (non-streaming)
                with st.spinner("Generating JSON response..."):
                    result = engine.structured_response(prompt)

                if result.get("type") == "error":
                    st.error(f"API Error: {result['error']}")
                else:
                    if result.get("parsed"):
                        st.json(result["parsed"])
                    else:
                        st.code(result["content"], language="json")

                    s = result["stats"]
                    cols = st.columns(4)
                    cols[0].caption(f"Total: {s['total_ms']}ms")
                    cols[1].caption(f"Cached: {s['cached_tokens']}/{s['prompt_tokens']}")
                    cols[2].caption(f"Output: {s['completion_tokens']} tokens")
                    cols[3].caption("Mode: JSON")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["content"],
                        "stats": result["stats"],
                    })

    # Cache stats in sidebar
    with st.sidebar:
        st.divider()
        st.subheader("Cache Stats")
        if model_id in [e.model_id for e in st.session_state.engines.values()]:
            for e in st.session_state.engines.values():
                if e.model_id == model_id:
                    cs = e.get_cache_stats()
                    st.caption(f"Requests: {cs['total_requests']} | Hits: {cs['cache_hits']}")
                    st.caption(f"Cached: {cs['total_cached_tokens']:,}/{cs['total_prompt_tokens']:,} tokens")
                    if cs["total_prompt_tokens"] > 0:
                        st.progress(cs["cache_rate"] / 100, text=f"Cache rate: {cs['cache_rate']}%")
                    st.caption(f"Est. savings: ${cs['estimated_savings']:.6f}")
                    break
        else:
            st.caption("No requests yet")


# ── Compare Mode ────────────────────────────────────────────────────────────

elif mode == "Compare Models":
    st.header("Model Comparison — Side by Side")
    st.caption("Same prompt sent to all 3 models in parallel")

    # Show previous comparisons
    for comp in st.session_state.compare_results:
        st.divider()
        st.markdown(f"**You:** {comp['prompt']}")
        cols = st.columns(3)
        for i, (mid, result) in enumerate(comp["results"].items()):
            with cols[i]:
                name = MODELS[mid]["name"]
                if result.get("type") == "error":
                    st.error(f"{name}: {result['error']}")
                else:
                    st.subheader(name)
                    stats = result.get("stats", {})

                    # Metrics row
                    m1, m2 = st.columns(2)
                    m1.metric("TTFT", f"{stats.get('ttft_ms', stats.get('total_ms', '-'))}ms")
                    m2.metric("Total", f"{stats.get('total_ms', '-')}ms")

                    # Reasoning
                    reasoning = result.get("reasoning") or stats.get("reasoning", "")
                    if reasoning and show_reasoning:
                        with st.expander("💭 Reasoning"):
                            st.markdown(reasoning)

                    # Content
                    content = result.get("content") or stats.get("content", "")
                    if output_mode == "JSON":
                        try:
                            st.json(json.loads(content))
                        except (json.JSONDecodeError, TypeError):
                            st.markdown(content)
                    else:
                        st.markdown(content)

                    # Cache info
                    cached = stats.get("cached_tokens", 0)
                    prompt_t = stats.get("prompt_tokens", 0)
                    st.caption(f"Cache: {cached}/{prompt_t} | Output: {stats.get('completion_tokens', '-')} tokens")

    # Compare input
    if prompt := st.chat_input("Type a prompt to compare across all models..."):
        st.divider()
        st.markdown(f"**You:** {prompt}")

        cols = st.columns(3)
        model_ids = list(MODELS.keys())

        # Create placeholders for each column
        placeholders = {}
        for i, mid in enumerate(model_ids):
            with cols[i]:
                st.subheader(MODELS[mid]["name"])
                st.caption(f"{MODELS[mid]['speed']} | {MODELS[mid]['tier']}")
                placeholders[mid] = {
                    "status": st.empty(),
                    "reasoning": st.empty(),
                    "content": st.empty(),
                    "metrics": st.empty(),
                    "cache": st.empty(),
                }
                placeholders[mid]["status"].info("⏳ Waiting...")

        # Create engines in main thread (session_state not accessible from threads)
        engines_for_compare = {}
        for mid in model_ids:
            effort = reasoning_effort if mid != "qwen/qwen3-32b" else ("default" if reasoning_effort not in ["none", "default"] else reasoning_effort)
            engines_for_compare[mid] = get_engine(
                mid,
                reasoning_effort=effort,
                show_reasoning=show_reasoning,
                max_completion_tokens=max_tokens,
                json_mode=(output_mode == "JSON"),
                json_schema=json_schema,
            )

        # Run all 3 models in parallel using threads
        def run_model(mid: str, engine: GroqEngine) -> dict:
            if output_mode == "JSON":
                return engine.structured_response(prompt)
            else:
                content = ""
                reasoning = ""
                final_stats = None
                for chunk in engine.stream_response(prompt):
                    if chunk["type"] == "reasoning":
                        reasoning += chunk["text"]
                    elif chunk["type"] == "content":
                        content += chunk["text"]
                    elif chunk["type"] == "done":
                        final_stats = chunk["stats"]
                    elif chunk["type"] == "error":
                        return {"type": "error", "error": chunk["error"]}
                return {
                    "type": "done",
                    "content": content,
                    "reasoning": reasoning,
                    "stats": final_stats or {},
                }

        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(run_model, mid, engines_for_compare[mid]): mid for mid in model_ids}
            for future in as_completed(futures):
                mid = futures[future]
                try:
                    result = future.result()
                    results[mid] = result
                except Exception as e:
                    results[mid] = {"type": "error", "error": str(e)}

                # Update the placeholder for this model
                ph = placeholders[mid]
                ph["status"].empty()

                if results[mid].get("type") == "error":
                    ph["content"].error(f"Error: {results[mid]['error']}")
                else:
                    stats = results[mid].get("stats", {})
                    reasoning = results[mid].get("reasoning", "")
                    content = results[mid].get("content", "")

                    # Metrics
                    with ph["metrics"].container():
                        m1, m2 = st.columns(2)
                        ttft = stats.get("ttft_ms", stats.get("total_ms", "-"))
                        m1.metric("TTFT", f"{ttft}ms")
                        m2.metric("Total", f"{stats.get('total_ms', '-')}ms")

                    # Reasoning
                    if reasoning and show_reasoning:
                        with ph["reasoning"].expander("💭 Reasoning"):
                            st.markdown(reasoning)

                    # Content
                    if output_mode == "JSON":
                        try:
                            ph["content"].json(json.loads(content))
                        except (json.JSONDecodeError, TypeError):
                            ph["content"].markdown(content)
                    else:
                        ph["content"].markdown(content)

                    # Cache
                    cached = stats.get("cached_tokens", 0)
                    prompt_t = stats.get("prompt_tokens", 0)
                    ph["cache"].caption(f"Cache: {cached}/{prompt_t} | Output: {stats.get('completion_tokens', '-')} tokens")

        # Summary table
        st.divider()
        st.subheader("Comparison Summary")

        summary_data = []
        for mid in model_ids:
            r = results.get(mid, {})
            stats = r.get("stats", {})
            summary_data.append({
                "Model": MODELS[mid]["name"],
                "TTFT (ms)": stats.get("ttft_ms", stats.get("total_ms", "-")),
                "Total (ms)": stats.get("total_ms", "-"),
                "Output Tokens": stats.get("completion_tokens", "-"),
                "Cached Tokens": stats.get("cached_tokens", 0),
                "Prompt Tokens": stats.get("prompt_tokens", 0),
                "Status": "✅" if r.get("type") != "error" else "❌",
            })

        st.dataframe(summary_data, width="stretch", hide_index=True)

        # Save comparison
        st.session_state.compare_results.append({
            "prompt": prompt,
            "results": results,
        })

    # Cache stats for all models
    with st.sidebar:
        st.divider()
        st.subheader("Cache Stats (All Models)")
        for engine in st.session_state.engines.values():
            cs = engine.get_cache_stats()
            if cs["total_requests"] > 0:
                st.caption(f"**{MODELS[engine.model_id]['name']}**")
                st.caption(f"  Requests: {cs['total_requests']} | Hits: {cs['cache_hits']} | Rate: {cs['cache_rate']}%")
                st.caption(f"  Cached: {cs['total_cached_tokens']:,} tokens | Savings: ${cs['estimated_savings']:.6f}")
