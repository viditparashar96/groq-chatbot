# Groq Reasoning Chatbot 🚀

A CLI chatbot with **streaming + reasoning** support for Groq's fastest LLM models. Switch between models on-the-fly, see the AI's thinking process in real-time, and chat in 119+ languages.

## Features

- ⚡ **Real-time streaming** — tokens appear as they're generated
- 🧠 **Reasoning display** — see the model's chain-of-thought thinking
- 🌍 **Multilingual** — supports Hindi, Arabic (8 dialects), African languages, and 100+ more
- 🔄 **Hot-swap models** — switch between Qwen3, GPT-OSS 20B, and GPT-OSS 120B mid-conversation
- 📊 **Performance metrics** — TTFT, total time, reasoning stats per message
- 🎛️ **Configurable reasoning effort** — tune speed vs. depth

---

## Model Comparison

| Feature              | qwen/qwen3-32b          | openai/gpt-oss-20b      | openai/gpt-oss-120b     |
|----------------------|-------------------------|-------------------------|-------------------------|
| **Output Speed**     | ~400 t/s                | ~1000 t/s ⚡             | ~500 t/s                |
| **TTFT**             | Fast                    | Ultra-fast ⚡            | Fast                    |
| **Tier**             | Preview                 | Production ✓            | Production ✓            |
| **Streaming**        | ✅ Yes                  | ✅ Yes                   | ✅ Yes                  |
| **Reasoning**        | ✅ Yes                  | ✅ Yes                   | ✅ Yes                  |
| **Reasoning Effort** | none / default          | low / medium / high     | low / medium / high     |
| **Context Window**   | 131K tokens             | 131K tokens             | 131K tokens             |
| **Max Output**       | 40,960 tokens           | 65,536 tokens           | 65,536 tokens           |
| **Architecture**     | Dense 32.8B             | MoE 21B (3.6B active)   | MoE 117B (5.1B active)  |
| **Multilingual**     | 119 langs ★★★★★         | 14+ langs ★★☆☆☆         | 81+ langs ★★★☆☆         |
| **Hindi**            | ✅ Excellent            | ⚠️ Basic                | ⚠️ Moderate             |
| **Arabic**           | ✅ 8 dialects           | ❌ Poor                 | ⚠️ Moderate             |
| **African langs**    | ✅ Afrikaans, Swahili+  | ❌ Limited              | ⚠️ Some                 |
| **Price (in/1M)**    | $0.29                   | $0.075 💰               | $0.15                   |
| **Price (out/1M)**   | $0.59                   | $0.30 💰                | $0.60                   |
| **Best For**         | Multilingual + Reasoning| Speed-critical English  | Complex reasoning       |

**Key Takeaway:** Qwen3-32B is the only model that checks all 3 boxes — streaming + reasoning + strong multilingual (Hindi/Arabic/African).

---

## Quick Start

### 1. Get your API key (free)

Go to [console.groq.com/keys](https://console.groq.com/keys) and create a free API key.

### 2. Set your environment

```bash
# Linux/Mac
export GROQ_API_KEY=gsk_your_key_here

# Windows (PowerShell)
$env:GROQ_API_KEY = "gsk_your_key_here"

# Windows (CMD)
set GROQ_API_KEY=gsk_your_key_here
```

### 3. Install & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run with interactive model picker
python chatbot.py

# Or specify a model directly
python chatbot.py --model qwen/qwen3-32b
python chatbot.py --model openai/gpt-oss-20b
python chatbot.py --model openai/gpt-oss-120b --reasoning-effort high
```

---

## Usage Examples

```bash
# Fastest model (English-focused)
python chatbot.py -m openai/gpt-oss-20b

# Best multilingual (Hindi, Arabic, etc.)
python chatbot.py -m qwen/qwen3-32b

# Best reasoning quality with high effort
python chatbot.py -m openai/gpt-oss-120b -e high

# Hide reasoning (show only answers)
python chatbot.py -m openai/gpt-oss-20b --no-reasoning

# Show comparison table
python chatbot.py --show-comparison
```

## In-Chat Commands

| Command           | Description                                  |
|--------------------|----------------------------------------------|
| `/model [name]`    | List or switch models                        |
| `/effort [level]`  | Set reasoning effort (low/medium/high)       |
| `/reasoning`       | Toggle reasoning display ON/OFF              |
| `/clear`           | Clear conversation history                   |
| `/compare`         | Show model comparison table                  |
| `/help`            | Show all commands                            |
| `/quit`            | Exit                                         |

## Multilingual Examples

Just type in any language:

```
You ❯ मुझे भारत के इतिहास के बारे में बताओ
You ❯ اشرح لي الذكاء الاصطناعي باللغة العربية
You ❯ Explain quantum computing in Hindi
You ❯ Qu'est-ce que l'intelligence artificielle?
```

---

## How Reasoning Works

### Qwen3-32B
- Uses `reasoning_format` parameter: `parsed` (separate field) or `raw` (`<think>` tags) or `hidden`
- `reasoning_effort`: `none` (skip thinking) or `default` (think)

### GPT-OSS 20B / 120B
- Uses `include_reasoning` parameter (true/false)
- `reasoning_effort`: `low`, `medium`, `high`
- Reasoning comes in `delta.reasoning` field during streaming

---

## Requirements

- Python 3.10+
- `groq` Python SDK (v0.28+)
- Free Groq API key

## License

MIT — do whatever you want with it.
