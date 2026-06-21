<p align="center">
  <img src="https://raw.githubusercontent.com/skrrt-sh/raif-vllm/main/assets/banner.jpg" alt="RAIF" width="640">
</p>

<h1 align="center">raif-vllm</h1>

<p align="center"><strong>One vLLM plugin for transparent RAIF token savings</strong></p>

<p align="center">
  Install it and existing OpenAI clients get <a href="https://github.com/skrrt-sh/raif-standard">RAIF</a> on<br>
  <code>tools</code> and <code>response_format</code> — no proxy, no client changes, no vLLM fork.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License: Apache-2.0"></a>
  <a href="https://pypi.org/project/raif-vllm/"><img src="https://img.shields.io/pypi/v/raif-vllm?label=PyPI&color=3775a9" alt="raif-vllm on PyPI"></a>
  <img src="https://img.shields.io/badge/vLLM-0.19.x-ff6f00" alt="vLLM 0.19.x">
  <a href="https://huggingface.co/skrrt-sh/raif-llama-3.2-3b-lora"><img src="https://img.shields.io/badge/model-Hugging%20Face-ffb000" alt="Model on Hugging Face"></a>
</p>

---

A vLLM endpoint normally speaks JSON. This plugin makes it speak
[RAIF](https://github.com/skrrt-sh/raif-standard) — the ~10%-lighter, self-repairing
wire format — **without any client changes**. The fine-tuned model emits compact
RAIF-G; the plugin decodes it back to JSON at the request/response boundary, so a
stock OpenAI client gets RAIF on `tools` and `response_format` transparently. No
proxy, no fork — one `pip install` and an entry point.

<p align="center">
  <img src="https://raw.githubusercontent.com/skrrt-sh/raif-vllm/main/assets/demo-vllm.gif" alt="RAIF × vLLM demo — the round trip by levels: a stock OpenAI client sends a normal request, the plugin injects a compact schema cue, the model emits RAIF-G on the wire (fewer tokens), and the plugin decodes it back to JSON at the boundary — the client never sees RAIF" width="820">
</p>

<p align="center"><sub>The integration by levels — client → inject → RAIF-G on the wire → decode → clean JSON. Driven by the real plugin helpers (no GPU); rebuild with <code>vhs assets/demo-vllm.tape</code>.</sub></p>

## Install

```sh
pip install raif-vllm
```

It pulls [`raif-format`](https://pypi.org/project/raif-format/) `>=0.6` from PyPI
automatically.

vLLM itself is provided by the serving host (it pins CUDA/torch); target
**`vllm>=0.19,<0.20`** — v0.19 is the last CUDA-12 vLLM and carries the hooks the
plugin needs. `pip install "raif-vllm[vllm]"` pulls a compatible engine for local
experiments.

## Serve

The plugin is model-agnostic — it works with **all three published RAIF adapters**.
The chat templates ship *inside* the wheel, so `pip install raif-vllm` is all you
need; resolve one for a model with `raif-vllm-chat-template <name>`:

```sh
# Llama-3.2-3B  (the flagship)
VLLM_PLUGINS=raif vllm serve unsloth/Llama-3.2-3B-Instruct \
  --enable-lora --lora-modules raif=skrrt-sh/raif-llama-3.2-3b-lora \
  --max-lora-rank 32 --max-model-len 8192 \
  --chat-template "$(raif-vllm-chat-template llama-3b)" \
  --reasoning-parser raif --enable-auto-tool-choice --tool-call-parser raif

# Qwen3-4B-Instruct  (deployable agent model, ~14 GB)
VLLM_PLUGINS=raif vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --enable-lora --lora-modules raif=skrrt-sh/raif-qwen3-4b-lora \
  --max-lora-rank 32 --max-model-len 8192 \
  --chat-template "$(raif-vllm-chat-template qwen-4b)" \
  --reasoning-parser raif --enable-auto-tool-choice --tool-call-parser raif

# Qwen2.5-0.5B  (tiny & fast)
VLLM_PLUGINS=raif vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --enable-lora --lora-modules raif=skrrt-sh/raif-qwen2.5-0.5b-lora \
  --max-lora-rank 32 --max-model-len 8192 \
  --chat-template "$(raif-vllm-chat-template qwen-0.5b)" \
  --reasoning-parser raif --enable-auto-tool-choice --tool-call-parser raif
```

| model | base | adapter | template name |
|---|---|---|---|
| `llama-3b` | `unsloth/Llama-3.2-3B-Instruct` | [`raif-llama-3.2-3b-lora`](https://huggingface.co/skrrt-sh/raif-llama-3.2-3b-lora) | `llama-3b` |
| `qwen-4b` | `Qwen/Qwen3-4B-Instruct-2507` | [`raif-qwen3-4b-lora`](https://huggingface.co/skrrt-sh/raif-qwen3-4b-lora) | `qwen-4b` |
| `qwen-0.5b` | `Qwen/Qwen2.5-0.5B-Instruct` | [`raif-qwen2.5-0.5b-lora`](https://huggingface.co/skrrt-sh/raif-qwen2.5-0.5b-lora) | `qwen-0.5b` |

- `VLLM_PLUGINS=raif` runs the entry point, which registers the `raif` reasoning +
  tool parsers **and** installs the `render_chat` inject hook (the seam that adds
  the compact `<schema>` cue before chat-templating).
- `--tool-call-parser raif` decodes the tools path into `tool_calls`;
  `--reasoning-parser raif` decodes the `response_format` path into
  `message.content`.
- `--chat-template "$(raif-vllm-chat-template <name>)"` is load-bearing: each
  template renders messages only and ignores the `tools` variable, so the served
  prompt matches training. Without it the LoRA echoes the verbose OpenAI tool-def
  JSON. The Qwen3 adapter prepends a `<think>` block; the plugin strips it at the
  decode boundary, so no client change is needed.

## What a plain OpenAI client gets

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# tools -> JSON tool_calls
client.chat.completions.create(model="raif", tools=[...], tool_choice="auto",
    messages=[{"role": "user", "content": "Weather in Oslo?"}])

# response_format -> JSON content (use non-streaming — see below)
client.chat.completions.create(model="raif",
    response_format={"type": "json_schema", "json_schema": {...}},
    messages=[{"role": "user", "content": "..."}])
```

| OpenAI path | Behavior |
|---|---|
| plain chat | passthrough, untouched |
| `tools` | RAIF-G → JSON `tool_calls` (streaming + non-streaming) |
| `response_format` (`json_schema` / `json_object`) | RAIF-G → JSON `message.content` |
| plain chat **streaming** | passthrough |

### Known limitation: streaming `response_format`

Streaming a `response_format` request is **not decoded** — the client receives raw
RAIF-G. (vLLM's streaming seam passes the parser no schema, and the shared
`is_reasoning_end` flag must stay `True` so the *tools* streaming path keeps
working.) **Use non-streaming `response_format` for structured output** — it
decodes fully. Tool-call streaming is unaffected. See
[`docs/vllm_e2e_results.md`](docs/vllm_e2e_results.md).

## Verified end-to-end

Smoked on an A40 (vLLM 0.19) across every OpenAI path a stock client uses — plain
chat, `tools`, `response_format` (`json_schema` + `json_object`) — on **all three
published adapters**:

| model | plain | `tools` | `response_format` | streaming |
|---|---|---|---|---|
| `llama-3b` | PASS | PASS | PASS | known limitation |
| `qwen-4b` | PASS | PASS | PASS | known limitation |
| `qwen-0.5b` | PASS | PASS | PASS | known limitation |

Every non-streaming path decodes RAIF-G correctly with **no client awareness**.
Token cost is **shape- and tokenizer-dependent** (see the [benchmarks](https://github.com/skrrt-sh/raif-standard/tree/main/benchmarks)):
the win shows on tables and arrays-of-objects, while a lone flat record is roughly
break-even — on the single-record smoke probe, `llama-3b` came in at −19% vs the
equivalent JSON, the Qwen tokenizers near break-even.

Reproduce any model with
[`scripts/serve_smoke.sh`](scripts/serve_smoke.sh) (`MODEL=llama-3b|qwen-0.5b|qwen-4b`)
+ [`examples/smoke_plugin.py`](examples/smoke_plugin.py); full results in
[`docs/vllm_e2e_results.md`](docs/vllm_e2e_results.md).

## Project layout

```
raif_vllm/                    the plugin: reasoning + tool parsers, render_chat inject hook
raif_vllm/chat_templates/     tools-ignoring templates (llama32, qwen25, qwen3) — shipped in the wheel
raif_vllm/templates.py        chat-template resolver (raif-vllm-chat-template CLI)
scripts/serve_smoke.sh        end-to-end GPU smoke (MODEL=llama-3b|qwen-0.5b|qwen-4b)
scripts/make_chat_template.py derive a tools-ignoring template from a base's stock one
examples/smoke_plugin.py      the e2e client (plain · tools · response_format · streaming)
examples/demo_vllm.py         the animated terminal demo (assets/demo-vllm.tape records it)
docs/                         serving guide, e2e results, RunPod runbook
tests/                        unit tests for the parsers + inject hook
```

## More

- Serving guide + the chat-template fix: [`docs/vllm_tool_calling.md`](docs/vllm_tool_calling.md).
- RunPod GPU runbook: [`docs/runpod_testing.md`](docs/runpod_testing.md).
- The models: the [`raif-llama-3.2-3b-lora`](https://huggingface.co/skrrt-sh/raif-llama-3.2-3b-lora), [`raif-qwen3-4b-lora`](https://huggingface.co/skrrt-sh/raif-qwen3-4b-lora), and [`raif-qwen2.5-0.5b-lora`](https://huggingface.co/skrrt-sh/raif-qwen2.5-0.5b-lora) adapters, trained in [`skrrt-sh/raif-lora`](https://github.com/skrrt-sh/raif-lora). The codec: [`raif-format`](https://github.com/skrrt-sh/raif-standard).

## License

[Apache-2.0](LICENSE) for the plugin. The adapters it serves carry their base
model's license: the Llama-3.2 adapter is a derivative of Llama 3.2 (the **Llama
3.2 Community License** applies — "Built with Llama"); the Qwen2.5 / Qwen3 adapters
are Apache-2.0.
