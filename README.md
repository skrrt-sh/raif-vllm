<p align="center">
  <img src="assets/banner.jpg" alt="RAIF" width="640">
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

```sh
VLLM_PLUGINS=raif vllm serve unsloth/Llama-3.2-3B-Instruct \
  --enable-lora --lora-modules raif=skrrt-sh/raif-llama-3.2-3b-lora \
  --max-lora-rank 32 --max-model-len 8192 \
  --chat-template raif_llama32.jinja \
  --reasoning-parser raif \
  --enable-auto-tool-choice --tool-call-parser raif
```

- `VLLM_PLUGINS=raif` runs the entry point, which registers the `raif` reasoning +
  tool parsers **and** installs the `render_chat` inject hook (the seam that adds
  the compact `<schema>` cue before chat-templating).
- `--tool-call-parser raif` decodes the tools path into `tool_calls`;
  `--reasoning-parser raif` decodes the `response_format` path into
  `message.content`.
- `--chat-template raif_llama32.jinja` (in `chat_templates/`) is load-bearing: it
  renders messages only and ignores the `tools` variable, so the served prompt
  matches training. Without it the LoRA echoes the verbose OpenAI tool-def JSON.

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

Smoked on an A40 (vLLM 0.19, base `unsloth/Llama-3.2-3B-Instruct` + the
`skrrt-sh/raif-llama-3.2-3b-lora` adapter) across every OpenAI path a stock client
uses — plain chat, `tools`, `response_format` (`json_schema` + `json_object`):

| path | result |
|---|---|
| plain chat | **PASS** — passthrough, untouched |
| `tools` | **PASS** — RAIF-G → JSON `tool_calls` |
| `response_format` (`json_schema`) | **PASS** — decoded to JSON content, **−19%** tokens vs the equivalent JSON |
| `response_format` (`json_object`) | **PASS** — schemaless decode |
| `response_format` **streaming** | known limitation (raw RAIF-G — use non-streaming) |

Reproduce with [`scripts/serve_smoke.sh`](scripts/serve_smoke.sh) +
[`examples/smoke_plugin.py`](examples/smoke_plugin.py); full results in
[`docs/vllm_e2e_results.md`](docs/vllm_e2e_results.md).

## Project layout

```
raif_vllm/               the plugin: reasoning + tool parsers, render_chat inject hook
chat_templates/          raif_llama32.jinja — tools-ignoring template (training parity)
scripts/serve_smoke.sh   end-to-end GPU smoke (install → serve → all OpenAI paths)
examples/smoke_plugin.py the e2e client (plain · tools · response_format · streaming)
docs/                    serving guide, e2e results, RunPod runbook
tests/                   unit tests for the parsers + inject hook
```

## More

- Serving guide + the chat-template fix: [`docs/vllm_tool_calling.md`](docs/vllm_tool_calling.md).
- RunPod GPU runbook: [`docs/runpod_testing.md`](docs/runpod_testing.md).
- The model: the [`skrrt-sh/raif-llama-3.2-3b-lora`](https://huggingface.co/skrrt-sh/raif-llama-3.2-3b-lora) adapter, trained in [`skrrt-sh/raif-lora`](https://github.com/skrrt-sh/raif-lora). The codec: [`raif-format`](https://github.com/skrrt-sh/raif-standard).

## License

[Apache-2.0](LICENSE). The trained adapter it serves is a derivative of Llama 3.2
(the **Llama 3.2 Community License** applies — "Built with Llama").
