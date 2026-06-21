"""Generate a tools-ignoring RAIF chat template from a base model's stock one.

Each RAIF LoRA was trained on its base model's own chat template; serving must
match that prompt EXACTLY except that the `tools` variable is neutralized — so the
model sees only the injected `<schema>` cue, never the verbose OpenAI tool-def
JSON (which would otherwise be echoed). Rather than hand-port each base's markers,
this takes the base tokenizer's stock `chat_template` verbatim and prepends a
single `{%- set tools = none %}`, guaranteeing training parity by construction.

    python scripts/make_chat_template.py Qwen/Qwen2.5-0.5B-Instruct \
        raif_vllm/chat_templates/raif_qwen25.jinja

Run once per base on a box with `transformers` + Hub access; commit the result.
The Llama-3.2 template (raif_llama32.jinja) was produced the same way.
"""

from __future__ import annotations

import sys
from pathlib import Path

_NEUTRALIZE = (
    "{#- RAIF: ignore the tools variable entirely so the LoRA sees only the "
    "<schema> cue (training parity). #}\n"
    "{%- set tools = none %}\n"
)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: make_chat_template.py <base_model> <out.jinja>", file=sys.stderr)
        return 2
    base, out_path = args[0], Path(args[1])

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    stock = tok.chat_template
    if not stock:
        print(f"base {base!r} has no chat_template", file=sys.stderr)
        return 1
    # Already neutralized upstream? keep verbatim; else prepend the tools=none set.
    body = stock if "set tools = none" in stock else _NEUTRALIZE + stock
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body)
    print(f"wrote {out_path} ({len(body)} chars; tools neutralized) from {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
