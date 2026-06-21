"""Animated terminal demo of the RAIF vLLM integration — the *principle*, by levels.

Paced for screen recording (rebuilt to gif/mp4 via `assets/demo-vllm.tape`). It runs
against the REAL plugin code: the chat-template `<schema>` inject, the reasoning- and
tool-parser decode seams, and the `raif` codec the model is trained to emit. No GPU and
no server are needed — the one thing a GPU would do (emit RAIF-G) is reproduced with
`raif.encode`, which is exactly the wire format the LoRA learns to produce. Token counts
are real (cl100k via tiktoken).

    uv run --with raif-format --with tiktoken python examples/demo_vllm.py
    DEMO_FAST=1 ...   # no animation, for iterating
"""

from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace

# Import the plugin's pure helpers from the repo (no vLLM, no GPU). The path insert
# must precede these imports, so they carry noqa: E402 deliberately.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raif import decode, encode  # noqa: E402  the codec the model emits / decodes

from raif_vllm.reasoning import route_and_decode  # noqa: E402  response_format seam
from raif_vllm.structured import (  # noqa: E402
    build_response_schema_block,  # the <schema> cue injected pre-template
    strip_reasoning_prefix,  # Qwen3 <think> handling
)
from raif_vllm.tool_parser import decode_arguments  # noqa: E402  the tools seam

FAST = bool(os.environ.get("DEMO_FAST"))

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
    def toks(s: str) -> int:
        return len(_ENC.encode(s))
except Exception:  # pragma: no cover - fallback if tiktoken/cache is unavailable
    def toks(s: str) -> int:
        return max(1, round(len(s) / 4))


# ── tiny animation kit (ANSI truecolor, typewriter) ──────────────────────────
def out(s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.flush()


def nl(n: int = 1) -> None:
    out("\n" * n)


def clear() -> None:
    out("\x1b[2J\x1b[3J\x1b[H")


def sleep(ms: int) -> None:
    if not FAST:
        time.sleep(ms / 1000)


def type_(text: str, ms: int = 18) -> None:
    if FAST:
        out(text)
        return
    for ch in text:
        out(ch)
        time.sleep(ms / 1000)


def typeln(text: str, ms: int = 18) -> None:
    type_(text, ms)
    nl()


def padr(s: str, w: int) -> str:
    return s if len(s) >= w else s + " " * (w - len(s))


def padl(s: str, w: int) -> str:
    return s if len(s) >= w else " " * (w - len(s)) + s


def paint(r: int, g: int, b: int):
    return lambda s: f"\x1b[38;2;{r};{g};{b}m{s}\x1b[0m"


blue = paint(96, 165, 250)
cyan = paint(34, 211, 238)
green = paint(74, 222, 128)
red = paint(248, 113, 113)
amber = paint(251, 191, 36)
purple = paint(167, 139, 250)
gray = paint(120, 128, 140)
white = paint(231, 233, 238)


def bold(s: str) -> str:
    return f"\x1b[1m{s}\x1b[0m"


def chip(n: str) -> str:
    return f"\x1b[1m\x1b[38;2;12;18;32m\x1b[48;2;96;165;250m {n} \x1b[0m"


PAD = "   "

WORDMARK = [
    "██████╗  █████╗ ██╗███████╗   ██╗   ██╗██╗     ██╗     ███╗   ███╗",
    "██╔══██╗██╔══██╗██║██╔════╝   ██║   ██║██║     ██║     ████╗ ████║",
    "██████╔╝███████║██║█████╗     ██║   ██║██║     ██║     ██╔████╔██║",
    "██╔══██╗██╔══██║██║██╔══╝     ╚██╗ ██╔╝██║     ██║     ██║╚██╔╝██║",
    "██║  ██║██║  ██║██║██║         ╚████╔╝ ███████╗███████╗██║ ╚═╝ ██║",
    "╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝          ╚═══╝  ╚══════╝╚══════╝╚═╝     ╚═╝",
]
GRADIENT = [paint(96, 165, 250), paint(110, 145, 251), paint(130, 138, 250),
            paint(150, 132, 250), paint(160, 130, 250), paint(167, 139, 250)]


def wordmark() -> None:
    for i, line in enumerate(WORDMARK):
        out(PAD + GRADIENT[i](line))
        nl()
        sleep(70)


def slide(n: str, title: str, desc: str) -> None:
    clear()
    nl(2)
    out(f"{PAD}{chip(n)}  ")
    type_(bold(white(title)), 22)
    nl(2)
    typeln(PAD + gray(desc), 11)
    nl()
    sleep(450)


# ── the running example: a structured `response_format` request ──────────────
# A 3-day forecast — a small table, the shape where RAIF's declare-keys-once wins.
SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "unit": {"type": "string"},
        "days": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "high": {"type": "number"},
                    "low": {"type": "number"},
                    "summary": {"type": "string"},
                },
            },
        },
    },
}
VALUE = {
    "city": "Oslo",
    "unit": "celsius",
    "days": [
        {"date": "2026-06-22", "high": 21, "low": 12, "summary": "clear"},
        {"date": "2026-06-23", "high": 19, "low": 11, "summary": "rain"},
        {"date": "2026-06-24", "high": 23, "low": 14, "summary": "cloudy"},
    ],
}
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "forecast", "schema": SCHEMA}}


def intro() -> None:
    clear()
    nl(2)
    wordmark()
    nl()
    type_(PAD + gray("Transparent RAIF for any OpenAI client — one vLLM plugin."), 14)
    nl(2)
    sleep(1500)


def slide_principle() -> None:
    slide("01", "The principle — five levels, one round trip",
          "Install the plugin. A stock OpenAI client gets RAIF — no proxy, no fork.")
    levels = [
        (purple, "① client", "sends a normal request  (tools / response_format)"),
        (blue,   "② inject", "plugin appends a tiny <schema> cue to the user message"),
        (amber,  "③ model",  "emits RAIF-G on the wire  — fewer tokens"),
        (green,  "④ decode", "plugin → JSON at the request/response boundary"),
        (cyan,   "⑤ client", "gets clean tool_calls / content — never saw RAIF"),
    ]
    for color, lvl, desc in levels:
        out(f"{PAD}  {bold(color(padr(lvl, 10)))}{gray(desc)}\n")
        sleep(420)
    nl()
    type_(PAD + bold(white("The client code never changes. The wire just gets lighter.")), 14)
    sleep(2200)


def slide_request() -> None:
    slide("02", "Level ① — what the client sends",
          "A plain OpenAI call. No RAIF import, no awareness. This is the whole client.")
    code = [
        (cyan, "client = OpenAI(base_url=\"http://localhost:8000/v1\")"),
        (white, ""),
        (white, "client.chat.completions.create("),
        (white, "    model=\"raif\","),
        (white, "    messages=[{\"role\": \"user\", \"content\": \"3-day forecast for Oslo?\"}],"),
        (amber, "    response_format={\"type\": \"json_schema\", ...},"),
        (white, ")"),
    ]
    for color, line in code:
        out(f"{PAD}  {color(line) if line else ''}\n")
        sleep(170)
    nl()
    type_(PAD + gray("Stock client, stock shapes. The plugin does the rest server-side."), 13)
    sleep(2000)


def slide_inject() -> None:
    slide("03", "Level ② — inject the schema cue",
          "The cue is the request's response_format schema, rendered as RAIF — not the question.")
    # Plain-language gloss for RAIF's compact schema shorthand (name:type, ? = optional).
    gloss = {"city:s?": "string, optional", "unit:s?": "string, optional",
             "days[]:o?": "array of objects"}
    cue_lines = build_response_schema_block(RESPONSE_FORMAT).split("\n")

    # Two independent inputs already in the request (slide ①) — keep them distinct.
    out(f"{PAD}  {gray('the request carried two separate things:')}\n")
    out(f"{PAD}    {gray('• the question      ')}{white('3-day forecast for Oslo?')}"
        f"{gray('   (free text)')}\n")
    out(f"{PAD}    {gray('• response_format   ')}{amber('a JSON Schema the developer set')}"
        f"{gray('  → city, unit, days[]')}\n")
    nl()
    sleep(700)
    out(f"{PAD}  {gray('the plugin renders that ')}{amber('schema')}"
        f"{gray(' to RAIF and appends it to the message:')}\n")
    nl()
    out(f"{PAD}    {white('user:  3-day forecast for Oslo?')}\n")
    for ln in cue_lines:
        note = ""
        if ln == "<schema>":
            note = amber("   ← the response_format schema, as RAIF")
        elif ln in gloss:
            note = gray("   " + gloss[ln])
        out(f"{PAD}           {blue(padr(ln, 12))}{note}\n")
        sleep(160)
    nl()
    type_(PAD + gray("Never the question — the cue is a deterministic render of response_format."), 12)
    sleep(2200)


def slide_wire() -> None:
    slide("04", "Level ③ — RAIF-G on the wire",
          "The model answers in RAIF-G: keys declared once, one compact row per record.")
    wire = encode(VALUE)                       # exactly what the trained model emits
    json_str = json.dumps(VALUE, separators=(",", ":"))
    jt, rt = toks(json_str), toks(wire)
    saved = round(100 * (jt - rt) / jt)

    out(f"{PAD}  {bold(amber('what the model emits'))}  {gray(f'— RAIF-G, {rt} tokens')}\n")
    for line in wire.split("\n"):
        note = gray("   ← the header: field names, written once") if line.startswith("days::") else ""
        out(f"{PAD}    {white(line)}{note}\n")
        sleep(90)
    nl()
    out(f"{PAD}  {gray('the same data as JSON')}  {gray(f'— {jt} tokens, minified')}\n")
    out(f"{PAD}    {gray(json_str[:58] + ' …')}\n")
    nl()
    sleep(500)
    out(f"{PAD}  {green('▸ ')}{bold(white(f'{saved}% fewer tokens'))}"
        f"{gray('  — lossless, and the win grows with table size')}\n")
    nl()
    type_(PAD + gray("Same data, same meaning — just lighter on the wire."), 13)
    sleep(2200)


def slide_decode() -> None:
    slide("05", "Level ④ — decode back to JSON",
          "At the boundary the plugin runs the real codec. The client gets plain JSON.")
    wire = encode(VALUE)
    # The exact seam vLLM calls: reasoning_parser.extract_reasoning(output, request).
    request = SimpleNamespace(response_format=RESPONSE_FORMAT, tools=None,
                              tool_choice=None, vllm_xargs=None)
    _, content = route_and_decode(wire, request)
    out(f"{PAD}  {white('city=Oslo  days::…')}  {cyan('→ decode →')}  "
        f"{green(json.dumps(json.loads(content))[:30] + ' …')}\n")
    out(f"{PAD}  {gray('  the client receives ordinary OpenAI JSON content — no RAIF awareness.')}\n")
    nl()
    sleep(900)

    # Self-heal: show the ACTUAL fumbled wire (a code fence + a ':' slip) and the fix.
    out(f"{PAD}  {bold(white('And it self-heals.'))} "
        f"{gray('suppose it arrives fenced, with ')}{amber('city:')}{gray(' instead of ')}{amber('city=')}\n")
    nl()
    broken = [
        ("```", amber("   ← a code fence the model added — not RAIF")),
        ("city:Oslo", amber("   ← a ':' where RAIF uses '='")),
        ("days::date,high,low,summary", ""),
        ("```", ""),
    ]
    for ln, note in broken:
        out(f"{PAD}      {gray(ln)}{note}\n")
        sleep(170)
    nl()
    fumbled = "```\n" + wire.replace("city=", "city:") + "\n```"
    res = decode(fumbled)
    kinds = {r["kind"] for r in res["repairs"]}
    ok = kinds == {"markdown_stripped", "separator_coerced"}
    fence_desc = gray("drops the code fence")
    sep_desc = gray("turns ':' back into '='")
    out(f"{PAD}  {green('✓ ') if ok else red('✗ ')}"
        f"{white('decode() repairs both — values never touched:')}\n")
    out(f"{PAD}      {cyan(padr('markdown_stripped', 20))}{fence_desc}\n")
    out(f"{PAD}      {cyan(padr('separator_coerced', 20))}{sep_desc}\n")
    out(f"{PAD}      {green('→ ')}{white(json.dumps(res['value'])[:36] + ' …')}\n")
    nl()
    type_(PAD + bold(white("Clean JSON out — even when the model fumbles the syntax.")), 13)
    sleep(2400)


def slide_models() -> None:
    slide("06", "Every model — one plugin",
          "Llama-3.2-3B · Qwen3-4B · Qwen2.5-0.5B, all served transparently.")
    # Qwen3 prepends a reasoning preamble (sometimes bare closers) before the RAIF-G.
    wire = encode({"city": "Oslo", "unit": "celsius", "temperature": 12})
    qwen3_raw = "</tool_call>\n\n</think>\n\n" + wire
    out(f"{PAD}  {gray('Qwen3 emits a reasoning preamble before the answer:')}\n")
    out(f"{PAD}    {amber('</tool_call>')}  {amber('</think>')}  {white(wire.replace(chr(10), '  '))}\n")
    nl()
    sleep(700)
    stripped = strip_reasoning_prefix(qwen3_raw)
    args = decode_arguments(qwen3_raw)  # the tool seam strips + decodes in one step
    out(f"{PAD}  {cyan('strip_reasoning_prefix()')}{gray(' → ')}{white(stripped.replace(chr(10), '  '))}\n")
    out(f"{PAD}  {green('✓ ')}{gray('decoded args: ')}{white(json.dumps(args))}\n")
    nl()
    type_(PAD + bold(white("The plugin absorbs each base's quirks. Your client stays identical.")), 13)
    sleep(2400)


def outro() -> None:
    clear()
    nl(2)
    wordmark()
    nl()
    typeln(PAD + gray("Three layers, shipped independently:"), 14)
    out(f"{PAD}  {white('format')}  {gray('raif-format')}   {purple('·')}   "
        f"{white('model')}  {gray('raif-lora')}   {purple('·')}   "
        f"{white('serving')}  {gray('raif-vllm')}\n")
    nl()
    out(f"{PAD}{cyan('pip install raif-vllm')}\n")
    out(f"{PAD}{cyan('github.com/skrrt-sh/raif-vllm')}\n")
    nl()
    typeln(PAD + gray("Open source. Apache-2.0."), 14)
    nl()
    sleep(2600)


def main() -> None:
    intro()
    slide_principle()
    slide_request()
    slide_inject()
    slide_wire()
    slide_decode()
    slide_models()
    outro()


if __name__ == "__main__":
    main()
