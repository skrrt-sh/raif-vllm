"""Packaged RAIF chat templates + a resolver for `vllm serve --chat-template`.

The chat template is load-bearing: it renders messages only and ignores the
`tools` variable, so the served prompt matches training (without it the LoRA
echoes the verbose OpenAI tool-def JSON). The templates ship *inside* the wheel,
so `pip install raif-vllm` is enough — no repo checkout needed. Resolve a path:

    python -m raif_vllm.templates qwen-4b      # prints the absolute .jinja path

    vllm serve ... --chat-template "$(python -m raif_vllm.templates llama-3b)"

or in Python: `from raif_vllm import chat_template; chat_template("qwen-0.5b")`.
"""

from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).resolve().parent / "chat_templates"

# Map a model short-name (the names used across raif-lora / the README) and a few
# convenient aliases to the packaged template file. Each base family renders with
# different markers, so each needs its own tools-ignoring template.
_ALIASES: dict[str, str] = {
    # Llama-3.2-3B
    "llama-3b": "raif_llama32.jinja",
    "llama32": "raif_llama32.jinja",
    "llama": "raif_llama32.jinja",
    # Qwen2.5-0.5B
    "qwen-0.5b": "raif_qwen25.jinja",
    "qwen25": "raif_qwen25.jinja",
    "qwen2.5": "raif_qwen25.jinja",
    # Qwen3-4B-Instruct-2507
    "qwen-4b": "raif_qwen3.jinja",
    "qwen3": "raif_qwen3.jinja",
}


def template_path(name: str) -> str:
    """Absolute path to a packaged RAIF chat template for a model name/alias.

    Accepts a short model name (`llama-3b`, `qwen-0.5b`, `qwen-4b`), an alias, or
    a bare template filename. Raises FileNotFoundError if it isn't packaged.
    """
    key = name.lower()
    fname = _ALIASES.get(key, name if name.endswith(".jinja") else f"{name}.jinja")
    path = _DIR / fname
    if not path.is_file():
        have = sorted(set(_ALIASES) | {p.name for p in _DIR.glob("*.jinja")})
        raise FileNotFoundError(f"no packaged chat template for {name!r}; known: {have}")
    return str(path)


def available() -> list[str]:
    """Model names/aliases that resolve to a packaged template."""
    return sorted(k for k in _ALIASES if (_DIR / _ALIASES[k]).is_file())


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m raif_vllm.templates",
        description="Print the absolute path to a packaged RAIF chat template.",
    )
    ap.add_argument("name", nargs="?", help="model name/alias (e.g. llama-3b, qwen-4b)")
    ap.add_argument("--list", action="store_true", help="list available names")
    args = ap.parse_args(argv)
    if args.list or not args.name:
        print("\n".join(available()))
        return 0
    print(template_path(args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
