# Renders the RAIF × vLLM integration demo to MP4 + GIF. Rebuild from the repo root:
#   vhs assets/demo-vllm.tape
# Runs against the REAL plugin helpers (no GPU, no server) + the raif codec; token
# counts are real (cl100k via tiktoken). `uv` resolves raif-format + tiktoken; the
# Hide block pre-warms the env and the tiktoken cache so only the demo is recorded.
Output "assets/demo-vllm.mp4"
Output "assets/demo-vllm.gif"

Set Shell "bash"
Set FontSize 22
Set Width 1360
Set Height 940
Set Padding 28
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful

Hide
Type "export PS1='$ '"
Enter
Type "clear"
Enter
# Launch the demo off-camera — the script clears the screen on the first frame,
# so the recording opens straight on the intro, not the typed-out command.
Type "uv run -q --with raif-format --with tiktoken python examples/demo_vllm.py 2>/dev/null"
Sleep 700ms
Enter
Show

# Ends during the outro's ~2.6s hold (demo runs ~59s) — on the call-to-action,
# not the blinking shell prompt. Bump if a slower machine cuts the last slide.
Sleep 58s
