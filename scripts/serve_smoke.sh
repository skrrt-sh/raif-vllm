#!/usr/bin/env bash
# RAIF single-plugin smoke on a CUDA-12 GPU box (vLLM 0.19, entry-point model).
#
# Runs ON the GPU box, from a checkout of this repo. Installs vLLM 0.19 + the
# plugin (which pulls raif-format>=0.6 from PyPI), serves a base model + its RAIF
# LoRA with `VLLM_PLUGINS=raif`, waits for health, then runs the e2e client across
# all five OpenAI paths.
#
# Works for every published RAIF adapter — pick one with MODEL=:
#   MODEL=llama-3b   (default) unsloth/Llama-3.2-3B-Instruct  + raif-llama-3.2-3b-lora
#   MODEL=qwen-0.5b           Qwen/Qwen2.5-0.5B-Instruct      + raif-qwen2.5-0.5b-lora
#   MODEL=qwen-4b            Qwen/Qwen3-4B-Instruct-2507      + raif-qwen3-4b-lora
#
# v0.19.0 is the LAST CUDA-12 vLLM and carries every hook the plugin needs
# (reasoning-parser content decode + the render_chat inject seam).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKROOT="${WORKROOT:-/workspace/raif}"
MODEL="${MODEL:-llama-3b}"
PORT="${PORT:-8000}"
PY="${PY:-python3.12}"
VLLM_PIN="${VLLM_PIN:-vllm==0.19.0}"

# Per-model base / adapter / packaged-template name. Each base family renders with
# different markers, so each has its own tools-ignoring template (resolved below).
case "$MODEL" in
  llama-3b)
    BASE="${BASE:-unsloth/Llama-3.2-3B-Instruct}"
    ADAPTER="${ADAPTER:-skrrt-sh/raif-llama-3.2-3b-lora}"
    TPL_NAME="llama-3b" ;;
  qwen-0.5b)
    BASE="${BASE:-Qwen/Qwen2.5-0.5B-Instruct}"
    ADAPTER="${ADAPTER:-skrrt-sh/raif-qwen2.5-0.5b-lora}"
    TPL_NAME="qwen-0.5b" ;;
  qwen-4b)
    BASE="${BASE:-Qwen/Qwen3-4B-Instruct-2507}"
    ADAPTER="${ADAPTER:-skrrt-sh/raif-qwen3-4b-lora}"
    TPL_NAME="qwen-4b" ;;
  *) echo "unknown MODEL=$MODEL (want llama-3b|qwen-0.5b|qwen-4b)" >&2; exit 2 ;;
esac

SMOKE="$REPO_DIR/examples/smoke_plugin.py"

log() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }

log "0. GPU + interpreter  (MODEL=$MODEL)"
command -v nvidia-smi >/dev/null || die "no nvidia-smi — not a CUDA GPU box"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
command -v "$PY" >/dev/null || die "interpreter '$PY' not found (set PY=)"
"$PY" --version

mkdir -p "$WORKROOT"
export HF_HOME="${HF_HOME:-/workspace/.hf-cache}"
mkdir -p "$HF_HOME"

log "1. Install vLLM 0.19 (CUDA-12) + this plugin (editable; pulls raif-format from PyPI) + clients"
"$PY" -c 'import vllm' 2>/dev/null || "$PY" -m pip install -q "$VLLM_PIN"
# fastapi/vllm MUST be separate pip calls — a combined resolve is ResolutionImpossible.
# Pin fastapi 0.115.6 (starlette <0.42) or vLLM's prometheus instrumentator 500s /health.
"$PY" -m pip install -q openai "fastapi==0.115.6" -e "$REPO_DIR"
"$PY" -c 'import vllm, raif, raif_vllm; print("vllm", vllm.__version__, "| raif", raif.__version__, "| raif_vllm OK")'

log "1b. Resolve the packaged tools-ignoring chat template (generate it if missing)"
# Templates ship inside the wheel; qwen ones are derived from the base's stock
# template (parity-safe). Generate on first use, then resolve the packaged path.
if ! CHAT_TEMPLATE="$("$PY" -m raif_vllm.templates "$TPL_NAME" 2>/dev/null)"; then
  PKG_TPL_DIR="$("$PY" -c 'import raif_vllm,os;print(os.path.join(os.path.dirname(raif_vllm.__file__),"chat_templates"))')"
  case "$MODEL" in
    qwen-0.5b) OUT="$PKG_TPL_DIR/raif_qwen25.jinja" ;;
    qwen-4b)   OUT="$PKG_TPL_DIR/raif_qwen3.jinja" ;;
    *) die "no template for $MODEL and no generator mapping" ;;
  esac
  "$PY" "$REPO_DIR/scripts/make_chat_template.py" "$BASE" "$OUT"
  CHAT_TEMPLATE="$("$PY" -m raif_vllm.templates "$TPL_NAME")"
fi
echo "chat template: $CHAT_TEMPLATE"
[ -f "$CHAT_TEMPLATE" ] || die "chat template not found at $CHAT_TEMPLATE"

log "1c. Detect the adapter's LoRA rank (max-lora-rank must be >= it)"
MAX_RANK="$("$PY" - "$ADAPTER" <<'PYEOF'
import json, sys
from huggingface_hub import hf_hub_download
cfg = json.load(open(hf_hub_download(sys.argv[1], "adapter_config.json")))
r = int(cfg.get("r", 32))
print(next(c for c in (8, 16, 32, 64, 128, 256) if c >= r))  # round up to a vLLM choice
PYEOF
)"
echo "adapter r -> --max-lora-rank $MAX_RANK"

log "2. Serve $BASE + LoRA '$ADAPTER' with the raif plugin (VLLM_PLUGINS=raif, port $PORT)"
VLLM_PLUGINS=raif "$PY" -m vllm.entrypoints.openai.api_server --model "$BASE" \
  --enable-lora --lora-modules "raif=$ADAPTER" \
  --max-lora-rank "$MAX_RANK" --max-model-len 8192 --enforce-eager \
  --chat-template "$CHAT_TEMPLATE" \
  --reasoning-parser raif \
  --enable-auto-tool-choice --tool-call-parser raif \
  --port "$PORT" >"$WORKROOT/vllm-serve.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

log "3. Wait for health (up to ~6 min for model load)"
for _ in $(seq 1 120); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SERVER 2>/dev/null || die "vllm exited early — see $WORKROOT/vllm-serve.log"
  sleep 3
done
curl -sf "http://localhost:$PORT/health" >/dev/null || die "server never became healthy"

log "4. Confirm the plugin actually loaded (render patch + parser registration)"
grep -E "RAIF plugin:|RAIF render patch|no pre-template inject seam" "$WORKROOT/vllm-serve.log" || \
  printf '\033[1;33mWARN: no RAIF plugin log lines found — entry point may not have run.\033[0m\n'

log "5. Smoke: plain chat + tools + response_format + json_object (+ token saving)"
set +e
"$PY" "$SMOKE" --base-url "http://localhost:$PORT/v1" --model raif
SMOKE_RC=$?
set -e

log "OVERALL ($MODEL)"
if [ "$SMOKE_RC" -eq 0 ]; then
  printf '\033[1;32mOVERALL: PASS — RAIF single plugin verified on %s.\033[0m\n' "$MODEL"
  printf '(server log: %s)\n' "$WORKROOT/vllm-serve.log"
  echo "EXITCODE=0"
  exit 0
fi
printf '\033[1;31mOVERALL: FAIL (%s, smoke rc=%d) — see %s\033[0m\n' "$MODEL" "$SMOKE_RC" "$WORKROOT/vllm-serve.log"
echo "EXITCODE=$SMOKE_RC"
exit 1
