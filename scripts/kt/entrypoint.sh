#!/usr/bin/env bash
# kt-llama.cpp container entrypoint: downloads the model files on first start (one stream each) and runs
# llama-server with a measured profile. Extra arguments to `docker run IMAGE ...` are passed to llama-server.
set -euo pipefail

MODELS_DIR=${MODELS_DIR:-/models}; PORT=${PORT:-8080}; HOST=${HOST:-0.0.0.0}
PROFILE=${PROFILE:-auto}; LORA=${LORA:-0}; ALIAS=${ALIAS:-qwen3.8-27b-ktopt}
KT_MODEL_REPO=${KT_MODEL_REPO:-wiklif/Qwen3.8-27B-KTopt-GGUF}
KT_MODEL_FILE=${KT_MODEL_FILE:-Qwen3.8-27B-KTopt.gguf}
KT_LORA_FILE=${KT_LORA_FILE:-Qwen3.8-27B-KTopt-eora-output-r64.gguf}
DRAFT_REPO=${DRAFT_REPO:-analogalok/Qwen3.8-27B-DFlash2-Q2_K-GGUF}
DRAFT_FILE=${DRAFT_FILE:-Qwen3.8-27B-DFlash2-Q2_K.gguf}
HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}

log() { echo "[kt] $*" >&2; }

fetch() {  # <repo> <file>: download into $MODELS_DIR unless present
    local dst="$MODELS_DIR/$2"
    [ -s "$dst" ] && return 0
    if [[ "$1" == TODO/* ]]; then
        log "missing $dst and no model repository is configured yet."
        log "Mount the file into $MODELS_DIR (e.g. -v /path/Qwen3.8-27B-KTopt.gguf:$dst:ro) or set KT_MODEL_REPO."
        exit 1
    fi
    mkdir -p "$MODELS_DIR"
    log "downloading $1/$2 into $MODELS_DIR (single stream, resumable)"
    curl -fL --retry 5 --retry-delay 10 -C - ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
        -o "$dst.part" "$HF_ENDPOINT/$1/resolve/main/$2"
    mv "$dst.part" "$dst"
}

# Free GPU memory at start (whole card, including other processes such as a desktop).
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_INDEX:-0}" 2>/dev/null | head -1 | tr -dc '0-9')
FREE=${FREE:-0}

if [ "$PROFILE" = auto ]; then
    # Thresholds = measured peak of the profile + ~480 MiB the driver keeps + ~200 MiB margin (docs/kt/docker.md).
    ADJ16=0; ADJ12=0
    [ "$LORA" = 1 ] && ADJ16=300 && ADJ12=50
    if   [ "$FREE" -ge $((15600 + ADJ16)) ]; then PROFILE=200k
    elif [ "$FREE" -ge $((14950 + ADJ16)) ]; then PROFILE=160k
    elif [ "$FREE" -ge $((11900 + ADJ12)) ]; then PROFILE=96k
    elif [ "$FREE" -ge $((11450 + ADJ12)) ]; then PROFILE=64k
    elif [ "$FREE" -ge $((11050 + ADJ12)) ]; then PROFILE=32k
    else log "only ${FREE} MiB of GPU memory is free; at least about 11 GB is needed."; exit 1; fi
    log "free GPU memory ${FREE} MiB -> profile ${PROFILE}"
fi

case "$PROFILE" in
    200k)    CTX=200000; TIER=t3; DRAFT=1 ;;
    160k)    CTX=163840; TIER=t3; DRAFT=1 ;;
    160k-t4) CTX=172032; TIER=t4; DRAFT=1 ;;
    262k)    CTX=262144; TIER=t2; DRAFT=1 ;;
    96k)     CTX=98304;  TIER=t2; DRAFT=0 ;;
    64k)     CTX=65536;  TIER=t2; DRAFT=0 ;;
    32k)     CTX=32768;  TIER=t2; DRAFT=0 ;;
    custom)  CTX=${CTX:?set CTX}; TIER=${TIER:?set TIER}; DRAFT=${DRAFT:-1} ;;
    *) log "unknown PROFILE=$PROFILE (auto, 200k, 160k, 160k-t4, 262k, 96k, 64k, 32k, custom)"; exit 1 ;;
esac

fetch "$KT_MODEL_REPO" "$KT_MODEL_FILE"
[ "$LORA" = 1 ] && fetch "$KT_MODEL_REPO" "$KT_LORA_FILE"
[ "$DRAFT" = 1 ] && fetch "$DRAFT_REPO" "$DRAFT_FILE"

ARGS=(-m "$MODELS_DIR/$KT_MODEL_FILE" -c "$CTX" -ctk vbr --vbr-budget "$TIER"
      -ctxcp 2 -np 1 -ngl 99 -ub 256 -b 1024 -fa on
      --jinja --reasoning-format deepseek --alias "$ALIAS"
      --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0
      --host "$HOST" --port "$PORT")
[ "$LORA" = 1 ] && ARGS+=(--lora "$MODELS_DIR/$KT_LORA_FILE")
[ "$DRAFT" = 1 ] && ARGS+=(-md "$MODELS_DIR/$DRAFT_FILE" -cd 0 -ctkd q4_0 -ctvd q4_0 --spec-draft-n-max 2)

export VBR_VRAM_HEADROOM_MIB=${VBR_VRAM_HEADROOM_MIB:-256}
log "profile ${PROFILE}: context ${CTX}, cache ${TIER}, drafter ${DRAFT}, adapter ${LORA}, port ${PORT}"
exec "${LLAMA_SERVER:-/app/llama-server}" "${ARGS[@]}" "$@"
