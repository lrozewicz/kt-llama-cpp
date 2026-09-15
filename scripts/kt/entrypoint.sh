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
gb() { awk -v n="$1" 'BEGIN { printf "%.2f GB", n / 1e9 }'; }

progress() {  # <file> <bytes so far> <total bytes, 0 = unknown> <bytes per second>
    local s
    s=$(awk -v f="$1" -v n="$2" -v t="$3" -v r="$4" 'BEGIN {
        s = sprintf("%s: %.2f", f, n / 1e9)
        s = s (t > 0 ? sprintf(" of %.2f GB (%d%%)", t / 1e9, 100 * n / t) : " GB")
        s = s sprintf(", %.1f MB/s", r / 1e6)
        if (t > 0 && r > 0) { m = (t - n) / r / 60; s = s (m < 1 ? ", under a minute left" : sprintf(", about %d min left", m + 0.5)) }
        print s }')
    log "$s"
    status "Downloading model files: $s"
}

# While files download, socat answers every request on $PORT with 503 and the progress (like llama-server's
# "Loading model"), so a client sees why the server is not ready instead of a connection reset.
export KT_STATUS_FILE=${KT_STATUS_FILE:-/tmp/kt-status}
status_pid=""
status() { printf '%s\n' "$*" > "$KT_STATUS_FILE.tmp" && mv "$KT_STATUS_FILE.tmp" "$KT_STATUS_FILE"; }
status_start() {
    [ -n "$status_pid" ] && return 0
    command -v socat >/dev/null || return 0
    socat "TCP-LISTEN:$PORT,reuseaddr,fork" EXEC:/app/kt-status-http.sh 2>/dev/null &
    status_pid=$!
}
status_stop() {  # frees $PORT for llama-server
    [ -n "$status_pid" ] || return 0
    kill "$status_pid" 2>/dev/null; wait "$status_pid" 2>/dev/null || true
    status_pid=""
}

fetch() {  # <repo> <file>: download into $MODELS_DIR unless present
    local dst="$MODELS_DIR/$2" url="$HF_ENDPOINT/$1/resolve/main/$2"
    [ -s "$dst" ] && return 0
    if [[ "$1" == TODO/* ]]; then
        log "missing $dst and no model repository is configured yet."
        log "Mount the file into $MODELS_DIR (e.g. -v /path/Qwen3.8-27B-KTopt.gguf:$dst:ro) or set KT_MODEL_REPO."
        exit 1
    fi
    mkdir -p "$MODELS_DIR"
    status_start
    # Size from a HEAD request: the last Content-Length after the redirects, 0 when unknown.
    local total have size="size unknown" from=""
    total=$(curl -fsIL ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} "$url" | tr -d '\r' \
        | awk 'tolower($1) == "content-length:" { n = $2 } END { print n + 0 }') || total=0
    have=$(stat -c %s "$dst.part" 2>/dev/null || echo 0)
    [ "$total" -gt 0 ] && size=$(gb "$total")
    [ "$have" -gt 0 ] && from=", resuming at $(gb "$have")"
    log "downloading $1/$2 ($size) into $MODELS_DIR$from; one stream, progress every 10 s"
    status "Downloading model files: $2 ($size)${from:-, starting}"

    # curl's own meter is a single \r-joined line in `docker logs`, so it runs silent and we print our own lines.
    curl -fL -sS --retry 5 --retry-delay 10 -C - ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
        -o "$dst.part" "$url" &
    local pid=$! tick fin rc prev=$have now
    while :; do
        sleep 10 & tick=$!
        rc=0; wait -n -p fin "$pid" "$tick" || rc=$?
        if [ "$fin" = "$pid" ]; then kill "$tick" 2>/dev/null; wait "$tick" 2>/dev/null || true; break; fi
        now=$(stat -c %s "$dst.part" 2>/dev/null || echo 0)
        progress "$2" "$now" "$total" $(( (now - prev) / 10 )); prev=$now
    done
    if [ "$rc" -ne 0 ]; then
        log "downloading $2 failed (curl exit code $rc, error above). If it is not the network, check KT_MODEL_REPO" \
            "and HF_TOKEN; then start the container again and the download resumes where it stopped."
        exit "$rc"
    fi
    mv "$dst.part" "$dst"
    log "downloaded $2"
}

# Free GPU memory at start (whole card, including other processes such as a desktop).
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_INDEX:-0}" 2>/dev/null | head -1 | tr -dc '0-9')
FREE=${FREE:-0}

if [ "$PROFILE" = auto ]; then
    # nvidia-smi "free" already excludes the driver reserve. Threshold = measured peak of the profile + ~250 MiB
    # (a 200 MiB safety margin plus the gap between the reserve and the point where allocations start to fail).
    ADJ16=0; ADJ12=0
    [ "$LORA" = 1 ] && ADJ16=300 && ADJ12=50
    if   [ "$FREE" -ge $((15150 + ADJ16)) ]; then PROFILE=200k
    elif [ "$FREE" -ge $((14500 + ADJ16)) ]; then PROFILE=160k
    elif [ "$FREE" -ge $((11450 + ADJ12)) ]; then PROFILE=96k
    elif [ "$FREE" -ge $((11000 + ADJ12)) ]; then PROFILE=64k
    elif [ "$FREE" -ge $((10600 + ADJ12)) ]; then PROFILE=32k
    else log "only ${FREE} MiB of GPU memory is free; at least about 10.6 GB is needed."; exit 1; fi
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
status_stop

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
