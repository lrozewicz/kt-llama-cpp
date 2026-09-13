# Running and tuning

[← back to README](../../README.md)

All numbers on this page come from an RTX 4070 Ti Super 16 GB on Linux. One model file serves every profile;
only the flags change.

## Profiles for 16 GB GPUs

Shared flags:

```bash
M=models/Qwen3.8-27B-KTopt.gguf
LORA="--lora models/Qwen3.8-27B-KTopt-eora-output-r64.gguf"          # optional, see below
DRAFT="-md models/Qwen3.8-27B-DFlash2-Q2_K.gguf -cd 0 -ctkd q4_0 -ctvd q4_0 --spec-draft-n-max 2"
COMMON="-ctxcp 2 -np 1 -ngl 99 -ub 256 -b 1024 -fa on --jinja --reasoning-format deepseek \
        --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --host 127.0.0.1 --port 8080"
```

| profile | command | context |
|---|---|---|
| 200k, t3 cache (default) | `VBR_VRAM_HEADROOM_MIB=256 build/bin/llama-server -m $M -c 200000 -ctk vbr --vbr-budget t3 $DRAFT $COMMON` | 200k |
| 160k, t3 cache (busy desktop) | same with `-c 163840` | 160k |
| 160k, t4 cache (most precise) | `... -c 172032 -ctk vbr --vbr-budget t4 ...` | 160k |
| 262k, t2 cache (full native window) | `... -c 262144 -ctk vbr --vbr-budget t2 ...` | 262k |

Measured:

| profile | adapter | desktop VRAM | prompt | prefill | generation | peak GPU memory | quality checks |
|---|---|---|---|---|---|---|---|
| 200k t3 | no | ~0.6 GB | 144k / 188k | 692 t/s | 38.6 t/s | 15.28 / 15.51 GB | HumanEval@188k 93.3%, needle 5/5 |
| 200k t3 | r64 | ~0.6 GB | 144k / 188k | 683 t/s | 36.1 t/s | 15.56 / 15.83 GB | HumanEval@188k 94.5%, needle 5/5 |
| 160k t3 | no | ~1.2 GB | 160k | 658 t/s | 31.1 t/s | 15.45 GB | |
| 160k t3 | r64 | ~1.2 GB | 160k | 659 t/s | 31.3 t/s | 15.72 GB | too close to the limit |
| 160k t4 | no | ~0.6 GB | 160k | 805 t/s ¹ | 39.2 t/s ¹ | 15.64 GB ¹ | needle 5/5 |
| 262k t2 | no | ~0.6 GB | 250k | 593 t/s | 34.9 t/s | 15.49 GB | needle @250k 5/5 |

"Peak GPU memory" is the whole card, desktop included. On this GPU, allocations start to fail at about
15.9 GB. Generation speed falls as the context fills, so the 160k rows measured at the end of a 160k prompt
are slower than the 200k rows measured at 144k.

¹ Measured on the predecessor recipe KTsmall, whose file is 34 MB smaller.

**How much your desktop uses matters.** Browsers and IDEs with GPU acceleration each take a few hundred MB.
If a profile runs out of memory, lower `-c`: each 1k tokens of context costs about 21 MB with the t3 cache.

## GPUs with 12 GB

Use the t2 cache without the drafter; the drafter costs about 1.5 GB and was slower on our prompt.

```bash
VBR_VRAM_HEADROOM_MIB=256 build/bin/llama-server -m $M $LORA -c 65536 -ctk vbr --vbr-budget t2 $COMMON
```

| context | adapter | prompt | prefill | generation | server process memory | needle |
|---|---|---|---|---|---|---|
| 32k | r64 | 29k | 1322 t/s | 41.4 t/s | 10.35 GB | 5/5 @28k |
| 64k | r64 | 58k | 1149 t/s | 38.5 t/s | 10.76 GB | |
| 96k | r64 | 86k | 1017 t/s | 36.2 t/s | 11.18 GB | 5/5 @84k |

These are the memory of the server process alone, measured on the 16 GB card. 64k leaves about 1.2 GB for the
desktop and the driver; 96k needs other processes to use no more than about 0.9 GB. A real 12 GB card has less
memory bandwidth and will generate more slowly.

## The EoRA adapter

`Qwen3.8-27B-KTopt-eora-output-r64.gguf` corrects the quantization error of the output projection
(see [The KTopt model](quantization.md#eora-adapter)). It lowers KL divergence by 2–3% and costs about 6%
generation speed. Its GPU cost is about 0.3 GB at long context, most of it a logits-sized buffer, and only about
30 MB at short context. Load it with `--lora`. At 200k it fits only on a light desktop; for the 160k t4 and 262k
t2 profiles lower `-c` by about 16k when you add it (an estimate, not measured).

## Flags that matter

- **Use static cache tiers only** (`-ctk vbr --vbr-budget t2|t3|t4`). The fork's dynamic mode (`--vbr-floor`) and
  mixed K/V tiers broke long-context quality in our tests: HumanEval@188k fell to 0–26%.
- **`-ub 256 -b 1024`.** Larger micro-batches were about 15% slower at prefill on this model and cost about
  0.65 GB more.
- **`--spec-draft-n-max 2`.** Three draft tokens lowered acceptance and speed.
- **`-ctxcp 2`** keeps two context checkpoints for prompt caching. `-np 1` keeps a single slot, which is what
  the memory budget allows.
- **`VBR_VRAM_HEADROOM_MIB=256`** keeps a small reserve for the VBR scratch buffers.
- **MTP speculation** is not usable at long context in this fork; its sidecar keeps an uncompressed cache.
  At 32k or less it is the fastest option (about 120 t/s on our coding prompt), but this model file has no MTP layer.

## Reasoning effort and sampling

The Qwen3.8 chat template accepts `reasoning_effort` of `low`, `medium` or `xhigh` (default) in
`chat_template_kwargs`; any other value is rejected with HTTP 500. `xhigh` adds an instruction to think carefully
and check assumptions, `low` asks for brief thinking, `medium` adds nothing. The benchmarks used `xhigh`.
`"enable_thinking": false` turns thinking off. Qwen's recommended sampling for thinking mode is temperature 1.0,
top_p 0.95, top_k 20, min_p 0.

## Clients

**curl:**

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "local", "max_tokens": 4096,
  "messages": [{"role": "user", "content": "Write a quicksort in Python."}],
  "chat_template_kwargs": {"reasoning_effort": "low"}}'
```

With `--reasoning-format deepseek` the thinking arrives in `reasoning_content`, separate from `content`.

**[Oh My Pi](https://github.com/can1357/oh-my-pi)**, in `~/.omp/agent/models.yml`. A custom provider name does not
pick up omp's built-in Qwen rule, so the thinking settings must be spelled out:

```yaml
providers:
  local-kt:
    baseUrl: http://127.0.0.1:8080/v1
    apiKey: sk-local-no-auth
    api: openai-completions
    compat:
      supportsStore: false
      supportsDeveloperRole: false
    models:
      - id: qwen3.8-27b-ktopt
        name: "Qwen3.8-27B KTopt (local)"
        reasoning: true
        thinking: { mode: effort, efforts: [low, medium, xhigh], defaultLevel: low, requiresEffort: true }
        input: [text]
        tokenizer: qwen3
        contextWindow: 163840
        maxTokens: 32768
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
        compat:
          thinkingFormat: qwen
          supportsReasoningEffort: true
          qwenTemplateReasoningEffort: true
          supportsReasoningParams: true
          reasoningContentField: reasoning_content
          extraBody: { temperature: 1.0, top_p: 0.95, top_k: 20, min_p: 0.0 }
```

Start the server with `--alias qwen3.8-27b-ktopt` so the model id matches. Shift+Tab in omp cycles the effort level.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| out-of-memory errors deep into a long prompt | the desktop uses more GPU memory than the profile allows; lower `-c` or close GPU-accelerated apps |
| `Illegal instruction`, empty log | the binary was built with `GGML_NATIVE=ON` on a CPU with other instruction sets; rebuild with `GGML_NATIVE=OFF` |
| HTTP 500 "Unexpected reasoning effort" | the client sent an effort other than `low`, `medium` or `xhigh` |
| the server loads another llama.cpp's libraries | `export LD_LIBRARY_PATH=$PWD/build/bin` |
| "couldn't bind HTTP server socket" | the port is taken; choose another `--port` |
