# kt-llama.cpp

**Qwen3.8-27B with 160k–262k tokens of context on a single 16 GB GPU.**

kt-llama.cpp is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) built on
[buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp). It brings together three pieces that were not
available in one engine:

- **Trellis ("KT") weight quantization.** The `iq2_kt`, `iq3_kt`, `iq4_kss` and `iq4_ks` types from
  [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), ported to the CUDA MMQ and MMVQ kernels.
- **A compressed KV cache.** buun-llama-cpp's VBR / TurboQuant tiers (2.25–4.125 bits per value) make
  a 200k-token cache fit next to the weights.
- **DFlash2 speculative decoding**, which keeps generation at roughly 35–40 tokens/s at long context.

The tree is merged with mainline llama.cpp (September 2026) and fixes two CUDA performance regressions of the
underlying fork. Its companion model is **Qwen3.8-27B-KTopt**, a 10.3 GB GGUF whose per-tensor recipe was
chosen by measured sensitivity.

> **Status: experimental.** KT types run on CUDA only. Tested on an RTX 4070 Ti Super (sm_89) and on RTX PRO
> 4000 / 4500 / 6000 Blackwell (sm_120), on Linux.

## Results at a glance

**Quality.** All 629 questions of GPQA Diamond, IFBench and LiveCodeBench v6, the benchmarks from the Qwen
model card. Thinking mode, one sample per question, 200k output budget, identical engine and settings for
every model:

| model | size | GPQA Diamond | IFBench | LiveCodeBench v6 | average |
|---|---|---|---|---|---|
| Q8_0 (reference) | 29.1 GB | 92.4 | 81.7 | 88.5 | 87.5 |
| **KTopt + EoRA r64** | **10.3 GB** | 89.4 | 85.3 | 85.5 | **86.7** |
| Unsloth UD-IQ3_XXS | 10.9 GB | 87.9 | 83.0 | 87.0 | 86.0 |
| Qwen model card (BF16) | 54 GB | 89.2 | 79.5 | 90.3 | |

None of the differences in the average is statistically significant. KTopt reaches its answers with about 20–30% fewer
reasoning tokens than UD-IQ3_XXS; under a 64k output budget that makes it 2.8 points better, which is
significant. [Methodology and full tables →](docs/kt/benchmarks.md)

**Long context on a 16 GB card.** HumanEval with all 164 tasks asked after a 188k-token filler: 93.3–94.5%.
Polish needle-in-a-haystack at five depths: 5/5 at 185k and 5/5 at 250k.

**Speed** on an RTX 4070 Ti Super 16 GB, with DFlash2, measured on one long documentation prompt:

| profile | context | prompt tested | prefill | generation | peak GPU memory |
|---|---|---|---|---|---|
| t3 cache, default | 200k | 144k | 692 t/s | 38.6 t/s | 15.3 GB (15.5 GB at 188k) |
| t3 cache, busy desktop | 160k | 160k | 658 t/s | 31.1 t/s | 15.4 GB |
| t2 cache, full native window | 262k | 250k | 593 t/s | 34.9 t/s | 15.5 GB |
| t2 cache, 12 GB GPUs, no drafter | 64k / 96k | 58k / 86k | 1149 / 1017 t/s | 38.5 / 36.2 t/s | 10.8 / 11.2 GB |

## Quick start

### 1. Build (Linux, CUDA)

```bash
git clone -b main https://github.com/lrozewicz/kt-llama-cpp
cd kt-llama-cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="89;120" \
  "-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16"
cmake --build build --config Release -j --target llama-server
```

Set `CMAKE_CUDA_ARCHITECTURES` for your GPU: 86 for RTX 30xx, 89 for RTX 40xx, 120 for RTX 50xx and RTX PRO
Blackwell. Only 89 and 120 have been tested.

### 2. Download the model

- **Qwen3.8-27B-KTopt-GGUF**: [Hugging Face — link coming soon](https://huggingface.co/) <!-- TODO: replace with the model repo URL -->.
  It contains `Qwen3.8-27B-KTopt.gguf` (10.3 GB) and the optional adapter `Qwen3.8-27B-KTopt-eora-output-r64.gguf` (32 MB).
- **DFlash2 drafter** (705 MB), for the long-context profiles:

```bash
hf download analogalok/Qwen3.8-27B-DFlash2-Q2_K-GGUF --local-dir models
```

### 3. Run

```bash
VBR_VRAM_HEADROOM_MIB=256 build/bin/llama-server \
  -m models/Qwen3.8-27B-KTopt.gguf -c 200000 \
  -ctk vbr --vbr-budget t3 -ctxcp 2 -np 1 -ngl 99 -ub 256 -b 1024 -fa on \
  -md models/Qwen3.8-27B-DFlash2-Q2_K.gguf -cd 0 -ctkd q4_0 -ctvd q4_0 --spec-draft-n-max 2 \
  --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
  --host 127.0.0.1 --port 8080
```

The 200k profile needs a light desktop, using at most about 0.6 GB of GPU memory. With a browser and an IDE
open, use `-c 163840`. The other profiles (t4 at 160k, t2 at 262k, 12 GB GPUs, the EoRA adapter) are described
in [Running and tuning](docs/kt/running.md).

### 4. Connect

The server speaks the OpenAI API at `http://127.0.0.1:8080/v1`. The Qwen3.8 chat template takes a
`reasoning_effort` of `low`, `medium` or `xhigh` (the default) through `chat_template_kwargs`. Examples for
curl and for the Oh My Pi coding agent are in [Running and tuning](docs/kt/running.md#clients).

## Documentation

| page | what it covers |
|---|---|
| [Running and tuning](docs/kt/running.md) | every profile with measured memory and speed, the flags that matter, 12 GB GPUs, clients, troubleshooting |
| [Benchmarks](docs/kt/benchmarks.md) | methodology, results at 64k, 128k and 200k output budgets, paired statistics, comparison with the Qwen card |
| [The KTopt model](docs/kt/quantization.md) | the quantization recipe and how it was found, KL-divergence tables, the EoRA adapter, how to reproduce it |
| [Engine changes](docs/kt/engine.md) | what differs from buun-llama-cpp and mainline, how it was verified, known limitations |
| [buun-llama-cpp README](docs/buun-llama-cpp.md), [VBR](docs/vbr.md) | the underlying fork's KV cache codecs and options |

## Credits

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp): the engine.
- [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp): the VBR / TurboQuant KV cache, TCQ codecs and DFlash2 support.
- [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp): the trellis quantization types and the quantizer used for KTopt.
- [Qwen](https://huggingface.co/Qwen) for Qwen3.8-27B, [Unsloth](https://huggingface.co/unsloth) for the reference GGUFs,
  [analogalok](https://huggingface.co/analogalok/Qwen3.8-27B-DFlash2-Q2_K-GGUF) for the DFlash2 drafter GGUF.

MIT license, like llama.cpp. See [LICENSE](LICENSE).
