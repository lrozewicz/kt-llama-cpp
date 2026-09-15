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

**What you need**

- Linux, or Windows with WSL2.
- An NVIDIA GPU with at least 12 GB of memory; 16 GB for 160k context or more.
- NVIDIA driver 570 or newer.
- About 12 GB of free disk space.

The model uses quantization types that mainline llama.cpp, LM Studio and Ollama cannot load. Run it with the
kt-llama.cpp engine: as a ready Docker image (option A) or built from source (option B).

### Option A: Docker (recommended)

**1. Install Docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).**
Check that containers can see your GPU; the command should print a table with your card:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

**2. Start the server:**

```bash
docker run -d --name kt-llama --gpus all -p 8080:8080 -v kt-models:/models ghcr.io/lrozewicz/kt-llama-cpp:cuda
```

**3. Wait until it is ready.** The first start downloads about 11 GB (the model and the DFlash2 drafter) into the
`kt-models` Docker volume; later starts reuse the files. Follow the progress with `docker logs -f kt-llama`; during
the download it prints a line every 10 seconds, such as
`[kt] Qwen3.8-27B-KTopt.gguf: 5.83 of 10.30 GB (56%), 10.6 MB/s, about 7 min left`. Ctrl+C stops following the log,
not the server. The line `[kt] free GPU memory ... -> profile 160k` shows the context size that was chosen. Until the
server is up, every request to port 8080 gets HTTP 503 with the reason: the download progress, then `Loading model`
for about 30 seconds. It is ready when this command prints `{"status":"ok"}`:

```bash
curl http://127.0.0.1:8080/health
```

**4. Stop it and start it again later:**

```bash
docker stop kt-llama
docker start kt-llama
```

The context size is chosen from the GPU memory that is free when the container starts:

| your GPU | chosen automatically | to choose it yourself, add to step 2 |
|---|---|---|
| 16 GB, light desktop (under about 0.8 GB of GPU memory in use) | 200k | `-e PROFILE=200k` |
| 16 GB, browser and IDE open | 160k | `-e PROFILE=160k` |
| 12 GB | 96k, 64k or 32k, without the drafter | `-e PROFILE=96k` |
| 16 GB, full native window, light desktop | never chosen automatically | `-e PROFILE=262k` |

To change it later, remove the container with `docker rm -f kt-llama` and run step 2 again with the `-e` option; the
downloaded files stay in the volume. If port 8080 is taken, use `-p 8081:8080` and port 8081 in the URLs below.
All settings are in [the Docker guide](docs/kt/docker.md).

### Option B: build from source

**1. Build the engine.** You need the CUDA toolkit (12.8 or newer for RTX 50xx and RTX PRO Blackwell), CMake and a C++
compiler. Set `CMAKE_CUDA_ARCHITECTURES` to 86 for RTX 30xx, 89 for RTX 40xx or 120 for RTX 50xx and RTX PRO Blackwell.

```bash
git clone https://github.com/lrozewicz/kt-llama-cpp
cd kt-llama-cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  "-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16"
cmake --build build --config Release -j --target llama-server
```

**2. Download the model and the drafter.** The `hf` command comes with `pip install -U huggingface_hub`.

```bash
hf download wiklif/Qwen3.8-27B-KTopt-GGUF Qwen3.8-27B-KTopt.gguf --local-dir models
hf download analogalok/Qwen3.8-27B-DFlash2-Q2_K-GGUF Qwen3.8-27B-DFlash2-Q2_K.gguf --local-dir models
```

**3. Start the server.** On a 16 GB GPU, 160k context (works with a browser and an IDE open):

```bash
VBR_VRAM_HEADROOM_MIB=256 build/bin/llama-server \
  -m models/Qwen3.8-27B-KTopt.gguf -c 163840 -ctk vbr --vbr-budget t3 \
  -md models/Qwen3.8-27B-DFlash2-Q2_K.gguf -cd 0 -ctkd q4_0 -ctvd q4_0 --spec-draft-n-max 2 \
  -ctxcp 2 -np 1 -ngl 99 -ub 256 -b 1024 -fa on \
  --jinja --reasoning-format deepseek --alias qwen3.8-27b-ktopt \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --host 127.0.0.1 --port 8080
```

With a light desktop you can use `-c 200000` for 200k, or `-c 262144 --vbr-budget t2` for the full 262k.

On a 12 GB GPU, 96k context without the drafter:

```bash
VBR_VRAM_HEADROOM_MIB=256 build/bin/llama-server \
  -m models/Qwen3.8-27B-KTopt.gguf -c 98304 -ctk vbr --vbr-budget t2 \
  -ctxcp 2 -np 1 -ngl 99 -ub 256 -b 1024 -fa on \
  --jinja --reasoning-format deepseek --alias qwen3.8-27b-ktopt \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --host 127.0.0.1 --port 8080
```

**4. Check that it is ready.** Loading takes about 30 seconds; then `curl http://127.0.0.1:8080/health` prints
`{"status":"ok"}`. Flags, other profiles and troubleshooting: [Running and tuning](docs/kt/running.md).

### Test the API from Python

The server speaks the OpenAI Chat Completions API at `http://127.0.0.1:8080/v1`. The model id is `qwen3.8-27b-ktopt`, and
the sampling settings recommended by Qwen are already set on the server.

**A simple request**, with `pip install requests`:

```python
import requests

BASE_URL = "http://127.0.0.1:8080/v1"

resp = requests.post(
    f"{BASE_URL}/chat/completions",
    json={
        "model": "qwen3.8-27b-ktopt",
        "messages": [{"role": "user", "content": "Write a Python function that checks whether a number is prime."}],
        "max_tokens": 4096,
        "chat_template_kwargs": {"reasoning_effort": "low"},  # low | medium | xhigh
    },
    timeout=600,
)
resp.raise_for_status()
message = resp.json()["choices"][0]["message"]
print("--- reasoning ---")
print(message.get("reasoning_content", ""))
print("--- answer ---")
print(message["content"])
```

**Streaming with the official OpenAI client**, with `pip install openai`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")

stream = client.chat.completions.create(
    model="qwen3.8-27b-ktopt",
    messages=[{"role": "user", "content": "Explain in three sentences what a KV cache is."}],
    max_tokens=4096,
    stream=True,
    extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
)

answering = False
for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    thinking = getattr(delta, "reasoning_content", None)
    if thinking:
        print(thinking, end="", flush=True)      # the model's reasoning comes first
    if delta.content:
        if not answering:
            print("\n--- answer ---")
            answering = True
        print(delta.content, end="", flush=True)
print()
```

- `reasoning_effort` sets how long the model thinks: `low` (fastest), `medium`, or `xhigh` (the default, used in the
  benchmarks). Any other value returns HTTP 500.
- The thinking arrives in `reasoning_content`, separately from the answer in `content`.
- Long answers can take minutes, so keep the timeout generous.

## Documentation

| page | what it covers |
|---|---|
| [Docker](docs/kt/docker.md) | the container image, automatic profile selection, settings, building it yourself |
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
