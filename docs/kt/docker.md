# Docker

[← back to README](../../README.md)

The image `ghcr.io/lrozewicz/kt-llama-cpp:cuda` contains `llama-server` built for sm_86, sm_89 and sm_120, plus an
entrypoint that downloads the model on first start and picks a measured profile for your GPU.

## Requirements

- Linux, or Windows with WSL2 and Docker Desktop's GPU support.
- NVIDIA driver 570 or newer (the image uses CUDA 12.8).
- The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- About 11 GB of free GPU memory, and 11 GB of disk for the model files.

## Run

```bash
docker run --gpus all -p 8080:8080 -v kt-models:/models ghcr.io/lrozewicz/kt-llama-cpp:cuda
```

On first start the container downloads `Qwen3.8-27B-KTopt.gguf` (10.3 GB) and, for the long-context profiles, the
DFlash2 drafter (705 MB) into the `kt-models` volume, one file at a time. Later starts reuse them. The server is
ready when `curl http://127.0.0.1:8080/health` returns `{"status":"ok"}`, and speaks the OpenAI API at
`http://127.0.0.1:8080/v1` with the model id `qwen3.8-27b-ktopt`.

To use files you already have, mount them instead of the volume:

```bash
docker run --gpus all -p 8080:8080 \
  -v /path/to/Qwen3.8-27B-KTopt.gguf:/models/Qwen3.8-27B-KTopt.gguf:ro \
  -v /path/to/Qwen3.8-27B-DFlash2-Q2_K.gguf:/models/Qwen3.8-27B-DFlash2-Q2_K.gguf:ro \
  ghcr.io/lrozewicz/kt-llama-cpp:cuda
```

With Compose: `docker compose up -d` using the [docker-compose.yml](../../docker-compose.yml) in the repository.

## Profiles

`PROFILE=auto` (the default) reads the free GPU memory when the container starts and picks the largest profile
that fits. Each threshold is the measured peak of the profile plus about 480 MiB that the driver keeps and a
200 MiB margin.

| profile | context | cache | drafter | picked by `auto` when free GPU memory is at least |
|---|---|---|---|---|
| `200k` | 200,000 | t3 | yes | 15,600 MiB (a 16 GB card with a light desktop) |
| `160k` | 163,840 | t3 | yes | 14,950 MiB |
| `96k` | 98,304 | t2 | no | 11,900 MiB (a 12 GB card) |
| `64k` | 65,536 | t2 | no | 11,450 MiB |
| `32k` | 32,768 | t2 | no | 11,050 MiB |
| `160k-t4` | 172,032 | t4 | yes | manual only |
| `262k` | 262,144 | t2 | yes | manual only |
| `custom` | `CTX` | `TIER` | `DRAFT` | manual only |

`auto` only sees the memory free at start. If you open a browser or an IDE later, the margin shrinks; choose a
smaller profile by hand if you often do. Speeds and quality checks for each profile are in
[Running and tuning](running.md).

## Settings

| variable | default | meaning |
|---|---|---|
| `PROFILE` | `auto` | see above |
| `LORA` | `0` | `1` loads the EoRA r64 adapter; `auto` then asks for about 300 MiB more on 16 GB profiles |
| `PORT` | `8080` | server port inside the container |
| `ALIAS` | `qwen3.8-27b-ktopt` | model id reported by the API |
| `KT_MODEL_REPO` | the published model repository | Hugging Face repository with `Qwen3.8-27B-KTopt.gguf` |
| `HF_TOKEN` | unset | token for gated or private repositories |
| `GPU_INDEX` | `0` | GPU whose free memory `auto` reads |

Arguments after the image name go straight to `llama-server`, for example `--api-key secret` or `--metrics`.

## Building the image yourself

```bash
docker build -f .devops/kt-cuda.Dockerfile -t kt-llama-cpp:cuda .
docker build -f .devops/kt-cuda.Dockerfile --build-arg CUDA_ARCHS=89 -t kt-llama-cpp:cuda .   # faster, one GPU family
```

The GitHub Actions workflow `.github/workflows/kt-docker.yml` builds and publishes the image on every push to
`main` that changes code. The workflows inherited from llama.cpp are renamed to `*.yml.disabled`.
