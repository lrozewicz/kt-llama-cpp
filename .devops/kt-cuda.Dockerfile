# kt-llama.cpp CUDA image: llama-server built for sm_86 / sm_89 / sm_120 plus an entrypoint that downloads
# Qwen3.8-27B-KTopt on first start and runs one of the measured profiles. See docs/kt/docker.md.
ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=12.8.1
ARG BASE_CUDA_DEV_CONTAINER=docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_CUDA_RUN_CONTAINER=docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_DEV_CONTAINER} AS build
ARG CUDA_ARCHS="86;89;120"
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates libssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
        "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}" \
        "-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16" \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --config Release -j"$(nproc)" --target llama-server \
    && mkdir -p /out && cp -a build/bin/llama-server /out/ && find build -name "*.so*" -exec cp -P {} /out/ \;

FROM ${BASE_CUDA_RUN_CONTAINER}
ARG APP_REVISION=N/A
LABEL org.opencontainers.image.title="kt-llama.cpp" \
      org.opencontainers.image.description="Qwen3.8-27B with 160k-262k context on one 16 GB GPU (llama.cpp fork with KT quants and VBR KV cache)" \
      org.opencontainers.image.source="https://github.com/lrozewicz/kt-llama-cpp" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.revision=$APP_REVISION
# socat serves the download progress on the server port until llama-server starts (scripts/kt/status-http.sh).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libssl3 curl ca-certificates socat \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/ /app/
COPY scripts/kt/entrypoint.sh /app/kt-entrypoint.sh
COPY scripts/kt/status-http.sh /app/kt-status-http.sh
ENV LD_LIBRARY_PATH=/app NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    MODELS_DIR=/models PORT=8080 HOST=0.0.0.0 PROFILE=auto LORA=0
VOLUME /models
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20m CMD curl -fs "http://127.0.0.1:${PORT}/health" || exit 1
ENTRYPOINT ["/app/kt-entrypoint.sh"]
