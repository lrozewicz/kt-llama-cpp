# Engine changes

[← back to README](../../README.md)

## Base

- [buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) at `d0f82fd41`: the VBR / TurboQuant KV cache, TCQ
  codecs and DFlash2 speculative decoding.
- Merged with mainline [llama.cpp](https://github.com/ggml-org/llama.cpp) through `434ddbbc0`: 124 upstream commits,
  26 conflicts. The fork's flash-attention CUDA files were kept, and the fork's attention callers adapted to the
  new upstream signature.
- buun-llama-cpp has moved on since (193 commits as of 2026-09-13). They are not merged here, so that this tree
  stays the exact code that was benchmarked.

## Changes on top of the merge

| commit | change |
|---|---|
| `6f526e454` | CUDA port of ik_llama.cpp's `iq2_kt`, `iq3_kt`, `iq4_kss`, `iq4_ks` (details below) |
| `be9ab5d75` | The flash-attention vector kernel's "Sparse V" dequant skip now applies only to Turbo/TCQ V types. Applied unconditionally, it cost about 25% decode speed with a plain q4_0 cache at long context. |
| `a94a13112` | The MMQ J=64 tile cap for IQ2/IQ3/Q3_K on Ada is now opt-in (`GGML_CUDA_MMQ_ADA_JCAP=1`). It cost 21–31% per operation and about 12% prefill on sm_89. |
| `2dc116965`, `f25832911`, `03dfcb78a` | test fixes after the merge |

## The KT port

- **File compatibility.** The types keep ik_llama.cpp's numeric GGUF ids (144, 146, 153, 154), so KT files are
  interchangeable between the two engines.
- **Row header.** Every row of these types starts with a 4-byte float scale that mainline's core does not know.
  A new `ggml_row_meta_size()` is wired into `ggml_row_size`, `ggml_nbytes`, the row stride, the contiguity checks
  and the GGUF reader.
- **MMVQ** addresses rows of these types in bytes and hands the dot product a row pointer, as ik_llama.cpp does.
- **MMQ** gets new tile loaders (`mmq-load-tiles.cuh`) that decode (row, block) from the linear block index. The
  shared-memory layout and tile configuration follow IQ4_XS.
- **CPU** has a generic dequantize-then-dot fallback, for correctness only.
- **Limitations.** Weights must be 2D, which covers dense models; MoE expert tensors are not supported. There is no
  quantizer, so `test-backend-ops` cannot test these types.

## Verification

| check | result |
|---|---|
| KL divergence vs ik_llama.cpp on the same file | code 0.032625 vs 0.032573; prose 0.0600 vs 0.0604 |
| same file, MMQ vs MMVQ path | equal within one standard error |
| llama-bench, KT-mix file vs UD-IQ3_XXS on this engine | pp2048 1654 vs 1605 t/s; tg128 48.1 vs 49.7 t/s |
| perplexity, prose, UD-IQ3_XXS: this engine vs mainline vs original fork | 8.4609 vs 8.4543 vs 8.4795 |
| prefill after both CUDA fixes, pp2048 | 1624 t/s, mainline 1622 t/s |
| KT file on RTX PRO Blackwell (sm_120) vs RTX 4070 Ti Super (sm_89) | KLD 0.04311 vs 0.04291 |

## Known limitations

- KT types have CUDA kernels only; other backends fall back to the slow CPU path.
- Use static VBR tiers for long context. The dynamic mode and mixed K/V tiers broke long-context quality in our
  tests.
- The MTP sidecar keeps an uncompressed cache and is not usable at 150–200k context.
- A binary built with `GGML_NATIVE=ON` fails with "Illegal instruction" on CPUs with a different instruction set.
  Build shared binaries with `GGML_NATIVE=OFF`.
- Tested on Linux only: CUDA 12.0 on sm_89 and CUDA 12.8 on sm_120.
