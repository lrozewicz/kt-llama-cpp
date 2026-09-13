# The KTopt model

[← back to README](../../README.md)

`Qwen3.8-27B-KTopt.gguf` is 10.31 GB, about 3.06 bits per weight. It uses trellis quantization types from
ik_llama.cpp with a per-tensor recipe chosen from measured sensitivity, plus an optional 32 MB EoRA adapter.

## Recipe

Quantized from BF16 with `llama-quantize` from ik_llama.cpp. The base type is `IQ3_KT`. The importance matrix was
computed from a mixed calibration text (GitHub documentation, English chat prose, Polish Wikipedia and arXiv)
with 8192-token chunks.

| tensors | type |
|---|---|
| FFN, layers 0–23 | `iq2_kt` |
| FFN, layers 24–55 | `iq3_kt` |
| FFN, layers 56–63 | `iq4_kss` |
| attention and Gated DeltaNet projections | `iq3_kt` |
| `attn_output` | `iq3_kt` |
| `ssm_out` | `iq4_kss` |
| `attn_k`, `attn_v` | `q5_K` |
| `ssm_alpha`, `ssm_beta` | `q8_0` |
| `output` | `iq4_ks` |
| `token_embd` | `q2_K` |

To reproduce with ik_llama.cpp, use the exact rule string we used. The first matching rule wins, so the later
`token_embd=q3_K` has no effect; `nextn=q6_K` only matters for BF16 files that include the MTP layer, and ours did not.

```bash
llama-quantize --imatrix imatrix.dat --custom-q \
  'blk\.(5[6-9]|6[0-3])\.ffn_=iq4_kss,token_embd=q2_K,attn_output=iq3_kt,ssm_alpha=q8_0,ssm_beta=q8_0,attn_k=q5_K,attn_v=q5_K,output=iq4_ks,token_embd=q3_K,nextn=q6_K,blk\.([0-9]|1[0-9]|2[0-3])\.ffn_=iq2_kt,ssm_out=iq4_kss' \
  Qwen3.8-27B-BF16.gguf Qwen3.8-27B-KTopt.gguf IQ3_KT 24
```

This engine runs the KT types but does not quantize them: quantization stays in ik_llama.cpp.

## How the recipe was found

Starting from a baseline recipe, we ran 23 single-group ablations. Each one moved one group of tensors up or down
one type step, and each was scored by KL divergence against BF16 on code and on prose. The main findings:

- **Late FFN layers and `output` matter most.** Upgrading FFN 56–63 to `iq4_kss` was the best purchase per GB;
  downgrading `output` or FFN 48–63 was the most expensive cut.
- **Some cuts look cheap on average but are fragile.** Early attention / GDN layers (0–15) and `ssm_out` at lower
  precision barely moved the average KLD, but multiplied the error 2–5× on one hard, LaTeX-heavy prose chunk.
  Their effects compound when combined.
- **Effects are not additive.** A recipe optimised on the average (code KLD 0.0409) had a prose error of 0.504 on
  the hard chunk and was rejected. KTopt keeps only changes that were neutral on hard text.
- **Prose needs at least 19 evaluation chunks.** With 8 chunks one hard chunk dominated the result.
- **At long context, cache precision matters more than weight bits.** A 0.74 GB smaller file with a t3 cache beat
  a larger file with a t2 cache at 188k.

## KL divergence

Mean KLD of the next-token distribution against BF16 at context 4096. The code corpus is 14 GitHub repositories
created after the model's release (16 chunks); prose is recent arXiv abstracts plus Polish Wikipedia (8 chunks).
Lower is better.

| model | file size | KLD code | KLD prose |
|---|---|---|---|
| Unsloth UD-IQ4_XS | 14.25 GB | 0.0138 | 0.0187 |
| Unsloth UD-Q3_K_XL | 13.15 GB | 0.0228 | 0.0287 |
| Unsloth UD-IQ3_S | 12.04 GB | 0.0330 | 0.0507 |
| **KTopt + EoRA r64** | 10.31 GB + 32 MB | **0.0419** | **0.0644** |
| **KTopt** | 10.31 GB | 0.0429 | 0.0662 |
| Unsloth UD-IQ3_XXS | 10.93 GB | 0.0487 | 0.0781 |
| ISTA-DASLab GSQ-RCO IQ3_XXS | 10.09 GB | 0.0602 | 0.0783 |
| Unsloth UD-Q2_K_XL | 9.83 GB | 0.0741 | 0.1061 |

## EoRA adapter

`Qwen3.8-27B-KTopt-eora-output-r64.gguf` is a training-free, rank-64 correction of the quantization error of
`output.weight`, in the eigenspace of its input activations
([EoRA, Liu et al. 2024](https://arxiv.org/abs/2410.21271)). Activations came from the BF16 model on 131k
calibration tokens. It captures about 50% of the weighted error energy of that tensor.

| | KLD code | KLD prose | HumanEval@188k | generation |
|---|---|---|---|---|
| KTopt | 0.0429 | 0.0662 | 93.3% | 38.6 t/s |
| KTopt + EoRA r64 | 0.0419 (−2.3%) | 0.0644 (−2.8%) | 94.5% | 36.1 t/s (−6.4%) |

What did not work: adapters on all layers or on the late layers helped less per MB and cost more speed and memory.
The trellis error is nearly full-rank (rank 16 captures about 13% of the weighted energy). Distilling the output
adapter from BF16 logits overfitted the calibration text and did not beat plain EoRA on held-out text.
