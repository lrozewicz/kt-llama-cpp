# Benchmarks

[← back to README](../../README.md)

## Setup

- **Benchmarks:** the three text benchmarks from the [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
  that can run without an external judge or sandboxed agents. All questions: GPQA Diamond (198), IFBench (300),
  LiveCodeBench v6 (131 problems, window 2025-02-01 … 2025-04-06).
- **Scoring:** GPQA with the simple-evals prompt ("Answer: $LETTER", options shuffled per question), IFBench with
  the authors' `run_eval.py`, LiveCodeBench with the official `codegen_metrics`. Only the final answer is scored,
  never the reasoning. An answer cut off at the output limit counts as wrong.
- **Generation:** thinking mode with the card's sampling (temperature 1.0, top_p 0.95, top_k 20), one sample per
  question, plain q8_0 KV cache, no drafter. The same engine build and settings for every model.
- **Models:** `unsloth/Qwen3.8-27B-GGUF` Q8_0 as the reference, KTopt with the EoRA r64 adapter, and Unsloth
  UD-IQ3_XXS as a quant of similar size.
- **Hardware:** RTX PRO 4000 / 4500 / 6000 Blackwell GPUs on RunPod, about 80 GPU-hours in total.

We first ran with a 64k output limit. Every answer cut off at 64k was then continued from its saved reasoning,
first to 128k and then to 200k tokens. Continuing is statistically the same as a larger limit from the start,
because the text written before the old limit does not change. Three answers stuck in a degenerate repetition
loop (one KTopt, two UD-IQ3_XXS) were not continued and count as wrong.

## Results at a 200k output limit

| benchmark | Q8_0 | KTopt + EoRA r64 | UD-IQ3_XXS | Qwen card (BF16) |
|---|---|---|---|---|
| GPQA Diamond, accuracy | 92.4 ± 3.7 | 89.4 ± 4.3 | 87.9 ± 4.5 | 89.2 |
| IFBench, prompt-level loose / strict | 81.7 / 79.0 | 85.3 / 81.7 | 83.0 / 80.3 | 79.5 |
| LiveCodeBench v6, pass@1 | 88.5 ± 5.5 | 85.5 ± 6.0 | 87.0 ± 5.8 | 90.3 |
| LCB by difficulty: easy / medium / hard | 100 / 92.3 / 80.3 | 100 / 92.3 / 73.8 | 100 / 92.3 / 77.0 | |
| average of the three | 87.5 | 86.7 | 86.0 | |
| mean output tokens: GPQA / IFBench / LCB | 15.7k / 8.5k / 29.1k | 14.9k / 7.7k / 29.7k | 20.7k / 10.3k / 36.3k | |

± is the half-width of a 95% confidence interval.

Paired differences on the same questions, bootstrap 95% CI with 10,000 resamples, stratified by benchmark for
the average:

| pair | GPQA | IFBench | LCB | average |
|---|---|---|---|---|
| KTopt − UD-IQ3_XXS | +1.5 [−1.5, +4.5] | +2.3 [−1.7, +6.3] | −1.5 [−6.9, +3.8] | +0.8 [−1.6, +3.1] |
| KTopt − Q8_0 | −3.0 [−6.1, 0.0] | +3.7 [0.0, +7.7] | −3.1 [−9.2, +3.1] | −0.8 [−3.3, +1.8] |
| UD-IQ3_XXS − Q8_0 | −4.5 [−8.1, −1.0] | +1.3 [−2.7, +5.7] | −1.5 [−6.1, +3.1] | −1.6 [−4.0, +0.8] |

## How the output limit changes the picture

| output limit | Q8_0 | KTopt + r64 | UD-IQ3_XXS | KTopt − UD-IQ3_XXS | LCB answers cut off (Q8 / KTopt / IQ3) |
|---|---|---|---|---|---|
| 64k | 82.2 | 82.1 | 79.3 | **+2.8 [+0.2, +5.4]** | 22 / 22 / 32 of 131 |
| 128k | 87.3 | 86.5 | 85.7 | +0.8 [−1.6, +3.1] | 2 / 2 / 3 |
| 200k | 87.5 | 86.7 | 86.0 | +0.8 [−1.6, +3.1] | 0 / 0 / 0 |

## Reading the results

- **With a large budget, all three are close.** The averages lie within 1.5 points and no average difference is
  significant. KTopt sits 0.8 points below Q8_0 and 0.8 points above UD-IQ3_XXS.
- **With a tight budget, token efficiency decides.** UD-IQ3_XXS reasons 20–35% longer than Q8_0 on the same
  questions and runs out of budget more often; at 64k KTopt is 2.8 points ahead of it. KTopt's answer lengths
  match Q8_0, which also means faster answers and less context used on a 16 GB card.
- **Single-benchmark differences of 3–4 points are noise** at one sample per question. KTopt's IFBench lead over
  Q8_0 is most likely noise; we do not claim the quant beats the reference.
- **The harness agrees with the Qwen card.** With the large budget the Q8_0 reference matches the card on all three
  benchmarks within the confidence intervals. The earlier LiveCodeBench gap (74.8 vs 90.3 at 64k) came from the
  output limit: most of the 22 cut-off answers finished between 64k and 128k tokens. Independent reports agree on
  GPQA ([ISTA-DASLab, BF16 89.9](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF)). The card does not
  state its token budget, sample count or LCB window.
- **What this measures.** The runs use a plain q8_0 cache and moderate prompt lengths, so they measure the weights
  and the adapter. The compressed long-context cache is tested separately below.

## Long-context checks (16 GB profiles)

| check | result |
|---|---|
| HumanEval, all 164 tasks asked after a 188k-token filler, t3 cache | 93.3% without adapter, 94.5% with EoRA r64. UD-IQ3_XXS on the same engine (t2): 91.5% |
| Polish needle-in-a-haystack, 5 depths (10–90%) at 185k, t3 | 5/5 |
| same at 250k, t2 | 5/5 |
| dense sweep 90k–190k in 8k steps, 5 needles each (predecessor recipe KTsmall, t3) | 65/65, no empty answers |

The last row concerns [llama.cpp#27756](https://github.com/ggml-org/llama.cpp/issues/27756), silent EOS on prompts
above about 130k in mainline. We did not observe it in this engine.

KL-divergence measurements against BF16 are on [The KTopt model](quantization.md#kl-divergence) page.
