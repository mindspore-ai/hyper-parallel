# Qwen3 MoE + Muon: Q/K RMSNorm fusion and two negative results

> **Evidence provenance:** A training experiment.
> Results apply to the measured configuration. Operator speedups must not be
> extrapolated into model-throughput speedups.

## Configuration and measurement protocol

Qwen3-30B-A3B with loaded weights, four nodes and 32 die, HSDP shard8/replica4, and
TP/CP/EP/PP all 1. Global batch 32, microbatch 1, BF16, native routing, and recomputation
of the first 16 layers. Muon used five asym5 Newton–Schulz iterations with an AdamW path.
The environment was CANN 9.0.0, Torch 2.9.0, torch_npu 2.9.0.post2, and Transformers 5.20.0.
Each run completed 50 steps, with a primary window of steps 11–50 and an auxiliary
window of 31–50. The declared peak was 376 TFLOPS/die, with the full S×S PaLM convention.
The benchmark disabled profiling. A separate 50-step run profiled steps 11–13 on ranks
0, 8, 16, and 24.

The MFU numerator includes useful model work only. HFU in this case adds 16 layers of
recomputation and excludes Muon/AdamW. The user approved module relative RMS error ≤5%,
normalized maximum error ≤5%, and whole-model scalar tolerances of rtol 5% / atol 0.01.
These were short-run criteria, not general convergence criteria. The performance
retention threshold was fixed separately at 2%.

## Verified intervention

The model already used fused backbone RMSNorm, FlashAttention, RoPE, grouped matmul,
and SwiGLU, while GQAAttention still used HF Q/K RMSNorm. Traces showed FP32
Pow/Mul/Reduce/Cast chains for shapes [1,S,32,128] and [1,S,4,128]. Replacing these
128-dimensional normalization calls with `npu_rms_norm` and its runtime backward
preserved gamma identity, epsilon, BF16, and noncontiguous input layouts. `wraps`
preserved the original forward interface.

- Baseline `000-a8e78934`: MFU 13.5283%, 59,078.37 tokens/s, 2.132071 s/step.
- Q/K fusion `003-ef80585b`: MFU 13.9239%, 60,805.91 tokens/s, 2.071497 s/step,
  a relative improvement of 2.92%.
- Peak allocated memory decreased from 43.0972 to 40.7218 GiB/die.
- Across three profiled steps on rank 0, 5664 scalar-chain invocations for the Q/K
  shapes became 384 RmsNorm and 288 RmsNormGrad calls. Summed device-kernel duration
  fell from approximately 323 ms to 53 ms. Sums across streams are not critical-path shares.
- Forward/backward probes at lengths 3776, 3777, and 4096, using actual strides and
  non-unit gamma, measured Q speedups of approximately 5.6–5.7× and K speedups of
  approximately 1.9–2.0×. End-to-end gains were much smaller.
- Inputs matched across 1600 rank-steps. Maximum relative differences from baseline
  were 0.38% for loss and 3.01% for gradient norm, within the approved thresholds.
- Independent confirmation `006-dd4473f5` passed with MFU 13.9232% and 60,802.86 tokens/s,
  approximately -0.005% from the first Q/K result. Inputs, loss, and gradients matched
  bitwise across 1600 rank-steps. The run stopped after exhausting six candidate and
  confirmation trials. The 20% target was not achieved.

## Verified negative results

1. `004-8a261598` retained Q/K fusion, made transposed Muon inputs contiguous, and used
   foreach to batch two independent coefficient scalings. Newton–Schulz coefficients,
   iteration count, BF16, and matrix-multiplication count were unchanged. Whole-model
   inputs, loss, and gradients were bitwise equal to the Q/K parent. Traces removed some
   Transpose/Muls work while the dominant BatchMatMul work remained almost unchanged.
   MFU was 13.9527%, only +0.21% over the parent, so the candidate was discarded.
2. `005-cae21f04` verified pure causal masks elementwise before using FlashAttention
   `sparse_mode2` with a compressed causal mask. Padding and non-pure-causal masks used
   the original implementation. Prechecks covered Q/K/V gradients and padding.
   The path engaged for 102400 calls, and whole-model scalars were bitwise equal.
   MFU was 13.9884%, only +0.46% over the parent, so the candidate was discarded.
   The original implementation already set `next_tockens=0`; changing the mode did
   not introduce causal sparse computation for the first time.

## Applicability and evidence

The official 700-branch RMSNorm and FlashAttention pages provide semantic references,
but do not alone prove compatibility with the 2.9 runtime. Check the installed training
graph and backward registration, then measure forward and backward on the current device.
Inference-only APIs are not direct replacements for training operators.

Baseline and candidate runs occurred on different days, so environment changes are an
alternative explanation for small gains. Independent confirmation remains necessary.
Fifty steps do not establish long-term convergence.

- `REPORT.md` and `all_trials.csv`: all results, failures, interruptions, and confirmation runs.
- `profile_comparison_000-a8e78934_to_003-ef80585b.json`: before/after operator evidence for four ranks.
- `planning/qk_microbenchmark.json`: independent synchronized microbenchmarks and numerical comparisons.
- `planning/ns_kernel_comparison.json` and `planning/attention_kernel_comparison.json`: mechanisms behind negative results.
- `loop/trials/*/source.tar.gz` and `context.json`: complete source hashes and configurations.
- `communication_compute_timeline.html`: device compute, communication, and idle intervals.
  Idle time with PP1 is not described as a pipeline-parallel bubble.

First identify which normalization paths remain unfused. Existing FlashAttention or
GMM integration does not imply that every subpath is fused. Operator engagement, local
speedup, and end-to-end improvement each require their own evidence.
