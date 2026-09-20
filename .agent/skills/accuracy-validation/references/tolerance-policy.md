# Numerical Tolerance Policy

Choose one tier before execution. These defaults are guardrails, not claims that every model or kernel naturally meets
them. A different threshold requires a written numerical rationale and a new manifest before results are observed.

## Pointwise Tiers

For `|candidate - baseline| <= atol + rtol * |baseline|`:

| Tier | Typical use | Output/loss `rtol` | Output/loss `atol` | Gradient `rtol` | Gradient `atol` |
| --- | --- | ---: | ---: | ---: | ---: |
| `strict` | FP32 unit, operator, deterministic one-step parity | `1e-5` | `1e-6` | `1e-4` | `1e-5` |
| `standard` | BF16/FP16 FSDP, TP, CP, EP, Trainer component parity | `1e-3` | `1e-3` | `2e-3` | `1e-3` |
| `relaxed` | Fused/grouped kernels, low precision, numerically reordered hybrid paths | `1e-2` | `1e-2` | `2e-2` | `2e-3` |

Use exact equality for IDs, token counts, masks, integer routing decisions when ties are excluded, group membership, and
sample coverage. Never apply floating-point tolerances to hide ownership or counting errors.

## Zero-Centered Loss Trajectory

For comparable optimizer steps define:

```text
delta[t] = candidate_loss[t] - baseline_loss[t]
relative_error[t] = abs(delta[t]) / max(abs(baseline_loss[t]), 1e-8)
bias = abs(mean(delta)) / max(mean(abs(baseline_loss)), 1e-8)
```

After any explicitly declared warmup exclusion, require all limits for at least 20 steps:

| Tier | Median relative error | P95 relative error | Maximum relative error | Relative signed bias |
| --- | ---: | ---: | ---: | ---: |
| `strict` | `3e-4` (0.03%) | `1e-3` (0.1%) | `2e-3` (0.2%) | `2e-4` (0.02%) |
| `standard` | `2e-3` (0.2%) | `5e-3` (0.5%) | `1e-2` (1.0%) | `2e-3` (0.2%) |
| `relaxed` | `5e-3` (0.5%) | `1e-2` (1.0%) | `2e-2` (2.0%) | `5e-3` (0.5%) |

The signed-bias limit enforces fluctuation around the zero-difference axis instead of permitting a persistent one-sided
drift. Also report positive/negative delta counts. A short run in which all deltas have the same sign is not automatically
a failure, but it requires inspection even if the aggregate limits pass.

## Parallel-Specific Minimums

- FSDP/HSDP, TP, and CP: use at least `standard`; do not relax merely because collectives reorder summation.
- EP with the same expert implementation: use `standard`. Use `relaxed` only for a declared grouped/fused expert kernel.
- High-performance module plus parallelism: every individual axis must first pass its own tier. The combined path may use
  `relaxed` only when the optimized kernel changes accumulation order or precision.
- FP8 or other quantized training requires a separate calibration and capability benchmark. The `relaxed` tier alone is
  insufficient evidence of model quality.

## Benchmark Tolerance

For deterministic classification demos, require exact sample coverage, at least 99.9% prediction agreement, and no more
than 0.1 percentage-point absolute score loss. Any disagreement with a small candidate-score margin must retain the raw
scores for diagnosis. Generation/judge tasks use the benchmark's own uncertainty policy and repeated judge runs; do not
apply tensor `allclose` thresholds to text scores.

## Failure Handling

Report the first failing step and boundary, maximum absolute/relative error, reference magnitude, affected rank/group,
and whether the mismatch begins in inputs, forward, loss reduction, backward, gradient sync, clipping, optimizer, or
resume. Do not average away a single-rank ownership defect.
