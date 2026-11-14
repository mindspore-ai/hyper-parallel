# Single-Node Failure Patterns

Use these observed failure patterns to form a falsifiable hypothesis and
localize the initiating error before editing code.

## Environment failures that resemble logic bugs

| Symptom | Distinguishing evidence | Investigation |
|---|---|---|
| CPU unit-test backward raises `aclInit 507008` | Non-reentrant checkpoint replay restores autocast for a registered NPU device, reaching `is_bf16_supported` and `_lazy_init` on a host without an NPU. | Confirm the optional torch_npu dependency and device-probe path. Isolate the device-specific probe in a CPU test rather than changing training semantics. |
| Step 0 takes more than 20 minutes | Initial AOT compilation for a new shape can be slow. | Inspect kernel-cache timestamps and CPU activity. Adjust the execution timeout when compilation is confirmed. |
| Some ranks have no visible traceback | torchrun output may be truncated. | Read full per-rank logs and the resolved runtime configuration. |
| A source change appears to pass immediately | Stale `.pyc` files can execute the previous source. | Verify the imported revision and positive engagement markers; clean the owned checkout's bytecode cache if necessary. |
| A device failure escapes an operator-level `try/except` | Asynchronous errors such as `ACL stream synchronize failed` surface at synchronization, outside the initiating call. | Synchronize after a suspected region for diagnosis or use device logs to locate the source. |

## Numerical failures

| Symptom | Candidate cause | Investigation |
|---|---|---|
| Flat loss or zero hit rate | Labels shifted before input and shifted again inside the module | Compare target values and their offset from `input_ids`. |
| Changed selections or routing with slow non-convergence | Reduced-precision scores amplify cancellation and alter rankings | Compare with the reference FP32 scoring path and quantify BF16/FP16 selection changes. |
| Downstream NaN values | An upstream division, overflow, or invalid index | Use overflow detection to find the first nonfinite operation. |
| Recomputation changes results or gradients | A cached tensor was modified in place by `masked_fill_`, `scatter_`, `copy_`, or a similar operation | Compare selective recomputation on/off and inspect the first mutation of a replayed tensor. |

## Single-device performance failures

| Symptom | Candidate cause | Investigation |
|---|---|---|
| Slow post-processing | Integer ArgSort or scatter_reduce falls back to AI CPU | Check `npu_cpu_fallback` and supported native sorting/top-k implementations, preserving ordering semantics. |
| Gather bandwidth is unexpectedly low | Per-position advanced indexing takes a slow path | Compare an equivalent `index_select` implementation in an isolated paired test. |
| An optimization has no measured effect | The intended path never engaged | Verify positive path markers before concluding that the intervention is ineffective. |

## Experimental discipline

- Test a specific causal hypothesis before changing kernel initialization,
  padding, or other behavior.
- Synchronization can charge neighboring asynchronous work to the wrong
  interval. Confirm regional timing with an isolated paired test. An
  observed 90 ms einsum interval became 2 ms when measured independently.
- Distinguish an inactive intervention from an active intervention with no
  benefit. Both can produce an unchanged end-to-end metric.
