# Optimization techniques by bottleneck

> **Evidence provenance:** Mixed. General tuning practice, with measured
> counterexamples from operator and postprocessing optimization in this repository.
> See [evidence provenance](../SKILL.md#evidence-provenance).

Map the bottleneck identified by profiling or memory analysis to a tuning technique.
**Obtain the breakdown before consulting the table.** Each technique has applicability
conditions and counterexamples. Change one technique at a time and re-measure on the
same node in consecutive runs.

## Bottleneck categories

### Compute bottlenecks (high Cube, Vector, or FlashAttention share)

| Technique | Applicable scenario | Counterexample or constraint |
|---|---|---|
| Replace an operator chain with a fused implementation | Chains of small operators, fragmented postprocessing, or AI CPU fallback | Unsupported head counts, dtypes, or shapes cause per-call fallback and remove the benefit. See `fused-ops.md`. |
| Use a TopK kernel for sorting | Ascending-order selection or selecting the first k elements | Integer ArgSort can fall back to AI CPU when no AI Core implementation exists. Sort FP32 keys and convert back where equivalent. |
| Adjust dtype or layout | Unnecessary FP32 promotion or noncontiguous tensors | Do not reduce score precision without validation; changed tie ordering can exceed the tolerance. |
| Remove synchronization and Host readbacks | `.item()`, `bool(x.any())`, and similar forced stream synchronization | Data-dependent Host checks synchronize the pipeline. A branch-free `where` expression is often preferable. |

### Communication bottlenecks (high exposed TP, DP, EP, CP, or PP communication)

| Technique | Applicable scenario | Counterexample or constraint |
|---|---|---|
| Overlap computation and communication | Communication remains exposed or execution waits serially | Overlap requires additional buffers; assess the memory tradeoff. |
| Change the parallel layout | Communication along one parallel dimension is disproportionately expensive | Parallel degrees affect memory and operator support. Re-measure under a consistent protocol. |
| Fixed routing or EP scheduling | Imbalanced MoE routing or dispatch hotspots | Assess the tradeoff between load balance and numerical equivalence. |
| Select the communication group or algorithm | All-to-All versus ring communication, Ulysses versus hybrid layouts | Results across different topologies are not directly comparable. Use consecutive runs on the same hardware. |

### Device-memory bottlenecks (peak capacity or OOM)

| Technique | Applicable scenario | Counterexample or constraint |
|---|---|---|
| Full or selective activation recomputation | Large activation footprint with spare compute capacity | Selective activation checkpointing must not mutate cached tensors in place. CPU-only runs do not establish the device-memory benefit. |
| Activation or parameter offload | Limited HBM with available Host bandwidth | Preserve `wait_load`/`wait_offload` ordering and stream synchronization. |
| Row or block chunking | One operator creates oversized intermediate tensors | Too many chunks expose fixed overhead. Sweep the chunk size. |
| Reuse outputs or use in-place operations cautiously | The affected tensor is known not to be cached | In-place mutation of a cached tensor is rejected by strict selective activation checkpointing checks. |

### Device-idle or optimizer bottlenecks

| Technique | Applicable scenario | Counterexample or constraint |
|---|---|---|
| Optimizer-state offload | Optimizer states dominate device memory | Include Host transfer costs during the update phase. |
| Improve scheduling or Host dispatch | High device-idle (`Free`) share or launch gaps | Use profiling to distinguish Host dispatch delays from waits. |

## Automated tuning

Use the `autoresearch` skill for bounded experiments when the candidate space requires
systematic exploration. Declare the target files, benchmark command, and correctness
gate. Record each candidate as `keep`, `discard`, or `crash` with an audit trail.
This is suitable for operator or postprocessing changes with a bounded code surface,
a deterministic gate, and a machine-readable metric.

## Experiment discipline

- Change one technique at a time and use consecutive runs on the same node.
  A/B comparisons across time periods or resource pools are invalid.
- Check the mechanism before implementation. Estimate the targeted bottleneck's
  fraction of step time and the maximum possible gain. Investigate the measurement
  when the claimed improvement is inconsistent with this bound.
- Check negative results carefully. Confirm the candidate path actually executed
  using positive evidence before concluding that the technique is ineffective.
