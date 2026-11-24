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

For attention launch fan-out across many independent short windows, batch
equal-sized windows into one SDPA invocation. Confirm that each window keeps
its original positions, causal policy, and output order. Validate outputs,
input and weight gradients, and heterogeneous fallbacks. Use explicit
sequence boundaries with a verified variable-length kernel for unequal
windows. Measure end-to-end improvement after verifying the launch reduction.

For a hot square causal mask, compare explicit masking with the backend's
implicit causal dispatch before replacing the kernel. Verify the actual
shapes, scale, forward and backward paths. The plain implicit option requires
one complete unpadded sequence per row and full causal attention, with a
verified backend contract. Preserve padding, segment boundaries and local
attention windows through a compatible mask or variable-length representation.
Use a full numerical gate and an unprofiled same-workload repeat for retention.

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

If the peak contains full-vocabulary logits or FP32 cross-entropy buffers,
test the model family's supported chunked linear cross-entropy path. Preserve
pre-shifted targets, ignored positions and global token weighting. Check the
loss and gradients of hidden states and head weights at representative shapes.
Log the peak allocation and complete-step throughput at fixed work first.
Test reduced rematerialization at fixed work. Use a separate acceptance
series when the reclaimed capacity enables a larger workload.
If the peak comes from FP32 cross-entropy intermediates, first keep the full
vocabulary projection and test token-wise blocks of the existing FP32 loss.
Return the complete logits gradient to the original head backward so that the
weight-gradient GEMM keeps its accumulation order. A fused loss kernel is
another candidate. Test it with the actual logits shape and dtype, including
representative low-loss targets, ignored positions, and the scaled upstream
gradient. Hold target alignment, reduction and token normalization fixed.
Compare gradients after the cast to the logits dtype and record peak workspace
and separate forward/backward timings. Close scalar loss alone can hide
gradient differences after rounding.

Before testing a smaller loss tile, calculate which tensors are live at the
same time from the local shard shape and storage dtypes. Include the complete
logits gradient returned to the head and any layout or collective buffers
that overlap the loss tile. Compare that estimate with capacity after optimizer
state initialization, then measure full-step memory on the target runtime. Compare
the source and candidate on the same inputs and run an independent performance
benchmark. Record workspace savings separately from step speed. Exercising a
longer context establishes capacity for that workload and requires its own
qualification before use as an accepted result.

After a memory-saving head change, test reduced replay at the same physical
microbatch shape. The allocation peak can move from the output head to saved
decoder activations. If the new replay policy exceeds capacity, record the
first failing operator, phase, request size, and allocated and reserved
memory. Keep the last validated policy and use that evidence to choose the
next allocation to investigate.

### Device-idle or optimizer bottlenecks

| Technique | Applicable scenario | Counterexample or constraint |
|---|---|---|
| Optimizer-state offload | Optimizer states dominate device memory | Include Host transfer costs during the update phase. |
| Improve scheduling or Host dispatch | High device-idle (`Free`) share or launch gaps | Use profiling to distinguish Host dispatch delays from waits. |

### Input-pipeline stalls

If stalls align with batch acquisition, measure exposed wait at the trainer's
consumer boundary by update and rank. Check whether CPU decoding, tokenization,
or collation limits the producer before increasing data-loader worker count.
Compare worker count and prefetch depth at fixed model, packing, and workload.
Track Host memory, multiprocessing overhead, and the slowest consumer ranks.
Check the ordered input tensors, labels, and packed geometry at the trainer
boundary. Confirm throughput on matching unprofiled windows. On two equivalent
nodes, run both variants on each node and reverse their order across nodes to
check whether the result survives node and run-order effects.

Verify that the model-specific training loop actually uses the configured
prefetcher. Time iterator wait separately from position construction and device
transfer. When a rank-local loader reads ahead, checkpoint the consumed batch's
cursor and preserve the collective sequence of distributed loaders.

After prefetch reduces iterator wait, profile batch preparation on the training
thread. For packed multimodal inputs, repeated segment geometry can make
position construction a cache candidate. Check the pinned model implementation
for every input that affects positions, such as modality order, grid dimensions,
padding, and packed segment boundaries. Bound the cache, reuse only immutable
outputs, and use the original path for unsupported layouts. Compare cached
positions directly against recomputed tensors during the correctness gate.
Measure end-to-end benefit on representative input geometry and record the
cache hit rate when comparing workloads.

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
