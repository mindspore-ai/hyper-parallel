# Parallel-Strategy Cost Model Reference

Use analytical estimates to reject infeasible layouts and shortlist
experiments. Accept a strategy only after a complete benchmark under the
declared workload and hardware configuration.

## 1. Implementation and scope

Executable optimization requires `--psa-config` and calls
`hyper_parallel.auto_parallel.config_adapter.search_strategies`.
The controller then checks the returned layout against the Trainer's
parallel domains and adds a small neighborhood of valid alternatives.
Without a PSA configuration, neighborhood generation is planning-only; it
does not constitute an ND search.

The model described here separates computation, communication, memory,
and pipeline scheduling. Exact model support, calibration, and ranking
confidence come from the configured estimator and its current records.
For the ND-backed workflow, follow
[`sapp-parallel-strategy`](../../sapp-parallel-strategy/SKILL.md) and
[its validation status](../../sapp-parallel-strategy/references/status.md).

## 2. Communication estimates

A latency-bandwidth model has the general form:

```text
communication_time ≈ transferred_bytes / effective_bandwidth
                     + communication_rounds × latency
```

Transferred bytes and rounds depend on the collective algorithm, group
size, topology, dtype, and tensor layout. Nominal link bandwidth alone does
not predict exposed communication time.

| Parallel mechanism | Principal communication | Inputs needed by the estimate |
|---|---|---|
| DP | Gradient AllReduce | Gradient bytes, group size, collective algorithm, overlap with backward |
| FSDP / HSDP | Parameter AllGather and gradient ReduceScatter, plus replica synchronization where configured | Shard and replica groups, parameter buckets, prefetch, reshard policy |
| TP | Layer-level AllReduce, AllGather, or ReduceScatter | Sharded activation shapes, implementation, forward/backward call counts |
| EP | Token dispatch and combine, plus any required weight/gradient synchronization | Actual top-k assignments, payload width and dtype, expert distribution, topology |
| PP | Point-to-point activation and gradient transfers | Boundary shapes, microbatch count, stage placement, schedule |
| CP | Attention-specific exchange | Attention type, local sequence length, KV representation, selected CP algorithm |

For example, transmitting 60 GB at an effective 100 GB/s takes 0.6 seconds
before latency. If a model accounts for 120 GB of aggregate send/receive
traffic, the corresponding bandwidth term is 1.2 seconds. State which
traffic definition the estimate uses.

EP is not universally cheaper than TP. Top-k, routing imbalance, message
size, and the fraction of MoE layers determine the comparison. Likewise,
a fixed sequence-length threshold cannot establish whether CP is useful.

## 3. Computation estimates

Count the operations implied by the actual architecture and local tensor
shapes. A dense matrix product with dimensions `m × k` and `k × n`
requires approximately `2mnk` FLOPs. Sum the required projections,
attention products, selected expert products, and backward operations.

For a conventional dense attention layer, attention QK/PV products add
sequence-dependent work. Sparse attention, MLA, compressed attention, and
other specialized architectures require their own shape-aware accounting.

For MoE, count the token rows assigned to each **selected** expert.
Using the total number of resident experts as though every token executes
every expert overstates useful computation. Expert parallelism changes
where that work executes; routing imbalance changes the slowest rank.

An idealized compute-time lower bound divides work by aggregate device
peak. A practical estimate also needs observed kernel efficiency, launch
overhead, and critical-path scheduling. Linear speedup from TP, DP, or EP
is a limiting case, not a guaranteed efficiency.

For utilization reporting, use the declared model-FLOP convention and
framework telemetry. MFU excludes additional execution work such as
recomputation; HFU includes only the executed-work terms actually supplied
by the measurement. See
[MFU targets and accounting](../../perf-playbook/references/mfu-targets.md).

## 4. Pipeline scheduling

For a simple non-interleaved pipeline with `p` stages and `m`
microbatches, a common idealized bubble fraction is:

```text
bubble_fraction ≈ (p - 1) / (m + p - 1)
```

With two stages and eight microbatches, this is `1/9`, approximately
11%. This approximation assumes balanced stage service times and the
corresponding schedule. Interleaving, uneven stage work, P2P delays, and
activation-memory limits require more detailed accounting.

Increasing microbatch count can reduce bubbles while increasing memory
pressure or communication overhead. Validate both effects.

## 5. Memory estimates

Estimate the peak set of simultaneously resident allocations:

```text
peak_memory ≈ parameter_shards + materialized_parameters + gradients
              + optimizer_states + live_activations + communication_buffers
              + kernel_workspaces + allocator_reserve
```

Use each allocation's actual dtype and lifetime. BF16 parameters do not
imply BF16 gradients, FP32 main weights, or optimizer states; those depend
on the recipe and implementation.

A 30-billion-element BF16 parameter set occupies 60 GB before sharding.
Even partitioning across two TP ranks gives 30 GB per rank for those
parameters, and sharding across four ranks gives 15 GB. These are parameter
storage examples, not training-memory totals. Replicas, temporary
AllGather buffers, optimizer state, and activations must be added according
to their actual layouts.

Activation memory depends on architecture, microbatch size, sequence
length, checkpointing, fusion, and tensor lifetimes. A tensor of
`batch × sequence × hidden` elements occupies that element count times
its dtype size; multiplying it by layer count is only a coarse inventory,
not a peak-memory prediction.

Measure allocated and reserved peaks. Do not replace allocator and
workspace accounting with a universal fixed allowance.

## 6. Layout constraints

The executable controller's `_fit_to_devices` checks the Trainer's
domains as follows:

```text
world_size = DP × TP × CP × PP
stage_size = world_size / PP
EP divides stage_size
dense_shard_domain = DP × CP
dp_shard divides dense_shard_domain
edp_shard divides stage_size / EP
```

EP overlaps the stage's rank domain; it is **not** another independent
factor in the world-size product. Degrees and shard widths must be
positive integers. Model-specific expert-count divisibility and operator
support must also be validated.

Neither EP >= 2 nor DP >= 2 is a universal requirement. A particular
recipe can impose additional constraints, which should be recorded with
that recipe. The current executable strategy scheduler supports one node;
its PSA cluster size must match `--devices`.

Treat the following as hypotheses for screening, not hard rules:

- Small local workloads can lose performance from TP or PP overhead.
- CP value depends on the attention implementation and local sequence work.
- Large models may need combinations of sharding, recomputation, offload,
  TP, or PP to fit; parameter count alone does not select the degrees.

## 7. Calibration and error analysis

Sources of prediction error include effective network bandwidth,
communication overlap, routing imbalance, kernel-shape efficiency,
compilation, allocator fragmentation, and incomplete modeling of
specialized layers.

ND can fit component ratios from measured profiles and use them to convert
scores to estimated time. Calibration is valid only for the model family,
hardware, software, and operating region for which it was tested. Outside
that region, retain an explicit unvalidated status and use estimates only
to select measurements.

The earlier Qwen3-30B-A3B note recorded the following comparison:

| Candidate | Estimated tokens/s | Reported measured tokens/s | Relative difference |
|---|---|---|---|
| 1 | 3,800 | 4,274 | +12% |
| 2 | 3,500 | 4,486 | +28% |
| 3 | 3,200 | 1,081 | -66% |

These historical figures are not accompanied here by the workload
manifest, command, commit, or raw logs. They illustrate that both gain
magnitude and ordering can be wrong; they are not a current calibration
record or an acceptance threshold. In particular, candidates 1 and 2 are
reversed by measurement.

## 8. Search and measured validation

```text
Model/configuration checks
  -> valid layouts and analytical estimates
  -> small candidate set with explicit uncertainty
  -> complete hardware benchmarks
  -> numerical and memory checks
  -> measured selection and independent confirmation
```

Use the model to reject invalid configurations and prioritize a small
candidate set. Avoid treating an estimated MFU range, a fixed error band,
or a predicted winner as a measured result. Record failed and incomplete
runs; neither can be exported as the winner.

Further improvements can include calibrated kernel-efficiency tables,
topology measurements, overlap models, and additional measured layouts.
Any learned predictor or runtime-adaptive search is a separate extension
and must not be described as already implemented by this controller.

## References

- [Configuration templates](config-templates.md)
- [Decision rules](decision-rules.md)
- [Validation metrics](validation-metrics.md)
- [Parallel-strategy analyzer](../../parallel-strategy-analyzer/SKILL.md)
- [ND-backed strategy workflow](../../sapp-parallel-strategy/SKILL.md)

Historical note: the previous reference was dated 2026-10-21 and described
a Qwen3-30B-A3B experiment on eight Ascend 910B devices.
