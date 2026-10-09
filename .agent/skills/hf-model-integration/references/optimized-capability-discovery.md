# High-Performance Capability Discovery and Reuse Design

Complete this stage after authoritative-model inventory and before scaffolding or
implementation. Its purpose is to prevent duplicate kernels, family-private copies
of generic communication, and unverified claims that a replacement is faster.

## Discover Existing Capabilities

Start from the authoritative module's mathematics, not its class name. List the
attention layout, normalization, position encoding, recurrent/shared state, router,
expert activation, modality path, supported dtype, and target accelerator. Then
search at least these repository surfaces:

```bash
rg --files hyper_parallel/components/modules
rg --files hyper_parallel/distributed/context_parallel
rg --files hyper_parallel/distributed/expert_parallel
rg --files hyper_parallel/models | \
  rg 'adapter/(conversion|distributed|policies)|registration.py'
rg -n 'replacements=|context_parallel=|expert_parallel=' \
  hyper_parallel/models/*/adapter/registration.py
```

Inspect implementations and tests, not filenames alone. A GQA, MLA, recurrent
attention, routed MoE, or convolution with a similar name may have incompatible
parameter packing, mask convention, state lifetime, activation, or checkpoint
identity. Record exact backend requirements and whether a missing optional kernel
causes an explicit failure or a fallback.

Search in this order:

1. a generic component with the same mathematical and state contract;
2. generic collectives, routing, packing, or dispatch primitives that permit a
   model-owned thin wrapper;
3. an existing family adapter demonstrating the same extension contract;
4. a family-private implementation whose reusable part should be extracted into a
   model-independent component;
5. a new generic component or, only for irreducible family semantics, a new
   model-owned implementation.

Do not import another family's private adapter to create accidental ownership.
Use it as a reference and extract genuine commonality. Do not generalize merely
because two modules have similar shapes; the shared API must be name-independent
and support both contracts without family branches.

## Produce the Capability Inventory

Write `analysis/optimized_capability_inventory.json` beside the integration
evidence. Include one row for every performance-relevant final module and CP/EP
path with these fields:

- final FQN pattern and authoritative source file/class/function;
- mathematical signature: inputs, outputs, parameter/state layout, dtype, mask,
  routing, and backward requirements;
- target hardware/backend and optional kernel dependencies;
- existing generic component candidates and existing adapter references;
- compatibility result for forward, backward, state dict, materialization,
  TP/CP/EP/FSDP ownership, and checkpoint conversion;
- decision: `direct_reuse`, `thin_wrapper`, `extract_generic`, `new_generic`,
  `model_owned`, or `unsupported`;
- rejected alternatives and concrete incompatibilities;
- expected benefit: compute, communication, overlap, memory, or scalability;
- fallback behavior and how runtime evidence will detect it;
- linked precision and performance experiment IDs.

An empty candidate list is not sufficient evidence that nothing exists: include
the searched paths and symbols. `unsupported` requires an observed limitation and
an issue or remediation plan.

## Design the Reuse Boundary

Prefer direct construction when the optimized module preserves official FQNs,
aliases, initialization, materialization, checkpoint, and planner contracts. Use
replacement plus explicit weight transforms when it does not. Keep the model
adapter thin:

- generic kernels, collectives, token permutation, expert dispatch, and overlap
  machinery live in generic components;
- family-specific parameter mapping, router interpretation, shared-state wiring,
  mask adaptation, and combine semantics live in the adapter;
- recipe `plan_overrides` selects the implementation and topology;
- the reference path remains selectable for paired experiments and is removed
  only if an independent authoritative oracle remains available.

For every direct reuse, prove that the candidate contract matches rather than
assuming an existing module is correct. For every extraction or new generic API,
test provider-present and provider-absent behavior and exercise the abstraction
with the motivating model plus a synthetic or second existing consumer when
practical.

## CP Discovery and Design

Compare applicable algorithms such as local-query KV AllGather, Ulysses
sequence-to-head AllToAll, ring/P2P recurrent state, and hybrid decompositions.
Record:

- Q and KV head divisibility, sequence and head dimensions, and TP compatibility;
- communicated tensors, volume, collective order, differentiability, and layout;
- causal/global/packed mask convention, position offsets, and output restoration;
- asynchronous launch points, the first consumer that waits, and intended compute
  overlap;
- cross-layer state ownership and whether communication can occur without replay;
- backend/kernel requirements and fallback behavior.

The CP wrapper must reuse the selected optimized attention mathematics. If an
existing family implementation contains a reusable collective schedule, extract
the schedule or primitive; do not copy its attention formula into the new adapter.

## EP Discovery and Design

Compare existing grouped-expert modules, routing adapters, token permutation,
AllToAll, local-expert binding, grouped GEMM, and combine primitives. Record:

- learned-router output contract, score transform, top-k, capacity, and aux loss;
- expert parameter packing, activation/gate semantics, shared experts, and latent
  projections;
- local expert ownership and optimizer-state reduction across EP/FSDP;
- dispatch/combine ordering, empty experts, load imbalance, and backward path;
- asynchronous communication or grouped-compute opportunities;
- why a fused/grouped kernel is valid or invalid for the model's exact activation.

EP degree one and greater than one must use the same expert mathematics. An EP1 to
EPn throughput comparison measures scaling; it does not by itself prove that one
EP implementation is faster than another at the same topology.

## Write the Optimization Plan

Write `analysis/optimization_plan.yaml` before implementation. For every inventory
row, identify:

- correctness reference and optimized candidate;
- how identical weights, inputs, dtype, state, and output/gradient observations
  will be established;
- module microbenchmark shape and target accelerator;
- paired end-to-end reference/optimized recipe or override at the same topology;
- CP/EP same-topology implementation A/B and any separate scaling experiment;
- warm-up and measured-step policy, synchronization boundaries, and repeated-run
  policy;
- predeclared precision tolerances and performance thresholds;
- required metrics and profiler/communication-overlap evidence;
- scope limitation for a crop, unavailable kernel, or unsupported topology.

Do not invent a universal speedup threshold. Use a user/project target or justify
the threshold from the replacement's purpose. If neither exists, run an explicitly
labeled baseline pilot to estimate the run-to-run envelope, freeze the formal plan,
and require the candidate improvement to exceed that envelope without an
unexplained p90 or memory regression. Do not choose a threshold after seeing the
formal candidate result. If no stable same-topology baseline can be constructed,
mark performance acceptance `INCONCLUSIVE`; correctness can still be evaluated
independently.
