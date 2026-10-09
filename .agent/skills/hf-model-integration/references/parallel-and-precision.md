# Parallel and Precision Contracts

## Mandatory Precision Route

Use this resolved policy for onboarding examples and handoff validation:

```yaml
model_init_dtype: float32
optimizer:
  fp32_main_params: true
fsdp_config:
  mix_precision:
    param_dtype: bfloat16
    reduce_dtype: float32
    cast_forward_inputs: false
```

Leave `output_dtype` absent so model-owned output dtypes are preserved. In
particular, a model may produce BF16 activations and an FP32 scalar objective in
the same structured output.

Verify behavior, not only YAML text:

- model load/from-config finalization applies FP32 initialization dtype;
- Trainer wraps the optimizer with `Float16OptimizerWithFloat16Params`;
- trainable parameters expose FP32 `main_param` and `main_grad` handling;
- FSDP uses `apply_grad_on_fp32_main_grad=true` and FP32 reduction;
- forward inputs are not automatically cast by FSDP;
- resolved FSDP `output_dtype` is `None` and the scalar loss remains FP32;
- the optimizer updates main parameters and synchronizes model parameters as
  required by their dtype.

If a model parameter is FP32, its `main_param` legitimately aliases the model
parameter, so no additional master-weight tensor or mixed-precision checkpoint
leaf is required. If a model parameter is BF16/FP16, the optimizer must own a
distinct FP32 main parameter and persist it in the optimizer checkpoint.

## Tensor Parallel

Derive placements from the actual tensor semantics. For GQA, validate divisibility
of Q and KV heads independently. The head dimension remains replicated unless the
implementation explicitly proves another layout. A fused QKV transform must split
whole Q, K, and V groups rather than treating all output rows uniformly.

Pay particular attention to q/k norm: after column-wise Q/K projection, the norm
placement must follow the sharded head axis. A local-tensor Production run is not
proof that the declared DTensor placement is correct.

Declare semantic parameter roles even when TP size is one. The declaration also
drives planner validation, future TP sizes, FSDP interaction, and checkpoint
layout. `REPLICATED` is correct only when every TP rank needs the complete value;
do not use it merely because the first smoke uses TP1. For embedding tables,
experts, recurrent/shared scalars, and gates, justify TP/EP/FSDP ownership from the
single-rank formula and report per-rank parameter plus optimizer-state bytes.

If reducing FSDP shard count or enabling TP unexpectedly raises memory, inventory
every parameter by FQN, logical/local shape, TP/EP/FSDP placement, dtype, and
optimizer-state ownership. TP saves memory only for weights actually sharded on
that axis; replicated or excluded subtrees can dominate after FSDP changes.

Before any distributed launch, evaluate the final cropped configuration against
every topology in the validation manifest. Check head/group/KV divisibility,
projection split sizes, expert ownership, sequence-sharding constraints, and each
declared `tp_divide_attrs` value. Run a build-only or one-step smoke for the
highest-risk nontrivial topology before starting the full matrix. A TP1 baseline
cannot validate a TP rule whose divisor is inactive at TP1.

Derive the data-parallel batch divisor from the actual launcher world size and
`tp * cp * pp`; do not remove EP from or add EP to that divisor by intuition.
Require `global_batch_size` to be divisible by
`micro_batch_size * resolved_dp_world_size` for every recipe/topology pair, and
verify that the observed runtime rank count equals the launcher declaration.

## FSDP Uneven Shards and FP32 Main Gradients

Keep three shapes separate when analyzing hybrid parallelism:

1. the model parameter's logical global shape;
2. the TP/EP-local tensor presented to the FSDP axis;
3. the final rank-local FSDP shard shape.

An empty FSDP shard is valid when the source dimension is shorter than the FSDP
shard group. For example, a logical eight-element attention-sink vector under
TP1/FSDP16 has eight local `(1,)` shards and eight local `(0,)` shards while every
rank must still report logical global shape `(8,)`. Under TP2, apply the same
reasoning to the TP-local source dimension before FSDP. Validate that every rank
reports the same logical shape even when its local shard is empty.

`replicate_params` is the current way to keep selected parameters replicated but
FSDP-managed with synchronized gradients. Its Trainer configuration accepts exact
parameter FQNs, not globs or a numeric size threshold. Replicating tiny sinks,
scales, or gates may be a reasonable performance choice. Keep an uneven-shard
validation case whenever other parameters can still produce empty local shards.

When deciding which parameters to list, do not use total `numel` alone. Empty
shards depend on `source.size(shard_dim) < shard_world_size`; memory and
communication benefit depend on the whole tensor and optimizer state. Document
the exact FQNs and the reason each remains replicated.

For the FP32-main route, keep the DTensor layout end to end:

```text
FSDP model parameter
  -> FP32 DTensor main_param
  -> FP32 DTensor main_grad
  -> DTensor optimizer moments with the same global shape/mesh/placements
  -> standard DCP layout metadata
```

Select this route with `optimizer.fp32_main_params: true` and use HyperParallel's
distributed checkpointer. Model adapters do not transform gradients or optimizer
state. Verify local/global shape, mesh topology, placements, dtype, and device at
the parameter, gradient, and optimizer-state observation points. Treat
higher-order-gradient training as unsupported unless it is explicitly included in
the requested validation scope.

## Local-Region Boundary Usage

For a custom `local_compute_fn`, declare complete input, output, and parameter
placements in `plan_overrides`. The callable consumes and returns plain local
tensors in Production; `out_src` describes the logical output used for boundary
redistribution and Validate checks. Preserve the public output structure exactly,
including nested tuple/list results.

Pair every custom local region with a strict Validate case. Production verifies
the executable path; Validate verifies that the declared placements match the
same mathematics.

## Context Parallel

Apply CP after the optimized attention replacement. The CP adapter must wrap the
same optimized attention module and replace only the attention interface or
communication boundary; do not maintain a second attention implementation.

Validate implicit causal masks and externally supplied global masks. State the
model-facing boolean convention and perform any kernel-specific inversion exactly
once at the kernel boundary. Validate packed-document boundaries and offset-aware
causality.

For Ulysses, check that both Q heads and KV heads are divisible by CP size. A
`region_dispatch: false` custom communication region may hide internal DTensor
validation; never use it to conceal ordinary TP, q/k norm, RoPE, or mask placement
errors. Keep a strict CP=1 + TP Validate case that exposes attention internals.

For cross-layer shared attention, communication wrappers may exchange the tensors
stored in a per-forward shared state, but they must not replace the model's state
lifetime or silently replay a producer layer. Compare Full, Reindex, and Reuse (or
the model's equivalent roles) independently before combined CP.

## Expert Parallel

Use the same grouped-expert mathematical implementation with and without EP.
EP may dispatch tokens, execute local experts, and combine results, but it must
not change router logits, top-k selection, expert weights, scaling, or output
ordering.

Formal acceptance uses the learned router. Fixed or sequential routing is a
diagnostic tool only and must be labeled as such.

Do not assume a new gate function matches an older family. Compare router logits,
the exact score transform, top-k indices, normalized weights, auxiliary loss, and
logit gradients against the authoritative source. Probe composed operations on the
target accelerator; a dedicated fused operator is not required when supported
primitive operations preserve the reference mathematics.
