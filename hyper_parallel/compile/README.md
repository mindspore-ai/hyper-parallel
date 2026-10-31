# HyperParallel Graph Mode

Graph-mode architecture for automatic parallelization with FSDP.

> **Note**: The `compile/` subpackage relies on a patched autograd engine for
> joint-graph capture (see Limitations). Its `torch.*` imports — including
> `torch.fx.experimental` and `torch._guards` — are therefore intentional and
> carry a `forbidden-backend-import` suppression in
> `.jenkins/check/config/filter_pylint.txt`.

## Core Concept

**User**: Write model code + parallel configuration
**Framework**: Graph capture → DP/PP partitioning → Communication-compute overlap → Execution

## Architecture

### Layer 1: Configuration

```python
from hyper_parallel.compile import GraphParallelPlan, PassConfig

# Configure which modules to mark for FSDP
parallel_plan = GraphParallelPlan()
parallel_plan.fsdp_mark("tok_embeddings")
parallel_plan.fsdp_mark_pattern("layers.*")

# Parallel configuration
pass_config = PassConfig(enable_overlap=True)
```

### Layer 2: Graph Capture

`trace_model_graph` captures forward + backward into a joint FX graph:

- Parameters/buffers are static inputs (placeholders), not `get_attr` nodes
- `torch.autograd.grad` runs inside the traced function
- Uses `FakeTensorMode` + `make_fx` for symbolic tracing

### Layer 3: Pass Pipeline

```text
DeadCodeElimination → CanonicalizeGraph → FSDPPass
    → AutoOverlapPass → FsdpOverlapPass → [PP: PpPass] → [Inductor: InductorPass]
```

`PpPass` runs last among the graph-level passes so every pass still sees the
joint graph; it then replaces it with the schedule stub. `InductorPass`
(when `inductor_mode` is enabled) is TERMINAL: it lowers the graph to
TorchInductor kernels, after which the FX graph wraps the compiled callable
and no further graph-level pass may run. With PP enabled it compiles the
schedule's per-stage forward/backward subgraphs instead of the stub (see
below).

**FSDPPass** (data-parallel, `PassConfig.dp_mode`):

- Identifies DP parameter placeholders via GraphParallelPlan
- Sinks `all_gather` to each parameter's first forward use (Shard → Replicate)
- Frees the replicated parameter after its last forward read, then re-gathers
  it for the backward and rematerializes any saved forward view of it
  (`reshard_after_forward`), so peak memory tracks the forward working set
- Inserts gradient reduction on gradient outputs — `reduce_scatter`
  (Replicate → Shard) for `"fsdp"` / `"hsdp"`, `all_reduce` on the
  `dp_replicate` axis for `"ddp"` / `"hsdp"`
- Physically shards live model parameters (dim 0) so optimizer is FSDP-agnostic
  (`"ddp"` replicates instead; a non-divisible dim-0 param is pad-sharded like
  eager `fully_shard` — dim 0 zero-padded to a multiple of the degree — rather
  than erroring; a 0-dim scalar stays replicated and is `all_reduce`d)
- Falls back to DDP for a trainable parameter a partial plan leaves unmarked
  (kept replicated, gradient all-reduced over the DP group) and logs the
  fallback

Disable reshard with `PassConfig(fsdp_reshard_after_forward=False)`.

**PpPass** (pipeline-parallel, `PassConfig.pp_enabled`):

- Splits the (post-FSDP) joint graph to this rank's stage(s) at module-FQN
  boundaries, exchanging boundary activations/gradients via P2P
- Installs a self-contained GPipe / 1F1B / Interleaved-1F1B (VPP) schedule
  (`pp_schedule`) as a `call_module` stub, so the trainer needs no PP wiring
- `pp_chunks > 1` enables VPP: the model is split into
  `pp_degree * pp_chunks` virtual stages, rank `r` owns
  `r, r + pp_degree, ...`, and `pp_schedule: vpp` (alias
  `interleaved_1f1b`) pipelines them with the classic interleaved-1F1B
  warmup / steady / cooldown order
- Pure PP: `fsdp_enabled=False`, `pp_degree == world_size`; the PP+FSDP /
  PP+DDP hybrid uses the 2-D `(pp, DP)` mesh built by the compiler, which
  registers the intra-stage DP group (`"fsdp"` / `"dp_replicate"`) and the
  cross-stage `"pp"` group — `FSDPPass` shards parameters over the
  intra-stage group (its collectives for foreign stages are never copied
  into the stage slices) and `PpPass` pipelines each DP replica with the
  same DP replica of the neighbouring stages. PP+HSDP and PP composed with
  an external `MeshContext` are not supported yet.

**AutoOverlapPass**:

- **Placeholder no-op today**: `enable_overlap` is accepted, but the pass
  does not yet reorder `wait_tensor` nodes. Sinking waits past independent
  compute is planned.

**InductorPass** (terminal backend compilation, `PassConfig.inductor_mode`):

- `"regional"` compiles tagged compute regions with TorchInductor (kernel
  fusion, Inductor caching) while keeping `_c10d_functional` collectives,
  their waits, and side-effectful nodes (the FSDP reshard storage free) as
  eager FX nodes — the comm scheduling surface the overlap passes built stays
  observable at runtime.
- `"full"` compiles the whole joint graph as one region (c10d functional
  collectives included, functionalized by Inductor). It always sets
  `reorder_for_peak_memory=True` so AOTAutograd keeps the fwd/bwd
  interleaving.
- Both modes force `torch._inductor.config.eager_numerics.division_rounding`,
  so compiled code is bitwise-equal to eager (unit-tested loss/gradient
  parity), and restore the positional calling convention that
  `run_traced_graph` relies on (`regional_inductor` otherwise switches the
  graph to boxed codegen).
- Nodes inserted by earlier passes carry no fake metadata; the pass
  re-propagates shapes with `FakeTensorProp` on the graph's own
  `FakeTensorMode` before compiling.
- Requires `example_inputs` (the tracer's fake inputs) for the non-PP joint
  graph. With active PP, `PpPass` first replaces the joint graph with the
  schedule stub; the pass then compiles every virtual stage's forward and
  backward subgraph independently (both `regional` and `full`), deriving each
  subgraph's fake inputs from its own placeholders. The stub stays the outer
  entry point and the schedule's P2P remains outside the compiled regions —
  cross-stage fusion is impossible by construction. Stage forward subgraphs
  can export multi-result ops (e.g. `nll_loss_forward`) as saved values; their
  nested output is flattened to `getitem` leaves before compilation and
  rebuilt by a wrapper, since AOTAutograd's regional path requires flat
  outputs.
  `inductor_configs` (a `PassConfig` field) forwards
  `torch._inductor.config` overrides to the compiled regions.

Inductor settings live on `PassConfig` (like the PP settings), not in the
YAML-facing `compile:` section:

```python
pass_config = PassConfig(
    inductor_mode="regional",                  # or "full"
    inductor_configs={"reorder_for_peak_memory": True},  # optional overrides
)
```

**FsdpOverlapPass** (independent FSDP graph controls):

- `fsdp_allgather_prefetch=True` prefetches tagged parameter all-gathers in
  forward/backward and delays their wait/view DAGs.
- `fsdp_overlap_grad_reduce=True` independently advances tagged gradient
  reduce-scatter launches and delays their wait/view DAGs.
- Both switches default to `True`, independently of the legacy `enable_overlap`
  flag. The legacy flag keeps selecting `AutoOverlapPass`.
- `PpPass` runs last, so this pass always sees the FSDP collectives before
  the split; after the split they live inside the schedule's per-stage
  submodules where a graph-level pass could no longer reach them. The
  reordering is value-level and dataflow-preserving, so the stage slices
  inherit it unchanged.
- `max_in_flight_gb=0.25` retains the previous 256 MiB speculative AG limit.
  The name and binary-unit conversion follow PyTorch's overlap scheduler;
  this pass accounts for early AG replicas until first use in each phase,
  not total device memory or gradient communication buffers. A zero limit
  disables early AG launches while retaining AG wait/view sinking.
- Consumer-group lookahead and collective reordering are internal policies.
  Input, effect, phase,
  and parameter storage-release constraints still apply. Other collective
  launch nodes retain their relative order; participating ranks must execute
  compatible static graphs and configuration.
- Eager `fsdp_config.forward_prefetch_depth` / `backward_prefetch_depth` still
  count module prefetch targets. They no longer configure graph consumer groups.

Trainer configuration:

```yaml
compile:
  enabled: true
  use_joint_graph: true
  fsdp_allgather_prefetch: true
  fsdp_overlap_grad_reduce: true
  max_in_flight_gb: 0.25
```

To disable both FSDP scheduling passes, set both FSDP switches to `false`.
Migrate `compile.enable_overlap` to the independent switches above and convert
`fsdp_prefetch_budget_bytes` to `max_in_flight_gb` by dividing by `1024**3`.
The graph-only depth and collective-reordering fields have been removed from
YAML and `PassConfig`. Lookahead depths remain mutable variables inside
`passes/overlap/overlap_param_gather.py`, both defaulting to `1`. Change their
source values or patch the module before compilation:

```python
from hyper_parallel.compile.passes.overlap import overlap_param_gather as prefetch_pass

prefetch_pass.fsdp_forward_prefetch_depth = 5
prefetch_pass.fsdp_backward_prefetch_depth = 2
```

Use identical values on every participating rank. Patching affects subsequent
pass runs, not graphs already compiled. Depth counts preceding parameter-consumer
groups, not eager modules; attention-interval anchors may select earlier compute.
Zero disables early all-gathers for that phase, including attention-interval
prefetch, while retaining wait/view sinking. All depths remain subject to the
same memory budget and dependency constraints.

### Layer 4: Execution

```python
from hyper_parallel.compile import GraphTrainer

trainer = GraphTrainer(
    model=model,
    train_fn=train_fn,
    pass_config=pass_config,
    parallel_plan=parallel_plan,
)

# Compile on first batch, then run forward + backward + optimizer
trainer.train(dataloader, max_steps=100, log_interval=10)
```

The compiled graph is a `GraphModule` that:

- Takes sharded parameters as inputs
- Gathers them via AllGather at each step
- Computes forward + backward
- Scatters gradients via ReduceScatter
- Outputs loss + grads

## Usage

```python
from hyper_parallel.compile import GraphParallelPlan, GraphTrainer, PassConfig

# 1. Model
model = Llama3Model(config)

# 2. Parallel plan
parallel_plan = GraphParallelPlan()
parallel_plan.fsdp_mark_pattern("layers.*")

# 3. Trainer
trainer = GraphTrainer(
    model=model,
    train_fn=lambda m, x, y: m(x).loss(y),
    pass_config=PassConfig(enable_overlap=True),
    parallel_plan=parallel_plan,
)

# 4. Training
trainer.train(dataloader, max_steps=1000)
```

## Dynamic input shapes

Static tracing remains the default. Enable symbolic user inputs with one option:

```python
compiler = GraphCompiler(model=model, train_fn=train_fn, dynamic=True)
```

For precise control, pass `dynamic_arg_dims` to `GraphCompiler` or `GraphTrainer`:

```python
compiler = GraphCompiler(
    model=model,
    train_fn=lambda m, x, y: ((m(x) - y) ** 2).mean(),
    dynamic_arg_dims={"x": [0, 1], "y": [0, 1]},
)
# x: [batch, sequence, input_features]; y: [batch, sequence, output_features]
loss, loss_dict = compiler.forward_backward(x=x, y=y)
```

The first call captures one symbolic forward/backward graph. Later batches with
compatible batch/sequence lengths reuse it; gradients still accumulate into the
live parameters. `GraphTrainer` exposes the same options and owns optimizer steps.
Numerical and graph-reuse regressions live in
[the CPU unit tests](../../tests/ut/compile/test_dynamic_shapes.py).

For the existing **BaseTrainer + GraphCompiler** path, configure:

```yaml
compile:
  enabled: true
  use_joint_graph: true
  dynamic_arg_dims:
    model_inputs.input_ids: [0, 1]
    labels: [0, 1]
```

Add paths for other tensors whose sizes vary, such as
`model_inputs.attention_mask: [0, 1]`. Paths refer to the keyword arguments of
`train_fn`; BaseTrainer supplies `model_inputs` and `labels`. Use `dynamic: true`
instead of the mapping to automatically symbolize all user tensor dimensions.

- `dynamic_arg_dims` overrides `dynamic`; unlisted dimensions stay static.
  Nested input paths and negative dimension indices are supported.
- Model parameters and buffers keep static shapes. Runtime guards reject
  incompatible shapes, metadata and Python constants without recompiling.
- Use a representative first batch with dynamic dimensions **greater than 1**;
  sizes 0/1 and shape-dependent Python branches may specialize the graph.
- Dynamic shapes support joint forward/backward graphs, gradient accumulation
  and optimizer updates. Pipeline parallel and data-dependent output shapes
  are not supported.

The approach follows MagiCompiler's separation of dynamic and static dimensions
and reuse of a general symbolic graph, adapted to HyperParallel's joint-graph
tracer. It adds no runtime dependency on the separate MagiCompiler checkout.

## Key Design Decisions

1. **Static Inputs**: Parameters are graph inputs, not `get_attr`. This allows passes to split the graph by reshaping placeholders.

2. **Joint Graph**: Forward + backward captured together via `torch.autograd.grad` inside the traced function.

3. **FSDP-Agnostic Optimizer**: FSDPPass shards the live model's parameters in place. `model.parameters()` returns shards, so optimizer needs no FSDP awareness.

4. **Declarative Sharding**: GraphParallelPlan uses FQN patterns (`layers.*`) instead of imperative module wrapping.

## Parallelism support

| Axis | Status | Notes |
|------|--------|-------|
| FSDP / DDP / HSDP | Implemented | `FSDPPass`; `PassConfig.dp_mode` selects the mode |
| PP | Implemented | `PpPass` + `pp_schedule` (`gpipe` / `1f1b` / `interleaved_1f1b` / `vpp`), including VPP via `pp_chunks > 1`; pure PP and the PP+FSDP / PP+DDP hybrid (2-D `(pp, DP)` mesh) |
| TP (+ SP / LP) | Via `mesh_context` | TP collectives are baked into automodel boundary forwards; graph mode reuses the mesh |
| PP + HSDP / PP + TP | Planned | HSDP needs a 3-D mesh; TP-in-stage needs per-stage TP collectives in the graph |
| EP | Planned | — |

## Limitations

- **Non-divisible parameters are pad-sharded, not errors**: a parameter whose
  dim 0 is not divisible by the DP shard degree is zero-padded to a multiple of
  the degree and sharded like any other (mirrors eager `fully_shard`): the graph
  narrows the gathered padded full tensor back to the real size, and the
  backward pads the gradient back before `reduce_scatter`. A 0-dim scalar cannot
  be padded (eager `fully_shard` rejects scalars too); it stays replicated and
  its gradient is `all_reduce`d over the shard axis (and the `dp_replicate` axis
  for HSDP), so replicas stay synchronized instead of silently diverging.
- **Partial plans fall back to DDP**: with an explicit `GraphParallelPlan` that
  leaves some trainable parameters unmarked, those parameters are not
  FSDP-sharded; they stay replicated and their gradient is `all_reduce`d over
  the DP group (DDP semantics, so replicas stay in sync). `FSDPPass` logs the
  fallback naming them.
- **PP pure / hybrid split**: pure PP runs with `fsdp_enabled=False` and
  `pp_degree == world_size`. The PP+FSDP / PP+DDP hybrid is supported
  through the 2-D `(pp, DP)` mesh: set `fsdp_enabled=True`, `pp_enabled=True`
  and `pp_degree` plus `fsdp_degree` (or `dp_replicate_degree` for DDP) with
  `pp_degree * dp_degree == world_size`; the compiler registers the
  intra-stage DP group and the cross-stage `"pp"` group. PP+HSDP (3-D mesh)
  and PP composed with an external `MeshContext` are not supported yet.
  Without a pp group (direct `PpPass` use), `fsdp_enabled=True` is rejected.
- **VPP stage plan**: with `pp_chunks > 1`, a manual `GraphParallelPlan`
  must declare `pp_degree * pp_chunks` exact module FQNs (one list per
  virtual stage); auto split distributes layers evenly across the virtual
  stages. The interleaved schedule ships boundary values between adjacent
  virtual stages, including the last-stage -> rank-0 wrap edge.
- **`AutoOverlapPass` is a placeholder no-op** — see the pass note above.
- **`InductorPass` composes with PP (and everything else)**: `regional` /
  `full` backend compilation runs after `PpPass` and compiles each virtual
  stage's forward/backward subgraph through the schedule's
  `map_stage_graphs` hook; the schedule stub itself is never compiled and its
  P2P stays eager. Both modes require a torch build that ships
  `torch.fx.passes.regional_inductor` (newer than 2.9; the supported 2.6 /
  2.7 / 2.9 builds do not). On older builds the pass logs a warning, leaves
  the graph untouched and records `inductor_compiled="unavailable"` instead of
  raising. Compiled regions are keyed to the traced static input shapes —
  exactly like stock
  `torch.compile` static tracing; dynamic-shape graphs keep working (Inductor
  compiles the symbolic graph with its own guards), but the combination is
  validated on CPU only, and `dynamic` plus PP remains rejected by
  `GraphCompiler` (independent of Inductor).
- **Compilation is not idempotent**: the partitioning passes mutate the live
  model in place, so `GraphCompiler.compile` may only be called once — a
  second call silently re-traces the already-partitioned model and yields an
  incorrect graph (it does not raise; guard the call site yourself).
- **Requires torch with a patched autograd engine** for joint-graph capture.
  On stock torch the backward half is annotated structurally instead (see
  `_annotate_autograd_backward` in `tracer/graph_tracer.py`).
