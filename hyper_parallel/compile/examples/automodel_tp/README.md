# TP + FSDP Graph-Mode Demo

Combines **automodel TP sharding** with the **graph-mode FSDP pass**: automodel
shards the model along the TP axis and bakes TP collectives into the boundary
forwards; `GraphTrainer` then traces a joint fwd+bwd FX graph and runs
`FSDPPass` (FSDP on the dp sub-mesh) + `AutoOverlapPass`.

This is the "FSDP2Manager off, FSDPPass sees only the FX graph" path — the two
communication domains are orthogonal (FSDP on params/grads, TP on activations),
so `FSDPPass` is TP-agnostic.

## Quick start

```bash
# 4 cards (TP=2, DP/FSDP=2)
bash run.sh
# or directly
torchrun --nproc_per_node=4 train.py --config config.yaml
```

NPU (`hccl`) is used when available; otherwise falls back to `gloo` on CPU
(useful for a smoke test of the trace + comm-node insertion).

## Pipeline

```
automodel                                    GraphTrainer
─────────                                    ────────────
MeshContext.build_meshs            ──►      reuse mesh:
  (dp, cp, tp) + (fsdp_replicate,            only register dp sub-mesh
  fsdp_shard, tp)                             as "fsdp" group; tp group is
                                             already wired into the boundary
ShardingPlanner.plan(tp)                      forwards.
  → TP ShardingPlan
                                          trace_model_graph (make_fx)
apply_sharding_plan(production)            → joint fwd+bwd FX graph; params
  → params = plain TP shards                 are static inputs shaped as the
  → boundary forwards insert TP              TP shard
    collectives on activations
                                          FSDPPass
                                            → dim-0 shard each TP-sharded
                                              param again (FSDP axis ⊥ TP axis)
                                            → all_gather on param placeholders
                                              (recovers the TP shard)
                                            → reduce_scatter on grad outputs
                                          AutoOverlapPass
                                            → move wait_tensor for overlap
```

## Why it works (orthogonal comms)

| | FSDPPass | TP (boundary forwards) |
|---|---|---|
| acts on | parameters / gradients | activations |
| group | `"fsdp"` (dp sub-mesh) | `"tp"` group object (held by boundary) |
| op in graph | all_gather / reduce_scatter (functional) | dist.nn.functional / fc API |
| inserted by | FSDPPass (graph transform) | automodel (forward wrapping, baked in) |

`FSDPPass` does not know TP exists — it only sees parameter placeholders whose
shape is the TP shard; dim-0 chunk + all_gather recovers that TP shard, which is
exactly what the (boundary-wrapped) forward consumes.

## Code

```python
from hyper_parallel.compile import GraphTrainer, ParallelConfig, ShardingPlan
from hyper_parallel.auto_models.components.distributed.infrastructure import MeshContext
from hyper_parallel.auto_models.components.distributed.sharding_planner import ShardingPlanner
from hyper_parallel.auto_models.components.distributed.sharding_applier import apply_sharding_plan

# 1. automodel mesh + TP sharding (FSDP2Manager NOT used)
mesh_ctx = MeshContext(tp_size=2, dp_size=2, dp_shard_size=2, ...)
mesh_ctx.build_meshs(device_type, world_size)
plan = ShardingPlanner().plan(model, mesh_ctx.device_mesh, tp_size=2,
                              sequence_parallel=False, loss_parallel=False)
model, _ = apply_sharding_plan(model, plan, mesh_ctx, validate_mode=False)

# 2. GraphTrainer reuses the automodel mesh
fsdp_plan = ShardingPlan(); fsdp_plan.fsdp_wrap_pattern("*")
trainer = GraphTrainer(
    model=model, train_fn=train_fn,
    parallel_config=ParallelConfig(enable_overlap=True, fsdp_degree=2, tp_size=2),
    sharding_plan=fsdp_plan, mesh_context=mesh_ctx, device=device,
)
trainer.compile(sample_input, sample_label)   # inspect_graph shows both comm sets
trainer.train(data_iter, max_steps=10)
```

## Prerequisites

- `torch` with the joint-trace backward engine
  (`torch.compiler._patch_engine_backward`) — needed for `make_fx` to capture
  the backward half of the joint graph (TP + FSDP grad collectives).
- PR #1278 (functional-collective adaptation of
  `differentiable_all_gather_concat` / `differentiable_all_reduce`) merged, so
  the boundary TP collectives trace into the FX graph cleanly.
- `transformers` (for the tiny Llama config — no download).

## Limitations

- `reduce_scatter` on the TP forward path (loss_parallel) is not yet adapted to
  the functional-collective API — keep `loss_parallel: false`.
- FSDPPass shards along dim 0; every TP-sharded param's dim 0 must be divisible
  by `dp_size` (the demo's tiny Llama is sized so `hidden / (tp*dp)` and
  `intermediate / tp` are integers).
- Gradient scaling convention: both TP `all_reduce` and FSDP `reduce_scatter`
  use plain `sum`; the optimizer LR must account for `tp*dp` (no
  `1/(dp*cp*tp_loss_replica)` bookkeeping as FSDP2Manager does).
