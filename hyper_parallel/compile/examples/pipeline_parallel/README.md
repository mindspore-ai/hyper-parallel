# Pipeline Parallel Example (Graph-Pass PP Split)

Pipeline-parallel training with HyperParallel graph mode. The joint
forward+backward FX graph is sliced into per-rank stages by `PpPass` —
**at graph level**, in contrast to torchtitan, which splits the model
itself (deepcopy + layer deletion) before wrapping chunks in
`PipelineStage`.

## Quick start

```bash
cd hyper_parallel/compile/examples/pipeline_parallel

# 2 PP stages on 2 cards (v1 contract: world_size == pp_degree)
bash run_2cards.sh

# or manually:
torchrun --nproc_per_node=2 train.py --config config.yaml
```

On an NPU node the process group uses hccl automatically; other platforms
fall back to gloo.

## How it works

```text
joint graph (fwd+bwd, params as static placeholders)
        │  FSDPPass (off here) / PpPass
        ▼
per-rank stage slices: fwd_gm + bwd_gm  ── boundary act/grad via isend/irecv
        │
        ▼
ScheduleGPipe installed as a call_module node in the graph stub
        │
        ▼
GraphTrainer runs graph_module(*state, input, label) unchanged
```

1. **Node → stage attribution** — every FX node is mapped to a stage via
   its `nn_module_stack` metadata; parameters (static placeholders) are
   attributed by FQN ancestor walk against the stage plan.
2. **Boundary detection** — the single activation crossing stage k → k+1
   (and the single backward gradient crossing back) is located.
3. **Graph surgery** — stage k's nodes are sliced into a forward subgraph
   and a backward subgraph; forward outputs (boundary activation + every
   intermediate the backward needs) become backward placeholders, so all
   forwards can run before all backwards (GPipe).
4. **Live-model pruning** — foreign-stage submodules are removed in place,
   so `model.parameters()` / the optimizer only see this rank's stage.
5. **Schedule install** — `ScheduleGPipe` (eager `dist.isend`/`irecv`,
   microbatched, grads averaged over microbatches) is installed as a
   `call_module` node, surviving recompiles; the trainer needs zero PP
   awareness.

## Stage split options

**Automatic (default)** — the first top-level `nn.ModuleList` with at
least `pp_degree` entries is the layer container; it is distributed evenly,
modules before it go to stage 0, modules after it to the last stage:

```yaml
pp: {}   # omit stages -> auto even-by-layers
```

**Manual** — exact module FQNs per stage (no wildcards):

```yaml
pp:
  stages:
    - [tok_embeddings, layers.0, layers.1]
    - [layers.2, layers.3, norm, lm_head]
```

Both come from `PassPlan.pp_stage(stage_idx, fqns)` /
`create_sharding_plan_from_yaml`.

## Configuration reference (`parallel:` section)

| Key | Default | Meaning |
|-----|---------|---------|
| `pp_enabled` | `false` | Opt in to the PP pass |
| `pp_degree` | `null` | `null` → `world_size`; v1 requires `pp_degree == world_size` |
| `pp_microbatch_size` | `1` | Samples per microbatch; batch must be divisible |
| `enable_overlap` | `true` | No-op under PP (the schedule overlaps at Python level) |

## Current constraints (v1)

- **Pure PP only**: `fsdp_enabled=False`. PP+FSDP hybrid requires a device
  mesh expressing both dims (`mesh_context` path); the 1-D fallback FSDP
  mesh cannot host PP and is rejected with a clear error.
- **Microbatch-shaped compile**: the trainer must `compile()` once with a
  microbatch-shaped sample (see `train.py`) — the traced graph bakes view
  sizes for one microbatch, and the schedule slices every full batch into
  matching microbatches at runtime.
- **Multi-value stage cuts**: every value crossing a cut ships over P2P
  (tensors as-is, int scalars packed as 0-d int64). Gradients flow back
  during the backward sweep.
- **GPipe schedule**: forward sweep then backward sweep. 1F1B / ZB-V land
  in `passes/parallel/pp_schedule.py` as additional schedule classes.
- Loss is only real on the last stage; other stages return a zero scalar
  (see `Last-stage loss:` in the run output for convergence).
