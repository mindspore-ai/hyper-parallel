# HyperParallel Dry-run Design

## Goal

Dry-run drives the unmodified training step — `TextTrainer.train_step`, the
real loss aggregation, gradient normalization, clipping, and optimizer
sequence — over FakeTensors. It reads the first real CPU batch for shape and
token statistics, then executes one logical training step without loading
model weights, allocating device memory, binding an accelerator, or creating
real communication domains.

The guiding principle is **identical by construction, not by imitation**:
dry-run owns only the environment (fake process group, FakeTensor
materialization, host-scalar resolution, device-effect suppression); the step
logic itself is the trainer's own code. Whatever the trainer's step loop does
— per-micro-step FSDP gradient-sync flags, reshard scheduling, aux-loss
binding, token-weighted global loss — dry-run inherits automatically.

The tracker records tensor storage lifetimes only. It does not classify tensors
as parameters, gradients, optimizer state, activations, forward, backward, or
recompute. The snapshot follows the `segments` and `device_traces` shape used by
PyTorch memory visualization.

## Package Layout

All dry-run implementation lives under `hyper_parallel/tools/dryrun`:

- `runner.py`: environment preparation and real-trainer orchestration.
- `runtime.py`: value replay, distributed emulation, host-scalar shim,
  batch replay iterator, and tensor collection.
- `memory.py`: logical storage tracker and snapshot writer.
- `compatibility.py`: scoped AutoModel/optimizer-subgroup/Muon/CSA2 patches.
- `operator_support.py`: built-in and user-defined FakeTensor kernels.
- `config.py`: YAML configuration dataclasses.
- `data.py`: one-shot CPU data probe.
- `examples/deepseek_v41.py`: DeepSeek-specific scoped replay handlers.

The reconciliation ST lives in `tests/torch/tools/test_dryrun_memory_sync.py`:
it runs one recipe through both the simulator and a real 8-card training step,
then asserts the logical peak stays inside a tolerance band of the real trace
peak and that every large real allocation call site appears in the dry-run
trace. A step-loop divergence surfaces there as a named `file:line` before it
is visible in the peak.

## Configuration Interface

```yaml
dry_run:
  enabled: true
  target_device: npu
  ranks: [0, 7]
  virtual_world_size: 8
  snapshot:
    enabled: true
    path: outputs/dryrun/deepseek_v41
    capture_stacks: true
    stack_depth: 64
```

`ranks` defaults to `[0]`. `virtual_world_size` defaults to 1 without torchrun.
`target_device` states the accelerator the plan is for and takes precedence
over environment detection, so capacity planning runs on hosts without that
accelerator — including hosts with no accelerator at all (simulation then
falls back to CPU FakeTensors and reports `simulation_device` in metadata;
the logical peak is identical to the accelerator-lying simulation).
`snapshot.path` is an output root, not a file template. Dry-run creates:

```text
<snapshot.path>/rank_<rank>/memory.pickle
```

No separate report or JSON file is generated. Run metadata is stored under the
snapshot `metadata` key. Progress and errors use standard Python logging.

## Launch Modes

Single process, sequential selected ranks:

```bash
python -m hyper_parallel.tools.dryrun config.yaml
```

Existing torchrun launch:

```bash
torchrun --nproc_per_node=8 -m hyper_parallel.tools.dryrun config.yaml
```

All torchrun workers parse the config, but only listed ranks build and execute
the fake model. In torchrun mode, a configured `virtual_world_size` must equal
`WORLD_SIZE`.

## Execution Principle

1. Resolve and validate the dry-run config and selected ranks.
2. Build a CPU data path through the real TextTrainer data stages and save the
   first batch structure and token counts.
3. Initialize a fake process group with the configured logical topology and
   enable the simulated device runtime (`set_device`/`synchronize`/
   `empty_cache` and allocator history become no-ops; the process never binds
   an accelerator).
4. Build and parallelize the model without materializing or loading weights.
5. Replace meta state with FakeTensors.
6. Construct the real BaseTrainer state through its own build stages (loss,
   optimizer, lr scheduler, callbacks, model-integration session) and run one
   unmodified `TextTrainer.train_step` over a replayed mock batch.
7. Track storage allocation, alias, resize, and release.
8. Write one native-compatible protocol-4 pickle per simulated rank.

## Host Scalars

The real step loop reads a handful of host scalars (`loss.item()` for logging
accumulators, token counts for the gradient-normalization denominator,
`grad_norm` casts for diagnostics). FakeTensors have no values, so a scoped
`TorchFunctionMode` shim resolves `item`/`__float__`/`__int__`/`__bool__` on
FakeTensors: logging-only scalars read as zero without touching tensor shapes,
while token counts — the only value-bearing quantities the step needs — are
pinned to the true values the CPU probe observed. Real tensors pass through
untouched.

## Value-Dependent Replay

FakeTensors preserve shape and dtype but not runtime values. Data-dependent
operations therefore cannot always execute unchanged.

MoE forced balancing (`fix_router=True`) fixes the intended expert assignment,
but it is not sufficient by itself. The generic EP path still needs concrete
values for:

- `argsort` permutations.
- Expert token counts.
- `.tolist()` host integers.
- Value-sized slicing and output shapes.

Registered value-dependency handlers automatically discover supported targets.
Each handler declares a default replay path. The built-in `moe_routing` handler
uses `balanced`; DeepSeek V4.1 registers a model-specific Engram handler when
its example module is imported.

There is no user-facing `value_dependencies` rule list. The handlers are scoped
to the fake step and restore patched methods afterward.

## Custom Operators

Users register a shape-only kernel with:

```python
from hyper_parallel.tools.dryrun import register_fake_op


@register_fake_op("custom::my_forward")
def my_forward_fake(input_tensor, weight):
    return torch.empty_like(input_tensor)
```

Registration occurs immediately when the defining module is imported. There is
no `op_plugins` enable switch. Model integration code must import the registration
module before dry-run constructs the model.

When FakeTensor encounters an unsupported operator, the error explains how to
register a kernel and lists currently registered operators.

## Compatibility Boundary

DTensor supports FakeTensor locals natively: ``DTensorBase.__new__`` builds a
storage-less meta shell for a FakeTensor local, and every metadata, grad, and
storage accessor answers for the local tensor. Dry-run and real training share
the same dispatch path; no DTensor methods are patched.

The runner installs the remaining process-local compatibility patches before
model build:

- Deferred, shape-only AutoModel materialization.
- Fake process-group subgroup creation for optimizer replicas.
- Layout-derived Muon gather metadata for uneven and empty shards.
- CSA2 host-value validation bypasses limited to tensors with `fake_mode`.

Normal components do not import dry-run config or DeepSeek modules.

Storage alias checks in core (Muon, FSDP) compare `untyped_storage()` identity
directly. This works for real tensors and FakeTensors. `data_ptr()` is not used
because FakeTensors have no physical address and can return identical zero
values for independent storages (empty shards behave the same way).

## Snapshot Semantics

Each live storage produces a logical synthetic segment. The tracker records
requested storage bytes and native-style alloc/free events. It does not simulate:

- Caching allocator segments or fragmentation.
- Device workspace and communication library reserves (fused vendor kernels'
  aclnn workspaces are the main unaccounted item when `use_fused_ops` paths
  are active).
- Asynchronous device-stream overlap (the host-visible pipelining of async
  communication — buffers issued in one micro-step and waited in the next —
  **is** reproduced, because the real scheduler code runs; only the
  kernel-time interleaving on device streams is out of scope).
- HCCL/NCCL persistent allocations.
- Activation swap host/device lifetimes.

The resulting peak is a logical capacity estimate, not a reservation guarantee.
Measured against 8-die DeepSeek-V4.1 validation crops, the logical peak lands
at 94–100% of the real trace-level peak across EP/FSDP/TP+SP/CP mixes.

## Unsupported Scope

The current implementation rejects:

- Pipeline parallelism.
- CPU offload.
- PEFT mutation.
- `torch.compile`.
- Activation swap.

Dry-run must fail fast when a configuration requests behavior whose memory
lifetime cannot be represented by the current logical tracker.
