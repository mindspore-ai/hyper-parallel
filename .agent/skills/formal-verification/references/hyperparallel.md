# HyperParallel training context

HyperParallel is a PyTorch distributed training framework targeting Ascend NPU and NVIDIA GPU.
For this integration, focus on training-core behavior: tensor partitioning, forward/backward,
gradient synchronization, parameter/activation lifetime, pipeline scheduling and checkpoints.

Canonical constraints are supplied separately as domain-knowledge inputs from
[project-overview.md](../../../rules/project-overview.md) and
[distributed.md](../../../rules/distributed.md). Consult the current versions rather than
assuming that every heuristic below is a public API guarantee.

## Entry points and useful references

Paths below are relative to the framework repository, including in the isolated snapshot.

| Profile | Implementation | Existing regression evidence |
|---------|----------------|------------------------------|
| `dtensor` | `hyper_parallel/core/dtensor/`, `hyper_parallel/core/utils/` | `tests/ut/core/test_dtensor.py`, `tests/ut/core/test_device_mesh.py`, `tests/ut/core/test_tensor_redistribution.py`, `tests/ut/core/utils/test_shape_utils.py` |
| `fsdp` | `hyper_parallel/core/fully_shard/` | `tests/ut/core/fully_shard/`, `tests/torch/fully_shard/` |
| `pipeline` | `hyper_parallel/core/pipeline_parallel/` | `tests/ut/core/pipeline_parallel/`, `tests/torch/pipeline_parallel/` |
| `activation` | `hyper_parallel/core/activation_memory/` (swap and checkpoint APIs) | `tests/ut/core/activation_memory/`, `tests/ut/auto_models/distributed/test_activation_checkpoint.py` |
| `tensor-parallel` | `hyper_parallel/core/tensor_parallel/`, `hyper_parallel/core/shard/` | `tests/ut/core/shard/`, `tests/torch/tensor_parallel/` |
| `checkpoint` | `hyper_parallel/core/distributed_checkpoint/` | `tests/ut/core/distributed_checkpoint/` |

Start with shard geometry or state transitions before expanding to a larger coupled subsystem.
The `training-core` profile includes all of `hyper_parallel/core/` and `hyper_parallel/collectives/`.

Repository rules for test authoring are [testing.md](../../../rules/testing.md),
[unit-test.md](../../../rules/unit-test.md) and
[code-style.md](../../../rules/code-style.md). Distributed test launchers use
`tests.common.distributed_launcher.torchrun_case` or the existing parallel-case helper;
their parent modules must remain free of Torch/HyperParallel imports.

The profile directories constrain source extraction. Calls into other modules or native
code may need an expanded explicit `--submodule` selection. Record unresolved callee
contracts rather than silently assuming correctness. A verified isolated snapshot can
include uncommitted changes; preserve the snapshot path from the official engine's log
alongside the original HEAD revision when reporting results.
