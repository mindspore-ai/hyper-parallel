# HyperParallel Dry-run

`hyper_parallel.tools.dryrun` executes the configured training path with
FakeTensors and writes a PyTorch-memory-viz-compatible logical snapshot. Model
weights and activations are not allocated on an accelerator.

## Run

```bash
python -m hyper_parallel.tools.dryrun path/to/training.yaml
```

For an existing torchrun launch, each worker reads the same YAML and only ranks
listed in `dry_run.ranks` execute:

```bash
torchrun --nproc_per_node=8 -m hyper_parallel.tools.dryrun path/to/training.yaml
```

Without torchrun, the CLI runs the configured ranks sequentially in one process,
using `virtual_world_size` as the logical topology. The default is rank 0.

The YAML uses the normal trainer sections plus:

```yaml
dry_run:
  enabled: true
  target_device: npu
  ranks: [0, 7]
  virtual_world_size: 8
  snapshot:
    enabled: true
    path: "outputs/dry_run"
    capture_stacks: true
    stack_depth: 64
```

`snapshot.path` is the output root. Dry-run creates
`rank_{rank}/memory.pickle` under it automatically. Setting
`snapshot.enabled: false` keeps the in-memory result but writes no pickle.
`capture_stacks` and `stack_depth` control Python allocation stacks in the
snapshot and can reduce overhead for very large models.
`enabled: false` makes the dry-run CLI exit without constructing a model.

`ranks` must be unique, non-negative, and smaller than the logical world size.
Under torchrun, `virtual_world_size`, when set, must equal the launched
`WORLD_SIZE`.

The only dry-run data product is the pickle selected by `snapshot.path`; run
metadata is embedded under the snapshot's `metadata` key. Progress and errors
use the normal Python logger and therefore follow the launch process's console
or trainer logging configuration.

## Add A Custom Operator

If dry-run reports `UnsupportedOperatorException`, add a shape-only FakeTensor
kernel for the missing operator. Do not copy values or launch device work.

```python
# my_project/dryrun_ops.py
import torch

from hyper_parallel.tools.dryrun import register_fake_op


@register_fake_op("custom::my_forward")
def my_forward_fake(input_tensor, weight, *, scale=1.0):
    del weight, scale
    return torch.empty_like(input_tensor)
```

There is no separate enable switch. The decorator registers immediately when
the defining module is imported. Import that module from the model integration
code before dry-run constructs the model.

## Value Dependencies

FakeTensor removes runtime values, so data-dependent routing and control flow
cannot execute unchanged. Dry-run automatically discovers targets supported by
registered value-dependency handlers and applies each handler's default balanced
path. There is no user-facing `value_dependencies` switch or rule list.

For MoE, `fix_router=True` is necessary but not sufficient: it fixes the intended
expert assignments, while FakeTensor still cannot materialize `argsort`
permutations, expert counts, `.tolist()` host ints, or value-sized slices.
The automatic MoE handler supplies those shapes during the fake step.

DeepSeek V4.1 additionally registers its Engram replay handler when the example
model module is imported. Models without value-dependent branches simply have no
discovered targets.

The built-in DeepSeek-V4.1 example is in
`hyper_parallel/tools/dryrun/examples/deepseek_v41.yaml`.
The full interface and execution design is in
`hyper_parallel/tools/dryrun/DESIGN.md`.
