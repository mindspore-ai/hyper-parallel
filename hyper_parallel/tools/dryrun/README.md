# HyperParallel Dry-run

`hyper_parallel.tools.dryrun` executes the configured training path with
FakeTensors and writes a PyTorch-memory-viz-compatible logical snapshot. Model
weights and activations are not allocated on an accelerator.

## Run

Launch with `torchrun` matching the target `WORLD_SIZE`; each worker reads the
same YAML and only ranks listed in `dry_run.ranks` (default `[0]`) execute:

```bash
torchrun --standalone --nproc_per_node=8 -m hyper_parallel.tools.dryrun path/to/training.yaml
```

The YAML uses the normal trainer sections plus:

```yaml
dry_run:
  enabled: true
  steps: 2
  ranks: [0, 7]
  snapshot:
    enabled: true
    path: "outputs/dry_run"
    capture_stacks: true
    stack_depth: 64
```

`target_device` defaults to `npu` and can be omitted unless simulating GPU
(`target_device: cuda`). `steps` defaults to `2` so Step 0 materializes lazy
optimizer states and Step 1 captures steady-state peak memory.

`snapshot.path` is the output root. Dry-run creates
`rank_{rank}/memory.pickle` under it automatically. Setting
`snapshot.enabled: false` keeps the in-memory result but writes no pickle.
`capture_stacks` and `stack_depth` control Python allocation stacks in the
snapshot and can reduce overhead for very large models.
`enabled: false` makes the dry-run CLI exit without constructing a model.

`ranks` must be unique, non-negative, and smaller than the launched `WORLD_SIZE`.

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
cannot execute unchanged. Dry-run exposes plain install functions; the model
integration code calls the one it needs. There is no rule list and no
`dry_run` configuration key — the model owner decides, because the same module
(such as Engram) can appear in several models whose replay needs differ.

```python
from hyper_parallel.tools.dryrun import install_moe_replay


def build_my_model(distributed_setup, **kwargs):
    model = ...  # parallelized model
    install_moe_replay(model, distributed_setup.mesh_context, "balanced")
    return model
```

Replay modes for routed experts are `balanced`, `hotspot` and `explicit`
(the last one takes `source_expert_loads`); the choice is made at the call
site. Model-specific replay beyond MoE is a normal `xxx.forward = xxx` /
class-attribute patch in the model integration module, installed once and
never restored — dry-run owns the process it runs in.

For MoE, `fix_router=True` is necessary but not sufficient: it fixes the intended
expert assignments, while FakeTensor still cannot materialize `argsort`
permutations, expert counts, `.tolist()` host ints, or value-sized slices.
The automatic MoE handler supplies those shapes during the fake step.

DeepSeek V4.1 registers its Engram replay handler when the example model module
is imported, and its recipe selects that handler explicitly. Models without
value-dependent branches simply have no targets.

The built-in DeepSeek-V4.1 example is in
`hyper_parallel/tools/dryrun/examples/deepseek_v41.yaml`.
