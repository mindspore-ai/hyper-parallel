# Build and train a model with public module families

**English** | [简体中文](README.zh.md)

This example contains a complete causal attention algorithm, model-owned routing, and SwiGLU expert computation. Importing the public model package registers its families and model specification. Users construct the model and train it through the public API; they do not call registration or preparation helpers.

The example uses portable Torch computation and FP32. Mega and low-precision members require their own implementation and numerical acceptance.

Use a PyTorch environment compatible with this repository and install the checkout in editable mode with `pip install -e .`.

## Single-process training

Run the following Python code from the repository root. Construct an ordinary `nn.Module`; its public families select their default local members.

```python
import torch
from examples.torch.module_family import DecoderConfig, FamilyDecoder

config = DecoderConfig()
model = FamilyDecoder(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
x = torch.randn(2, 4, config.hidden_size, device=next(model.parameters()).device)
loss = model(x).square().mean()
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
print(tuple(model(x).shape))  # (2, 4, 8)
```

## Parallel training

Models and public families accept the existing complete job `mesh` directly. No additional assembly call is required. After initializing the process group and selecting the current device:

```python
import torch

from examples.torch.module_family import DecoderConfig, FamilyDecoder
from hyper_parallel.distributed.mesh import MeshContext

# Four ranks: dense DP2 x CP2, expert view EP2.
job_mesh = MeshContext(dp_size=2, cp_size=2, ep_size=2,
                       dp_shard_size=4, edp_shard_size=2)
job_mesh.build_meshs(device_type="npu", world_size=4)
with torch.device("npu"):
    model = FamilyDecoder(DecoderConfig(), mesh=job_mesh)
```

The model calls `CausalAttention(config=config, mesh=mesh)` and `GroupedExperts(config=config, mesh=mesh)`. Each family selects a concrete class from the actual CP/EP domains, applies its computation layout, and borrows communication resources before returning. The mesh stays outside static requests and `state_dict`. Meta parameters stay meta; an offline logical mesh records layouts for the framework to finish at runtime.

Families require the exact axis names `cp` and `ep` when the corresponding parallel size exceeds one. CP uses `device_mesh`; EP uses `fsdp_moe_mesh`, falling back to `device_mesh` when no expert view is supplied. Names and sizes are validated before member construction, including offline logical meshes.

`HyperAutoModelForCausalLM.from_config(config, model_factory=FamilyDecoder, distributed_setup=setup)` is an optional managed entry. The runnable [train.py](train.py) creates the job and uses that entry to continue with main-repository FSDP2, materialization, and training:

```bash
python -m examples.torch.module_family.train
torchrun --standalone --nproc-per-node=4 -m examples.torch.module_family.train --cp-size 2
torchrun --standalone --nproc-per-node=4 -m examples.torch.module_family.train --ep-size 2
torchrun --standalone --nproc-per-node=4 -m examples.torch.module_family.train --cp-size 2 --ep-size 2
```

Four-device training uses main-repository HyperParallel FSDP2. The optimizer step follows the existing Trainer convention, `SkipDTensorDispatch(no_skip={torch.zeros_like})`, to retain DTensor metadata in Adam state. An existing Trainer handles this step for its users. CP/EP members do not add parameter-gradient reductions.

CP2+EP2 reuses the same four ranks: the dense topology is DP2 x CP2, and EP2 is an expert view over that domain.

Construction follows this sequence:

```text
Import the public package
  -> Register family declarations and the model specification
  -> Construct the model directly, or let the facade pass its existing mesh
  -> Families select members, apply component layouts, and borrow CP/EP resources
  -> The framework checks shared parameters and reuses completed layouts and bindings
  -> Apply layouts for the remaining model parameters
  -> Apply main-repository FSDP2
  -> Materialize or initialize weights
  -> Create the optimizer and train
```

Family declarations live in their definition files and run once through Python's import lifecycle. User scripts do not call `register_*()`.

## Member selection

A family selects a concrete `nn.Module` from its declared target arguments and the job's existing CP/EP configuration. Direct construction and module replacement use the same selection function. Arguments such as `kernel`, `precision`, or `backend` are ordinary selection parameters when a family declares them; each supported combination points directly to a concrete member.

This example exposes one Torch algorithm per family. With default job settings, `CausalAttention` returns `CausalAttentionModule` and `GroupedExperts` returns `GroupedExpertsModule`. CP2 selects `CPCausalAttention`; EP2 selects `EPGroupedExperts`. These example entries therefore need no `backend` parameter. With a mesh, selection happens during construction. The framework owns global shared-parameter checks, FSDP2, the optimizer, and DCP; families add no parameter-gradient reductions.

## Ownership and layout

```text
module_family/
├── __init__.py                      Public exports; loads family and model declarations
├── config.py                        Algorithm dimensions and structural constraints
├── model.py                         Model composition, router, and model specification
├── attention/
│   ├── family.py                    Public entry, member selection, and layout declaration
│   ├── attention.py                 Complete default algorithm and identity stages
│   └── cp_attention.py              Inherits the algorithm; exchanges K/V and sequence coordinates
├── grouped_experts/
│   ├── family.py                    Public entry and expert layout declaration
│   ├── grouped_experts.py           dispatch/group_experts/combine and SwiGLU computation
│   └── ep_grouped_experts.py        Inherits the algorithm and delegates transport to EPContext
└── train.py                         Job entry, meshes, main-repository FSDP2, and training loop
```

Models call the public family entries. Module owners maintain family declarations; model owners maintain model specifications. Importing the public package loads both.

## Scope and validation

The CP example uses equal-length, contiguous sequence partitions and a causal mask. It does not implement RoPE, KV cache, or packed-document attention. The expert algorithm implements SwiGLU with per-slot batched matrix multiplication as a portable example; it makes no fused-kernel performance claim.

The [distributed acceptance case](../../../tests/torch/distributed/test_module_family_public_example.py) constructs models directly and through the public facade for CP2, EP2, and CP2+EP2. It compares two steps of outputs, global parameters, and Adam moments against a dense reference. A four-process CPU case also checks direct-construction outputs and input gradients.

The direct single-process example uses CPU; the managed entry selects an available local device. Distributed training requires a GPU or NPU supported by main-repository FSDP2.
