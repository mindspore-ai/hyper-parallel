# FSDP / HSDP Data Parallelism Guide

**English** | [简体中文](fsdp.zh.md)

HyperParallel implements Fully Sharded Data Parallel (FSDP) and Hybrid Sharded Data Parallel (HSDP) through `fully_shard`. It shards parameters, gradients, and optimizer states across devices in the data parallel domain, significantly reducing per-device memory usage.

## Core Concepts

`fully_shard` supports FSDP and HSDP, selected by the dimensionality of the supplied `mesh`. They differ in whether the data parallel domain is divided into sharding and replication groups:

| Mode | Mesh | Parameter sharding scope | Gradient communication |
|------|------|--------------|----------|
| FSDP | 1-D `(shard,)` | Sharded across the entire DP domain; each device holds `1/N` of the parameters | ReduceScatter across the entire DP domain |
| HSDP | 2-D `(replicate, shard)` | Sharded within each sharding group, with identical parameter copies across replicas | ReduceScatter within sharding groups + AllReduce across replicas |

- **FSDP**: Parameters are evenly sharded across all DP ranks, providing the finest sharding and greatest memory savings. However, every forward/backward requires AllGather across the full DP domain to reconstruct parameters, which can be costly across nodes.
- **HSDP**: The DP domain is partitioned into replicated groups, with FSDP-style sharding inside each group and parameter replication between groups. AllGather stays within each sharding group, typically within a node; only one gradient AllReduce crosses nodes. This balances memory savings and communication cost. `(replicate, shard)` means the number of replicas × shards per replica; `(2, 4)` means two replicas, each with four-way sharding.

Generally, use FSDP on a single node or when communication is not a bottleneck. Use HSDP across multiple nodes when you want to keep AllGather within nodes.

## API Overview

The `fully_shard` entry point applies FSDP/HSDP sharding to a module, modifies it in place, and returns it. Pass policy objects (`MixedPrecisionPolicy`, `OffloadPolicy` / `CPUOffloadPolicy`) to control mixed precision and offload. Optimizer states follow parameter sharding automatically, with no additional configuration.

```python
fully_shard(
    module,                       # Module or list of modules, combined into one sharding unit.
    *,
    mesh=None,                    # DeviceMesh: 1-D for FSDP, 2-D for HSDP.
    reshard_after_forward=True,   # Whether to reshard after forward.
    shard_placement_fn=None,      # Customize each parameter's sharding.
    mp_policy=MixedPrecisionPolicy(),   # Mixed-precision policy.
    offload_policy=OffloadPolicy(),     # Offload policy.
    ignored_params=None,          # Exclude from both sharding and gradient synchronization.
    replicate_params=None,        # Manage as replicated parameters with AllReduce gradients.
    comm_fusion=False,            # Enable AllGather / ReduceScatter fusion.
)
```

| Parameter | Description |
|------|------|
| `mesh` | Data parallel topology. 1-D selects FSDP; 2-D `(replicate, shard)` selects HSDP. `None` creates a default 1-D mesh over the global world size. |
| `reshard_after_forward` | Whether to reshard after forward; see performance optimization below. |
| `shard_placement_fn` | Callable returning `Shard` to customize each parameter's sharding dimension; `None` uses dimension 0. |
| `mp_policy` | Mixed-precision policy controlling parameter computation and gradient reduction dtypes. |
| `offload_policy` | Pass `CPUOffloadPolicy()` to offload sharded parameters/gradients to CPU for additional device memory savings. |
| `ignored_params` | Parameters remain in their original modules, without sharding or gradient synchronization. |
| `replicate_params` | Parameters remain replicated but managed, with DDP-style AllReduce gradient synchronization, unlike `ignored_params`. |
| `comm_fusion` | AllGather / ReduceScatter / AllReduce fusion. |

### Runtime Controls

After `fully_shard`, the module is dynamically extended with runtime methods to adjust sharding and communication during training:

| Method | Description |
|------|------|
| `model.set_modules_to_forward_prefetch(modules)` | Set the modules to prefetch during forward. |
| `model.set_modules_to_backward_prefetch(modules)` | Set the modules to prefetch during backward. |
| `model.set_requires_gradient_sync(flag)` | Control gradient synchronization for this backward, for gradient accumulation. |
| `model.set_gradient_scaling_factor(factor)` | Set the gradient scaling factor. |
| `model.set_reduce_op_type("avg" / "sum")` | Set the gradient reduction operation; defaults to `avg`. |
| `model.set_reshard_after_forward(flag)` | Toggle resharding after forward at runtime. |
| `model.unshard()` / `model.reshard()` | Manually reconstruct / reshard parameters. |

---

## Basic Usage

### 1. Minimal FSDP Example

```python
from hyper_parallel import fully_shard, init_device_mesh

# A 1-D mesh selects FSDP, sharding across the entire DP domain.
mesh = init_device_mesh(device_type="npu", mesh_shape=(dp_size,), mesh_dim_names=("dp",))
model = fully_shard(model, mesh=mesh)

# Standard training flow.
output = model(input)
loss = criterion(output)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

### 2. HSDP Example

```python
# A 2-D mesh selects HSDP: (number of replicas, shards per replica).
mesh = init_device_mesh(
    device_type="npu",
    mesh_shape=(replicate_size, shard_size),
    mesh_dim_names=("replicate", "shard"),
)
# Pass the 2-D mesh directly without selecting a submesh.
model = fully_shard(model, mesh=mesh)
```

### 3. Mixed Precision

```python
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
import torch

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,    # Parameter computation dtype.
    reduce_dtype=torch.float32,    # Gradient reduction dtype.
)
model = fully_shard(model, mesh=mesh, mp_policy=mp_policy)
```

### 4. Gradient Scaling Factor

CP/PP configurations may require gradient scaling:

```python
model = fully_shard(model, mesh=mesh)
model.set_gradient_scaling_factor(0.5)   # None disables scaling.
```

---

## FSDP Performance Optimization

`fully_shard` hides parameter AllGather and gradient reduction overhead through **compute-communication overlap**, primarily using prefetch and concurrent gradient communication.

### Forward/Backward Prefetch

Unshard subsequent modules' parameters early to overlap the current layer's computation with the next layer's parameter AllGather. Prefetch must be registered explicitly: specify which modules to reconstruct while the current module computes.

```python
# Forward: AllGather layer_{i+1} parameters when entering layer_i.
layer_i.set_modules_to_forward_prefetch([layer_i+1])

# Backward: AllGather layer_{i-1} parameters when entering layer_i backward.
layer_i.set_modules_to_backward_prefetch([layer_i-1])
```

Once registered, parameter AllGather triggered for a module overlaps computation in the preceding/following module, hiding communication latency.

### Overlap Backward Gradient Communication with Computation

During backward, each layer's gradient reduction is issued asynchronously once its gradients are computed. It overlaps later layers' backward computation, which can continue without waiting for communication:

- **FSDP**: Gradient ReduceScatter is issued asynchronously and overlaps computation.
- **HSDP**: Gradients first undergo ReduceScatter within sharding groups, then AllReduce across replicas. These operations are pipelined so one layer's AllReduce can overlap the next layer's ReduceScatter and backward computation, further hiding cross-group communication costs.

### Communication Fusion

`comm_fusion=True` combines small communications into larger ones, reducing the number of communication calls. This can benefit host-bound or cross-node training.

```python
model = fully_shard(model, mesh=mesh, comm_fusion=True)
```

### reshard_after_forward

`reshard_after_forward=True` (default) immediately reshards parameters after forward, freeing parameter memory between forward and backward. Set it to `False` to retain full parameters and avoid backward AllGather, trading device memory for fewer FSDP communications.

---

## Combine with Other Parallel Strategies

### FSDP + TP

Construct a two-dimensional `(dp, tp)` mesh, apply tensor parallelism on the `tp` subdimension first, then FSDP on `dp`:

```python
from hyper_parallel import init_device_mesh, fully_shard, parallelize_module

model = ...  # model that developer created

mesh = init_device_mesh("npu", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

# Apply TP first.
parallelize_module(model, mesh["tp"], tp_plan)

# Then apply FSDP on the DP subdimension.
model = fully_shard(model, mesh=mesh["dp"])
```

### FSDP + PP

Construct a mesh containing a `dp` dimension and apply FSDP separately to each PP stage submodule:

```python
from hyper_parallel import PipelineStage, fully_shard, init_device_mesh

split_models = ...  # PP stage models that developer split

mesh = init_device_mesh("npu", (pp_size, dp_size), mesh_dim_names=("pp", "dp"))

# Apply FSDP inside each PP stage.
for stage_model in split_models:
    fully_shard(stage_model, mesh=mesh["dp"])

stage = PipelineStage(stage_model, stage_index, stage_num=pp_size)
```

### FSDP + EP

Construct a two-dimensional `(fsdp, ep)` mesh, apply expert parallelism to expert layers on `ep`, then apply FSDP to the entire MoE module:

```python
from hyper_parallel import init_device_mesh, fully_shard
from hyper_parallel.core.expert_parallel import ExpertParallel

moe = ...  # MoE model that developer created

mesh = init_device_mesh("npu", (fsdp_size, ep_size), mesh_dim_names=("fsdp", "ep"))

# EP shards the expert layers along the expert dimension.
ExpertParallel().apply(moe.experts, mesh["ep"])

# FSDP shards the entire MoE module along the fsdp dimension.
fully_shard(moe, mesh=mesh["fsdp"])
```
