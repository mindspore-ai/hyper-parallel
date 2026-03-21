# MindSpore Fully Sharded Implementation Guide

[中文](./README_CN.md) | [English](./README.md)

This document explains the core differences between MindSpore `fully_shard` and PyTorch, and describes the MindSpore implementation approach. It aims to help developers understand the design choices and key mechanisms.

## 1. Quick Guide

- **Parameter Object Immutability**: PyTorch switches module parameters between `sharded_param` and `unsharded_param` at runtime via `setattr`; however, MindSpore's `value_and_grad` binds the differentiation objects at initialization and will not differentiate a new object if swapped. Therefore, MindSpore fully_shard always keeps the module's parameter as the same Python object, i.e., `sharded_param`.

- **State Switching = In-place Data Update**: When switching between `sharded`/`unsharded` states, the `sharded_param` object itself remains unchanged; only its underlying data and type are updated in-place. `sharded_param.data` toggles between the local shard and the full all-gathered parameter; `_local_tensor.data` always retains the sharded data and does not change with the state.

- **Gradients Need Manual Attachment**: MindSpore's functional differentiation returns gradients separate from the `Parameter` objects. Here, hooks are used to attach gradients to `unsharded_param.grad` while returning a placeholder `_return_grad`; post-processing performs communication operations similar to PyTorch, updates `sharded_param.grad`, and then copies the result back to `_return_grad.data`, ensuring the differentiation function obtains the correct sharded gradients.

## 2. Core Design Principles

### 2.1 Differentiation Differences and Object Immutability

MindSpore's `value_and_grad` captures parameter objects at initialization, relying on Python object identity; PyTorch's `loss.backward` performs automatic differentiation at runtime based on the computation graph and allows parameter object replacement during forward passes.

```python
fully_shard(net, mesh=mesh, mp_policy=mp_policy)
grad_fn = ms.value_and_grad(net, None, net.trainable_params()) # Differentiation objects must be determined at initialization
optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate)
```

After `fully_shard` wrapping, `value_and_grad` captures `sharded_param`, but the forward computation actually uses unsharded data. If we replaced `sharded_param` with `unsharded_param` at runtime as in PyTorch, MindSpore would not differentiate the new object.

Therefore, MindSpore `fully_shard` **cannot** switch parameter objects between `sharded` and `unsharded` states like PyTorch; it must keep `sharded_param` unchanged and only update its underlying data.

### 2.2 In-place Data Switching

`sharded_param` is of type `ParameterDTensor`, internally maintaining two data references:

- `self.data`: the data participating in computation, pointing to different views depending on the state.
- `_local_tensor.data`: always holds the original locally sharded data.

State switching operations:

- `to_sharded()`: points `self.data` to `_local_tensor.data`;
- `to_unsharded()`: performs all-gather to obtain the full parameter `unsharded_param`, and points `self.data` to `unsharded_param.data`.

### 2.3 Dynamic Class Switching (`__class__` switching)

When used alone, `fully_shard` makes `sharded_param` a `ParameterDTensor` type (a mix of Parameter and DTensor), while `unsharded_param` is a plain `Parameter` type. Therefore, state switching also requires synchronously changing `__class__`:

- **`_switch_param_to_dtensor(data)`**: restores `__class__` to `ParameterDTensor` and points `self.data` to `_local_tensor`.
- **`_switch_param_to_parameter(data)`**: downgrades `__class__` to `Parameter` and points `self.data` to the unsharded data.

**Attribute Synchronization**: `DTensorBase` overrides certain instance attributes of `Parameter` (e.g., `has_init`, `init`) with `@property`, and the attribute values are stored in `_local_tensor`. After switching to `Parameter`, these properties disappear from the MRO and must be manually copied to the instance dictionary. When switching back to DTensor, the property descriptors automatically shadow the old values in the instance dictionary (data descriptors have higher priority), and the instance dictionary is actively cleaned to keep `__dict__` tidy.

## 3. Gradient Attachment and Post-Processing

MindSpore returns gradients via differentiation functions rather than attaching them directly to parameters as in PyTorch. To reuse PyTorch fully_shard's post-processing logic, a bridging mechanism is required.

The bridge is built using `param.register_hook(...)`, transforming MindSpore's gradient flow into a path similar to PyTorch fully_shard's post-processing.

### 3.1 Gradient Bridging Mechanism

1. **Hook Captures Full Gradient**: A hook is registered on `sharded_param`. Due to the object immutability described in Section 2.1, when in the unsharded state, the hook receives the full gradient (because the forward pass used the full parameter). During backward, the gradient is cached in `unsharded_param.grad`.
2. **Return Placeholder Gradient**: The hook also returns `_return_grad` (a placeholder Tensor), which is ultimately returned to the caller of `value_and_grad`.
3. **Post-processing (`post_backward()`)**: Performs reduce-scatter or all-reduce operations on `unsharded_param.grad` to obtain the local gradient.
4. **Update Sharded Gradient**: Writes the reduced gradient to `sharded_param.grad` (kept as a `DTensor`).
5. **Backfill Return Value**: Points `_return_grad.data` to `sharded_param.grad.to_local()`, ensuring the user finally obtains the correctly sharded gradient view. Note that due to framework mechanics, the gradient returned by the differentiation function is a plain `Tensor`, not a `DTensor`, while the gradient attached to the parameter is of type `DTensor`.

The roles can be simplified as:

- `unsharded_param.grad`: internal full-gradient cache
- `sharded_param.grad`: sharded `DTensor` gradient state maintained internally by fully_shard
- `_return_grad`: plain `Tensor` returned to `value_and_grad` and eventually consumed by the optimizer

## 4. Notes on Reduce-like Communication Operators

MindSpore's reduce-like operators (e.g., `AllReduce`, `ReduceScatter`) **do not support the `AVG` mode**. Therefore, when implementing gradient averaging, division by the communication group size must be done manually.

## 5. Summary

MindSpore fully_shard achieves PyTorch-equivalent fully sharded semantics within the functional differentiation paradigm by keeping parameter objects unchanged, updating data in-place, dynamically switching class behavior, and bridging gradients via hooks. Understanding these differences aids in troubleshooting and further feature extension.
