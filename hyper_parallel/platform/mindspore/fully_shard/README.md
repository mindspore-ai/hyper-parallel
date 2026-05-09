# MindSpore Fully Sharded Torch Differences

[中文](./README_CN.md) | [English](./README.md)

This note only records the parts of MindSpore `fully_shard` that differ from the torch backend.

## 1. Autograd Entry

Torch can rely on its native `Tensor.backward()` behavior directly.

MindSpore needs an explicit PyNative compatibility layer before `fully_shard` can follow the same backward-style flow:

- `fully_shard()` automatically enables `_pynative_executor.set_grad_flag(True)`
- it patches torch-like tensor interfaces such as `requires_grad`, `grad`, and `backward()`
- training is expected to use `loss.backward()` instead of the old `value_and_grad` bridge

## 2. Unsharded Parameter Construction

Torch can swap in an unsharded parameter view without worrying about `Parameter(tensor)` creating a new underlying tensor object.

MindSpore cannot rely on `Parameter(tensor)` here, because that may materialize a new tensor instead of reusing the all-gather buffer. For this reason, `fully_shard` creates the full parameter as:

- first `Parameter([], requires_grad=False)`
- then `unsharded_param.data = unsharded_tensor`
- finally restore `requires_grad` if the sharded parameter needs gradients

This keeps `unsharded_param` sharing the all-gather storage while avoiding MindSpore
recording backward metadata from the temporary empty shape.

## 3. Reduce Semantics

Torch backends may expose average-style reduction directly in collectives.

MindSpore reduce-like collectives do not provide native `AVG`, so `fully_shard` implements average reduction as:

- `SUM`
- then explicit division by group size

## 4. Version-Preserving Buffer Updates

Tensor `_version` participates in consistency checks for autograd, selective
recomputation, and similar flows. `fully_shard` should stay transparent to those
mechanisms and must not introduce extra `_version` bumps on its own.

Torch can handle this directly with
`torch.autograd._unsafe_preserve_version_counter(...)`.

MindSpore does not provide an equivalent API, but it can return a new tensor via
`.data` that shares the same storage as the original one. Updating that new tensor
with `copy_` or other in-place writes lets `fully_shard` refresh the underlying
contents without bumping the original tensor version. That is why all-gather
copy-out style buffer refreshes use `tensor.data.copy_(src)`.

This should still be distinguished from `Parameter` / DTensor rebasing, which
must continue to use `set_data(...)`. DTensor-backed parameters may maintain
multiple synchronized storages internally, and `.data = ...` only updates one of
them, while `set_data(...)` updates the full parameter state correctly.
