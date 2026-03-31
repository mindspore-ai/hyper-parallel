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

- `Parameter([])`
- then `unsharded_param.data = unsharded_tensor`

This keeps `unsharded_param` sharing the all-gather storage.

## 3. Reduce Semantics

Torch backends may expose average-style reduction directly in collectives.

MindSpore reduce-like collectives do not provide native `AVG`, so `fully_shard` implements average reduction as:

- `SUM`
- then explicit division by group size
