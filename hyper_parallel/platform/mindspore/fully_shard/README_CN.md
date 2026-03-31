# MindSpore Fully Sharded 与 Torch 的差异

[中文](./README_CN.md) | [English](./README.md)

本文只记录 MindSpore `fully_shard` 相对 torch 后端不同的部分。

## 1. 自动求导入口

torch 可以直接依赖原生的 `Tensor.backward()` 语义。

MindSpore 这里需要先补一层 PyNative 兼容能力，`fully_shard` 才能走同样的 backward 风格路径：

- `fully_shard()` 会自动开启 `_pynative_executor.set_grad_flag(True)`
- 会给 Tensor 补齐 `requires_grad`、`grad`、`backward()` 等 torch 风格接口
- 训练入口改为 `loss.backward()`，不再沿用旧的 `value_and_grad` 桥接方案

## 2. Unsharded 参数构造方式

torch 在切换到 unsharded 参数时，可以直接依赖后端已有的参数视图语义。

MindSpore 这里不能直接用 `Parameter(tensor)`，因为这一步可能重新创建底层 Tensor，导致无法复用 all-gather 出来的存储。所以 `fully_shard` 现在用下面的方式构造完整参数：

- 先创建 `Parameter([])`
- 再执行 `unsharded_param.data = unsharded_tensor`

这样 `unsharded_param` 才能和 all-gather buffer 共享底层存储。

## 3. Reduce 语义

torch 后端通常可以直接使用带平均语义的通信实现。

MindSpore 的 reduce 类通信算子没有原生 `AVG`，所以 `fully_shard` 里的平均梯度仍然是：

- 先做 `SUM`
- 再手动除以 group size
