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

- 先创建 `Parameter([], requires_grad=False)`
- 再执行 `unsharded_param.data = unsharded_tensor`
- 最后按需恢复 `requires_grad`

这样 `unsharded_param` 才能和 all-gather buffer 共享底层存储，同时避免 MindSpore
在临时空 shape 上记录错误的反向元信息。

## 3. Reduce 语义

torch 后端通常可以直接使用带平均语义的通信实现。

MindSpore 的 reduce 类通信算子没有原生 `AVG`，所以 `fully_shard` 里的平均梯度仍然是：

- 先做 `SUM`
- 再手动除以 group size

## 4. 保持 Version 不变的 Buffer 更新方式

tensor 的 `_version` 会在自动微分、选择重计算等场景中参与一致性校验。
`fully_shard` 对 tensor 的处理应该尽量保持透明，不能额外引入 `_version` 变化，
否则会影响这些校验语义。

torch 在这类路径上可以直接使用
`torch.autograd._unsafe_preserve_version_counter(...)` 来抑制 version 递增。

MindSpore 没有对应接口，但可以通过 `.data` 取到一个和原始 tensor 共享内存的新
tensor，再对这个新 tensor 做 `copy_` 等修改，从而实现对原始 tensor version
递增的抑制。因此 `fully_shard` 在 all-gather copy-out 这类 buffer 刷新路径上，
会使用 `tensor.data.copy_(src)`。

需要区分的是，`Parameter` / DTensor rebasing 仍然要继续使用 `set_data(...)`。
因为 DTensor 参数内部可能维护多份需要同步的存储，`.data = ...` 只会更新其中一份，
而 `set_data(...)` 才能正确更新整个参数状态。
