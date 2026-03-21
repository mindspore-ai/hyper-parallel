# MindSpore Fully Sharded 实现说明

[中文](./README_CN.md) | [English](./README.md)

本文档阐述 MindSpore `fully_shard` 与 PyTorch 的核心差异，并解释 MindSpore 的实现方案。旨在帮助开发者理解设计选择与关键机制。

## 1. 快速导读

- **参数对象不可替换**：PyTorch 在运行时通过 `setattr` 将模块参数在 `sharded_param` 和 `unsharded_param` 之间切换；但 MindSpore 的 `value_and_grad` 在初始化时绑定求导对象，换掉后不会对新对象求导。因此 MindSpore fully_shard 始终保持模块的参数是同一个 Python 对象，即 `sharded_param`。

- **状态切换 = 原地换数据**：`sharded` / `unsharded` 切换时，`sharded_param` 对象本身不变，只是原地更新底层数据和类型。`sharded_param.data` 在本地分片和 all-gather 完整参数之间切换；`_local_tensor.data` 始终保留分片数据，不随状态变化。

- **梯度需要手动挂载**：MindSpore 函数式求导返回的梯度与 `Parameter` 对象是分离的。这里通过 hook 将梯度挂载到 `unsharded_param.grad`，同时返回占位的 `_return_grad`；后处理执行类似 PyTorch 的通信操作，更新 `sharded_param.grad`，再将结果回填到 `_return_grad.data`，使求导函数获取切分后正确的值。

## 2. 核心设计原则

### 2.1 微分差异与对象不变性

MindSpore 的 `value_and_grad` 在初始化时捕获参数对象，依赖 Python 对象 identity；PyTorch 的 `loss.backward` 在运行时根据计算图自动求导，可以在前向中替换参数对象。

```python
fully_shard(net, mesh=mesh, mp_policy=mp_policy)
grad_fn = ms.value_and_grad(net, None, net.trainable_params()) # 求导对象需要在初始化时确定
optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate)
```

`fully_shard` 包装后，`value_and_grad` 捕获的是 `sharded_param`，但前向计算阶段实际使用的是 unsharded 数据。若像 PyTorch 那样在运行时直接把 `sharded_param` 替换成 `unsharded_param`，MindSpore 不会对新对象求导。

因此，MindSpore `fully_shard` **不能**像 PyTorch 那样在 `sharded` 和 `unsharded` 状态间切换参数对象，而必须保持 `sharded_param` 不变，仅更新其底层数据。

### 2.2 数据原地切换

`sharded_param` 是 `ParameterDTensor` 类型，内部维护两份数据：

- `self.data`：参与计算的数据，随状态切换指向不同视图。
- `_local_tensor.data`：始终保存本地切分后的原始数据。

状态切换操作：

- `to_sharded()`：将 `self.data` 指向 `_local_tensor.data`；
- `to_unsharded()`：执行 all-gather 获得完整参数 `unsharded_param`，并将 `self.data` 指向 `unsharded_param.data`。

### 2.3 动态类型切换（`__class__` switching）

fully_shard 单独使用时，`sharded_param` 为 `ParameterDTensor` 类型（Parameter + DTensor 混合类），`unsharded_param` 为 `Parameter` 类型。为此，切换状态时还需同步修改 `__class__`：

- **`_switch_param_to_dtensor(data)`**：`__class__` 恢复为 `ParameterDTensor`，`self.data` 指向 `_local_tensor`。
- **`_switch_param_to_parameter(data)`**：`__class__` 降级为 `Parameter`，`self.data` 指向 unsharded 数据。

**属性同步**：`DTensorBase` 用 `@property` 重写了 `Parameter` 的普通实例属性（`has_init`、`init`），属性值存储在 `_local_tensor` 中。切换为 `Parameter` 后这些 property 从 MRO 消失，需手动将其拷贝到实例字典。切换回 DTensor 后，property 描述符自动遮蔽实例字典中的旧值（data descriptor 优先级更高），同时主动清理实例字典保持 `__dict__` 整洁。

## 3. 梯度挂载与后处理

MindSpore 通过求导函数返回梯度，而非像 PyTorch 那样将梯度直接挂载在参数上。为复用 PyTorch fully_shard 的后处理逻辑，需要建立桥接机制。

通过 `param.register_hook(...)` 做桥接，把 MindSpore 的梯度流转成类似 PyTorch fully_shard 的后处理路径。

### 3.1 梯度桥接机制

1. **Hook 捕获完整梯度**：在 `sharded_param` 上注册 hook。由于 2.1 节所述的对象不变性，当处于 unsharded 状态时，hook 接收到的梯度是完整的（因为前向使用了完整参数）。反向时将梯度缓存至 `unsharded_param.grad`。
2. **返回占位梯度**：hook 同时返回 `_return_grad`（一个占位 Tensor），该值最终返回给 `value_and_grad` 调用者。
3. **后处理（`post_backward()`）**：执行 reduce-scatter 或 all-reduce 操作，从 `unsharded_param.grad` 计算得到本地梯度。
4. **更新 sharded 梯度**：将 reduce 后的梯度写入 `sharded_param.grad`（保持为 `DTensor`）。
5. **回填返回值**：将 `_return_grad.data` 指向到 `sharded_param.grad.to_local()`，确保用户最终获取的梯度是正确的分片视图。注意，由于框架机制，求导函数返回的梯度是普通`Tensor`，不是`DTensor`，参数上挂载的 grad 是 `DTensor` 类型。

角色分工可以简化成：

- `unsharded_param.grad`：内部 full-grad 缓存
- `sharded_param.grad`：fully_shard 内部维护的 sharded `DTensor` 梯度状态
- `_return_grad`：返回给 `value_and_grad`，并最终被用户传给优化器的普通 `Tensor`

## 4. Reduce 类通信算子说明

MindSpore 的 Reduce 类算子（如 `AllReduce`、`ReduceScatter`）**不支持 `AVG` 模式**。因此在实现梯度平均时，需要手动除以通信组大小。

## 5. 总结

MindSpore fully_shard 通过保持参数对象不变、原地更新数据、动态切换类行为，以及 hook 桥接梯度，在函数式微分范式下实现了与 PyTorch 等效的 fully sharded 语义。理解这些差异有助于排查问题并进一步扩展功能。
