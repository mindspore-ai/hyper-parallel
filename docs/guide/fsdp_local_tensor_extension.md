# FSDP Local Tensor Extension

本机制让外部 Tensor 包接入 HyperParallel 的 Torch `fully_shard`，而不让 HyperParallel 依赖任何具体量化格式。HP 负责逻辑切分和集合通信；外部实现只负责逻辑 Tensor 与物理通信 Tensor 之间的转换。

## 协议

外部 local tensor 必须显式设置 `_hp_fsdp_extension = True`，并实现以下方法：

```python
class MyLocalTensor(torch.Tensor):
    _hp_fsdp_extension = True

    def hp_fsdp_to_dtensor(self, mesh, placements) -> torch.Tensor:
        # 返回可安装为 nn.Parameter 的 DTensor-compatible wrapper。
        # 返回对象必须暴露 layout 和 to_local()。
        ...

    def fsdp_pre_all_gather(self, context):
        # 返回 (physical_tensors, metadata)。
        # physical_tensors 是一个或多个待通信的 torch.Tensor。
        ...

    def fsdp_post_all_gather(self, outputs, metadata, *, out=None):
        # 返回 (unsharded_tensor, inner_tensor)。
        # out 非空时必须原样返回该对象。
        ...
```

`FSDPGatherContext` 提供：

- `phase`：`forward` 或 `backward`
- `reshard_after_forward`：当前 FSDP 调度策略
- `param_fqn`：用于定位问题的参数名

## 职责边界

| HyperParallel | Extension 实现 |
| --- | --- |
| 使用 `torch.chunk` 执行逻辑分片、选择 rank、拒绝不均匀分片 | 保持 `chunk`、`clone`、`contiguous` 后仍是有效 extension Tensor |
| 为每个物理 Tensor 发起 all-gather，等待全部 work | 生成物理通信 Tensor 与 metadata |
| 缓存/重建 all-gather buffer，并验证返回 shape | 根据 gathered outputs 重建 unsharded 表示 |
| 对 BF16/FP32 梯度使用普通 `DTensor.from_local()` | 参数存储 wrapper 不需要同时表达高精度梯度 |

非 flattenable extension 参数不会进入普通 comm-fusion buffer，而是走独立 all-gather 和梯度归约路径；普通参数仍保持既有 fusion 行为。

## out 规则

重复 unshard 时，HP 将已有 unsharded Parameter 作为 `out` 传入 `fsdp_post_all_gather()`。extension 必须原样返回同一个对象，而不是新对象。这保证 optimizer、autograd hook 和模块 Parameter 引用不被替换。

物理 Tensor 数量、dtype、numel 或 device 改变时，HP 会重建 all-gather buffer，不会复用不匹配的旧 buffer。

## 使用方式

该协议随 Torch fully_shard 导出：

```python
from hyper_parallel.platform.torch.fully_shard import (
    FSDPGatherContext,
    FSDPLocalTensorExtension,
    fsdp_post_all_gather,
    fsdp_pre_all_gather,
    fsdp_shard_tensor,
    fsdp_to_dtensor,
)
```

通常外部包只需要实现 local Tensor；将带该 Tensor 的 Parameter 交给现有 `fully_shard()` 即可。不要在外部实现第二套 rank 切分或 all-gather 调度。

## 验证

```bash
python -m pytest tests/ut/platform/torch/fully_shard -q
```

协议级测试覆盖多物理 Tensor、`out` 身份要求、wrapper 最小接口、buffer 契约变化和普通参数的 comm-fusion 回归。

