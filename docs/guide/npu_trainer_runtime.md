# NPU Trainer 运行时适配

本指南说明 Trainer 在 CUDA、NPU 和 CPU 环境中的设备与分布式初始化行为。它只覆盖运行时适配，不涉及低精度模块替换、精度诊断或 FSDP Tensor Extension。

## 设备选择

Torch 平台按以下优先级选择设备：

1. 可用 NPU：`npu`
2. 可用 CUDA：`cuda`
3. 其他环境：`cpu`

Trainer、DP/CP 规约和 SIGTERM 同步均通过 `platform.device()` 创建通信 Tensor。这样 HCCL 不会收到 CPU Tensor，CUDA 也继续保持原有行为。

## 分布式初始化

入口为 `hyper_models.components.distributed.infrastructure.initialize_distributed`：

```python
initialize_distributed()
initialize_distributed(backend="nccl")
```

函数仍接受历史调用方的 `backend` 参数，但实际后端由运行设备约束：

| 运行设备 | 初始化后端 | 说明 |
| --- | --- | --- |
| NPU | `hccl` | NPU 训练强制使用 HCCL；传入的 `backend` 不覆盖它。 |
| CUDA | 调用方传入值，默认 `nccl` | 保留 CUDA 调用方的选择。 |
| CPU | `gloo`（默认 `nccl` 时自动降级） | 用于本地和单元测试。 |

当环境未提供 `RANK` 和 `WORLD_SIZE` 时，函数不创建 process group，直接返回 `torch.distributed`。分布式启动时会使用 `LOCAL_RANK` 设置当前 CUDA/NPU 设备。

## HCCL 与信号同步

`DistributedSignalHandler.signals_received()` 使用 `platform.device()` 创建标记 Tensor，再执行 `all_gather`。因此所有 rank 必须在相同控制流位置调用 SIGTERM 查询；否则任意后端的集合通信都会等待其他 rank。

## 验证

以下测试覆盖 backend 兼容性和 NPU 设备选择：

```bash
python -m pytest \
  tests/hyper_models/components/distributed/test_infrastructure.py \
  tests/components/training/test_signal_handler.py -q
```

