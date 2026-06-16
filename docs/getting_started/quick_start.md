# 快速开始

## 概览

HyperParallel 是昇腾超节点亲和的分布式并行加速库，提供从集群级分布式并行到芯片内多核并行的一站式加速能力。

- **安装指南**：[installation.md](./installation.md)
- **特性使用指南**：详见 [docs/guide/](../guide/) 目录
- **API 参考**：详见 [docs/api/](../api/) 目录
- **FAQ 与故障排查**：[faq.md](../faq.md)

## 版本信息

- 当前版本：v1.0.0
- 发布日期：2026-06-30
- 支持后端：PyTorch（GPU/NPU）+ MindSpore（Ascend NPU）

## 设计理念

### 模型和系统优化解耦

HyperParallel 支持编程模型从"系统优化内嵌到模型脚本"演进到"模型和系统优化解耦"，通过声明式接口隐式注入并行、重计算、offload 等系统优化，算法代码无需感知系统优化细节。

### SPMD → MPMD → 集群+多核 MPMD

HyperParallel 支持并行范式从 SPMD 演进到 MPMD，支持集群 MPMD（Pipeline 并行）和多核 MPMD（芯片内多核并行 + 核级内存语义单边通信），增强通算掩盖和 MAC 利用率。

### Stateful → Stateless 计算状态分离

HyperParallel 支持存算关系从 Stateful 演进到 Stateless，通过远端和本地 Tensor 统一编程、远端 Tensor 预取和缓存，实现计算状态分离。

## 最小可运行示例

```python
import hyper_parallel as hp

# 创建设备 mesh
mesh = hp.init_device_mesh("npu", (dp_size,), mesh_dim_names=("dp",))

# 应用 FSDP
model = hp.fully_shard(model, mesh=mesh)

# 正常训练
output = model(input)
loss = criterion(output)
loss.backward()
optimizer.step()
```

更多示例请参考 [特性使用指南](../guide/) 和 `examples/` 目录。