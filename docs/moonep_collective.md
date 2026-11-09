# MoonEP collective 实验后端

HyperParallel 内部的 MoonEP 算法实验实现，使用 `torch.distributed`，不依赖外部 `moonep` 或 `npu_patch` 包。原始 `all_to_all` 保持默认后端。

## 工作方式

1. 收集各 rank 到全局专家的 assignment 数量，生成确定性的负载重分配计划。
2. 将过载 owner 的部分专家计算迁移到空闲 rank，按需复制专家权重，不丢弃 assignment。
3. 同一源 token 发往同一目标 rank 时只发送一次 hidden states，将路由权重和展开映射打包交换。
4. 接收端按专家整理 assignment，执行 packed SwiGLU，在执行 rank 加权预合并输出，再回传源 rank。
5. 自定义 autograd 逆向交换 token 梯度和专家参数梯度，副本梯度返回 owner 求和。

一个 top-k token 对应 k 条 assignment。计划使各 rank 的 assignment 数量相差不超过 1，但不保证实际执行耗时相等。

## 配置入口

Qwen3-MoE 的 `qwen3moe_ep_compute_fn` 支持以下参数：

```python
token_dispatcher="moonep_collective"
use_grouped_gemm=False
```

生产模型应通过现有模型 adapter/recipe 配置入口设置参数。专家参数必须是本地完整的 packed gate/up 与 down 权重；支持现有转置权重布局。`use_grouped_gemm=True` 使用 Ascend NPU grouped GEMM。显式容量丢弃和 overlap 参数不受支持。

## 可复现实例

先从本仓库执行 editable install，并确认导入位置：

```bash
pip install -e .
python -c 'import hyper_parallel; print(hyper_parallel.__file__)'
```

小模型示例需要 Ascend NPU、兼容的 Torch/torch_npu 和提供 Qwen3-MoE 的 Transformers。使用随机权重、合成 token、FP32、SGD，在同一进程组内交替比较两个后端：

```bash
mkdir -p output/moonep_example
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 \
  examples/torch/moe/benchmark_moonep_collective.py \
  --seq-length=4096 --micro-batch=1 --accumulation=2 \
  --warmup=10 --steps=50 --repeats=3 \
  --output=output/moonep_example/results.jsonl
```

结果包含完整 optimizer step time、tokens/s，以及全 rank 最大 allocated/reserved 峰值。示例使用手动稠密梯度平均及专家 owner 梯度缩放，不包含 FSDP、SAC 和真实数据加载；不能将其结果当作完整 Qwen3-30B 训练收益。输出文件按追加方式写入，重复实验应选择新的文件。

小模型完整梯度和三步 SGD 更新对照：

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 \
  examples/torch/moe/check_moonep_collective_model.py
```

## 测试

```bash
pytest -q tests/ut/core/expert_parallel

torchrun --standalone --nproc_per_node=2 \
  tests/torch/expert_parallel/_moonep_collective_parity.py --fp64

torchrun --standalone --nproc_per_node=2 \
  tests/torch/expert_parallel/_moonep_collective_parity.py --fp64 --assignment
```

UT 包含 planner 守恒、确定性、原生 host planner allocation fixture、稳定布局、packet 位保持和梯度检查。分布式 worker 对照完整专家梯度和 SGD/AdamW 参数更新，覆盖均匀、热点、稀疏、空 rank、全空、梯度累积和非重入 checkpoint。

Ascend EP8 launcher 位于 `tests/torch/expert_parallel/test_moonep_collective.py`。CPU FP64 对照通过不能替代 NPU 低精度验收。

## 已知限制

- 当前定位为实验后端，没有证明端到端训练加速。此前完整 30B 训练观察到吞吐退化。
- NPU grouped + AdamW 参数更新仍存在未关闭的数值差异。
- BF16/FP16 输出预合并可能在最终源 rank 合并前对部分和舍入，不保证与原始路径逐位一致。
- `HP_EP_CAPACITY_FACTOR` 环境变量和遗留专家容量属性尚未在 collective 接入层统一拒绝；对照实验必须禁用这些容量限制。
- 尚未验收多节点或专家 FSDP 度大于 1 的组合。
- 不包含 SHMEM/VMM、原生通信重排融合内核、通信计算重叠或有界专家预取槽位。

## 核心模块

| 文件 | 职责 |
| --- | --- |
| [moonep_collective_plan.py](../hyper_parallel/distributed/expert_parallel/moonep_collective_plan.py) | 确定性规划和 assignment 分段 |
| [moonep_collective.py](../hyper_parallel/distributed/expert_parallel/moonep_collective.py) | 可微交换、权重复制、批量布局和专家计算 |
| [moonep_collective_dedup.py](../hyper_parallel/distributed/expert_parallel/moonep_collective_dedup.py) | Token/rank 去重和输出预合并 |
| [moonep_collective_packet.py](../hyper_parallel/distributed/expert_parallel/moonep_collective_packet.py) | 混合类型打包及反向传输 |
