# Llama3 风格示例（Torch：TP、FSDP2 与上下文并行）

本目录提供在 **HyperParallel Torch 后端** 上、面向 **Llama3 风格解码器** 的最小可运行示例：**声明式张量并行（TP）+ 序列并行（SP）**，通过 `parallelize_module` 及 `ColwiseParallel`、`RowwiseParallel`、`SequenceParallel`、`PrepareModuleInput` 等辅助类实现；并可与 **`fully_shard`（FSDP2 风格的全分片数据并行）** 或 **`ContextParallel`（沿序列维的上下文并行）** 组合。TP+FSDP 使用二维 `DeviceMesh`，切出 `mesh["tp"]` 与 `mesh["dp"]`，分别承担 TP 与 FSDP。TP+CP 使用 `(tp, cp)` 二维网格：在 `mesh["tp"]` 上套用 `parallelize_llama3`，在 `mesh["cp"]` 上对注意力内的 BSHD SDPA 子模块挂 `ContextParallel`（Colossal 模式，`ulysses_degree=1`）。

> **说明：** 若你的文档树中有 `examples/distributed/tensor_parallelism/README.md`，可将其中环境、启动方式、约束等小节与本目录对照阅读。**本目录以本文档为说明入口。**

---

## 目录与文件

| 文件 | 说明 |
|------|------|
| `model.py` | 小型 Llama3 结构：`tok_embeddings`、`layers.*`、`attention`（wq/wk/wv/wo）、`feed_forward`（w1/w2/w3）、RMSNorm、`output`。 |
| `parallelize.py` | `parallelize_llama3()`：TorchTitan 风格 TP+SP（行切 embedding、序列维 `Shard`、注意力与 SwiGLU 的 Colwise/Rowwise 等）。 |
| `tensor_parallel_example.py` | **仅 TP：**一维 `DeviceMesh`，`world_size` 等于 TP 宽度；短训练循环。 |
| `fsdp_tp_example.py` | **TP + `fully_shard`：**二维 mesh `(dp, tp)`，TP 用 `mesh["tp"]`，FSDP 用 `mesh["dp"]`；对子层与根模块嵌套 `fully_shard`。 |
| `tp_cp_example.py` | **TP + CP：**构建 `Llama3Model`（`n_layers=2`），`parallelize_llama3(..., mesh["tp"])`，再对**每一层**的 `layer.attention.sdpa_core` 在 `mesh["cp"]` 上应用 `ContextParallel`。各 CP rank 处理长度为 `seq_len / cp` 的 token 窗口；通过 `rope_seq_start` 使 RoPE 与 embedding 后的 CP 窗口对齐。 |
| `__init__.py` | 再导出主要符号；示例脚本会把本目录加入 `sys.path`。 |

---

## 环境与依赖

与仓库根目录 [`examples/README.md`](../README.md) 中的 **PyTorch / CANN / HyperParallel** 要求一致。请加载 Ascend 或虚拟环境，确保可 `import hyper_parallel`、`torch` 以及（如适用）`torch_npu`。

| 组件 | 说明 |
|------|------|
| Python | >= 3.9 |
| CANN / 驱动 | 与当前 Ascend 栈版本匹配 |
| HyperParallel | `import hyper_parallel` 可用 |

---

## 运行一：仅张量并行（TP）

进程数必须等于 **TP 度数**；脚本在一维 `DeviceMesh` 上覆盖所有 rank。

```bash
cd examples/torch/llama3

torchrun --nnodes=1 --nproc_per_node=2 tensor_parallel_example.py
```

在仓库根目录执行：

```bash
torchrun --nnodes=1 --nproc_per_node=2 examples/torch/llama3/tensor_parallel_example.py
```

（若未安装到环境，可先设置 `PYTHONPATH` 指向仓库根目录。）

---

## 运行二：TP + FSDP（`fully_shard`）

总 rank 数 = `torchrun` 的 **`world_size`**。设 TP 宽度为 `TP`（环境变量 `LLAMA3_TP_SIZE`，默认 `2`），则：

- **`dp_size = world_size / TP`**（沿 `mesh["dp"]` 的数据并行 / FSDP）。
- 需满足 **`world_size % TP == 0`**。

示例：4 卡、`TP=2`、`DP=2`：

```bash
cd examples/torch/llama3
torchrun --nnodes=1 --nproc_per_node=4 fsdp_tp_example.py
```

可选环境变量：

| 变量 | 含义 | 默认 |
|------|------|------|
| `LLAMA3_TP_SIZE` | 张量并行宽度 | `2` |
| `LLAMA3_DEVICE_TYPE` | 设备类型：`npu` 或 `cuda` | `npu` |

脚本会设置 `HYPER_PARALLEL_PLATFORM=torch`。

---

## 运行三：TP + 上下文并行（`ContextParallel`）

总 rank 数须等于 **`LLAMA3_TP_SIZE * LLAMA3_CP_SIZE`**（默认 `2 × 2 → 4` 进程）。脚本从 `model.py` 构建 **`Llama3Model`**，调用 **`parallelize_llama3(model, mesh["tp"])`**（与运行一相同的 TP+SP 方案），再对**每个**解码层执行 **`ContextParallel(..., ulysses_degree=1).apply(layer.attention.sdpa_core, mesh["cp"])`**，在 BSHD SDPA 子模块（`model.py` 中的 `Llama3BshdSdpaCore`）上注册 Colossal CP 钩子。

各 CP rank 接收其上下文并行组对应的 token 切片；向 `Llama3Model.forward` 传入 **`rope_seq_start`**，使 RoPE 在 `tok_embeddings` 之后与该全局窗口对齐（模型按 **embedding 后的本地** 序列长度切 `freqs_cis`）。

```bash
cd examples/torch/llama3
torchrun --nnodes=1 --nproc_per_node=4 tp_cp_example.py
```

可选环境变量：

| 变量 | 含义 | 默认 |
|------|------|------|
| `LLAMA3_TP_SIZE` | 头维 TP 宽度（`parallelize_llama3` 所用 mesh） | `2` |
| `LLAMA3_CP_SIZE` | 上下文并行宽度 | `2` |
| `LLAMA3_DEVICE_TYPE` | `npu` 或 `cuda` | `npu` |

约束：**`seq_len % cp == 0`**，**`(seq_len / cp) % tp == 0`**（保证 Rowwise embedding 的序列分片均匀），且 **`n_heads` / `n_kv_heads` 能被 `tp` 整除**。

脚本会设置 `HYPER_PARALLEL_PLATFORM=torch`。

---

## 约束与说明

1. **序列并行：**激活在序列维使用 `Shard(1)`，**序列长度须能整除 TP 宽度**。
2. **注意力头数：** **`n_heads` 与 `n_kv_heads` 均须能整除 TP 宽度**（与 Colwise 切最后一维一致）。
3. **`parallelize_llama3`** 当前仅实现 **`enable_sequence_parallel=True`**；`enable_sequence_parallel=False` 未实现。
4. **训练循环：**仅对 **`loss.backward()` 与 `optimizer.step()`** 使用 `SkipDTensorDispatch`；**不要**把整个前向包进去，否则 TP 路径上 `DTensor` 行为可能异常。
5. **`fsdp_tp_example.py`：**在 DTensor + `fully_shard` 路径上调用 **`model.set_reduce_op_type("sum")`**；各 rank 使用**相同随机 batch** 做冒烟验证，确认组合栈能跑通，**不是**严格的单卡数值基准。
6. 配置为 **`Llama3DemoConfig` 教学用小模型**；数值与性能以本仓库 HyperParallel 实现为准。
7. **`tp_cp_example.py`** 使用 `model.py` 中的真实 **`Llama3Model` / `Llama3Attention`** 栈；SDPA 在 **`Llama3BshdSdpaCore`** 内执行，以便 `ContextParallel` 以 BSHD 布局挂接 `forward(q, k, v)`。未显式传入 `freqs_cis` 时，**`Llama3Model.forward(..., rope_seq_start=...)`** 会按 embedding 后的本地长度切 `self.freqs_cis`（CP 分窗与 TP 序列分片组合时必需）。

---

## 延伸阅读（API）

- 张量并行入口：`hyper_parallel` 包中的 `parallelize_module` 及各 `ParallelStyle` 子类（见源码与主项目文档）。
- 全分片训练：`hyper_parallel.fully_shard`（`core/fully_shard/api.py`）。
- 上下文并行：`hyper_parallel.ContextParallel`（`core/context_parallel/context_parallel.py`）。
