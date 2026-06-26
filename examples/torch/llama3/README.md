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
| `dp_tp_cp_sp_fsdp_example.py` | **8 卡综合：DP + TP + CP + SP + FSDP2。**4-D mesh `(dp, fsdp, cp, tp)`，TP+SP 用 `mesh["tp"]`，CP 用 `mesh["cp"]`，FSDP2/HSDP 用 `mesh[("dp", "fsdp")]`（参数在 `fsdp` 内分片、在 `dp` 上复制；`dp=1` 时退化为纯 FSDP2）。 |
| `demo_utils.py` | 共享 ``init_dist()`` / ``train_steps()``（默认 10 step，可用 ``LLAMA3_NUM_STEPS`` 覆盖）。 |
| `pipeline.py` | PP stage 构建：`build_llama3_pp_chunk()` 按 PP rank 切层；末 stage 支持 per-micro-batch CE。 |
| `pp_example.py` | **仅 PP：**1-D mesh `(pp,)`，`Schedule1F1B`，每 rank 一个 stage；stage 0 喂 token，末 stage 算 CE。 |
| `pp_tp_example.py` | **PP + TP：**2-D mesh `(pp, tp)`，stage 内 `parallelize_llama3`，`Schedule1F1B` + P2P。 |
| `pp_fsdp_tp_cp_sp_example.py` | **8 卡综合：PP + FSDP2 + TP + CP + SP。**4-D mesh `(pp, fsdp, cp, tp)`，默认 `(2,2,2,1)`。 |
| `pp_fsdp_cp_example.py` | **8 卡：PP + FSDP2 + CP。**3-D mesh `(pp, fsdp, cp)`，默认 `(2,2,2)`。 |
| `__init__.py` | 再导出主要符号；示例脚本会把本目录加入 `sys.path`。 |

---

## 环境与依赖

与仓库根目录 [`examples/README.md`](../README.md) 中的 **PyTorch / CANN / HyperParallel** 要求一致。请加载 Ascend 或虚拟环境，确保可 `import hyper_parallel`、`torch` 以及（如适用）`torch_npu`。

运行示例前请执行 ``export HYPER_PARALLEL_PLATFORM=torch``（HyperParallel 在 import 时读取该变量）。

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
export HYPER_PARALLEL_PLATFORM=torch

torchrun --nnodes=1 --nproc_per_node=2 tensor_parallel_example.py
```

在仓库根目录执行：

```bash
export HYPER_PARALLEL_PLATFORM=torch
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
export HYPER_PARALLEL_PLATFORM=torch
torchrun --nnodes=1 --nproc_per_node=4 fsdp_tp_example.py
```

可选环境变量：

| 变量 | 含义 | 默认 |
|------|------|------|
| `LLAMA3_TP_SIZE` | 张量并行宽度 | `2` |
| `LLAMA3_DEVICE_TYPE` | 设备类型：`npu` 或 `cuda` | `npu` |

---

## 运行三：TP + 上下文并行（`ContextParallel`）

总 rank 数须等于 **`LLAMA3_TP_SIZE * LLAMA3_CP_SIZE`**（默认 `2 × 2 → 4` 进程）。脚本从 `model.py` 构建 **`Llama3Model`**，调用 **`parallelize_llama3(model, mesh["tp"])`**（与运行一相同的 TP+SP 方案），再对**每个**解码层执行 **`ContextParallel(..., ulysses_degree=1).apply(layer.attention.sdpa_core, mesh["cp"])`**，在 BSHD SDPA 子模块（`model.py` 中的 `Llama3BshdSdpaCore`）上注册 Colossal CP 钩子。

各 CP rank 接收其上下文并行组对应的 token 切片；向 `Llama3Model.forward` 传入 **`rope_seq_start`**，使 RoPE 在 `tok_embeddings` 之后与该全局窗口对齐（模型按 **embedding 后的本地** 序列长度切 `freqs_cis`）。

```bash
cd examples/torch/llama3
export HYPER_PARALLEL_PLATFORM=torch
torchrun --nnodes=1 --nproc_per_node=4 tp_cp_example.py
```

可选环境变量：

| 变量 | 含义 | 默认 |
|------|------|------|
| `LLAMA3_TP_SIZE` | 头维 TP 宽度（`parallelize_llama3` 所用 mesh） | `2` |
| `LLAMA3_CP_SIZE` | 上下文并行宽度 | `2` |
| `LLAMA3_DEVICE_TYPE` | `npu` 或 `cuda` | `npu` |

约束：**`seq_len % cp == 0`**，**`(seq_len / cp) % tp == 0`**（保证 Rowwise embedding 的序列分片均匀），且 **`n_heads` / `n_kv_heads` 能被 `tp` 整除**。

---

## 运行四：8 卡综合（DP + TP + CP + SP + FSDP2）

`dp_tp_cp_sp_fsdp_example.py` 在 4-D `DeviceMesh` `(dp, fsdp, cp, tp)` 上同时启用：

| 维度 | 组件 | 说明 |
|------|------|------|
| `mesh["tp"]` | **TP + SP** | `parallelize_llama3` 的 TorchTitan 风格方案：`Colwise/Rowwise` 线性 + `SequenceParallel` 范数 + `Shard(1)` 序列维激活。 |
| `mesh["cp"]` | **CP** | `ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1)` 挂在每个 `layer.attention.sdpa_core`（Colossal CP，BSHD Q/K/V）。 |
| `mesh[("dp", "fsdp")]` | **FSDP2 + DP** | `fully_shard` 用 2-D HSDP 切片：参数在 `fsdp` 组内分片、在 `dp` 组上复制；`dp=1` 即纯 FSDP2，`dp>=2` 即 HSDP。 |

需满足 **`world_size == dp * fsdp * cp * tp`**。默认 `(dp, fsdp, cp, tp) = (1, 2, 2, 2)`，即 8 卡。

```bash
cd examples/torch/llama3
export HYPER_PARALLEL_PLATFORM=torch
torchrun --nnodes=1 --nproc_per_node=8 dp_tp_cp_sp_fsdp_example.py
```

也可在仓库根目录执行：

```bash
export HYPER_PARALLEL_PLATFORM=torch
torchrun --nnodes=1 --nproc_per_node=8 examples/torch/llama3/dp_tp_cp_sp_fsdp_example.py
```

可选环境变量（默认见括号）：

| 变量 | 含义 | 默认 |
|------|------|------|
| `LLAMA3_DP_SIZE` | 外层（HSDP 复制）DP 宽度 | `1` |
| `LLAMA3_FSDP_SIZE` | FSDP2 分片宽度 | `2` |
| `LLAMA3_CP_SIZE` | 上下文并行宽度 | `2` |
| `LLAMA3_TP_SIZE` | 张量并行宽度（含 SP） | `2` |
| `LLAMA3_DEVICE_TYPE` | `npu` 或 `cuda` | `npu` |

约束：**`n_heads` / `n_kv_heads` 能被 `tp` 整除**；**`seq_len % cp == 0`**；**`(seq_len / cp) % tp == 0`**（保证每个 CP 窗口内的 SP 序列分片均匀）。

8 卡常见组合（`dp * fsdp * cp * tp = 8`）：

| `dp` | `fsdp` | `cp` | `tp` | 说明 |
|------|--------|------|------|------|
| 1 | 2 | 2 | 2 | 默认；纯 FSDP2 + CP + TP/SP。 |
| 2 | 2 | 1 | 2 | HSDP（DP 复制 × FSDP 分片）+ TP/SP，关闭 CP。 |
| 1 | 4 | 1 | 2 | 大 FSDP 分片 + TP/SP。 |

> **注意：当 `fully_shard` 把 TP-DTensor 权重提升到 ≥3-D mesh 且活动 `(tp,)` 输入仍为 1-D 时（如 `dp=2, fsdp=1, cp=2, tp=2`），库内 layout-infer 路径还不支持权重 mesh 是输入 mesh 超集的情况：先在 `parallel_embedding.infer_layout` 触发 `int - tuple` 类型错误，进一步还会卡在 `parallel_matmul` 的 `x_mesh_shape != w_mesh_shape` 检查上。**如需"纯 DP（不分片参数）+ TP/SP/CP"，建议改用 `(dp=1, fsdp=2, cp=2, tp=2)`（默认）或上面的 HSDP 组合，等库内修复后再启用 `fsdp=1`。

脚本会在所有 rank 上对 `tokens` / `targets` 做一次广播确保同样输入（冒烟用法，与 `fsdp_tp_example.py` 一致；不是严格的单卡数值基准）。

---

## 运行五：流水线并行（PP）

`pp_example.py` 在 1-D `DeviceMesh` `(pp,)` 上用 **`Schedule1F1B`** 跑 Llama3 风格 decoder 的 PP 训练循环。每个 PP rank 持有一段连续层：rank 0 含 `tok_embeddings`，末 rank 含 `norm` + `output` 并在末 stage 上计算 per-micro-batch 交叉熵。

**`world_size` 必须等于 `LLAMA3_PP_SIZE`**（默认 `2`）。

```bash
cd examples/torch/llama3
export HYPER_PARALLEL_PLATFORM=torch
torchrun --nnodes=1 --nproc_per_node=2 pp_example.py
```

可选环境变量：

| 变量 | 含义 | 默认 |
|------|------|------|
| `LLAMA3_PP_SIZE` | 流水线并行宽度 | `2` |
| `LLAMA3_DEVICE_TYPE` | `npu` 或 `cuda` | `npu` |

约束：**`n_layers >= pp_size`**；**batch size 须能整除 micro-batch 数**（示例默认 `batch=8`、`micro_batches=4`）。

---

## 运行六：PP + TP

`pp_tp_example.py` 在 2-D mesh `(pp, tp)` 上组合 PP 与 TorchTitan 风格 TP+SP：每个 PP stage 内调用 `parallelize_llama3(..., mesh["tp"])`，stage 0 以 Replicate 布局的 `DTensor` token 批次作为输入。

**`world_size == LLAMA3_PP_SIZE * LLAMA3_TP_SIZE`**（默认 `2 × 2 → 4`）。

```bash
cd examples/torch/llama3
export HYPER_PARALLEL_PLATFORM=torch
torchrun --nnodes=1 --nproc_per_node=4 pp_tp_example.py
```

可选环境变量：

| 变量 | 含义 | 默认 |
|------|------|------|
| `LLAMA3_PP_SIZE` | 流水线并行宽度 | `2` |
| `LLAMA3_TP_SIZE` | 张量并行宽度 | `2` |
| `LLAMA3_DEVICE_TYPE` | `npu` 或 `cuda` | `npu` |

约束：**`n_layers >= pp_size`**；**`n_heads` / `n_kv_heads` 能被 `tp` 整除**；**`seq_len % tp == 0`**。

---

## 运行七：8 卡 PP + FSDP2 + TP + CP + SP

`pp_fsdp_tp_cp_sp_example.py` 在 4-D `DeviceMesh` `(pp, fsdp, cp, tp)` 上同时启用五种并行：

| 维度 | 组件 | 说明 |
|------|------|------|
| `mesh["pp"]` | **PP** | `Schedule1F1B`；每 PP rank 一段 stage chunk（`pipeline.py`）。 |
| `mesh["tp"]` | **TP + SP** | `parallelize_llama3` 的 TorchTitan 风格方案。 |
| `mesh["cp"]` | **CP** | `ContextParallel` 挂在每个 `layer.attention.sdpa_core`。 |
| `mesh["fsdp"]` | **FSDP2** | `fully_shard` 在每个 PP stage 内沿 `fsdp` 轴分片参数。 |

需满足 **`world_size == pp * fsdp * cp * tp`**。默认 `(pp, fsdp, cp, tp) = (2, 2, 2, 1)`，即 8 卡（四轴均为 2 需要 16 卡，见下方说明）。

```bash
cd examples/torch/llama3
export HYPER_PARALLEL_PLATFORM=torch
torchrun --nnodes=1 --nproc_per_node=8 pp_fsdp_tp_cp_sp_example.py
```

16 卡全开 TP+SP（`tp=2`）::

```bash
export HYPER_PARALLEL_PLATFORM=torch
LLAMA3_TP_SIZE=2 torchrun --nnodes=1 --nproc_per_node=16 pp_fsdp_tp_cp_sp_example.py
```

可选环境变量（默认见括号）：

| 变量 | 含义 | 默认 |
|------|------|------|
| `LLAMA3_PP_SIZE` | 流水线并行宽度 | `2` |
| `LLAMA3_FSDP_SIZE` | FSDP2 分片宽度（每个 PP stage 内的 decoder block） | `2` |
| `LLAMA3_CP_SIZE` | 上下文并行宽度 | `2` |
| `LLAMA3_TP_SIZE` | 张量并行宽度（含 SP） | `1`（8 卡）；16 卡可设 `2` |
| `LLAMA3_DEVICE_TYPE` | `npu` 或 `cuda` | `npu` |

约束：**`n_layers >= pp`**；**`n_heads` / `n_kv_heads` 能被 `tp` 整除**；**`seq_len % cp == 0`**；**`(seq_len / cp) % tp == 0`**；每 step 使用 **`micro_batch_num=1`**（与 PP+FSDP 调度兼容）。

> **8 卡说明：**`pp × fsdp × cp × tp = 8` 时至少一轴必须为 `1`。默认令 **`tp=1`**，PP/FSDP/CP 均为 2；`parallelize_llama3` 仍会调用（SP 在 tp=1 时为 no-op）。FSDP 仅包裹各 **decoder block**（不 wrap stage 根模块），避免 PP 调度器注入的显式 unshard/reshard 与 Torch autograd 冲突。若四轴均需 ≥2，请使用 16 卡 `(2,2,2,2)`。
>
> **`fsdp=1` 不可用：**当 `LLAMA3_FSDP_SIZE=1` 时，`fully_shard` 与 TP 的 DTensor mesh 维数不一致（`x_mesh_shape != w_mesh_shape`），8 卡下所有 `fsdp=1` 组合均会失败。请保持 **`fsdp >= 2`**。
>
> **8 卡已验证组合（`fsdp >= 2`，各跑 1 step 通过）：**
>
> | pp | fsdp | cp | tp | 说明 |
> |----|------|----|----|------|
> | 1 | 2 | 1 | 4 | 无 PP 切分，TP=4 |
> | 1 | 2 | 2 | 2 | 无 PP，TP+CP 均 2 |
> | 1 | 2 | 4 | 1 | 无 PP，CP=4 |
> | 1 | 4 | 1 | 2 | 无 PP，FSDP=4 |
> | 1 | 4 | 2 | 1 | 无 PP |
> | 1 | 8 | 1 | 1 | 无 PP，纯 FSDP8 |
> | 2 | 2 | 1 | 2 | PP+TP 均 2，无 CP |
> | 2 | 2 | 2 | 1 | **默认**，PP+CP 均 2 |
> | 2 | 4 | 1 | 1 | PP=2，FSDP=4 |
> | 4 | 2 | 1 | 1 | PP=4（每层一 stage） |

---

## 运行八：8 卡 PP + FSDP2 + CP

`pp_fsdp_cp_example.py` 在 3-D `DeviceMesh` `(pp, fsdp, cp)` 上同时启用三种并行：

| 维度 | 组件 | 说明 |
|------|------|------|
| `mesh["pp"]` | **PP** | `Schedule1F1B`；每 PP rank 一段 stage chunk（`pipeline.py`）。 |
| `mesh["cp"]` | **CP** | `ContextParallel` 挂在每个 `layer.attention.sdpa_core`。 |
| `mesh["fsdp"]` | **FSDP2** | `fully_shard` 在每个 PP stage 内沿 `fsdp` 轴分片参数。 |

需满足 **`world_size == pp * fsdp * cp`**。默认 `(pp, fsdp, cp) = (2, 2, 2)`，即 8 卡。

```bash
cd examples/torch/llama3
export HYPER_PARALLEL_PLATFORM=torch
torchrun --nnodes=1 --nproc_per_node=8 pp_fsdp_cp_example.py
```

可选环境变量（默认见括号）：

| 变量 | 含义 | 默认 |
|------|------|------|
| `LLAMA3_PP_SIZE` | 流水线并行宽度 | `2` |
| `LLAMA3_FSDP_SIZE` | FSDP2 分片宽度（每个 PP stage 内的 decoder block） | `2` |
| `LLAMA3_CP_SIZE` | 上下文并行宽度 | `2` |
| `LLAMA3_DEVICE_TYPE` | `npu` 或 `cuda` | `npu` |

约束：**`n_layers >= pp`**；**`seq_len % cp == 0`**；**`fsdp >= 2`**；每 step 使用 **`micro_batch_num=1`**（与 PP+FSDP 调度兼容）。

> **说明：**FSDP 仅包裹各 **decoder block**（不 wrap stage 根模块），避免 PP 调度器注入的显式 unshard/reshard 与 Torch autograd 冲突。`fsdp=1` 不可用（DTensor mesh_shape 不匹配）。

---

## 约束与说明

1. **序列并行：**激活在序列维使用 `Shard(1)`，**序列长度须能整除 TP 宽度**。
2. **注意力头数：** **`n_heads` 与 `n_kv_heads` 均须能整除 TP 宽度**（与 Colwise 切最后一维一致）。
3. **`parallelize_llama3`** 当前仅实现 **`enable_sequence_parallel=True`**；`enable_sequence_parallel=False` 未实现。
4. **训练循环：**仅对 **`loss.backward()` 与 `optimizer.step()`** 使用 `SkipDTensorDispatch`；**不要**把整个前向包进去，否则 TP 路径上 `DTensor` 行为可能异常。
5. **`fsdp_tp_example.py`：**在 DTensor + `fully_shard` 路径上调用 **`model.set_reduce_op_type("sum")`**；各 rank 使用**相同随机 batch** 做冒烟验证，确认组合栈能跑通，**不是**严格的单卡数值基准。
6. 配置为 **`Llama3DemoConfig` 教学用小模型**；数值与性能以本仓库 HyperParallel 实现为准。
7. **`tp_cp_example.py`** 使用 `model.py` 中的真实 **`Llama3Model` / `Llama3Attention`** 栈；SDPA 在 **`Llama3BshdSdpaCore`** 内执行，以便 `ContextParallel` 以 BSHD 布局挂接 `forward(q, k, v)`。未显式传入 `freqs_cis` 时，**`Llama3Model.forward(..., rope_seq_start=...)`** 会按 embedding 后的本地长度切 `self.freqs_cis`（CP 分窗与 TP 序列分片组合时必需）。
8. **训练步数：**所有示例默认 **10 step**（`demo_utils.TRAIN_STEPS` 或 ``LLAMA3_NUM_STEPS`` 环境变量）。

---

## 延伸阅读（API）

- 张量并行入口：`hyper_parallel` 包中的 `parallelize_module` 及各 `ParallelStyle` 子类（见源码与主项目文档）。
- 全分片训练：`hyper_parallel.fully_shard`（`core/fully_shard/api.py`）。
- 上下文并行：`hyper_parallel.ContextParallel`（`core/context_parallel/context_parallel.py`）。
