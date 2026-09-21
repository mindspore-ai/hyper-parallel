# DeepSeek-V4.1 MegaMoe 适配说明

## 变更范围

本实现将 `megamoe-deepseek-v4.1` (`e073f9ff`) 的 DSV4.1 MegaMoe 改动移植到
`megamoe-push-pull` (`d440ee6a`)，保留接收分支的 FSDP 和 activation memory
架构。源分支以 `upstream/trainer_dev` (`162aa8e1`) 为基线；下文的 NPU 精度和
整网性能数据是源分支的历史验收记录，不代表移植后的重新验收。四层训练 crop
用于验证完整训练链路，不能等同于官方 40 层模型。

MegaMoe 部分主要继承自 `megamoe-push-pull` (`2632ba92`) 的代码线，随后重放
到 `trainer_dev` 基线。由于重放时基线和周边平台代码不同，提交 SHA 已变化，
但核心 patch 可以按以下代表性映射核对：

| `megamoe-push-pull` | `megamoe-deepseek-v4.1` | 内容 |
| --- | --- | --- |
| `f7439d98` | `f3237463` | SHMEM one-sided runtime |
| `83be5c96` | `acb12a7f` | workspace/大任务图和显存优化 |
| `fda6c58c` | `2640a41d` | MegaKernel profiling |
| `ea0e29b7` | `d1f8ba84` | permutation gradient 的 ACLNN 适配 |
| `8ade7bb6` | `8a90fab8` | MegaMoe memory savings |
| `1b03f88c` | `c852d237` | push/pull transport |
| `5cb09343` | `343c9a8f` | 移除固定 expert 上限 |
| `148bc6a3` | `ba4c4f4a` | subgroup SHMEM bootstrap |
| `c1334800` | `9ecab8b4` | push/pull 集成测试 |
| `2632ba92` | `56223e9a` | optimizer parameter-copy 回退 |

在这条 MegaMoe 代码线上，源分支再叠加 DSV4.1 FP32 oracle、算子定位、
Trainer/FSDP replacement 和四层训练 benchmark；这些功能及后续 API 地址缓存、
profiling 容量修复一并移植。clipped SwiGLU 及其 native 构建依赖未纳入当前分支。

在显式选择无截断 SwiGLU 的验证配置下，当前实现保持 TopK router、shared
expert、attention、Engram、indexer、mHC 和优化器配置。TP/CP/PP 仍限制为 1，
EP 可以是 WORLD 或显式子组。

## 适配方式

### 独立 MoE block

`DeepseekV41MegaMoe` 保留源 block 的 router 和 shared experts，只替换 routed
experts。构造前要求完整 HF expert 权重，按 EP group-local rank 截取连续专家，
一次性将 HF 布局
`[E, 2I, H]`/`[E, H, I]` 转为 native 布局
`[E_local, H, 2I]`/`[E_local, I, H]`。运行时不复制旧参数，native executor
读取当前参数。

### Trainer/FSDP

`DeepseekV41TrainingExperts` 在参数分片和 optimizer 创建前完成 replacement，
保留 `gate_up_proj`、`down_proj` 名称，通过 `WeightConverter(Transpose)` 完成
checkpoint 双向布局转换。原 `mlp.experts` 仍是参数唯一持有者和嵌套 FSDP
单元；每次 forward 使用当前 unshard 参数，不缓存过期 view。

`deepseek_v41_megamoe_compute_fn` 只替换整个 routed 分支，router 和 shared
branch 仍由 DSV4.1 adapter 执行。所有兼容层在首次 native forward 前登记静态
规格，串行层可以共享一个 execution workspace；关闭必须在所有 backward 完成
后按层有序调用 `close()`。

MegaMoe recipe 与 `trainer_dev` 的数据接口保持一致：`DeepseekV41BatchAdapter`
统一负责 physical token cost、压缩对齐和 compact attention runtime metadata；
collate 和 `ParallelBatch` 不再接收旧的 `sample_alignment` 或嵌套 runtime adapter。
性能 benchmark 重建 dataloader 配置时也保留同一个 model-owned batch adapter。

### SwiGLU limit

MegaMoe 仅支持普通 SwiGLU。DSV4.1 released 配置的 `swiglu_limit=10`
不能直接接入；adapter 会在构造时拒绝正数 limit，避免静默改变计算语义。
独立验证和训练 crop 显式使用 `swiglu_limit=0`，routed/shared 两支均使用
无截断激活。该配置用于集成验证，不等同于原始 checkpoint 的激活语义。

### EP 梯度归约

owner EP 的 dispatch 将 token 发到 expert owner，owner 对收到的全局 token 计算
该 expert 的输出和 dW；同一 expert 的 dW 不再额外做一次 EP all-reduce。Trainer
中的 expert 副本归约沿用既有 EDP/FSDP mesh，通信 dtype 仍由框架的
`reduce_dtype` 控制。MegaMoe 负责 dispatch、owner compute、combine，不改变
通用 optimizer 或 dense DP 梯度归约。

### Push / pull

`MegaMoeExperts(..., dispatch_mode="push" | "pull")` 在构造时选定模式，所有
EP rank 必须一致，且不同模式不共享同一 workspace。

- `push`：按 lossless 最大接收容量分配对称 SHMEM receive 区，使用 PUT dispatch。
- `pull`：SHMEM 保存本地 source，接收 rows 使用普通 HBM，dispatch 使用 GET；
  combine 仍走 PUT。
- push 和 pull 的 dispatch/combine 通信任务均固定为 128 行，不根据接收负载
  构建或切换其他 plan。
- 每次独立 torchrun 使用新的
  `HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT`，进程组销毁前显式关闭所有
  MegaMoe executor。

## 精度验收

精度入口为
`hyper_parallel.core.multicore.examples.mega_moe.deepseek_v41_precision`，
Trainer 入口为
`examples.training_demo.benchmark_deepseek_v41_megamoe`。两条路径都固定初始
权重、输入和离散 route；比较 output、selected IDs、dX、router/shared 梯度、
每个 expert 梯度以及一步 optimizer update。

主判据是同状态独立 CPU FP32 oracle：每个 tensor 的 relative L2 不超过 1%，
最大绝对误差除以 reference 最大值不超过 2%；同时要求 shape、finite、梯度
存在性、初始权重和 route IDs 精确一致。HF BF16 逐元素比较仍作为诊断保留，
但不通过直接放宽阈值来掩盖 BF16 舍入差异。

CPU 回归覆盖 DSV4.1 model adapter、Multicore、replacement 和 oracle。
移除 clipped SwiGLU 后须重新验证当前配置；源分支正数 limit 的设备结果
不适用于当前分支。NPU 结果应记录源码 SHA、Torch/torch-npu/Transformers、
CANN 和 native payload hash，避免误用旧制品。

## 性能与 HBM 测试方法

性能比较只使用 fresh-process、同一 canonical rank-local BF16 权重和确定性
输入。EP8/E48 的含义是全局 48 个 routed experts、每卡 6 个；原始尺寸裁剪保持
`H=5120`、`I=2304`、`TopK=6`、每卡 4096 tokens，四层用于整网链路验证。

### 整网

先由 owner EP 写一次权重，再对 owner、MegaMoe push、MegaMoe pull 分别启动独立
torchrun。稳定测试建议 `warmup=8`、`steps=10`、`schedule_steps=18`：

```bash
MODEL_DIR=...
ENGRAM_ASSETS=...
WEIGHTS_DIR=...
RESULT_DIR=...

python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m examples.training_demo.benchmark_deepseek_v41_megamoe \
  --model-dir "$MODEL_DIR" --engram-assets "$ENGRAM_ASSETS" \
  --backend owner_ep --experts 48 --tokens 4096 \
  --warmup 8 --steps 10 --schedule-steps 18 \
  --weights "$WEIGHTS_DIR" --output "$RESULT_DIR/owner"

python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m examples.training_demo.benchmark_deepseek_v41_megamoe \
  --model-dir "$MODEL_DIR" --engram-assets "$ENGRAM_ASSETS" \
  --backend megamoe --dispatch-mode push --experts 48 --tokens 4096 \
  --warmup 8 --steps 10 --schedule-steps 18 \
  --weights "$WEIGHTS_DIR" --output "$RESULT_DIR/megamoe-push"

# pull 只需将 dispatch-mode 改为 pull，并使用独立 SHMEM endpoint/output。
```

每步在前后同步 NPU，按各 rank 最大 step time 汇总 mean/median 和 global
tokens/s。不要删除慢样本；出现外来或无法归属 NPU 进程时，整组 ABBA 失效并重跑。
MoE 加速比用相同输入和权重的模块边界诊断或
`deepseek_v41_benchmark` 计算，不能把单层 MoE 倍率直接外推为整网倍率。

### HBM 口径

结果 JSON 中的 `peak_allocated_bytes` 和 `peak_reserved_bytes` 是
`torch.npu.max_memory_*` 的 allocator 峰值；它们不包含所有 SHMEM/外部 allocator，
不能直接称为整卡 HBM。测试期间另用 `npu-smi info` 或 `npu-smi info -t proc-mem`
按秒采样每张卡的 HBM used/total 和进程归属，报告两类数值：Torch allocator
峰值，以及卡侧物理 HBM 峰值。两者必须注明采样边界和是否含 SHMEM heap。

### 当前性能边界

源分支使用 clipped SwiGLU 的 DSV4.1 整网和模块计时不适用于当前实现。
当前 push/pull 性能需要使用重新构建的 payload 和相同无截断配置实测。
Qwen 默认使用普通 SwiGLU，可以独立比较两种通信路径。

## 验收命令

```bash
HYPER_PARALLEL_PLATFORM=torch python -m pytest -q \
  tests/ut/auto_models/models/deepseek_v41 \
  tests/ut/core/multicore \
  tests/ut/auto_models/trainer/test_mixed_precision_optimizer.py

git diff --check
```

NPU block、Trainer smoke、checkpoint round-trip、push/pull 和性能结果分别归档；
一个通过的 block oracle 不代表完整 40 层训练收敛，也不代表 TP/CP/PP/VLM 已验收。
