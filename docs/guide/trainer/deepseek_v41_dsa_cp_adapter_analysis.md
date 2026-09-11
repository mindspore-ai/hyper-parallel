# DeepSeek-V4.1-Flash CSA2/DSA 高性能模块与并行适配报告

## 1. 结论

DeepSeek-V4.1-Flash 的 Compressed Sparse Attention 2（CSA2）不能由现有
`mla_dsa_ulysses_cp_wrapper` 正确表达。本次实现以官方 V4.1 技术报告和推理源码为算法基线，
以 PanGu 可训练 DSA 为训练反向和并行实现参考，完成了：

- Full、Reindex、Reuse 三种 CSA2 模式；
- decoder Hierarchical Sparse Indexer 的候选块生成和候选域 Reindex；
- PanGu 风格、只更新 Indexer 的 selected-TopK KL 辅助训练；
- sample-level packed attention 隔离，且不构造 `[S,S]` 稠密 mask；
- Indexer Q/merge 的 TP head 切分、score all-reduce；
- raw KV、shared main KV、indexer K 的异步 KV-all-gather CP；
- Ascend sparse flash attention forward/backward 和 scalar sink；
- 16-die Online 4K 的 TP1/FSDP16/EP16 与 TP2/FSDP8/EP16 单步验证。

官方资料：

- [DeepSeek-V4.1-Flash 技术报告][v41-report]
- [DeepSeek-V4 论文](https://arxiv.org/abs/2606.19348)
- [DeepSeek-V3.2-Exp 官方实现](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)

## 2. 技术报告核对：哪些训练时必须接入

| 项目 | 报告证据 | 当前结论 |
| --- | --- | --- |
| Full/Reindex/Reuse | §2.3.1 定义三种静态模式；§4.2.1 给出 40 层排布 | 模型结构，训练与推理都需要；已接入 |
| Hierarchical candidate | §2.3.2 说明在 post-training 引入，训练和推理采用相同候选域 | post-training 必需；已接入 |
| sample-level mask / packed | §4.1/§4.2.2 | 预训练必需；已用 compact 边界接入 |
| Indexer 训练目标 | Top-K 是离散选择；V4.1 未公布损失公式和系数 | 随机 Indexer 不能只靠主 loss 训练；按 V3.2/PanGu KL 路径接入，系数保持可配置 |
| shadow indexer / PP payload | §3.1.2 专门用于共享源和消费者跨 pipeline stage | PP>1 必需；当前配方 PP1，不是本次单机验证阻塞项 |
| KV-cache / bounded replay | §3.2 是推理和 rollout 状态管理 | 普通整段训练不需要 runtime KV-cache；未接入是正确边界 |
| FP4 main KV / Indexer QK QAT | §2.4.4 | 正式 post-training 精度需要；当前 BF16 crop 未实现 |

V4.1 §4.2.2 还说明：模型从 64K 开始直接用 sparse attention 训练，在 34T token 时扩到 1M，
没有 dense-attention warmup。当前 4K 用例只是结构、梯度和混合并行 smoke，不是官方训练日程复现。

### 2.1 四层裁剪如何覆盖 decoder 的 Reindex

正式 40 层结构如下：

- encoder 0–19：前两层 SWA；其余按三组六层组织，每组 `Full + 5×Reuse`，压缩率 2；
- decoder 20–39：五组四层，第一组 `Full + 3×Reuse`，其余四组
  `Reindex + 3×Reuse`，压缩率 1；
- decoder 第一个 Full 为 candidate source；每个 query 最多选 2,048 个 block，
  每 block 8 个位置，后续 Reindex 在最多 16,384 个候选位置中选 Top-512。

四层裁剪不可能保留 encoder 20 层后再保留 decoder 20 层。因此验证配方把正式 decoder 的
`Full -> Reindex` 角色映射到 crop layer 2 -> 3，并把压缩率保留为 2，以同时覆盖压缩感知因果边界。
4K 下把 candidate blocks 缩到 128，使候选位置为 1,024，仍大于最终 Top-512；这属于一致缩放，
不是把 Reindex 退化为 Full。

## 3. 单卡算法和高性能实现

### 3.1 CSA2 数据流

```text
hidden[S]
  ├─ current-layer raw SWA K=V projection ───────── raw KV[S]
  ├─ current-layer main Q projection ────────────── Q[S,H]
  └─ Full only: non-overlap compressor ──────────── main KV[S/r]
       └─ project indexer K

Full:    local index Q × shared index K over all visible positions
         -> Top-K + block-max candidate pool
Reindex: local index Q × shared index K over candidate positions only
         -> fresh Top-K
Reuse:   reuse latest Top-K directly

output = SparseFlashAttention(Q, raw KV || selected main KV, scalar sink)
```

压缩位置 `j` 代表原序列 `[j*r,(j+1)*r)`，query `i` 只能看到：

```text
j < floor((i + 1) / r)
```

该边界与普通 Q/K 同坐标 DSA 不同。

### 3.2 Hierarchical candidate 和 Reindex

`compressed_causal_topk_and_candidates` 在 Full 层的一次 score pass 中同时得到：

1. 当前层 Top-K；
2. 每个 block 的最大 score；
3. 最高分 candidate block id。

跨层状态保存 compact block id，而不是展开后的 position mask，状态宽度缩小 `block_size`
倍。Reindex 的 `compressed_candidate_topk` 只 gather 候选位置的 indexer K 并打分，不再构造
完整 `S × S/r` score。所有 score 路径按 query chunk 执行，限制临时峰值。

当前实现还把最新可见的未满 block 固定纳入候选集。这与官方推理源码的 causal candidate
处理一致，避免 block 尚未闭合时完全丢失最近位置。

### 3.3 Indexer KL 辅助训练

V4.1 报告没有公开 Indexer loss 的精确公式和系数，但 Indexer Top-K 选择不可微；随机初始化
训练时只执行离散 Top-K，主 attention loss 无法更新 Indexer Q/K/merge。为使裁剪模型成为
真正可训练结构，本次参考 PanGu 的：

- `pangu_dsa_self_attention.py`：detach Indexer 输入；
- `pangu_dsa_torch.py` / `pangu_dsa_core_attention.py`：用主 attention 分布作 teacher，
  只在 selected Top-K 上计算 KL，并预计算 Indexer 梯度。

实现中的 `_SharedCompressedIndexerKLLoss`：

- teacher 主 attention Q/K 和 scalar sink 全部 detach；
- 只保存 Indexer Q、K、merge 的预计算梯度，避免保留完整 teacher graph；
- 通过 `aux_loss_auto_scale` 把标量 loss 注入主图；
- TP 下先归并 index score 和 teacher 的 head sum，再计算一致的 KL；
- 单测将自定义 backward 与直接 autograd 的 loss/gradient 逐项比较。

配方中的 `indexer_loss_coeff=0.001` 仅为验证值。正式训练必须用内部训练配置或经实验重新标定，
不能把该值解释成 V4.1 官方超参数。

### 3.4 packed sample-level masking

Online collator 为每个样本输出 `cu_seq_lens`，并把每个样本长度补齐到 encoder 最大压缩率 2。
补位 token 的 label 为 `IGNORE_INDEX`。这样 compressor group 不会跨样本混合。

模型侧 `SharedCompressedPackedSequence` 仅携带：

- 全局 sample 边界；
- 当前 CP shard 的起点和长度；
- 全局序列长度。

attention 根据每个 query 的 sample start 计算最小可见 raw/main-KV index，Engram 用同一边界
阻止 n-gram 跨样本。整个流程不创建 `[S,S]` mask，适合继续扩到 64K/1M；当前仅验证了 4K。

### 3.5 NPU attention 路径

| 路径 | 实现 |
| --- | --- |
| Indexer | FP32 chunked matmul、ReLU、head merge、Top-K |
| sparse forward | `npu_sparse_flash_attention_enhance`，K=V，不展开 dense mask |
| sparse backward | `npu_sparse_flash_attention_grad_enhance`，合并 K/V gradient |
| scalar sink | 通过 sparse softmax max/sum 精确合并概率质量，并显式返回 sink gradient |
| CPU/GPU reference | 纯 Torch dense reference，仅用于数值测试 |

日志中的整数 `ArgSort` AiCPU 提示来自通用 EP dispatcher 的专家排序。新 Indexer 的 Top-K
没有调用整数 argsort，不能把该提示归因于 CSA2。

## 4. TP/CP/EP/FSDP 切分

### 4.1 TP

| 参数/张量 | TP placement | 通信 |
| --- | --- | --- |
| main attention Q/O heads | head-sharded | 常规 row/column parallel |
| Indexer `q_b_proj` | column shard on indexer heads | 无需 gather Q |
| Indexer `weights_proj` | column shard on merge heads | 无需 gather merge |
| Indexer `wk/k_norm` | replicate | K 无 head 维，避免复制 K 激活后重复存储权重切片 |
| Indexer score | partial sum | Top-K 前 all-reduce SUM |
| main KV / indexer K | replicate on TP | 与所有 local Q heads共享 |

即使 TP=1 也必须声明该边界：replacement、FSDP owner、参数覆盖检查和 TP2 的同配方升级都依赖
同一契约。size=1 的 `Replicate` 是恒等 placement，不产生通信。

### 4.2 CP

CP rank 持有连续 query shard。Q、index Q、Top-K 和 output 留在本地；raw SWA KV、main KV、
indexer K 执行 sequence all-gather。模块控制异步 launch/wait：

```text
raw KV ready        -> launch gather
local Q projections
main KV ready       -> launch gather
indexer K ready     -> launch gather
local index Q/merge
wait indexer K      -> Full/Reindex selection
wait raw/main KV    -> sparse attention
```

反向由可微 collective 汇总 global KV gradient 并取回 local shard。packed 边界携带 global/local
几何，CP query offset、压缩 causal boundary 和 sample boundary 同时生效。

### 4.3 TP2 为什么没有增加单卡优化器状态

16 个物理 rank 下：

- TP1 使用 `FSDP16 × TP1`；
- TP2 使用 `FSDP8 × TP2`；
- dense TP-sharded 参数的总分片因子都为 16；
- routed experts 和 Engram table 把 dense rank domain 展平为 virtual EP16；
- mHC、Engram fusion、Indexer K 等 TP-replicated 小参数由 FSDP8 切分，并按 source-layout
  gradient domain 去重/归并。

因此不能仅看到 `dp_shard_size: 8` 就推断优化器状态翻倍。最新实测 TP2 peak allocated
为 43.9169 GB，低于 TP1 的 50.0054 GB。HSDP/Muon 日志中的几十 billion `group_numels`
是 DTensor 全局逻辑 shape，不是单卡常驻参数量。

## 5. 为什么 `mla_dsa_ulysses_cp_wrapper` 不适配

### 5.1 坐标和状态模型不同

旧 wrapper 假定 Q/K/V 属于同层和同一序列坐标；CSA2 同时存在 `S` 的 raw SWA KV、
`S/r` 的 shared main KV、跨层 indexer K、candidate blocks 和 Top-K。仅修改
`actual_seq_qlen/kvlen` 不能表达 `floor((i+1)/r)` 的逐 query 因果边界。

### 5.2 Ulysses sequence-to-head 与 TP head shard 冲突

旧 wrapper 让 CP 临时接管 attention/indexer head 维。CSA2 已由 TP 切 attention heads 和
Indexer Q heads；再次 sequence-to-head 会改变 sink、grouped output projection 和跨层共享
Top-K 的解释。PanGu 对 DSA 同样选择 KV-all-gather CP，并拒绝 Ulysses CP。

### 5.3 生命周期不同

旧 wrapper 是逐层即时 DSA。CSA2 Full 发布 main KV、indexer K、candidate blocks 和 Top-K，
后续 Reindex/Reuse 消费同一 forward/micro-batch 的状态。逐层 wrapper 会重复 gather，或者把
local Top-K 错当 global index。PP>1 时还需要报告 §3.1.2 的 shadow indexer、pipeline payload
和 micro-batch state manager，不能由函数级全局 monkey patch 解决。

### 5.4 sink 和 batch 输入职责不同

旧路径面向参数化 sink key/value 并依赖 rescale；V4.1 是每 head 一个 scalar logit、sink
value 恒为零。旧 wrapper 还可能自行切 sequence，而当前 `ParallelBatch` 已统一切 input、label、
position 和 compact packed metadata，再切一次会破坏 Engram n-gram 边界。

## 6. 实现与验证

主要文件：

| 文件 | 作用 |
| --- | --- |
| `shared_compressed_dsa_attention.py` | candidate/Reindex、KL、packed、NPU sparse attention、TP/CP context |
| `hyper_parallel/models/deepseek_v41/modeling_deepseek_v41.py` | V4.1 参数结构、Full/Reindex/Reuse 调度、跨层状态 |
| `hyper_parallel/models/deepseek_v41/adapter/context_parallel.py` | TP score reduce 和异步 KV CP wrapper |
| `hyper_parallel/models/deepseek_v41/adapter/packed_sequence.py` | `ParallelBatch` 到 compact packed metadata 的适配 |
| `hyper_parallel/models/deepseek_v41/adapter/registration.py` | replacement 和 placement 注册 |

验证结果：

```text
18 passed  tests/ut/auto_models/models/deepseek_v41/test_deepseek_v41_crop.py
1 passed   tests/torch/context_parallel/test_deepseek_v41_dsa_cp.py
           (2-process Gloo: packed CP + TP Full/Reindex/KL gradient parity)
1 passed   tests/torch/expert_parallel/test_engram_parallel.py
4 passed   tests/ut/auto_models/_transformers/test_config_resolver.py

16-die NPU Online 4K, TP1/EP16/FSDP16:
  loss=12.8242, grad_norm=67.3168, step=19.8277s
  max_allocated=50.0054GB, max_reserved=60.0371GB

16-die NPU Online 4K, TP2/EP16/FSDP8/SP2:
  loss=12.6250, grad_norm=67.3998, step=15.9357s
  max_allocated=43.9169GB, max_reserved=58.9805GB
```

TP1 首次尝试在 backward 已完成后、既有 `clip_grad_norm_` communicator 懒初始化时遇到一次
HCCL `EJ0003` 端口占用；所有进程清理后，同代码重跑完整通过。该异常不是模型算子或 shape
错误，但正式长跑环境仍应避免并发作业复用 HCCL 端口范围。

## 7. 尚未声称完成的边界

1. **FP4 QAT**：正式 post-training 需要 main KV 的 RoPE 后 E2M1 fake quant（每 16 channel
   一个 E4M3 scale）以及 Indexer Q/K 的 MXFP4 QAT。公开报告未给出完整训练算子，PanGu 当前
   通用定点 QAT 也不等价；本次 BF16 crop 未伪装为生产 FP4 QAT。
2. **PP>1 attention sharing**：需要 shadow indexer、pipeline payload 和 micro-batch shared-state
   生命周期；当前目标拓扑 PP1，不影响本次验收。
3. **runtime KV-cache**：decode、persistent cache、SWA bounded replay 属于推理/rollout，不属于
   当前整段 backbone 训练 forward。
4. **正式长序列性能**：4K 单步验证了功能，不代表 64K/1M candidate state、CP overlap 和稳定
   吞吐已经验收；需要多步 profile 和 timeline。
5. **正式精度**：本地 checkpoint 仍是 LFS pointer，当前是随机初始化裁剪模型；不能替代真实
   权重、固定 batch、resume 和 TP1/TP2 loss/gradient 精度对齐。

[v41-report]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf
