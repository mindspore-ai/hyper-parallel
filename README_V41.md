# DeepSeek V4.1 Flash 预训练：CP 与融合算子优化

本分支基于 `trainer_dev` 的 `22fb803ba3126f5a3b9c03de5cdcb004cb0ffc34`，
整理 V4.1 Flash 的 CP 通信、数值修复和融合算子适配。保留 CSA2 的
Full/Reindex/Reuse、r1/r2 压缩和现有 KL teacher 定义。

本轮聚焦预训练，实验显式关闭 candidate pool。生产代码保留原有 candidate 行为，
不优化后训练 candidate/restricted Reindex，不对 Indexer K 分块。

## 修改与原因

| 修改 | 原因与行为 |
|---|---|
| packed 几何准备后复用 | 将样本边界、positions、SWA 索引、压缩下界和 Indexer 分段保存为 batch 快照，多层复用；正常原地修改边界后重新准备，旧消费者保留原快照 |
| SWA raw KV 默认 all-to-all-v halo | SWA 只需要左侧 `window-1` 个远端 token，无需每层收集整个 raw KV 序列；支持窗口跨多个 owner、零收发及反向梯度 SUM |
| 公共 AG backward 使用 SUM reduce-scatter | 直接将梯度归约到对应 owner，替换完整 all-reduce 后切片；同步/异步接口均保持输入 dtype，不额外除以 CP 大小 |
| 修复 Omni scalar sink bridge | 避免从 softmax 总质量中减去占主导的 dummy 概率所造成的消减误差；保留 sink 的正确前向及梯度 |
| 普通 Indexer 接入 CANN LI V2 | 显式适配 r1/r2 的因果前缀、packed 样本边界和 CP 中间 rank 的全局坐标；不改变评分或选集算法 |
| 完整选集 KL student 接入 CANN KL | 使用当前 teacher 计算融合梯度；partial/empty 行仍走参考实现，各子集按 query 数恢复原始全局均值 |
| 修复参考 KL 数值合同 | teacher/student 在 FP32 下计算；teacher 全部下溢为零时，手写梯度与标量目标一致；保留 autograd 上游 seed |
| 安全索引排序 | key ID 和 sentinel 最大值不超过 `2**24` 时转 FP32 排序再还原整数；超界保留整数排序，避免大 ID 精度损失 |
| 限制 teacher 临时张量 | 小 bank 的 FP32 预转换同时受大小和收益条件限制；selected KV/head 分块，及时释放上一 Q chunk 的临时张量，避免无条件转换超长 K bank |
| 910B 省略可选 metadata | 当前 LI/NoMask KL 不传 `seqused_q/k`，可使用 host tiling；省略这两个 custom AICPU 调用，其它设备保留 metadata |

main KV 和 Indexer K 仍按 source 全局 all-gather。异步 halo 在消费前 `wait()`，
handle 保留发送 storage；即使 rank 0 没有接收数据，也保留反向通信边。
各层共享的 KV、TopK 和梯度状态不跨 forward 缓存。

## 当前使用的融合算子

| 阶段 | 实现 |
|---|---|
| 主 attention | Omni `npu_sparse_flash_attention_enhance` 及对应 backward |
| 普通 Indexer | ops-transformer `LightningIndexerV2` |
| KL teacher | selected main+sink 的 FP32 小算子，按 head softmax、跨 head 聚合、最终 L1 归一化并 detach |
| KL student 完整行 | ops-transformer `SparseLightningIndexerKLLossGrad` |
| KL partial/empty 行、TP Indexer/KL | 保留参考实现 |

本轮没有切换到 V4 `SparseFlashMla`，也没有使用它的另一种 teacher 定义。
不支持的 CANN 输入形状回到参考实现；已经选择的 kernel 运行异常直接报错，
不通过捕获异常隐藏算子错误。NPU 主 attention 需要可用的 Omni 包。

## 实测收益与范围

测试环境为 8 张 Ascend 910B3、PyTorch/torch-npu 2.10.0、CANN 9.0.0。
模型为六层、59,390,788 参数、hidden512、main H8/D512、Indexer H32/D128、
TopK512、BF16、micro-batch 1、packed、CP8、S8192、SGD momentum。
各方案统一使用 reference mHC post 和确定性模式。

以下比较的是**本分支外围优化前的冻结快照与当前实现**。冻结快照已经包含 halo、
SUM reduce-scatter、sink 修复和第一版 CANN 接入；因此这不是相对原始 upstream
的全部收益，也不是单独 all-to-all-v 的收益。

每个方案预热 5 步，记录 7 步，重复两轮；step 时间取各 rank 最大值后求中位数，
包括 forward、backward、参数梯度 SUM all-reduce 和 optimizer.step。
峰值取测量步及各 rank 的最大 allocated，profile 单独采集。

| 轮次 | 冻结快照 step | 当前 step | step 缩短 | 冻结 allocated | 当前 allocated | 峰值减少 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 701.16 ms | 563.54 ms | 19.63% | 1422.03 MiB | 1220.15 MiB | 14.20% |
| 2 | 660.70 ms | 557.53 ms | 15.62% | 1415.14 MiB | 1220.15 MiB | 13.78% |

两轮 baseline 有约 6% 波动，故同时报告两轮。两边 reserved 均为 1684 MiB；
缓存池大小不等于活跃 tensor 峰值，也不等于 `npu-smi` 的进程占用。
这些是减层模型的本地测量记录，不是正式大模型训练的性能保证或 CI 门槛。
原始日志、冻结实验快照和 profiler 文件留在开发环境，未加入本仓库。

### 尚存的主要热点

关闭 candidate 后，当前 rank 0 的一次 profile 显示：

| 项目 | 当前设备 kernel 累计耗时 | 状态 |
|---|---:|---|
| 索引排序 | AI Core 约 0.14 ms | 原约 107 ms 的整数 SortAiCpu 路径已消失 |
| Omni attention 前反向 | 约 199 ms | 内核未改，仍是主要热点 |
| GatherV3 | 约 56 ms | 临时显存下降，耗时基本未降 |
| 确定性 ScatterAddWithSorted | 约 69 ms | partial KL 参考梯度仍有开销 |
| 原生 LI / KL | 约 0.49 / 1.24 ms | 融合核心本身占比较小 |

同名 gather/scatter 是整步聚合，不是独立 KL microbenchmark。以上 kernel 累计时间
不能直接与完整 step 相加。rank 7 的 Omni 前反向约 44 ms，并出现较长 collective
时长；后者包括等待其它 rank，不能直接当作网络传输成本。
rank 间差异的具体原因仍需相同输入换卡等实验排查。

teacher 的大副本减少后，rank 0 的 allocator 快照峰值已从 teacher 转到 Omni
backward。下一轮优先处理 Omni 慢 rank/反向临时量，以及普通 partial KL 的融合与
teacher 读写；candidate 不在当前范围内。

## 精度验证

| 验证层级 | 结果与覆盖 |
|---|---|
| 单元测试和模型回归 | 34 项通过：KL 标量与梯度、sink、packed 快照与边界更新、排序整数边界、有界转换和 teacher 分块 |
| 单卡真实 CANN | 2 项通过：r1/r2、packed、CP 中间位置的因果几何，完整/partial/empty KL、全局均值与非单位 upstream seed |
| CPU/Gloo 与 NPU/HCCL CP8 | 6 项通过：halo 原语与梯度、共享链、公共 RS 的 Qwen 回归、三步预训练模型更新 |
| 补充 S8192 冻结快照对照 | 8 rank × 3 步的 logits、loss 和 154 个参数梯度逐位一致，比较本轮增量优化前后的融合版本 |

S4096 的六层 CP8 测试轮换 packed 边界，保留普通 Full/Reindex/Reuse 和 r2→r1，
使用相同初值和三步 SGD momentum：

- halo reference 相对 AG reference：输出、loss、全部梯度及最终参数逐位一致。
- halo fused 相对 AG reference：logits、loss、主干梯度、sink 梯度和最终参数一致；
  Indexer 梯度最大相对 L2 为 **0.14449%**，最大绝对误差 **7.45e-9**，
  参考梯度最大值约 **2.55e-6**。
- 三步中真实 LI/KL 调用合计 **90/57 次（8 rank 总计）**，确认执行了融合路径。
  融合重复运行得到相同误差。

0.14449% 是观测值，不是新设定的容差。V4.1 使用已有 dtype 容差检查，
不新增统一“3%”门槛；Qwen 公共 RS 回归单独沿用该测试原有容差。
Indexer 梯度很小，BF16 参数更新相同不能单独证明其正确性；独立 KL 测试使用较大
loss 系数、非单位 seed，并对照可微标量目标与梯度。

### 验证入口

在正确安装当前 checkout 和匹配算子包的环境中运行：

```bash
python -m pytest -q tests/ut/components/functional/test_compressed_indexer.py \
  tests/ut/components/modules/test_shared_compressed_dsa_attention.py \
  tests/ut/core/context_parallel/test_sequence_halo.py \
  tests/ut/auto_models/models/deepseek_v41
python -m pytest -q tests/torch/context_parallel/test_v41_cann.py
python -m pytest -q tests/torch/context_parallel/test_sequence_halo.py
```

ST launcher 自动启动 CP8 worker。模型 ST 固定为 S4096；S8192 表格是补充本地
实验记录。CPU UT 不要求 NPU；NPU 算子测试需要真实 Omni/CANN 安装。

## Review 与遗留问题

本轮 review 核查了 halo 发送 storage 生命周期、消费前等待、反向零收发参与、
重复 owner 梯度 SUM、公共 AG 调用点、packed 快照失效、sink 梯度、KL 子集均值、
teacher detach、索引精度边界和 metadata 条件。在已测预训练合同内，未发现新的
阻断性数值或通信问题；这不代替下列组合验收。

1. **partial KL 尚未融合。** NoMask 对负索引槽位的分母语义不能直接用于 partial
   行；新的 causal mask3 实验未纳入本版。arch22 传 `seqused_q/k` 时必须提供
   metadata，不能复用当前无动态长度接口的省略条件。
2. **通用 AICPU/网络服务初始化问题未修复。** 最小复现不需要 Indexer 数学计算，
   空 AICPU 入口也能失败。910B 省略可选 metadata 只规避本模型这两种调用；
   其它设备仍保留 metadata，多 group/FSDP 初始化顺序仍需验证。
3. **GAS 辅助梯度缩放合同不变。** 完整 Trainer、TP/FSDP/多通信组、长程收敛和
   大规模训练尚未完成验收。packed 目前要求 micro-batch 1、连续等长 CP 分片和
   与压缩率对齐的样本边界；不支持 KV cache 训练路径。
4. **公共 RS 的影响超出 V4.1。** 已有原语和 Qwen CP8 回归；V3.2、通用 attention
   wrapper 及其它并行组合尚无完整模型验收，不能把原语通过扩大为全部模型通过。
5. **候选池、TP Indexer/KL 保留参考实现。** CUDA、其它 NPU 型号及完整 VLM/Engram
   并行组合未在本轮硬件上完成验收。
6. **静态检查不是全绿。** 仓库旧 backend 禁令与 Torch-only 规则、UT marker 规则
   存在冲突，且有长函数复杂度告警。本次修改文件的 pylint 报告有 5 条问题，
   均在 `collectives.py` 的上游基线中复现；未增加屏蔽规则。
   数值测试通过不等于全部仓库 CI 已通过。

## 依赖与文件范围

算子接口、具体约束、定向构建命令和安装注意事项见
[CP 与 CANN 适配说明](docs/deepseek-v41-cp-cann.md)。
ops-transformer 固定源码为 `0b08075141913daeaf1e856faf7480ed393c8949`，
随附的 [Python 包装补丁](docs/patches/ops-transformer-selected-wrapper.patch)
只修复定向导出及相对导入，不修改 kernel 算法。

本分支仅提交生产代码、正式测试、README/适配说明及上述文本补丁。
本地分析目录、性能日志、profile/内存快照、环境脚本、虚拟环境、算子源码构建目录、
wheel、动态库、模型权重和认证信息均不在本次改动中。
