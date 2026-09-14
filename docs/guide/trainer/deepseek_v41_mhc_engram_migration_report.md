# DeepSeek-V4.1 mHC / Engram 迁移与并行切分报告

## 1. 结论

本次迁移在 HyperParallel 中落地了两个可复用组件：

- `PipelinedMhcModule`：保持 DeepSeek-V4.1 跨子层流水式 mHC 语义，复用现有高性能 Sinkhorn 和 mHC post 算子。
- `EngramModule`：实现 V4.1 n-gram 哈希、压缩表查询、门控融合，以及 TP sequence parallel、CP 左上下文对齐、EP 稀疏行查询。

四层裁剪模型已经在 16 die 上完成 Online 4K 一步训练：

| 用例 | 物理网格 | SP | 结果 | 峰值 allocated | step time |
| --- | --- | --- | --- | ---: | ---: |
| TP1 | FSDP16 × TP1，虚拟 EP16 | 关闭 | 通过 | 50.0054 GB | 19.8277 s |
| TP2 | FSDP8 × TP2，虚拟 EP16 | 开启 | 通过 | 43.9169 GB | 15.9357 s |

16 die 无法同时构成彼此独立的 FSDP16 × TP2；该组合需要 32 个 rank。TP2 下使用 FSDP8，专家权重和 Engram 表仍把 `DP/FSDP8 × TP2` 展平为虚拟 EP16，因此它们的单卡分片数没有下降。

## 2. 分析基线

| 项目 | revision |
| --- | --- |
| HyperParallel 基线 | `08901cba5dbed4a849970c9ded8482cb974ab79a` |
| DeepSeek-V4.1-Flash | `dba1be0a40aa45a94ad051997016db3960a90277` |
| PanGu Megatron 参考 | `e53ba7f05c6936acf2f8e19d5f3ee4150daa0922` |

主要参考实现：

- V4.1 算法：`DeepSeek-V4.1-Flash/inference/model.py` 和 `inference/engram.py`。
- PanGu mHC：`PanGu/models/transformer/pangu_mhc.py`、`pangu_transformer_layer.py`。
- PanGu Engram：`PanGu/models/transformer/engram/`。
- PanGu TP 语义：`PanGu/distributed/tensor_parallel/layers.py` 中的 `SequenceParallelLinear`。

本地 Hugging Face 仓库只有源码、配置、tokenizer 和 Git LFS 指针。48 个 safetensors 文件不是实际权重，因此当前验证是按正式结构随机初始化的裁剪模型，不能作为预训练权重精度结论。

## 3. 为什么 TP=1 仍要声明模块边界

边界声明不是“只有 TP>1 才执行一次切分”的开关，而是模型并行契约。TP1 时仍需要它，原因如下：

1. **完整参数覆盖**：planner 会检查每个可训练参数是否恰好属于一个边界。V4.1 新增的 `fn/base/scale` 和 Engram 参数若没有边界，会在 TP1 就暴露漏接入问题。
2. **拓扑无关配方**：同一份配方先在 TP1 验证参数归属和替换，再切到 TP2；否则 TP2 会引入结构变化和并行变化两个变量。
3. **替换入口**：mHC/Engram 的 module replacement 和 Engram local compute 都挂在边界上。TP1 不分片，但仍需安装正确 forward。
4. **FSDP owner**：边界也决定 FSDP 包装、参数生命周期和预取单元，而不只描述 TP placement。
5. **语义审计**：显式写成 `tp: replicate` 表示这是分析后的选择；它与“planner 没识别到所以碰巧没切”不同。

因此，TP1 中声明 mHC/Engram 边界是正确的；`Replicate` 在 size=1 的 TP 轴上是恒等 placement，不产生 TP 通信。

## 4. 单卡算法分析

### 4.1 DeepSeek-V4.1 流水式 mHC

设残差流为 `X ∈ [B,S,C,H]`，其中 `C=hc_mult`。每个 attention/FFN mHC 参数组包含：

- `fn ∈ [(C+2)C, CH]`
- `base ∈ [(C+2)C]`
- `scale ∈ [3]`

先对 `X.flatten(C,H)` 做 RMSNorm，再线性投影并切为：

- `pre ∈ [B,S,C]`：`sigmoid(pre * scale[0] + bias) + eps`
- `post ∈ [B,S,C]`：`2 * sigmoid(post * scale[1] + bias)`
- `combine ∈ [B,S,C,C]`：仿射后执行 Sinkhorn，得到近似双随机残差混合矩阵

子层输入和输出分别为：

```text
collapse(X, pre) = Σ_c pre[c] · X[c]
post(Y, X)       = post[c] · Y + Σ_j combine[j,c] · X[j]
```

V4.1 的关键不是公式本身，而是系数的消费时序：

```text
上一层 FFN pre -> 当前 attention collapse
当前 attention pre -> 当前 FFN collapse
当前 FFN pre -> 下一层 attention collapse
```

PanGu 当前的 `MhcPreModule` 在同一次调用中同时生成 `h_pre` 并立即 collapse，现有 HyperParallel `MhcPreModule` 自定义算子也遵循这一时序。直接替换会把 V4.1 的跨子层流水依赖改成当前子层依赖，算法不等价。

本次采用的迁移方式是：

- 新增 `PipelinedMhcModule`，保留源模块 `input_norm/fn/base/scale` 的对象身份和 state-dict key，仅负责输出三组系数。
- Sinkhorn 在 NPU 自定义算子可用时走高性能路径，CPU/通用 Torch 走 `sinkhorn_knopps`。
- `pipelined_mhc_post` 复用现有 `mhc_post` 高性能实现。
- 不强行复用现有 `MhcPreModule`，因为其 fused 接口不能无损表达“本层生成、下一子层消费”的 `pre`。

### 4.2 Engram

Engram 单卡算法分为三段。

#### n-gram 哈希

1. 用 tokenizer 归一化表把 token id 压缩；不可参与哈希的 token 映射为 dead token。
2. 对每个位置构造 2-gram 到 `max_ngram_size`-gram；越过序列或 packed segment 左边界的位置用压缩后的 pad id 补齐。
3. token 与每层确定性奇数 multiplier 相乘，再逐级 XOR。
4. 每个 n-gram/head 用各自 prime 取模，加前缀 offset，得到互不重叠的全局表行号。

#### 表查询

哈希结果形状为 `[B,S,(N-1)×heads]`，每个行号查一个 `head_dim` 向量，拼接后送入 `wkv`。

#### 门控融合

`wkv` 输出一组共享 value 和每条 mHC stream 的 key：

```text
[key, value] = WKV(concat(embedding_rows))
key          ∈ [B,S,C,H]
value        ∈ [B,S,H]
```

对每条 stream 计算归一化相关性：

```text
dot  = sum_h(hidden * q_weight * k_weight * key)
dot *= rstd(hidden) * rstd(key) / sqrt(H)
gate = sigmoid(signed_sqrt(clamp_abs(dot)))
out  = hidden + gate * value
```

实现用 `where(dot < 0, -root, root)` 表达与 `copysign` 等价的 signed sqrt，避免 Ascend 上 `copysign` 回退 CPU；`dot=0` 时仍选择正根，和源实现一致。

## 5. PanGu 并行语义与 HyperParallel 映射

### 5.1 TP

PanGu 的 `SequenceParallelLinear` 明确分配完整 `[out,in]` 权重，没有设置 tensor-model-parallel shard 属性；它只给参数标记 `sequence_parallel`，让每个 TP rank 对自己的 token 段做相同线性变换并同步权重梯度。

因此按 PanGu 迁移时：

- mHC `fn/base/scale`：TP replicate。
- Engram `wkv/q_weight/k_weight`：TP replicate。
- TP2 必须开启 sequence parallel，才能让 replicated projection 处理互不重叠的 2K token 段，而不是两个 rank 重复处理完整 4K。
- Engram 先依据 TP rank 对 full `input_ids` 选取本地 token 窗口，并补最多 `max_ngram_size-1` 个左侧 token，保证 TP 分界处的 n-gram 与单卡一致。

把 `wkv` 改成列并行并非零成本替换：key/value 的 `H` 维会变为局部分片，当前每条完整 mHC stream 的门控和 residual add 需要额外 all-gather 或全面改成 hidden-sharded 数据流。PanGu 选择 token 维 SP + 权重复制，避免为每层 Engram 交换 `[B,S,5H]` 级别的激活。

### 5.2 EP

PanGu 的 Engram table 在 `tp_ep` group 上按行切分，并执行两次变长 all-to-all：

1. 根据 `owner = global_row // rows_per_rank` 稳定排序请求。
2. all-to-all 把全局 row id 发给 owner。
3. owner 做本地 embedding lookup。
4. 可微 all-to-all 把 value 返回请求方，再恢复原顺序。

HyperParallel 使用同样语义：`embed.weight: {ep: shard(0)}`，并把 16 die 的 dense region 展平为 expert mesh `{'edp_shard': 1, 'ep': 16}`。planner/source-shard 不再依赖 `experts.*` 名称，而是按 `EP: Shard` placement 识别任意虚拟 EP 参数。

裁剪表的逻辑行数为 100,776；为 EP16 补到 100,784，每卡 6,299 行。哈希永远只产生逻辑范围内的行，8 个 padding 行仅用于等分。

### 5.3 CP

mHC 和 Engram 融合都是 token-wise 算子，CP 不切权重。CP 的唯一额外语义来自 n-gram 左依赖：CP rank 的首 token 需要前一 rank 最多 `N-1` 个 token。

当前 `EngramModule` 为正确性先 all-gather 轻量 `input_ids/segment_starts`，再只对本地目标窗口加左 halo 并哈希；hidden activation 不做 CP all-gather。分布式测试覆盖了 CP 分界处输出、hidden grad 和 EP 本地表 grad 与单卡参考一致。

完整四层模型现在使用共享压缩注意力专用的异步 KV-all-gather CP，而不是 Ulysses。
Engram 组件级 CP 已完成分布式数值验证，`TP1 + CP2 + EP16 + FSDP16` 端到端 Online 4K
也已完成 16-die NPU 单步 smoke。

### 5.4 参数 placement 总表

| 参数 | TP | CP | EP/FSDP | 原因 |
| --- | --- | --- | --- | --- |
| mHC `fn/base/scale` | Replicate | Replicate | dense FSDP | full-hidden、token-wise；匹配 SequenceParallelLinear |
| Engram `wkv/q/k` | Replicate | Replicate | dense FSDP | token SP，避免大激活 all-gather |
| Engram `embed.weight` | 不再额外切 | 不再额外切 | virtual EP Shard(0) | TP rank 已包含在 EP16 中，避免重复切同一物理轴 |
| routed expert weights | TP 转入 virtual EP | Replicate | virtual EP Shard(0) | 384 experts / EP16 = 24 experts/rank |
| CSA2 Indexer `q_b/weights` | Column shard | Replicate | dense FSDP | Indexer head 局部计算，score 在 Top-K 前归并 |
| CSA2 Indexer `wk/k_norm` | Replicate | Replicate | dense FSDP | 单份共享 index K 没有可切 head 维 |
| attention/共享 expert 常规矩阵 | Column/Row shard | Replicate | FSDP8/16 | 标准 TP + FSDP |

## 6. TP2 显存问题复盘

### 6.1 `54,358,179,840` 不是单卡参数量

Muon 的 `[HSDP Batch] group_numels` 使用 DTensor 全局逻辑 shape 汇总。四层 routed experts 的逻辑参数量是 54.358B，但在 TP1 和 TP2 中都按 EP16 分片，单卡约 3.397B；该日志不能用来判断专家权重是否漏切。

对四层 meta model 的 planner placement 做完整盘点：

| 类别 | 全局逻辑参数量 | TP2 单卡主要分片因子 |
| --- | ---: | --- |
| virtual EP16（experts + Engram table） | 54,383,980,544 | 16 |
| TP Shard（embedding、lm_head、attention、shared experts、两个 Indexer Q/merge） | 1,945,960,704 | TP2 × FSDP8 = 16 |
| TP Replicate | 211,137,496 | FSDP8；Muon 再做参数 owner 去重 |

因此从 placement 看，没有 54B 级权重从 TP/EP 漏切。TP2 中未做 TP weight shard 的 216.59M 明细如下：

| 组 | 逻辑参数量 | 占 TP-replicate |
| --- | ---: | ---: |
| Engram `wkv.weight` | 157,286,400 | 74.5% |
| attention `q_a_proj`（4 层） | 26,214,400 | 12.4% |
| attention `kv_proj`（4 层） | 10,485,760 | 5.0% |
| compressed-attention compressor `wkv/wgate` | 5,242,880 | 2.5% |
| MoE router gate（4 层） | 7,864,320 | 3.7% |
| mHC `fn`（8 个子层） | 3,932,160 | 1.9% |
| Indexer `wk/k_norm` | 16,512 | <0.1% |
| Engram `q_weight/k_weight` | 40,960 | <0.1% |
| norm、mHC base/scale 等 | 54,104 | <0.1% |

两个 Indexer 的 `q_b_proj/weights_proj` 已按 head 做 TP Column shard，不在上述复制清单中。
其余 placement 中，Engram/mHC 是按 PanGu 算法有意复制；MLA 的低秩 A projection、router
和 norm 也是标准复制项。

### 6.2 优化器状态到底怎样切

HyperParallel Muon 会读取 DTensor 的 replicate mesh 维，按**完整参数**把 update/momentum owner 分配给不同 TP rank，owner 更新后广播参数。这避免所有 TP rank 都保留同一份 Muon momentum。

但它不是“把一个 replicated 参数内部再切两半”的 ZeRO：

- 157M 的 `engram.wkv.weight` 是一个参数。
- FSDP8 先让每个 TP rank 持有其 1/8 local shard。
- TP replicate owner 去重只能让一个 TP rank 负责这个 local shard 的 optimizer update；不能把同一个 local shard 再均匀拆给两个 owner。
- BF16 model shard、FP32 main shard 和准备好的 gradient 仍存在于两个 TP rank，单个大参数还会带来 owner rank 峰值不均衡。

所以“TP2 优化器状态总量应继续分摊”的方向是对的，当前也做了参数级分摊；但粒度是 parameter，不保证每个 rank 字节数完全相等。

### 6.3 首次 OOM 的直接原因与修正

首次 TP2 使用了 `sequence_parallel=false`。这与 replicated mHC/Engram projection 的设计不配套，两个 TP rank 都处理完整 4K token 序列；在已经接近 64GB 卡上限时，rank 5 最终在 `optimizer.step` 加载 `LpNormV2` 核时报告 `Memory_Allocation_Failure`。异步报错位置不是说范数本身占据了全部显存，只表示在那里发生了下一次失败的申请。

保持四层、4K、EP16、BF16 forward、FP32 reduce/main params 全部不变，仅启用 `sequence_parallel=true` 后：

- 一步训练通过；
- 峰值 allocated 为 43.9169GB，低于 TP1 的 50.0054GB；
- 没有通过减层数、关闭 FP32 主参数或缩短序列规避问题。

启动器现已让 TP2 默认开启 SP。

## 7. 实现文件

| 文件 | 作用 |
| --- | --- |
| `hyper_parallel/components/modules/mhc.py` | 新增流水式 mHC replacement 和复用 post 的适配函数 |
| `hyper_parallel/components/functional/mhc_post.py` | NPU 高性能实现 + 通用 Torch fallback |
| `hyper_parallel/components/modules/engram.py` | 哈希、融合、TP/CP 对齐和 EP sparse lookup |
| `hyper_parallel/components/modules/shared_compressed_dsa_attention.py` | CSA2 高性能 attention、candidate/Reindex、Indexer KL、packed 和 TP/CP 边界 |
| `hyper_parallel/models/deepseek_v41/modeling_deepseek_v41.py` | V4.1 裁剪结构和 replacement placeholder |
| `hyper_parallel/models/deepseek_v41/adapter/context_parallel.py` | V4.1 raw/compressed/index KV 异步 all-gather CP |
| `hyper_parallel/models/deepseek_v41/adapter/expert_parallel.py` | MoE 与 Engram local compute factory |
| `hyper_parallel/distributed/_builder/planner.py` | 任意显式 `EP: Shard` 参数的虚拟 EP 标记 |
| `hyper_parallel/distributed/_builder/parameter_sharding.py` | 按 placement 而非参数名选择 expert mesh |
| `hyper_parallel/distributed/_builder/source_shard.py` | 虚拟 EP source layout 元数据 |
| `examples/training_demo/train_deepseek_v41_online.yaml` | 模块替换、placement 和 Online 4K 配方 |
| `examples/training_demo/run_deepseek_v41_online.sh` | TP1 后 TP2 的 16 die 启动顺序 |

替换过程保留源参数对象和 state-dict key。Engram 表是唯一新增的虚拟 EP 行分片参数；mHC 和 Engram fusion 参数不依赖名称约定进入特殊 mesh。

## 8. 验证

已完成：

```text
18 passed tests/ut/auto_models/models/deepseek_v41/test_deepseek_v41_crop.py
1 passed  tests/torch/expert_parallel/test_engram_parallel.py（4-process Gloo）
1 passed  tests/torch/context_parallel/test_deepseek_v41_dsa_cp.py（2-process Gloo）
72 passed planner/virtual-EP 相关回归选择
TP1 EP16 FSDP16 Online 4K：1 step passed
TP2 EP16 FSDP8 SP2 Online 4K：1 step passed
TP1 CP2 EP16 FSDP16 Online 4K：1 step passed
```

TP2 成功指标：

```text
foundation_loss = 12.6250
grad_norm       = 67.3998
step_time       = 15.9357 s
max_allocated   = 43.9169 GB
max_reserved    = 58.9805 GB
```

TP1/TP2 的 global batch 分别是 16/8，且为随机初始化的 Online smoke，因此 loss/grad 不应互相做严格数值等价比较。

CP2 成功指标：

```text
foundation_loss = 13
grad_norm       = 67.8727
step_time       = 14.8038 s
max_allocated   = 44.9848 GB
max_reserved    = 58.9805 GB
```

该单步 smoke 验证的是功能和组合可运行性，未采集通信重叠 timeline，不能作为正式性能
对比结论。

## 9. 当前限制与后续正式验收

1. 本地 checkpoint 是 LFS pointer，尚未验证完整预训练权重加载；当前 `weight_transforms: none`。
2. 正式大表加载需补充 logical rows 到 padded rows 的 checkpoint pad/unpad transform，并确认源 Engram FP8 weight/scale 的训练格式；随机裁剪验证不覆盖这一点。
3. Engram CP 和共享压缩注意力 CP 已通过分布式数学验证及 16-die NPU CP2 单步 smoke；
   多步稳态吞吐和通信 overlap profile 尚未执行。
4. CSA2 Full/Reindex/Reuse、candidate pool、sample-level packed mask 和 PanGu 风格 Indexer KL
   已接入；PP>1 的 shadow indexer/pipeline payload 不在当前 PP1 配方内。
5. runtime KV-cache 与 SWA bounded replay 属于推理/rollout；正式 post-training 所需的 main KV
   与 Indexer Q/K FP4 QAT 尚未接入，当前 BF16 crop 不声明该精度能力。
6. 当前 Online 数据只用于功能 smoke；正式精度验收需要真实 checkpoint、固定离线 batch、TP1 baseline checkpoint 和多步 loss/gradient 对齐。
7. 成功 TP2/CP2 日志仍有一次通用 MoE dispatcher 的整数 `argsort` AiCPU 警告；Engram
   owner sort 已改为 float32 AiCore 路径，DSA Indexer 也不再排序离散 Top-K 索引，剩余
   告警不来自 Engram/DSA。
