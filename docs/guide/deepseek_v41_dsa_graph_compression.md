# DeepSeek-V4.1 DSA 静态图动静统一：Indexer 与 Compressor 集成

> 范围：在 hyperparallel 静态图（joint-graph）模式下打通 dsv4.1 DSA Attention 的
> Indexer 与 Compressor 执行路径，并使 eager 与静态图最大化复用同一套核心代码。
> 设计依据：`dsa_dynamic_static_unification_analysis.md`（torchtitan-npu 动静统一
> 方案向 hyperparallel 的迁移）。本文不追求性能与精度指标。

## 1. 总体架构

沿用 torchtitan-npu 的三条架构边界，全部落在现有 `GraphPackedSequence` 机制上：

```text
                (每个 microbatch，捕获之外)
batch(cu_seq_lens) ──> RuntimeInputAdapter.build ──> SharedCompressedPackedSequence
                              │ eager 直接使用（保留真实边界与缓存）
                              │ 图模式额外经过：
                              ▼
                 DeepseekV41Runtime.prepare_graph_inputs   ←—— 图边界 seam
                    · 预计算 per-ratio 压缩键元数据（minima）
                    · 校验文档对齐（lcm(ratios)），未对齐显式拒绝
                    · 产出 GraphPackedSequence（pytree 张量容器）
                              │
                              ▼
             make_fx 捕获整图（前向 + torch.autograd.grad 反向）
             图内只消费 [1, seq] 形张量：starts / positions / masks / doc_ids / minima
```

三个核心决策：

1. **数据相关推导全部图外化**：随 batch 变化的信息（每文档压缩键基址、因果可见上界）
   在图边界每步重算，以 `[1, seq]` 张量作为 pytree 叶子进图；图结构固定，值逐批更新。
2. **参考执行路径张量化**：TopK/候选/KL 的因果界不再依赖 Python 文档元组，
   eager 与图共用同一份张量推导（数值可证明等价）。
3. **形状策略放在数据侧**：collate 阶段将每个文档 pad 到压缩比 LCM 的整数倍，
   压缩 bank 形状恒为 `S // ratio`，eager 与图同形，无需容量 bank + 屏蔽机制。

## 2. 代码修改讲解

### 2.1 `hyper_parallel/components/modules/shared_compressed_dsa_attention.py`（核心）

| 修改 | 说明 |
| --- | --- |
| `minimum_key_indices` 纯张量化 | 原实现经 `document_boundaries()`（`.cpu().tolist()`）+ Python `accumulate` 构建文档偏移；现改为 `counts = diff // ratio` + exclusive cumsum + `index_select`，全部为张量算术。eager 与图边界调用同一函数，eager 顺带少一次 CPU 同步 |
| 新增 `causal_key_upper_bounds` | `minima + (positions - starts + 1) // ratio`，即每个 query 可见压缩键的独占上界 `[1, seq]`。可证明与原 `indexer_segments` Python 推导逐元素等价（含非对齐文档、CP 分片），UT 中对任意边界/比例组合逐一对拍 |
| `_causal_key_bounds` / `compressed_causal_topk` / `compressed_causal_topk_and_candidates` / `compressed_candidate_topk` 增加 `causal_upper` 形参 | 参考路径（含候选池）统一从张量界消费因果规则 `key < floor((query+1)/ratio)`；Python `query_segments` 仅保留给 eager fused 内核 |
| pytree schema 扩展 | `pytree_flatten` 增加第 5 个张量子树 `_prepared_minima`（按 `(device, ratio)` 索引），context 携带对齐标记 `("packing", alignment)`；schema 与文档数/长度无关，同图复用 |
| 图模式最小分支 | `block_plan` 对图输入恒返回 `None`（对齐已在图边界校验，连续压缩即精确）；`raw_document_starts` 返回 `()`；`indexer_segments` 显式 `NotImplementedError`；新增 `is_compression_aligned` / `compressed_key_length` 在图输入上走标志位/静态算术 |
| fused 算子能力门控 | `forward` 中对 `use_fused_ops` + 图打包输入显式 `NotImplementedError`（fused 内核需要 Python 文档 segments，无法进图），而非运行时才失败 |
| KL loss 函数式化 | `_SharedCompressedIndexerKLLoss.forward` 删除 `if not torch.any(valid_rows): continue` 数据分支与 `row_loss[valid_rows]` 布尔索引，改为掩码乘求和。空行贡献本就为 0、梯度本就为 0，eager 数值完全不变；图捕获不再触发数据依赖 guard |
| `prepare` 解除"packing-only"限制 | `_prepare_owned_metadata` 移除"图输入必须禁用压缩与索引"的报错；图输入若缺某比例的预计算元数据（adapter 配置不全），给出指向 `compress_ratios` 配置的明确错误 |
| `_plan_bank_dispatch` 收敛 | 仅在存在 CP 上下文时读取 Python 文档边界做通信路由；CP1 下 eager 与图都不再发生边界同步，逐文档计数改由 `compressed_key_length`（张量/静态）提供 |

### 2.2 `hyper_parallel/models/deepseek_v41/adapter/data/runtime.py`（图边界 seam）

`DeepseekV41Runtime` 新增 `compress_ratios: list[int] | None` 配置项（配置文件中必须
列出模型全部活跃压缩比，如 `[1, 2]`）。`prepare_graph_inputs` 在捕获之外：

- 调 `prepare(device, ratios)` 用真实 `cu_seq_lens` 预计算每个 ratio 的 minima；
- 以 `lcm(ratios)` 校验全部文档边界对齐，未对齐直接拒绝（提示配置 collate 的
  per-doc alignment padding）；
- 将对齐标记写入 pytree context，供图内 `is_compression_aligned` 使用。

eager 永远不经过该 hook，输入保持真实边界与缓存（与原行为一致）。

### 2.3 `hyper_parallel/data/batching/build_collate_fn.py`（数据侧形状策略）

`TextPackingCollator` / `build_online_text_collate_fn` 新增 `per_doc_alignment`
（取压缩比 LCM）：每个样本 pad 到完整块的整数倍（`input_ids` 补 0，`labels` 补
`IGNORE_INDEX`）。padding 落在文档末尾——损失被 mask、后续文档因果隔离、真实 token
相对位置不变。由此 eager 与静态图共用同一紧凑 bank 形状。配合已有的
`pad_to_length`（固定行长 S，chunk 循环与静态 shape guard 的 replay 契约）。

### 2.4 测试 `tests/ut/components/modules/test_packed_sequence_graph.py`

新增 `TestCompressionGraphInputs`：

- 图输入 minima / causal bounds 与 eager 逐批对拍（含 ratio=1 自然长度）；
- 张量因果界 vs 旧 Python segments 推导在**非对齐文档**上逐元素等价；
- 图边界对齐校验、恢复对象的图模式语义（block_plan/bank 长度/segments 拒绝）；
- pytree schema 跨批次稳定；
- 符号图（dynamic trace）跨不同文档数/长度复用同一张图，loss 与梯度逐批对拍；
- 参考 TopK 在两种界来源下选择结果一致。

### 2.5 示例配置与启动脚本

`hyper_parallel/compile/examples/automodel_text_graph/`：

- `train_deepseek_v41_graph_compression_fsdp2.yaml`：压缩/Indexer 全量激活
  （`v41_indexer_loss_coeff: 0.1`，KV/Index source 层来自模型 config），并配置
  `per_doc_alignment: 2`、`pad_to_length: 32`、runtime adapter `compress_ratios: [1, 2]`；
- `run_deepseek_v41_graph_compression_fsdp.sh`：与 packing 版启动范式一致。

## 3. 动静统一 vs 保留分支

### 3.1 已统一（同一份代码，两模式共用）

- 压缩键基址（minima）的构建函数与数值；
- 参考路径全部因果可见性：TopK、候选、KL、稀疏 attention 消费同一组张量界；
- packed 元数据对象类型：`GraphPackedSequence` 仅剩 pytree 注册标记，全部语义在
  `SharedCompressedPackedSequence` 基类；
- 数据形状契约：collate 对齐后，eager 连续压缩 fast-path 与图路径天然一致；
- KL/aux loss 等自定义 autograd.Function 的前反向（图捕获直接穿透）。

### 3.2 保留的分支差异（最小必要）

| 分支点 | 原因 |
| --- | --- |
| pytree 注册只挂在 `GraphPackedSequence` 子类 | eager FSDP 重建 pytree 时必须把 packed 输入当不透明叶子，保留真实边界与 per-device 缓存；基类一旦注册会破坏 eager 缓存语义 |
| `prepare_graph_inputs` 仅图模式调用 | 即"图外 seam"本身——tt-npu 中 trainer 侧公共 preprocess 的等价物，eager 的准备发生在 forward 内（带缓存） |
| `block_plan` / `raw_document_starts` / `indexer_segments` / `is_compression_aligned` / `compressed_key_length` 对图输入走标志位或静态算术 | 图输入不携带 Python 边界；这些查询的**值**与 eager 一致（对齐已在图边界校验），仅来源不同 |
| Python `query_segments` 仅存在于 eager fused 路径 | 原生 CANN 内核（lightning_indexer / sparse_flash_mla / fused KL）需要文档 segments 元组，进图会把首批边界固化；图模式使用参考实现并对 fused 显式报错 |
| eager 支持自然（非对齐）文档长度 | 经 `document_compression_plan` 的逐文档分组路径是 eager 能力；图模式将其声明为不支持配置（数据侧对齐解决），即 tt-npu 的"能力门控"而非第二套形状语义 |

## 4. 对 torchtitan-npu 的借鉴点

| tt-npu 机制 | 本次迁移落点 |
| --- | --- |
| 元数据在 trace 边界之外每步重建，pytree 张量进图（`preprocess_inputs → build_attention_metadata`） | `prepare_graph_inputs` 作为图边界 seam：per-ratio minima / 对齐标记在捕获外逐批重建，图内只消费 |
| 载体无状态、注册无副作用 | 缓存仍挂在输入对象上，但图输入是每步重建的 pytree 产物（unflatten），缓存生命周期=产物生命周期；注册收窄到子类，eager 语义零改动 |
| `per_doc_alignment = lcm(ratios)` 数据侧对齐，eager/图同形 | `TextPackingCollator.per_doc_alignment` + 图边界对齐校验，消除"容量 bank + 屏蔽"第二套形状语义的需求 |
| 图内写法函数式化（改同一份代码，而非 `if compiling` 分支） | KL loss 掩码化替代数据分支/布尔索引，eager 数值不变 |
| 能力门控（policy 决定是否进图，降级必告警） | fused 内核不支持图模式 → 进图前显式 `NotImplementedError`，配置错误（缺 `compress_ratios`）也给出指向性报错 |
| replay 契约显式化（静态 shape 必须一致，不自动 retrace） | `pad_to_length` 固定行长 S；文档数/长度变化仅体现在张量值上，同图复用 |

刻意**未**照搬的部分：tt-npu 的 `unique_consecutive(rows - flat)` 边界推导依赖其
position 编码约定，hyperparallel 保留自己的 `cu_seq_lens` 契约；CP 通信路由/负载均衡
（hyperparallel 强项）保持 eager 能力，未为图模式裁剪。

## 5. 调试与验证记录

环境：whh_titan 容器，Ascend 910。模型：官方 V4.1 配置裁剪（hidden 640，压缩比
{0,1,2}，KV/Index source 层齐全，KL 系数 0.1，候选池关闭），随机初始化。

| 验证项 | 结果 |
| --- | --- |
| UT：`test_packed_sequence_graph.py`（15 例，含新增压缩用例） | 全部通过 |
| 受影响面回归（DSA 模块 / KL / 数据批处理 / compile-graph 相关，共 800+ 用例） | 全部通过（3 个失败为清树即有的 megamoe/vision 无关失败） |
| 静态图 1P × 40 层（indexer+compressor+KL 激活，FSDP dp=1） | 30/30 迭代，loss 11.79 → 9.43，显存峰值 21.8GB |
| eager 2P × FSDP dp=2（同一配置/数据） | 30/30 迭代，loss → 8.71 |
| 静态图 2P × 12 层裁剪（图内 FSDP all-gather/reduce-scatter） | 30/30 迭代，loss → 8.49 |

调试中的关键结论：`range(0, SymInt_seq, chunk)` 会使符号图特化到首捕获长度——
故以 `pad_to_length` 固定行长（tt-npu replay 契约做法）；KL 中的
`torch.any` 数据分支会触发 `GuardOnDataDependentSymNode`——以函数式化解决；
40 层 + 2P 图模式在本机 HBM 装不下（40 层整图 + FSDP 缓冲），属环境容量问题，
12 层 2P 与 40 层 1P 均通过。

启动方式（容器内、仓库根目录）：

```bash
# 静态图（本例 1P；2P 去掉 dp 覆盖即可，需相应缩小模型裁剪深度）
MODEL_DIR=<裁剪模型目录> TOKENIZER_DIR=<tokenizer目录> \
WIKITEXT_TRAIN_FILE=<jsonl数据> NPROC_PER_NODE=1 \
bash hyper_parallel/compile/examples/automodel_text_graph/run_deepseek_v41_graph_compression_fsdp.sh \
  --fsdp_config.dp_shard_size=1 --training.global_batch_size=1

# eager 对照（同一配置）
... bash .../run_deepseek_v41_graph_compression_fsdp.sh --compile.enabled=false
```

## 6. 变更文件清单

- `hyper_parallel/components/modules/shared_compressed_dsa_attention.py`：核心动静统一
- `hyper_parallel/models/deepseek_v41/adapter/data/runtime.py`：图边界 seam 与配置项
- `hyper_parallel/data/batching/build_collate_fn.py`：per-doc 对齐 padding
- `tests/ut/components/modules/test_packed_sequence_graph.py`：等价性/重放/图语义 UT
- `hyper_parallel/compile/examples/automodel_text_graph/train_deepseek_v41_graph_compression_fsdp2.yaml`：压缩配置示例
- `hyper_parallel/compile/examples/automodel_text_graph/run_deepseek_v41_graph_compression_fsdp.sh`：启动脚本
