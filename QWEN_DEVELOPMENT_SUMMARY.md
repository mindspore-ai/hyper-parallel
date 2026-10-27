# Qwen 分布式训练开发改动总结

整理日期：2026-10-27。以下主体记录 rebase 前的代码快照：`qwen3.8_dev`，HEAD `75506890`。

> Rebase 更新：当前分支已基于 `mindspore-ai/hyper-parallel` master `30c8ce78`，
> HEAD 为 `8e00aa9d`。保留 38 个功能提交并增加 1 个 master 接口适配提交，共 39 个增量提交。
> 原历史中的 3 个提交因上游已实现或被 master 通用实现取代而不再重复应用；合并提交已展开。
> 数据源、batching、encoder runtime、配置及 checkpoint 以 master 为基础保留专项增量。
> 定向回归 323 项通过，补充模型构建/规划检查通过；两进程 CPU/Gloo 的视觉 CP、图文 CP、
> CP+Encoder-DP、chunk loss，以及 GDN P2P/KV all-gather（含重计算）验证通过。
> NPU 全模型训练尚未验证。备份分支：`backup/qwen3.8_dev-before-master-rebase-20261027`。
> 下文的提交哈希与统计仍对应原开发历史，便于查阅，不能作为当前 master 增量统计。

## 1. 统计范围与结论

本文以 Qwen 系列专项开发的首个提交 `820a170f` 为起点，统计其父提交 `052967ba` 之后、截至 `75506890` 的本地可达历史，即 `052967ba..75506890`。这是按功能起点确定的范围，不是简单截取最近 40 条日志。

| 项目 | 数量 / 说明 |
| --- | --- |
| 时间跨度 | 2026-09-18 至 2026-10-23（提交日期） |
| 总提交数 | 47 |
| 非合并提交 | 41 |
| 你的账号提交 | 39：`wcrzlh` 34 个、`Chaoran Wei` 5 个 |
| 协作者提交 | `mxysfive` 2 个，单独列出，不计作你的独立改动 |
| 合并提交 | 6 |
| 基线到 HEAD 的净差异 | 150 个文件，新增 16,066 行、删除 1,092 行 |

以上行数是整个范围的净差异，包含协作者修复、测试、文档、示例和两个二进制图片文件；不是每个 commit 的增删行数累加。Git 作者信息不能证明每一行是否由某次助手会话生成，所以本文按仓库记录总结这轮开发，不虚构会话归属。

这一轮的主要成果是：围绕 Qwen 文本与 MoE 训练补齐 Ascend 算子接入、CP/EP/FSDP 组合、在线会话数据监督、显存与优化器状态处理、HF 权重导出，然后进一步扩展到 Omni 多模态数据、视觉编码器并行和图文 CP + chunk loss。

目录名称与模型适配名称不同：运行示例位于 `examples/qwen3_8`、`examples/qwen3_6_moe`，底层复用 `models/qwen3_5`、`models/qwen3_5_moe` 的 HF 模型结构。本文沿用仓库命名，不将不同名称解释为不同的全新模型实现。

## 2. 开发阶段

| 阶段 | 时间 | 主要内容 |
| --- | --- | --- |
| 文本训练基础 | 09-18～09-22 | Qwen3.8 训练入口、GDN 后端、单机/多机启动、镜像、HF 导出、原生 chat template |
| 可用性与 MoE | 09-24～10-08 | 离线导出、定时 checkpoint、loss 绘图、system 消息修复、Qwen3.6 MoE 与 EP |
| 正确性与显存 | 10-09～10-20 | assistant mask、FP32 主权重、采样、混合 CP、chunk loss、FSDP 包装、MoE 导出修复 |
| 多模态扩展 | 10-21～10-23 | Omni、encoder parallel、HF 元数据、VLM CP、Parquet 图片数据、VLM chunk loss |

## 3. Qwen 文本训练与 Ascend GDN 接入

### 3.1 训练入口和部署环境

最初加入 Qwen3.8 的 `train_text.py`、离线/在线 YAML、依赖清单和单机八卡启动脚本，随后补上 32 节点 256 die 的启动流程。README 记录模型与数据准备、运行配置和训练参考曲线。

增加 `docker/Dockerfile.qwen3_8-a3`，将 Ascend A3 训练环境整理为可复用镜像配方，并更新 `.dockerignore` 和依赖版本。多机脚本的存在表示提供了启动配置，不等于本文重新验证过 256 die 训练。

主要文件：

- `examples/qwen3_8/train_text.py`、`train.yaml`、`train_online.yaml`
- `examples/qwen3_8/launch_1node_8dies.sh`、`launch_32nodes_256dies.sh`
- `examples/qwen3_8/requirements.txt`、`README.md`
- `docker/Dockerfile.qwen3_8-a3`

关联提交：`820a170f`、`937f381e`、`191053d4`。

### 3.2 Gated Delta Net 算子与模块替换

新增 AscendC GDN 功能封装及 KKT、三角矩阵相关实现，接入 GDN 模块替换和 CP wrapper。后续将训练示例切换到 Triton-Ascend GDN，并按后端明确区分 `gdn_ascendc` 与 `gdn_triton`，避免名称掩盖实际执行实现。

同时清理当时未使用的 Qwen3.8 TP/SwiGLU 配置与对应规划逻辑、测试，以及过渡性的高性能 YAML 和 wrapper。这里既有新增，也有撤回早期设计，不能把每个早期提交里的功能都当成最终保留能力。

主要文件：`components/functional/gated_delta_rule.py`、`_ascendc_gdn.py`、`_gdn_kkt.py`、`_gdn_tril.py`，以及 `components/modules/gdn_ascendc.py`、`gdn_triton.py`、`distributed/context_parallel/wrappers.py`（均位于 `hyper_parallel/` 下）。

关联提交：`820a170f`、`3c544c0a`、`85458dd6`、`5fb1ec77`、`937f381e`。

## 4. MoE 与混合并行

### 4.1 Qwen3.6 MoE 模型和 EP

新增 `qwen3_5_moe` adapter 注册和 Qwen3.6 MoE 示例，使模型进入现有 builder、模块替换和训练流程。继续加入模型级 expert parallel adapter，适配专家参数和 EP 执行路径。

专家输出聚合从手工加权与 `index_add_` 调整为 `moe_token_unpermute`，恢复 token 顺序并合并 top-k 专家输出。同步补充 adapter 合同与 EP local-region 测试。

示例给出 FSDP、FSDP + CP、FSDP + CP + EP 三种启动方式，并说明不复制专家参数时的拓扑关系：`ep_size × edp_shard_size = world_size / pp_size`。

仓库 README 留有一次八卡 Ascend 910B3 的历史参考：FSDP + CP + EP 在 steps 11–100 平均步时 7.853 秒、最大已分配显存 45.61 GB。这是该 README 的特定环境记录，不能据此推导普遍加速比，也不是本次重新测量。

关联提交：`c24eaec5`、`8ef8be6e`。

### 4.2 GDN P2P + Attention KV all-gather

为混合 GDN/attention 模型增加独立 CP 配方：GDN 使用 P2P，attention 使用 KV all-gather，并添加 `GDNCPDataCollator` 和 `GDNCPParallelBatch` 来约束输入与切片方式。

关键处理包括：

- 每个 microbatch 只接收一条完整会话，右侧补齐到适配 CP 和 GDN chunk 的长度。
- 按连续 token 区间切分 CP，保证本地长度满足 64-token chunk 约束。
- attention 的全局偏移 causal mask 交给 CP wrapper，避免 HF 按局部长度生成错误 mask。
- 禁止直接拼接独立会话：仅隔离 attention 不会自动重置 GDN recurrent state 和卷积历史。
- 该文本批处理路径要求 TP=1，并限制为完整 causal attention 场景。

主要文件：`hyper_parallel/models/qwen3_5/cp_batch.py`，以及两组示例目录中的 `train_online_p2p_kv_allgather.yaml`。

关联提交：`3e35b8e8`、`9c890b2d`。

## 5. 文本数据、chat template 和监督边界

### 5.1 使用 tokenizer 原生模板

在线文本数据处理接入 tokenizer 原生 `chat_template` 及模板参数，通过训练配置传递选项，减少手工拼接与模型真实推理格式之间的差异。

增加 tools 的显式启用机制，将工具定义和消息正确送入模板；修复开头包含 system 消息时的处理。Qwen3.6 示例后续加入 thinking 历史相关选项，并配置 scheduler 学习率。

关联提交：`f8a133ad`、`3ac14a0d`、`ad1c6d58`、`b84b9f34`。

### 5.2 assistant loss mask 的多轮修复

这一部分处理的是“模型到底学习哪些 token”，并非单纯的数据格式调整：

1. 将 assistant 监督起点与推理 `add_generation_prompt` 的前缀对齐。
2. 明确逐轮 assistant 的监督区间，补充多轮会话边界测试。
3. 修复只有 tool call、没有普通回答文本的 assistant 消息，保持 token 边界。
4. 增加 thinking 历史相关回归用例，防止模板选项变化改变监督语义。

主要文件：`hyper_parallel/data/text/chat_template.py`、`build_data_transform.py`、`hyper_parallel/trainer/config/data.py`、`hyper_parallel/trainer/text_trainer.py`。

关联提交：`ace55902`、`163b37c9`、`23582ba2`、`b84b9f34`。

### 5.3 训练前检查与采样修复

增加 `validate_chat_templates.py`，支持在训练前离线检查数据集的模板处理；后续修复为读取训练配置中的模板选项，使验证工具尽量沿用训练语义。

增加 `analyze_decif_lengths.py`，用于查看 Decif 数据长度分布，辅助选择最大长度及数据策略。

修复 cyclic sampler 的顺序：先打乱完整样本集合，再丢弃无法组成批次的尾部，避免在打乱前固定丢弃同一部分数据。

关联提交：`4534c6fb`、`b0b9abe3`、`6959373f`、`e192b177`。

## 6. 显存、FP32 主权重与 FSDP

### 6.1 FP32 main parameters 的 DTensor 状态

更新在线训练配置中的 FP32 主权重和学习率，并修复 mixed precision optimizer 对 DTensor 参数与状态的处理，使主权重路径保留分布式张量所需的信息。改动涉及优化器、DTensor 兼容层、model builder 和 trainer，而非仅修改 dtype 开关。

对应 `test_mixed_precision_optimizer.py` 有大量回归测试扩展；新增测试代码并不代表本文已在当前机器运行通过。

关联提交：`571bdaf6`、`ada81ddc`。

### 6.2 root FSDP 与 embedding 独立包装

新增 `fsdp_config.root_wrap_modules`，允许用模块完整名称指定额外独立包装单元，同时校验配置值。将 `reshard_after_forward` 的控制延伸到 root 剩余参数，并处理 chunk loss 与 LM head 包装的衔接。

随后在 Qwen 文本示例启用 embedding FSDP 和 chunk loss，适配 dense 与 MoE 文本模型。目的在于减少大 embedding/head 和全量 logits 带来的峰值显存；实际节省量需要具体配置下测量。

主要文件：`hyper_parallel/models/build_options.py`、`hyper_parallel/distributed/_builder/fsdp_adapter.py`、`hyper_parallel/models/qwen3_5/adapter/chunk_loss.py`。

关联提交：`846dbc5c`、`3d803e82`、`77352fd4`。

### 6.3 梯度释放与协作者修复

初始 Qwen 接入中，对满足条件的普通 FSDP 参数在已有 reduce-scatter 等待点及时应用梯度、清理通信输出引用，减少一直保留到 root finalization 的梯度分片。

另外两项来自协作者 `mxysfive`，属于当前分支结果但应独立归属：

| 提交 | 修复 | 影响 |
| --- | --- | --- |
| `34b955c8` | 及时释放 FSDP 梯度转换、重排后的源存储 | 减少无用梯度存储占用 |
| `84d21abd` | 不均匀 FSDP checkpoint 恢复时保留 DTensor layout | 保持恢复后的布局信息 |

## 7. Checkpoint 与 Hugging Face 导出

### 7.1 训练时导出、定时保存和保留数量

新增 `save_hf_weights`，允许 trainer 在 checkpoint 流程中额外导出 HF 模型权重。新增 `save_time_interval_minutes`，在达到时间间隔后的已完成 optimizer step 保存；新增 `save_total_limit`，控制已完成 step checkpoint 的保留数量。

回调增加相关流程与测试。HF 模型权重用于加载模型；完整训练恢复仍依赖 DCP、optimizer、scheduler、dataloader/RNG 等训练状态，二者不是同一份内容。

关联提交：`292371e0`、`58bd8517`。

### 7.2 单 CPU 进程离线 DCP → HF

新增 `examples/qwen3_8/tools/convert_dcp_to_hf.py`：从完成的 DCP step 目录读取模型权重，结合原始 HF config 构建模型结构，写入 safetensors 分片并保存 tokenizer。

CheckpointManager 新增 DCP 来源路径，可从已保存 checkpoint 重建权重，避免对训练中的 live model 发起全模型 gather。脚本检查输入模型目录和目标空目录，支持分片大小与专家权重布局选项。

关联提交：`21da0de4`。

### 7.3 导出正确性的后续补齐

| 问题 | 改动 | 提交 |
| --- | --- | --- |
| grouped GEMM 专家权重与 HF 布局不同 | 恢复专家维度布局；提供 `auto/hf/grouped_gemm` 选项处理歧义 | `3716b1d2` |
| fused MoE 权重被默认反向转换拆开 | 离线导出保留 fused expert 表达 | `41b6d8d7` |
| 非均匀 DTensor shard 无法按普通等长方式 gather | 扩展 redistribution/full-tensor 路径并补分布式测试 | `04357245` |
| HF 元信息或文本权重 key 不匹配 | 调整导出的模型元数据和 text 权重 key，支持重新加载 | `689ea7a2` |

因此这组工作并非只“写出一个权重文件”，还覆盖分片重建、专家布局和模型重新加载所需的信息。

## 8. Omni 多模态数据管线

### 8.1 替换旧 data/vlm

新增 `hyper_parallel/data/omni/`，迁移 VLMTrainer 到 Omni，删除旧 `data/vlm` 接口。按照目录 README，这是选择性移植上游 PR #884 的通用多模态数据组件并适配当前分支，不应归为全部从零实现。

| 组件 | 职责 |
| --- | --- |
| `build_dataset.py` | 本地数据索引与数据源构建 |
| `omni_transform.py` | processor 编码、媒体解析、监督标签 |
| `dynamic_batch.py`、`loader.py` | 候选选择、动态缓冲、迭代及状态恢复 |
| `packing.py` | 字段拼接、媒体偏移、`cu_seq_lens` |
| `collator.py` | token 堆叠与媒体字段拼接 |
| `parallel.py`、`get_batch.py` | TP 数据传输、CP token 切分和模型输入适配 |

数据源包括本地 JSON/JSONL，后续扩展 Parquet。缓冲区支持随 StatefulDataLoader 保存，并可按源索引重放未消费样本。索引重放要求数据、编码方式和 DP world size 等条件保持一致。

监督语义须特别区分：默认 Omni processor 路径仅监督最后一条 assistant；文本路径保留其多轮监督逻辑。旧 `data.vlm` 配置需要迁移，没有兼容别名。

关联提交：`3069fb76`。

### 8.2 Parquet 内嵌图片与八卡 VLM 启动

增加 Parquet row-group/row 索引，每个 worker 至多缓存一个 row group，索引时不解码图片。支持消息中的 `<image>` 与独立 `images` 列表按顺序对应，也支持 bytes/path 格式；bytes 优先，路径按数据文件目录解析，图片解码为 RGB。

校验图片占位符和图片数量，避免遗漏或额外图片被静默吞掉。加入 VLM 八卡启动脚本和依赖说明。另有小修复允许 assistant 回答中的 Unicode replacement character，避免合法样本因该字符被拒绝。

关联提交：`a43e6740`、`c339a613`。

## 9. 视觉编码器并行与 VLM CP

### 9.1 Encoder-DP / Encoder-CP

新增 `distributed/encoder_parallel/`，包含配置、mesh、任务规划、attention 交换、特征路由和 runtime，并通过模型 adapter 接入 Qwen vision。目录 README 标明通用 runtime 选择性来自上游 PR #1132。

执行流程：

1. 从 decoder CP × TP 的 rank 集合推导 encoder parent mesh。
2. 将整张图片分到 Encoder-DP bucket，按 patch 数平方估算负载进行分配。
3. Encoder-CP 切分图片 patch 序列，通过 Ulysses 交换 sequence/head 维度并处理 padding。
4. 收集每张图片 hidden states，调用原模型 spatial merger。
5. 通过可微通信将视觉特征路由回 decoder，恢复图片顺序。
6. 对无本地图片任务的 rank 加入 dummy work / 零值 autograd anchor，维持所需的梯度参与。

主要修改还包括 builder planner、forward rewriter、adapter spec、mesh 与 VLM trainer。该功能为实验性、显式启用能力。

关联提交：`d66a76f3`。

### 9.2 图文 decoder context parallel

新增 `QwenVLMParallelBatch` 和 CP forward bridge。它先在完整会话上计算 mRoPE 位置和 next-token targets，再按 CP rank 切分；图片特征只写入本 rank token 区间内的图片占位符。

这个次序非常关键：如果先切片再独立 shift labels，会丢失分片交界处的 next-token 目标。代码还检查图片特征行数与本地占位符数量，保留无图片行 rank 的 backward anchor，并对没有有效监督 token 的 rank 使用安全计数。

CP=1 保留原生 HF forward 作为基线路径；CP>1 通过适配桥连接视觉编码与 decoder。后续提供 Ulysses 配方及 GDN P2P + KV all-gather 配方。

关联提交：`403b2b0d`。

### 9.3 VLM chunk loss

新增 VLM 专用 chunk loss adapter，从 multimodal base model 获取 hidden states，再分块执行 LM head / cross entropy，不物化完整词表 logits。输出 `loss_sum` 和有效 token 数，保持 trainer 的归一化接口。

复用批处理阶段已全局 shift 的 targets，避免 chunk 内再次 shift；兼容 CP bridge 和 chunk adapter 的绑定顺序，保留普通 forward 的原有入口。

主要文件：`hyper_parallel/models/qwen3_5_moe/adapter/vlm_chunk_loss.py`、`vlm_cp.py`，以及 `examples/qwen3_6_moe/train_online_vlm_p2p_kv_allgather.yaml`。

关联提交：`e9e17785`。

## 10. 辅助工具和测试资产

新增 `plot_loss.py`，从训练日志绘制 loss，并支持原始值和平滑曲线叠加。仓库保留 Qwen3.8 和 Qwen3.6 MoE 的参考图片。

测试覆盖随功能一起扩展，主要分布如下：

| 领域 | 代表测试路径 | 主要验证对象 |
| --- | --- | --- |
| 文本模板 | `tests/ut/data/text/` | system、tools、thinking、assistant mask、验证工具 |
| 文本数据 | `tests/ut/auto_models/components/datasets/llm/test_build_data_transform.py` | 数据转换 |
| GDN/CP | `tests/ut/auto_models/components/`、`tests/torch/context_parallel/` | wrapper、分片、GDN 与 attention 组合 |
| FSDP/chunk loss | `tests/ut/auto_models/test_fsdp_root_wrap.py`、`test_qwen3_5_chunk_loss.py` | 包装策略、分块损失 |
| 优化器 | `tests/ut/auto_models/trainer/test_mixed_precision_optimizer.py` | FP32 main parameters 与 DTensor 状态 |
| 导出 | `tests/ut/auto_models/_transformers/test_checkpoint_manager_save.py` | HF 保存、布局、元信息 |
| 不均匀分片 | `tests/torch/shard/test_uneven_full_tensor.py` | full tensor 重建 |
| checkpoint 回调 | `tests/ut/trainer/test_checkpoint_callback.py` | 定时保存、保留策略 |
| Omni | `tests/ut/data/omni/` | 编码、packing、buffer resume、Parquet/图片 |
| Encoder parallel | `tests/ut/distributed/encoder_parallel/`、`tests/torch/encoder_parallel/` | 规划、特征和梯度比较、VLM CP |
| VLM chunk loss | `tests/ut/auto_models/test_qwen_vlm_chunk_loss.py`、`tests/torch/context_parallel/test_vlm_chunk_loss.py` | loss/gradient 对齐及不物化 logits |

**本次是提交审计和文档整理，没有重新运行这些测试，也没有进行 NPU/GPU 多卡训练。** 测试文件存在、历史 README 的运行记录、当前硬件回归通过是三种不同证据，不在本文中混为一谈。

## 11. 当前边界与后续需要注意的事项

- 这轮 Qwen/Omni/视觉适配主要是 Torch + Transformers 路径，不能解释为新增了同等 MindSpore 支持。
- GDN P2P 文本批处理要求单会话、TP=1 和 chunk 对齐，不支持任意多样本 packing。
- 当前 Qwen VLM CP 输入路径要求 TP=1、SP=false、单条未打包会话，以图片为目标；不支持视频及任意 padding 方式。
- VLM chunk loss 不支持 KV cache、非零 `logits_to_keep` 和 router auxiliary loss 聚合，属于训练接口。
- Encoder parallel 仍有视觉 activation checkpoint、编译、输入网格及 FSDP 包装边界等限制；可微路由和 CPU/Gloo 小模型测试不等于完整 NPU 模型训练已验收。
- Omni 的通用 packing 能力不意味着 Qwen VLM 适配器接受多会话 packing；具体模型 adapter 的限制仍然生效。
- `encoder_parallel/README.md` 保留了初始 decoder CP=1 的描述，但后续已有 VLM CP 适配；判断组合能力需要结合当前 adapter，而不能只读早期 README。
- 同一 README 引用了 `train_online_vlm_encoder_parallel.yaml`，本次快照未发现该文件；不要把这个链接作为已经交付的独立配置入口。
- commit body 中出现过 token throughput 修复及 revert；最终净变更不能算作保留了该吞吐修复。

## 12. 逐提交清单

以下按 Git 日志反向输出（整体从早期开发到后续合入），列出全部 41 个非合并提交。每条附 GitHub commit 链接和涉及文件，便于追溯；“协作者”条目独立标识。

### 01. feat: add Qwen3.8 distributed training support

- 提交：[`820a170f`](https://github.com/wcrzlh/hyper-parallel/commit/820a170fd54df5bbb89e0733e9cbf0656ec61e06)
- 日期：2026-09-18；作者：wcrzlh
- 涉及文件：

  - `examples/qwen3_8/launch_1node_8dies.sh`
  - `examples/qwen3_8/requirements.txt`
  - `examples/qwen3_8/train.yaml`
  - `examples/qwen3_8/train_high_performance.yaml`
  - `examples/qwen3_8/train_online.yaml`
  - `examples/qwen3_8/train_text.py`
  - `hyper_parallel/components/functional/_ascendc_gdn.py`
  - `hyper_parallel/components/functional/_veomni_gdn_kkt.py`
  - `hyper_parallel/components/functional/_veomni_gdn_tril.py`
  - `hyper_parallel/components/functional/gated_delta_rule.py`
  - `hyper_parallel/components/modules/__init__.py`
  - `hyper_parallel/components/modules/gdn_ascendc.py`
  - `hyper_parallel/components/modules/swiglu_mlp.py`
  - `hyper_parallel/core/fully_shard/hsdp_state.py`
  - `hyper_parallel/data/README.md`
  - `hyper_parallel/data/text/build_data_transform.py`
  - `hyper_parallel/distributed/_builder/default_templates.py`
  - `hyper_parallel/distributed/_builder/planner.py`
  - `hyper_parallel/distributed/context_parallel/__init__.py`
  - `hyper_parallel/distributed/context_parallel/wrappers.py`
  - `hyper_parallel/distributed/tensor_parallel/head_count.py`
  - `hyper_parallel/distributed/tensor_parallel/param_role.py`
  - `hyper_parallel/trainer/text_trainer.py`
  - `hyper_parallel/trainer/vlm_trainer.py`
  - `tests/ut/auto_models/components/datasets/llm/test_build_data_transform.py`
  - `tests/ut/auto_models/components/test_gated_delta_net.py`
  - `tests/ut/auto_models/components/test_gdn_cp_wrapper.py`
  - `tests/ut/auto_models/components/test_gdn_tp.py`
  - `tests/ut/core/fully_shard/test_state.py`
  - `tests/ut/dual_mode_dtensor/test_cp_attention_wrappers.py`
  - `tests/ut/dual_mode_dtensor/test_foundation_placement_role.py`

### 02. fix: use Triton-Ascend GDN for Qwen3.8 training

- 提交：[`3c544c0a`](https://github.com/wcrzlh/hyper-parallel/commit/3c544c0a14a380e7f48931f17764ca9ffcfcb2c6)
- 日期：2026-09-19；作者：wcrzlh
- 涉及文件：

  - `examples/qwen3_8/train.yaml`
  - `examples/qwen3_8/train_high_performance.yaml`
  - `examples/qwen3_8/train_online.yaml`
  - `hyper_parallel/components/modules/__init__.py`
  - `hyper_parallel/components/modules/gdn_triton.py`
  - `hyper_parallel/trainer/runtime/distributed.py`
  - `tests/ut/auto_models/components/test_gated_delta_net.py`

### 03. refactor: remove unused Qwen3.8 TP and SwiGLU options

- 提交：[`85458dd6`](https://github.com/wcrzlh/hyper-parallel/commit/85458dd6000b875f3cac57b7f5b988bfcd0e499f)
- 日期：2026-09-19；作者：wcrzlh
- 涉及文件：

  - `hyper_parallel/components/modules/swiglu_mlp.py`
  - `hyper_parallel/distributed/_builder/planner.py`
  - `hyper_parallel/distributed/tensor_parallel/head_count.py`
  - `hyper_parallel/distributed/tensor_parallel/param_role.py`
  - `tests/ut/auto_models/components/test_gdn_tp.py`
  - `tests/ut/dual_mode_dtensor/test_foundation_placement_role.py`
  - `tests/ut/dual_mode_dtensor/test_planner_golden_plans.py`

### 04. refactor: name GDN module replacements by backend

- 提交：[`5fb1ec77`](https://github.com/wcrzlh/hyper-parallel/commit/5fb1ec7790d63e5529280b1089ad704ee1b40505)
- 日期：2026-09-19；作者：wcrzlh
- 涉及文件：

  - `examples/qwen3_8/train.yaml`
  - `examples/qwen3_8/train_online.yaml`
  - `hyper_parallel/components/modules/__init__.py`
  - `hyper_parallel/components/modules/gdn_ascendc.py`
  - `hyper_parallel/components/modules/gdn_triton.py`
  - `tests/ut/auto_models/components/test_gated_delta_net.py`

### 05. feat: prepare Qwen3.8 training launch workflows

- 提交：[`937f381e`](https://github.com/wcrzlh/hyper-parallel/commit/937f381e375153af087373163c13f31908bbff50)
- 日期：2026-09-19；作者：wcrzlh
- 涉及文件：

  - `examples/qwen3_8/README.md`
  - `examples/qwen3_8/launch_1node_8dies.sh`
  - `examples/qwen3_8/launch_32nodes_256dies.sh`
  - `examples/qwen3_8/offline_4k_loss.png`
  - `examples/qwen3_8/requirements.txt`
  - `examples/qwen3_8/train.yaml`
  - `examples/qwen3_8/train_online.yaml`
  - `hyper_parallel/components/functional/_ascendc_gdn.py`
  - `hyper_parallel/components/functional/_gdn_kkt.py`
  - `hyper_parallel/components/functional/_gdn_tril.py`
  - `hyper_parallel/components/functional/_veomni_gdn_kkt.py`
  - `hyper_parallel/components/functional/_veomni_gdn_tril.py`
  - `hyper_parallel/distributed/context_parallel/__init__.py`
  - `hyper_parallel/distributed/context_parallel/wrappers.py`
  - `tests/ut/dual_mode_dtensor/test_cp_attention_wrappers.py`

### 06. feat: add Qwen3.8 A3 training image recipe

- 提交：[`191053d4`](https://github.com/wcrzlh/hyper-parallel/commit/191053d411300aabf6f7984198373ef5188aeb19)
- 日期：2026-09-21；作者：wcrzlh
- 涉及文件：

  - `.dockerignore`
  - `docker/Dockerfile.qwen3_8-a3`
  - `docker/README.md`
  - `examples/qwen3_8/README.md`
  - `examples/qwen3_8/requirements.txt`

### 07. feat: export Trainer checkpoints in Hugging Face format

- 提交：[`292371e0`](https://github.com/wcrzlh/hyper-parallel/commit/292371e05d06a9f6f682498b0bcf58941e3ee12c)
- 日期：2026-09-21；作者：wcrzlh
- 涉及文件：

  - `examples/qwen3_8/train.yaml`
  - `examples/qwen3_8/train_online.yaml`
  - `hyper_parallel/components/checkpoint/config.py`
  - `hyper_parallel/trainer/base.py`
  - `hyper_parallel/trainer/callbacks/checkpoint_callback.py`
  - `tests/ut/auto_models/checkpoint/test_hf_checkpoint_callback.py`

### 08. feat: support native tokenizer chat template options

- 提交：[`f8a133ad`](https://github.com/wcrzlh/hyper-parallel/commit/f8a133ad99c0685d8275fe6a95c115a8906141c1)
- 日期：2026-09-22；作者：wcrzlh
- 涉及文件：

  - `examples/qwen3_8/train_online.yaml`
  - `hyper_parallel/data/text/chat_template.py`
  - `hyper_parallel/trainer/config/data.py`
  - `hyper_parallel/trainer/text_trainer.py`
  - `tests/ut/data/text/test_chat_template.py`

### 09. feat: support opt-in tools in tokenizer templates

- 提交：[`3ac14a0d`](https://github.com/wcrzlh/hyper-parallel/commit/3ac14a0d58fb8c7c8f943334d07b9b02fcb3f6e1)
- 日期：2026-09-22；作者：wcrzlh
- 涉及文件：

  - `hyper_parallel/data/text/build_data_transform.py`
  - `hyper_parallel/data/text/chat_template.py`
  - `tests/ut/auto_models/components/datasets/llm/test_build_data_transform.py`
  - `tests/ut/data/text/test_chat_template.py`

### 10. feat: add offline DCP to Hugging Face conversion (#2)

- 提交：[`21da0de4`](https://github.com/wcrzlh/hyper-parallel/commit/21da0de49c19378e6ac8d3a0d788736649aef308)
- 日期：2026-09-24；作者：Chaoran Wei
- 涉及文件：

  - `examples/qwen3_8/tools/convert_dcp_to_hf.py`
  - `hyper_parallel/core/distributed_checkpoint/api.py`
  - `hyper_parallel/models/_transformers/checkpoint_loader.py`
  - `tests/ut/auto_models/_transformers/test_checkpoint_manager_save.py`
  - `tests/ut/core/distributed_checkpoint/test_api.py`

### 11. feat: support ploting loss curves/save ckpt by timestep (#3)

- 提交：[`58bd8517`](https://github.com/wcrzlh/hyper-parallel/commit/58bd85174505576463ccc74bcf68ac3da1b4c8d7)
- 日期：2026-09-24；作者：Chaoran Wei
- 涉及文件：

  - `examples/qwen3_8/tools/plot_loss.py`
  - `examples/qwen3_8/train_online.yaml`
  - `hyper_parallel/components/checkpoint/config.py`
  - `hyper_parallel/trainer/callbacks/checkpoint_callback.py`
  - `tests/ut/auto_models/checkpoint/test_hf_checkpoint_callback.py`
  - `tests/ut/trainer/test_checkpoint_callback.py`

### 12. fix: handle leading system messages in tokenizer templates (#4)

- 提交：[`ad1c6d58`](https://github.com/wcrzlh/hyper-parallel/commit/ad1c6d58a7529c37fdbd99a8d9715919b337fb56)
- 日期：2026-09-28；作者：Chaoran Wei
- 涉及文件：

  - `hyper_parallel/data/text/chat_template.py`
  - `tests/ut/data/text/test_chat_template.py`

### 13. feat: add Qwen3.6 support (#5)

- 提交：[`c24eaec5`](https://github.com/wcrzlh/hyper-parallel/commit/c24eaec573a5d63b9f6f1b6b5f8d3a329bcb6958)
- 日期：2026-09-30；作者：Chaoran Wei
- 涉及文件：

  - `examples/qwen3_6_moe/launch_1node_8dies.sh`
  - `examples/qwen3_6_moe/train_online.yaml`
  - `examples/qwen3_6_moe/train_text.py`
  - `examples/qwen3_8/train.yaml`
  - `examples/qwen3_8/train_online.yaml`
  - `hyper_parallel/distributed/_builder/planner.py`
  - `hyper_parallel/models/qwen3_5_moe/__init__.py`
  - `hyper_parallel/models/qwen3_5_moe/adapter/__init__.py`
  - `hyper_parallel/models/qwen3_5_moe/adapter/registration.py`
  - `hyper_parallel/models/registry.py`
  - `tests/ut/auto_models/models/qwen3_5_moe/test_adapter_contracts.py`

### 14. feat: add Qwen3.6 MoE expert parallel training (#6)

- 提交：[`8ef8be6e`](https://github.com/wcrzlh/hyper-parallel/commit/8ef8be6eacf701b5617283771732c297eed0f8fe)
- 日期：2026-10-08；作者：Chaoran Wei
- 涉及文件：

  - `examples/qwen3_6_moe/README.md`
  - `examples/qwen3_6_moe/loss_comparison.png`
  - `examples/qwen3_6_moe/train_online.yaml`
  - `hyper_parallel/distributed/expert_parallel/experts.py`
  - `hyper_parallel/models/qwen3_5_moe/adapter/distributed/__init__.py`
  - `hyper_parallel/models/qwen3_5_moe/adapter/distributed/expert_parallel.py`
  - `tests/ut/auto_models/models/qwen3_5_moe/test_adapter_contracts.py`
  - `tests/ut/dual_mode_dtensor/test_ep_moe_local_region.py`

### 15. fix(data): align assistant loss mask with inference prefix

- 提交：[`ace55902`](https://github.com/wcrzlh/hyper-parallel/commit/ace55902498a6a72ce13c1377089ae8493ce2868)
- 日期：2026-10-09；作者：wcrzlh
- 涉及文件：

  - `hyper_parallel/data/text/chat_template.py`
  - `tests/ut/data/conftest.py`
  - `tests/ut/data/text/test_chat_template.py`

### 16. feat: add Decif length distribution analyzer

- 提交：[`4534c6fb`](https://github.com/wcrzlh/hyper-parallel/commit/4534c6fbf7e2015616d7551c9b7a275d5844784c)
- 日期：2026-10-10；作者：wcrzlh
- 涉及文件：

  - `examples/qwen3_8/README.md`
  - `examples/qwen3_8/tools/analyze_decif_lengths.py`

### 17. chore: configure Qwen3.8 fp32 main weights and learning rate

- 提交：[`571bdaf6`](https://github.com/wcrzlh/hyper-parallel/commit/571bdaf64b6b065e8894c17325a5898553b9ac66)
- 日期：2026-10-12；作者：wcrzlh
- 涉及文件：

  - `examples/qwen3_8/train_online.yaml`

### 18. fix: preserve DTensor state with fp32 main parameters

- 提交：[`ada81ddc`](https://github.com/wcrzlh/hyper-parallel/commit/ada81ddc8e01f493c8a955403539ab1d56627bb9)
- 日期：2026-10-12；作者：wcrzlh
- 涉及文件：

  - `hyper_parallel/components/optim/mixed_precision_optimizer.py`
  - `hyper_parallel/core/optimizer/dtensor_compat.py`
  - `hyper_parallel/models/_transformers/model_builder.py`
  - `hyper_parallel/trainer/base.py`
  - `hyper_parallel/trainer/vlm_trainer.py`
  - `tests/ut/auto_models/trainer/test_mixed_precision_optimizer.py`

### 19. fix: shuffle all samples before dropping cyclic sampler tail

- 提交：[`e192b177`](https://github.com/wcrzlh/hyper-parallel/commit/e192b177f9dfd4e873cfc7e45a37ed45a6c5b2d1)
- 日期：2026-10-12；作者：wcrzlh
- 涉及文件：

  - `examples/qwen3_8/train_online.yaml`
  - `hyper_parallel/data/parallel/batch_sampler.py`
  - `tests/ut/data/test_cyclic_batch_sampler.py`

### 20. feat: add GDN P2P and attention KV all-gather CP configuration

- 提交：[`3e35b8e8`](https://github.com/wcrzlh/hyper-parallel/commit/3e35b8e8edb170ac7e295daa0c274494c53e43a4)
- 日期：2026-10-13；作者：wcrzlh
- 涉及文件：

  - `examples/qwen3_8/train_online_p2p_kv_allgather.yaml`
  - `hyper_parallel/models/qwen3_5/cp_batch.py`
  - `tests/torch/context_parallel/_test_gdn_p2p_kv_allgather.py`
  - `tests/torch/context_parallel/test_gdn_p2p_kv_allgather.py`
  - `tests/ut/auto_models/models/qwen3_5/test_cp_batch.py`

### 21. feat: add chunk loss adapter for Qwen3.5 text models

- 提交：[`846dbc5c`](https://github.com/wcrzlh/hyper-parallel/commit/846dbc5ca72565a57d097b7372908e5a94cb760d)
- 日期：2026-10-15；作者：wcrzlh
- 涉及文件：

  - `hyper_parallel/models/qwen3_5/adapter/chunk_loss.py`
  - `hyper_parallel/models/qwen3_5/adapter/registration.py`
  - `tests/ut/auto_models/test_qwen3_5_chunk_loss.py`

### 22. feat: support configurable root FSDP wrapping

- 提交：[`3d803e82`](https://github.com/wcrzlh/hyper-parallel/commit/3d803e826d88453458204cc57b7286ea1d04e5d2)
- 日期：2026-10-15；作者：wcrzlh
- 涉及文件：

  - `examples/qwen3_8/train_online.yaml`
  - `hyper_parallel/components/losses/chunked_cross_entropy.py`
  - `hyper_parallel/distributed/_builder/fsdp_adapter.py`
  - `hyper_parallel/models/build_options.py`
  - `tests/ut/auto_models/test_build_options.py`
  - `tests/ut/auto_models/test_fsdp_root_wrap.py`

### 23. fix: restore expert weight layout during offline HF export

- 提交：[`3716b1d2`](https://github.com/wcrzlh/hyper-parallel/commit/3716b1d28c76b9ec610665f2acd26075fb7eb7c9)
- 日期：2026-10-16；作者：wcrzlh
- 涉及文件：

  - `examples/qwen3_8/tools/convert_dcp_to_hf.py`
  - `hyper_parallel/models/_transformers/checkpoint_loader.py`
  - `tests/ut/auto_models/_transformers/test_checkpoint_manager_save.py`

### 24. refactor: clarify per-turn assistant loss mask boundaries

- 提交：[`163b37c9`](https://github.com/wcrzlh/hyper-parallel/commit/163b37c9c689a740da1b9b66a106eeccdd10724f)
- 日期：2026-10-16；作者：wcrzlh
- 涉及文件：

  - `hyper_parallel/data/text/chat_template.py`
  - `tests/ut/data/text/test_chat_template.py`

### 25. fix: 及时释放 FSDP 梯度转换和重排后的源存储

- 提交：[`34b955c8`](https://github.com/wcrzlh/hyper-parallel/commit/34b955c887dd3c7d182cd20e8aed655e7f7fbe3f)
- 日期：2026-10-16；作者：mxysfive（协作者修复）
- 涉及文件：

  - `hyper_parallel/core/fully_shard/hsdp_param.py`
  - `tests/ut/core/fully_shard/test_param.py`

### 26. debug: add offline chat template dataset validation

- 提交：[`b0b9abe3`](https://github.com/wcrzlh/hyper-parallel/commit/b0b9abe3d82c38960c042fbdbbd70ec275786f62)
- 日期：2026-10-16；作者：wcrzlh
- 涉及文件：

  - `examples/qwen3_8/tools/validate_chat_templates.py`
  - `tests/ut/data/text/test_validate_chat_templates.py`

### 27. fix: preserve token boundaries for tool-only assistant messages

- 提交：[`23582ba2`](https://github.com/wcrzlh/hyper-parallel/commit/23582ba26cfcca95474c867b06d073173c0bb0b8)
- 日期：2026-10-16；作者：wcrzlh
- 涉及文件：

  - `hyper_parallel/data/text/build_data_transform.py`
  - `hyper_parallel/data/text/chat_template.py`
  - `tests/ut/data/text/test_chat_template.py`

### 28. feat: add Qwen3.6 MoE hybrid CP training config

- 提交：[`9c890b2d`](https://github.com/wcrzlh/hyper-parallel/commit/9c890b2dc36e3411291821080b83db839a0e63a0)
- 日期：2026-10-16；作者：wcrzlh
- 涉及文件：

  - `examples/qwen3_6_moe/train_online.yaml`
  - `examples/qwen3_6_moe/train_online_p2p_kv_allgather.yaml`

### 29. fix: preserve fused MoE weights in offline HF export

- 提交：[`41b6d8d7`](https://github.com/wcrzlh/hyper-parallel/commit/41b6d8d7b92d6d9033d22c768740e93941f663ef)
- 日期：2026-10-16；作者：wcrzlh
- 涉及文件：

  - `examples/qwen3_8/tools/convert_dcp_to_hf.py`

### 30. feat: enable embedding FSDP and chunk loss for Qwen text models

- 提交：[`77352fd4`](https://github.com/wcrzlh/hyper-parallel/commit/77352fd437efc478c5e83251451dab57d8777909)
- 日期：2026-10-19；作者：wcrzlh
- 涉及文件：

  - `examples/qwen3_6_moe/train_online.yaml`
  - `examples/qwen3_6_moe/train_online_p2p_kv_allgather.yaml`
  - `examples/qwen3_8/train.yaml`
  - `examples/qwen3_8/train_online.yaml`
  - `examples/qwen3_8/train_online_p2p_kv_allgather.yaml`
  - `hyper_parallel/models/qwen3_5/adapter/chunk_loss.py`
  - `hyper_parallel/models/qwen3_5_moe/adapter/registration.py`
  - `tests/ut/auto_models/test_fsdp_root_wrap.py`
  - `tests/ut/auto_models/test_qwen3_5_chunk_loss.py`

### 31. fix: preserve DTensor layouts for uneven FSDP checkpoint resume

- 提交：[`84d21abd`](https://github.com/wcrzlh/hyper-parallel/commit/84d21abd13ffd3ba277556a69c0abb9b603c2278)
- 日期：2026-10-19；作者：mxysfive（协作者修复）
- 涉及文件：

  - `hyper_parallel/core/dtensor/dtensor.py`
  - `tests/ut/core/dtensor/test_dtensor_to.py`

### 32. fix: preserve Qwen3.6 thinking history and configure scheduler lr

- 提交：[`b84b9f34`](https://github.com/wcrzlh/hyper-parallel/commit/b84b9f3447d0539d43becd5b2c06977b90bdccda)
- 日期：2026-10-19；作者：wcrzlh
- 涉及文件：

  - `examples/qwen3_6_moe/train_online.yaml`
  - `examples/qwen3_6_moe/train_online_p2p_kv_allgather.yaml`
  - `tests/ut/data/text/test_chat_template.py`

### 33. fix: load training template options in validator

- 提交：[`6959373f`](https://github.com/wcrzlh/hyper-parallel/commit/6959373f8a07aba5775132395a9d5959db7d755f)
- 日期：2026-10-20；作者：wcrzlh
- 涉及文件：

  - `examples/qwen3_8/tools/validate_chat_templates.py`
  - `tests/ut/data/text/test_validate_chat_templates.py`

### 34. fix: support uneven DTensor gathering for weight export

- 提交：[`04357245`](https://github.com/wcrzlh/hyper-parallel/commit/04357245f6ff10f7edf7bc90ccaa07406f8bee93)
- 日期：2026-10-21；作者：wcrzlh
- 涉及文件：

  - `hyper_parallel/core/dtensor/tensor_redistribution.py`
  - `hyper_parallel/models/_transformers/checkpoint_loader.py`
  - `tests/torch/shard/_test_uneven_full_tensor.py`
  - `tests/torch/shard/test_uneven_full_tensor.py`
  - `tests/ut/auto_models/_transformers/test_checkpoint_manager_save.py`
  - `tests/ut/core/dtensor/test_tensor_redistribution.py`

### 35. refactor: migrate multimodal data processing to Omni

- 提交：[`3069fb76`](https://github.com/wcrzlh/hyper-parallel/commit/3069fb7684f562d30e36e2e7f5443445be0caaa4)
- 日期：2026-10-21；作者：wcrzlh
- 涉及文件：

  - `examples/data/omni_data.yaml`
  - `hyper_parallel/data/__init__.py`
  - `hyper_parallel/data/constants.py`
  - `hyper_parallel/data/omni/README.md`
  - `hyper_parallel/data/omni/__init__.py`
  - `hyper_parallel/data/omni/build_dataset.py`
  - `hyper_parallel/data/omni/collator.py`
  - `hyper_parallel/data/omni/constants.py`
  - `hyper_parallel/data/omni/dynamic_batch.py`
  - `hyper_parallel/data/omni/get_batch.py`
  - `hyper_parallel/data/omni/loader.py`
  - `hyper_parallel/data/omni/omni_transform.py`
  - `hyper_parallel/data/omni/packing.py`
  - `hyper_parallel/data/omni/parallel.py`
  - `hyper_parallel/data/vlm/__init__.py`
  - `hyper_parallel/data/vlm/build_data_transform.py`
  - `hyper_parallel/data/vlm/build_processor.py`
  - `hyper_parallel/data/vlm/collator.py`
  - `hyper_parallel/data/vlm/constants.py`
  - `hyper_parallel/data/vlm/dataset.py`
  - `hyper_parallel/data/vlm/get_batch.py`
  - `hyper_parallel/trainer/vlm_trainer.py`
  - `tests/ut/data/omni/test_omni_data.py`
  - `tests/ut/data/vlm/__init__.py`
  - `tests/ut/data/vlm/test_init.py`

### 36. feat: add encoder parallel support for Qwen VLM

- 提交：[`d66a76f3`](https://github.com/wcrzlh/hyper-parallel/commit/d66a76f30bae0a41ea6ee2fa8596624a2943b225)
- 日期：2026-10-21；作者：wcrzlh
- 涉及文件：

  - `hyper_parallel/distributed/_builder/forward_rewriter.py`
  - `hyper_parallel/distributed/_builder/planner.py`
  - `hyper_parallel/distributed/encoder_parallel/README.md`
  - `hyper_parallel/distributed/encoder_parallel/__init__.py`
  - `hyper_parallel/distributed/encoder_parallel/attention.py`
  - `hyper_parallel/distributed/encoder_parallel/batch_input.py`
  - `hyper_parallel/distributed/encoder_parallel/config.py`
  - `hyper_parallel/distributed/encoder_parallel/contracts.py`
  - `hyper_parallel/distributed/encoder_parallel/exchange.py`
  - `hyper_parallel/distributed/encoder_parallel/mesh.py`
  - `hyper_parallel/distributed/encoder_parallel/planning.py`
  - `hyper_parallel/distributed/encoder_parallel/runtime.py`
  - `hyper_parallel/distributed/mesh.py`
  - `hyper_parallel/models/_transformers/model_builder.py`
  - `hyper_parallel/models/adapter_spec.py`
  - `hyper_parallel/models/qwen3_5_moe/adapter/encoder_parallel.py`
  - `hyper_parallel/models/qwen3_5_moe/adapter/registration.py`
  - `hyper_parallel/models/replacement.py`
  - `hyper_parallel/trainer/config/parallelism.py`
  - `hyper_parallel/trainer/runtime/distributed.py`
  - `hyper_parallel/trainer/text_trainer.py`
  - `hyper_parallel/trainer/vlm_trainer.py`
  - `tests/torch/encoder_parallel/_qwen_encoder_impl.py`
  - `tests/torch/encoder_parallel/test_qwen_encoder_distributed.py`
  - `tests/ut/distributed/encoder_parallel/test_planning.py`
  - `tests/ut/distributed/encoder_parallel/test_qwen_encoder.py`

### 37. fix: export reloadable HF model metadata and text weight keys

- 提交：[`689ea7a2`](https://github.com/wcrzlh/hyper-parallel/commit/689ea7a26ecd32dda0022fa362e183a73a64b434)
- 日期：2026-10-21；作者：wcrzlh
- 涉及文件：

  - `hyper_parallel/models/_transformers/checkpoint_loader.py`
  - `tests/ut/auto_models/_transformers/test_checkpoint_manager_save.py`

### 38. feat: support Qwen VLM context parallel training

- 提交：[`403b2b0d`](https://github.com/wcrzlh/hyper-parallel/commit/403b2b0d87c00813c7978ad6a877747bdf7fee18)
- 日期：2026-10-22；作者：wcrzlh
- 涉及文件：

  - `examples/qwen3_6_moe/train_online_vlm.yaml`
  - `hyper_parallel/data/omni/omni_transform.py`
  - `hyper_parallel/distributed/_builder/fsdp_adapter.py`
  - `hyper_parallel/distributed/_builder/planner.py`
  - `hyper_parallel/models/adapter_spec.py`
  - `hyper_parallel/models/qwen3_5_moe/adapter/encoder_parallel.py`
  - `hyper_parallel/models/qwen3_5_moe/adapter/registration.py`
  - `hyper_parallel/models/qwen3_5_moe/vlm_cp.py`
  - `hyper_parallel/trainer/vlm_trainer.py`
  - `tests/torch/encoder_parallel/_qwen_vlm_cp_impl.py`
  - `tests/torch/encoder_parallel/test_qwen_vlm_cp.py`
  - `tests/ut/auto_models/test_fsdp_root_wrap.py`
  - `tests/ut/data/omni/test_omni_data.py`
  - `tests/ut/distributed/encoder_parallel/test_qwen_encoder.py`

### 39. feat: support parquet VLM data and add eight-card launch example

- 提交：[`a43e6740`](https://github.com/wcrzlh/hyper-parallel/commit/a43e674077f49df1a3fad9872660d40d461534a5)
- 日期：2026-10-22；作者：wcrzlh
- 涉及文件：

  - `.codespellrc`
  - `examples/qwen3_6_moe/README.md`
  - `examples/qwen3_6_moe/launch_1node_8dies.sh`
  - `examples/qwen3_6_moe/launch_vlm_1node_8dies.sh`
  - `examples/qwen3_6_moe/train_online_vlm.yaml`
  - `examples/qwen3_8/launch_1node_8dies.sh`
  - `examples/qwen3_8/launch_32nodes_256dies.sh`
  - `hyper_parallel/data/omni/README.md`
  - `hyper_parallel/data/omni/build_dataset.py`
  - `hyper_parallel/data/omni/omni_transform.py`
  - `tests/ut/data/omni/test_build_dataset.py`
  - `tests/ut/data/omni/test_omni_transform.py`

### 40. fix: allow replacement characters in VLM assistant responses

- 提交：[`c339a613`](https://github.com/wcrzlh/hyper-parallel/commit/c339a613347878276d27af6d38685f2c731ca893)
- 日期：2026-10-23；作者：wcrzlh
- 涉及文件：

  - `hyper_parallel/data/omni/omni_transform.py`

### 41. feat: support chunked loss for VLM context parallel training

- 提交：[`e9e17785`](https://github.com/wcrzlh/hyper-parallel/commit/e9e17785b16c504754ce0d9661b38d728e57b2dd)
- 日期：2026-10-23；作者：wcrzlh
- 涉及文件：

  - `examples/qwen3_6_moe/train_online_vlm.yaml`
  - `examples/qwen3_6_moe/train_online_vlm_p2p_kv_allgather.yaml`
  - `hyper_parallel/models/qwen3_5_moe/adapter/registration.py`
  - `hyper_parallel/models/qwen3_5_moe/adapter/vlm_chunk_loss.py`
  - `hyper_parallel/models/qwen3_5_moe/vlm_cp.py`
  - `tests/torch/context_parallel/_test_vlm_chunk_loss.py`
  - `tests/torch/context_parallel/test_vlm_chunk_loss.py`
  - `tests/ut/auto_models/test_qwen_vlm_chunk_loss.py`

## 13. 合并记录

合并提交用于记录集成关系，不再重复计算为独立功能。

| 提交 | 日期 | 合并说明 |
| --- | --- | --- |
| `d671cea2` | 2026-09-22 | Merge pull request #1 from wcrzlh/qwen3.8_dev_beta |
| `4ff51314` | 2026-10-13 | Merge pull request #7 from wcrzlh/qwen3.8_dev_beta |
| `65db7645` | 2026-10-19 | Merge pull request #9 from wcrzlh/qwen3.8_dev_beta |
| `6e646e9e` | 2026-10-19 | Merge pull request #10 from mxysfive/fix/fsdp-dcp-resume-layout |
| `a4755cca` | 2026-10-21 | Merge pull request #11 from wcrzlh/qwen3.8_dev_beta |
| `75506890` | 2026-10-23 | Merge pull request #16 from wcrzlh/qwen3.8_dev_beta |

## 14. 复核方法

```bash
git log --no-merges --reverse --stat 052967ba..75506890
git diff --stat 052967ba 75506890
git show <commit>
```

本文以本地 Git 历史、最终代码和仓库 README 为依据。没有重新查询远端 PR 状态；正文里的历史运行数字来自仓库记录。
