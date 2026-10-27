# Qwen rebase 设计与边界条件核对

日期：2026-10-27。基线：`upstream/master` / `30c8ce78`。
审查起点：`qwen3.8_dev` / `8e00aa9d`；本轮修复、测试与本文一并提交。

## 结论

**不能认定此前已覆盖所有 corner case。** 本次发现并修复了 7 类缺口，尤其是与 master
异步/分块 EP 的兼容问题。修复后定向回归、master EP 回归和 CPU/Gloo 数值测试通过。
NPU 算子、完整 FSDP 混合训练、故障恢复及性能仍有未验证项，不能用 CPU 结果替代。

历史归属核对：原分支 41 个非合并提交中，38 个在新基线上重放，3 个由 master 的已有实现
替代；另外有 1 个 rebase 接口适配提交。此处按 Git 历史追溯，不把协作者提交冒认为独立创作。
`git range-diff` 的逐提交结果见本文末尾。

## 文档依据与范围

当前仓库没有找到独立的 Qwen 设计原稿。已向用户询问额外路径；在得到该文件前，以以下
已有说明作为功能契约，**不声称已核对某份尚未提供的外部设计稿**：

| 文档 | 核对内容 |
| --- | --- |
| `QWEN_DEVELOPMENT_SUMMARY.md` | 原开发历史、最终保留能力及原始提交映射；它是回顾记录而非独立设计规范 |
| `examples/qwen3_8/README.md` | 文本训练、后端、配置、启动与历史环境 |
| `examples/qwen3_6_moe/README.md` | MoE、CP/EP/FSDP 组合、图文示例 |
| `hyper_parallel/data/README.md` | master 的 source → transform → batching → parallel → runtime 分层 |
| `docs/guide/data/omni_online_data_guide.md` | Omni 编码、标签、媒体字段及模型扩展边界 |
| `docs/guide/chunk_loss.md` | 全局对齐目标、sum/count、一阶梯度、避免完整 logits |
| `docs/guide/distributed_checkpoint.md` | DCP 保存/恢复、异步完成与重切分 |
| `docs/guide/trainer/yaml_config.md` | master 的配置解析契约 |
| `hyper_parallel/distributed/encoder_parallel/README.md` | Qwen vision bridge 与 encoder runtime 的支持边界 |
| `.agent/rules/distributed.md`、`.agent/skills/code-review/review-checklist.md` | 通信可微性、等待、内存、布局和共享调用方检查 |

检查包含提交归属、差异及调用方、文档与实现契约、边界测试和可运行的 CPU 数值验证。
AscendC/Triton 内核只检查接口与约束，不把未执行的设备代码标记为数值验证通过。

## 本轮发现与修复

### R1：EP 改动破坏了 master 的通用 dispatch 契约（高风险）

- 原因：本分支为 fused unpermute 移除了 `_EPDispatch.source_indices`，并缩减
  `_aggregate_ep_outputs` 的参数；master 新增的异步 dispatch/combine 和 chunk 路径仍依赖这些字段。
- 影响：异步路径访问不存在的字段或参数数量不匹配；capacity 丢弃部分路由后固定 top-k reshape
  不再成立；零 token 时 `view(0, -1)` 报错。
- 修复：恢复 source mapping 和原聚合签名。通用路径保留 master 的加权 scatter-add；只在无丢弃
  的同步完整路由、非空输入、NPU 分支显式启用 fused unpermute。
- 证据：静态检查发现真实调用方错误；新增空输入、重复路由、非均匀丢弃的输出/梯度测试；
  `tests/ut/core/expert_parallel` 中 master 的 chunk、capacity、异步/延迟等待等测试一并回归。
- 文件：`hyper_parallel/distributed/expert_parallel/experts.py`、
  `tests/ut/distributed/test_expert_aggregation.py`。

### R2：CP=1 提前返回绕过 runtime adapter

- 原因：`QwenVLMParallelBatch.__call__` 在 CP=1 分支没有执行 `_build_runtime_inputs`。
- 影响：配置了 encoder runtime 或其他模型扩展时，plan 不会送到 forward。
- 修复：CP=1 同样构建并合入 runtime inputs，保留公共碰撞检查；不污染 loss inputs。
- 测试：`test_cp1_calls_runtime_adapter`。

### R3：encoder bridge 不接受 chunk loss 输出

- 原因：dummy-work anchor 只加到 `output.loss`，而 chunk objective 返回 `loss_sum`。
- 影响：CP=1 encoder + chunk loss 组合报错，空 producer 的梯度连接也无法保留。
- 修复：优先支持 `loss_sum`，否则处理普通 `loss`；仍在 finally 中清理 feature context。
- 测试：`test_chunk_loss_anchor_and_cleanup`，检查 routed feature 和 dummy anchor 的梯度。

### R4：CP forward 对 HF 隐式选项校验不完整

- 原因：只看调用参数中的 `output_router_logits`，忽略 text config 的默认值。
- 影响：可能启用 router 输出，却在 CP 重算 loss 时忽略辅助目标；非完整 logits/tuple 输出也不满足
  桥接代码所需契约。
- 修复：检查有效 router 配置，明确拒绝不支持的辅助 loss、`return_dict=False`、非零/张量形式的
  `logits_to_keep`，并显式使用 `use_cache=False`。
- 测试：`test_cp_rejects_implicit_router_loss_and_partial_logits`；两进程 CP 数值回归通过。

### R5：序列对齐长度缺少边界校验

- 原因：仅检查整除，0、负数和浮点数未被正确拒绝。
- 修复：要求正整数且排除 bool，然后检查 `64 * cp_size` 整除。
- 测试：`test_invalid_alignment_is_rejected`。

### R6：本地 JSON 相对图片路径在 rebase 后遗漏

- 原因：共享 reader 中 JSONL 和新增 Parquet 保留来源信息，但 JSON 未附带来源文件路径。
- 影响：`images/a.png` 相对于进程目录而非 JSON 文件目录解析。
- 修复：在 master 的共享 loader 中为本地 JSON 与 Parquet 统一附加来源；保留远端路径原加载行为。
  不重新引入独立 Omni reader。
- 测试：`test_json_image_paths_use_source_directory` 先复现再通过；同时覆盖多文件、ratio split、
  explicit split、worker pickle、图片延迟解码及空源。

### R7：离线导出未拒绝缺失的模型权重

- 原因：只校验已提供 tensor 的名称/shape，没有校验模型 state_dict 所需的完整键集合。
- 影响：形状正确但缺权重的 DCP 可能被导出，随后加载时补出未训练权重。
- 修复：离线 layout normalization 先拒绝缺失参数或持久化 buffer；不改变 live export 路径。
- 测试：`test_dcp_missing_parameters_are_rejected`，使用缺 bias 的真实 Linear meta model。

## 逐条设计/功能对照

“通过”表示所列契约有实现和对应证据；不扩展为所有硬件/组合均通过。
“部分”表示接口或单元测试通过，仍有表后列出的系统验证缺口。

| # | 功能契约 | rebase 后实现/归属 | 重点边界与证据 | 结论 |
| --- | --- | --- | --- | --- |
| 1 | 训练入口、单机/多机 launch | 保留 Qwen examples 和镜像增量 | Python AST、bash 语法；配置解析已迁移 | 部分：未重建镜像/跑 256 die |
| 2 | AscendC/Triton GDN 替换保持权重结构 | 保留 module replacement 工厂 | 不改参数名、后端缺失、shape/dtype/状态拒绝；`test_gated_delta_net.py` | 部分：NPU 数值待验 |
| 3 | GDN Ulysses 的卷积 halo 与可微通信 | 保留 wrapper 增量，使用 master communication | 维度、head 整除、cache/packed 拒绝、primitive 恢复；`test_gdn_cp_wrapper.py` | 部分：设备路径待验 |
| 4 | GDN P2P + attention KV all-gather | 复用 master primitive 与新配方 | CP=2 输出/梯度、重计算、64-token 对齐；上轮 CPU ST 通过 | CPU 通过 |
| 5 | GDN 不允许独立对话直接 packing | 模型专用 collator/batch | 单样本、非空、labels 对齐、TP=1、local length；`test_cp_batch.py` | 通过 |
| 6 | Qwen MoE 注册与 EP 参数契约 | 模型 adapter 增量 | dense/text 与 VLM 入口、角色、fused experts、共享专家；adapter tests | 通过 |
| 7 | EP 输出恢复和 top-k 权重梯度 | R1 恢复 master 通用契约 | 空 rank、重复 token、capacity drop、异步/chunk；master EP UT + 新测试 | CPU 通过；NPU fused 待验 |
| 8 | tokenizer 原生模板选项 | 新参数接入 master text transform | 保留 thinking、非法/reserved 参数、显式 tools、前置 system | `test_chat_template.py` 通过 |
| 9 | assistant-only 多轮监督边界 | 原开发逻辑保留 | tool-only、空 thinking、BPE 合并、历史重写拒绝、loss_mask | 文本模板测试通过 |
| 10 | ShareGPT / instruction / tools 输入 | 扩展 master `build_text_transform` | 字段缺失、非法类型、JSON tools、无监督样本过滤 | transform UT 通过 |
| 11 | 不覆盖 master plaintext tail 行为 | master PlaintextTransform 保留 | 短尾保留/丢弃；`test_plaintext_tail_window.py` | 通过 |
| 12 | 离线模板校验与训练选项一致 | 工具改为 master 新导入路径 | YAML 参数、thinking/tools；`test_validate_chat_templates.py` | 通过 |
| 13 | cyclic sampler 先 shuffle 后 drop | 原采样增量保留 | 尾部公平性、索引与消费位置；`test_cyclic_batch_sampler.py` | 通过 |
| 14 | Omni source/transform/batching 分层 | master 作为实现基础 | 不恢复旧 reader/collator/runtime；独立 source provenance 增量 | 通过 |
| 15 | JSON/JSONL/Parquet 相对媒体路径 | R6 补全 JSON | 多目录、ratio/explicit split、过滤顺序、pickle | 真实文件 UT 通过 |
| 16 | 图片 bytes/path 与占位符绑定 | 保留 Omni transform 增量 | bytes 优先、空字段、数量不匹配、灰度 RGB、损坏字节、懒解码 | Omni transform UT 通过 |
| 17 | 最终 assistant 监督与媒体保护 | 保留 processor 边界补丁 | 合并边界、文本重写、替换字符、零监督、超长不截断 | Omni processor UT 通过；同步修正文档 |
| 18 | VLM 全局 shift 后 CP slice | 模型专用 geometry/targets | 图片跨界、单 rank 无监督、padding 目标屏蔽 | 两进程 VLM CP/Chunk ST 通过 |
| 19 | CP=1 基线与 runtime 组合 | R2 补运行时入口 | 扩展只进入 model inputs，chunk targets 仅 shift 一次 | 新旧 UT 通过 |
| 20 | CP forward 输入/输出契约 | R4/R5 补校验 | config router 默认、cache、tuple/partial logits、非法 alignment | 新增 UT 通过 |
| 21 | encoder runtime 使用 master 框架 | 仅 Qwen bridge/attention 为增量 | isolated visual subtree、TP/CP head 约束、视频/packing 拒绝 | encoder UT 通过 |
| 22 | encoder 特征/梯度与 dummy work | native vision children + master routing | 空 producer、padding 无梯度、图像间隔离、feature context 清理 | UT；上轮 encoder DP/CP ST 通过 |
| 23 | encoder + chunk loss 联用 | R3 支持 loss_sum anchor | feature 梯度、dummy 梯度、普通 loss 仍可用 | 新增 UT 通过 |
| 24 | dense/MoE/VLM chunk loss | 模型 loss adapter + master objective | 非整 chunk、空监督、masked target、logits=None、forward 保留 | 三组 chunk UT 通过 |
| 25 | LM head 不可独立 FSDP 后直接读权重 | objective bind 时拒绝 | 独立 wrapped head、bias、错误 model type/cache | root-wrap/chunk UT 通过 |
| 26 | 精确 token-weighted accumulation | step 级统计完整 microbatches | 1/3/0 token 的不等 microbatch 与梯度、metric hook | 本轮新增 trainer UT 通过 |
| 27 | root FSDP 模块与 reshard 策略 | 原配置增量叠加 master declared/excluded provider | tied/shared 参数拒绝、未知/空 root、owner 与 embedding-only | root-wrap UT 通过；完整 FSDP 待验 |
| 28 | FP32 main params / grad storage | 使用 master 已有实现，不重放重复补丁 | mixed precision optimizer、dtype/layout、存储释放相关单测 | UT 通过 |
| 29 | HSDP 提前应用已完成 RS 输出 | 原增量保留限制条件 | only plain FSDP、replicate size=1、同步与清理 | state UT 通过；stream/device 待验 |
| 30 | DTensor detach/to 保留 uneven layout | 原增量保留 | metadata 独立、Parameter→普通 DTensor、dtype 转换 | DTensor/state UT 通过 |
| 31 | uneven full_tensor 仅导出路径 | 新 detached gather，不冒充可微 redistribute | 空 shard、dim0、mixed/strided/reversed mesh、Partial/autograd/shape 拒绝 | UT + 本轮 4-rank Gloo ST 通过 |
| 32 | HF live export 元数据与权重布局 | 增量合入 master CheckpointManager | 动态 HSDP 类、失败不污染 config、text namespace、fused experts | export UT 通过 |
| 33 | offline DCP export | 保留 model-only CPU load + R7 | no_dist、rank0、错误 shape、方形权重歧义、缺参数 | DCP/export UT 通过 |
| 34 | checkpoint 时间/步数/epoch 保存 | 增量合入 master callback | deadline 跨 rank 一致、重复抑制、train_end、async wait/failure | callback/cadence UT 通过 |
| 35 | checkpoint retention | 原增量保留 | 非管理目录、symlink、不完整 checkpoint、活动保存保护 | cadence UT 通过；断电故障待验 |
| 36 | 保留 master 的共享行为 | EP 契约恢复、source 分层、原 checkpoint/diagnostic hooks | master EP 195 项相关测试，plaintext/runtime/配置/规划回归 | 所列回归通过 |

## 验证记录

| 验证 | 本轮结果 | 证据/说明 |
| --- | --- | --- |
| 扩展定向回归 | **387 passed** | `/tmp/qwen-audit-regression.log`；涵盖 data、Qwen、DTensor、FSDP state、checkpoint、trainer、planner 等 |
| master EP + 聚合 + local-region | **208 passed** | `/tmp/qwen-audit-ep-final.log`；与上一组有重叠，不能相加冒充独立总数 |
| VLM CP + Encoder-DP，2-rank Gloo | 每 rank **1 passed** | `/tmp/qwen-audit-cp-encoder.log`；特征/所有参数梯度对照 |
| VLM CP chunk loss，2-rank Gloo | 每 rank **1 passed** | `/tmp/qwen-audit-cp-chunk.log`；含空监督和图像跨边界 |
| uneven export，4-rank Gloo | 每 rank **1 passed** | `/tmp/qwen-audit-uneven.log`；真实 collectives、CPU offload、safetensors |
| AST、shell syntax、diff whitespace | 通过 | 对分支增量 Python 与 shell 检查，`git diff --check` |
| AGENTS catalog | 通过 | 10 skills / 13 agents |
| 受修复影响的生产文件 Pylint errors-only | 通过 | Python 3.12，`/tmp/qwen-audit-pylint-errors.log` |
| 分支全量 Pylint | **未全绿** | `/tmp/qwen-audit-pylint-direct.log`；见下节，不隐瞒为通过 |

新增 10 个边界测试：CP=1 runtime、alignment、router/output 约束、encoder chunk anchor、缺失权重、
JSON 相对路径、不等 microbatch token 权重，以及 EP 空输入/重复路由/capacity drop 三项。
新增测试以实际内容为准，未将已有 master 测试重复计为新增。

全量 Pylint 的标准 autogit 入口拒绝 GitHub origin（仅支持 GitCode）。因此直接调用同一
`run_pylint_review`，保留仓库 filter。它帮助找出了 R1 的真实错误。
剩余结果包含 `.pylintrc` 仍指定 Python 3.9 所致的 union 类型告警、旧 backend-import 规则与
master Torch-only 代码不一致、HF 动态构造参数的推断告警，以及分支/上游已有风格债。
未批量关闭检查或修改 filter 来制造绿灯；因此当前不宣称质量门禁全部通过。

## 尚未覆盖、需要独立验证的组合

1. **真实 NPU BF16/FP16**：AscendC 与 Triton GDN 前后向、NPU fused unpermute 的顺序/梯度、
   HCCL collective、混合精度误差边界；CPU fallback 和 mock 不能证明这些。
2. **完整训练组合**：FSDP/EP/CP/chunk loss/activation checkpoint/swap 组合，尤其是不同 DP rank
   的有图/无图 batch、视觉 FSDP child hooks 和空 producer 的 collective 顺序。现有 CPU ST
   不等价于完整 FSDP 训练。默认 YAML 的 activation checkpoint 与 encoder opt-in 限制也需按配置核对。
3. **恢复与故障**：真实训练保存后恢复 optimizer、scheduler、dataloader、RNG 并对照后续 loss；
   异步 I/O 错误、进程退出/断电、跨拓扑恢复。当前主要是单元合同与导出集成验证。
4. **性能与显存**：prefetch 全部 microbatches 的峰值、root reshard 策略对其他模型的影响、
   8 卡/256 die 吞吐和长跑稳定性、镜像可重建性。没有重新宣称历史 benchmark。
5. **支持范围之外**：Qwen VLM adapter 的 TP/SP、PP、视频、多样本 packing、router auxiliary loss
   等拒绝分支不是已实现功能；不应因底层通用 runtime 支持而推导该 adapter 已支持。
6. **设计原稿与 lint 债**：若存在仓库外设计稿，需要补做其原始条目映射；全量 lint 未绿仍需
   针对有效规则逐项治理，不能视作已通过发布门禁。

## 逐提交归属对照

`=` 表示补丁基本相同；`!` 表示重放时因 master 结构/接口调整而变化，不等于功能丢失。
“已替代”三项分别为 FP32 main parameter 状态、FSDP 梯度源存储释放、通用 Omni 管线。
本轮修复作为后续提交单独记录，不混入以下 39 个既有提交的统计。

| 原序号 | 原提交 | 新序号 | 新提交 | 关系 | 主题 |
| --- | --- | --- | --- | --- | --- |
| 1 | `820a170f` | 1 | `10087e80` | 适配 master | feat: add Qwen3.8 distributed training support |
| 2 | `3c544c0a` | 2 | `ef42c6b2` | 适配 master | fix: use Triton-Ascend GDN for Qwen3.8 training |
| 3 | `85458dd6` | 3 | `e577b890` | 适配 master | refactor: remove unused Qwen3.8 TP and SwiGLU options |
| 4 | `5fb1ec77` | 4 | `f4d01f0b` | 适配 master | refactor: name GDN module replacements by backend |
| 5 | `937f381e` | 5 | `474e29b1` | 保留 | feat: prepare Qwen3.8 training launch workflows |
| 6 | `191053d4` | 6 | `62c60adf` | 适配 master | feat: add Qwen3.8 A3 training image recipe |
| 7 | `292371e0` | 7 | `918351db` | 适配 master | feat: export Trainer checkpoints in Hugging Face format |
| 8 | `f8a133ad` | 8 | `7a2779b8` | 适配 master | feat: support native tokenizer chat template options |
| 9 | `3ac14a0d` | 9 | `945f13ee` | 适配 master | feat: support opt-in tools in tokenizer templates |
| 10 | `21da0de4` | 10 | `7424e1de` | 适配 master | feat: add offline DCP to Hugging Face conversion (#2) |
| 11 | `58bd8517` | 11 | `ed10dd04` | 适配 master | feat: support ploting loss curves/save ckpt by timestep (#3) |
| 12 | `ad1c6d58` | 12 | `8c67d85a` | 保留 | fix: handle leading system messages in tokenizer templates (#4) |
| 13 | `c24eaec5` | 13 | `39f65439` | 适配 master | feat: add Qwen3.6 support (#5) |
| 14 | `8ef8be6e` | 14 | `af85bb23` | 适配 master | feat: add Qwen3.6 MoE expert parallel training (#6) |
| 15 | `ace55902` | 15 | `2b21300e` | 保留 | fix(data): align assistant loss mask with inference prefix |
| 16 | `4534c6fb` | 16 | `5d93116b` | 保留 | feat: add Decif length distribution analyzer |
| 17 | `571bdaf6` | 17 | `fa672e6f` | 保留 | chore: configure Qwen3.8 fp32 main weights and learning rate |
| 18 | `ada81ddc` | - | `--------` | 已由 master 替代 | fix: preserve DTensor state with fp32 main parameters |
| 19 | `e192b177` | 18 | `6f1bf55d` | 保留 | fix: shuffle all samples before dropping cyclic sampler tail |
| 20 | `3e35b8e8` | 19 | `30ec98e3` | 保留 | feat: add GDN P2P and attention KV all-gather CP configuration |
| 21 | `846dbc5c` | 20 | `5c77571b` | 适配 master | feat: add chunk loss adapter for Qwen3.5 text models |
| 22 | `3d803e82` | 21 | `494c4ecc` | 适配 master | feat: support configurable root FSDP wrapping |
| 23 | `3716b1d2` | 22 | `22de55d5` | 适配 master | fix: restore expert weight layout during offline HF export |
| 24 | `163b37c9` | 23 | `ae87fde9` | 保留 | refactor: clarify per-turn assistant loss mask boundaries |
| 25 | `34b955c8` | - | `--------` | 已由 master 替代 | fix: 及时释放 FSDP 梯度转换和重排后的源存储 |
| 26 | `b0b9abe3` | 24 | `2b3fff2e` | 保留 | debug: add offline chat template dataset validation |
| 27 | `23582ba2` | 25 | `6579fd9b` | 适配 master | fix: preserve token boundaries for tool-only assistant messages |
| 28 | `9c890b2d` | 26 | `63aa8852` | 保留 | feat: add Qwen3.6 MoE hybrid CP training config |
| 29 | `41b6d8d7` | 27 | `46bdc78e` | 保留 | fix: preserve fused MoE weights in offline HF export |
| 30 | `77352fd4` | 28 | `459e9ca6` | 保留 | feat: enable embedding FSDP and chunk loss for Qwen text models |
| 31 | `84d21abd` | 29 | `199decbc` | 保留 | fix: preserve DTensor layouts for uneven FSDP checkpoint resume |
| 32 | `b84b9f34` | 30 | `8b6928b1` | 保留 | fix: preserve Qwen3.6 thinking history and configure scheduler lr |
| 33 | `6959373f` | 31 | `bb91b4ea` | 保留 | fix: load training template options in validator |
| 34 | `04357245` | 32 | `01d71b23` | 适配 master | fix: support uneven DTensor gathering for weight export |
| 35 | `3069fb76` | - | `--------` | 已由 master 替代 | refactor: migrate multimodal data processing to Omni |
| 36 | `d66a76f3` | 33 | `c46d06bb` | 适配 master | feat: add encoder parallel support for Qwen VLM |
| 37 | `689ea7a2` | 34 | `4a33d2b5` | 适配 master | fix: export reloadable HF model metadata and text weight keys |
| 38 | `403b2b0d` | 35 | `69606d1e` | 适配 master | feat: support Qwen VLM context parallel training |
| 39 | `a43e6740` | 36 | `3a7e0f14` | 适配 master | feat: support parquet VLM data and add eight-card launch example |
| 40 | `c339a613` | 37 | `ce5901ac` | 保留 | fix: allow replacement characters in VLM assistant responses |
| 41 | `e9e17785` | 38 | `c6c30961` | 适配 master | feat: support chunked loss for VLM context parallel training |
| - | `--------` | 39 | `8e00aa9d` | 新增适配提交 | fix: adapt Qwen training increments to current master |

## 文件增量清单

下面是相对于 master 的跟踪文件增量清单，便于复核范围；它不是逐文件宣称有硬件验证。
低层设备内核的验证边界见上文。本报告及本轮新增测试不在上述既有跟踪文件清单中。

```text
.dockerignore
docker/Dockerfile.qwen3_8-a3
docker/README.md
docs/guide/data/omni_online_data_guide.md
examples/qwen3_6_moe/README.md
examples/qwen3_6_moe/launch_1node_8dies.sh
examples/qwen3_6_moe/launch_vlm_1node_8dies.sh
examples/qwen3_6_moe/loss_comparison.png
examples/qwen3_6_moe/train_online.yaml
examples/qwen3_6_moe/train_online_p2p_kv_allgather.yaml
examples/qwen3_6_moe/train_online_vlm.yaml
examples/qwen3_6_moe/train_online_vlm_p2p_kv_allgather.yaml
examples/qwen3_6_moe/train_text.py
examples/qwen3_8/README.md
examples/qwen3_8/launch_1node_8dies.sh
examples/qwen3_8/launch_32nodes_256dies.sh
examples/qwen3_8/offline_4k_loss.png
examples/qwen3_8/requirements.txt
examples/qwen3_8/tools/analyze_decif_lengths.py
examples/qwen3_8/tools/convert_dcp_to_hf.py
examples/qwen3_8/tools/plot_loss.py
examples/qwen3_8/tools/validate_chat_templates.py
examples/qwen3_8/train.yaml
examples/qwen3_8/train_online.yaml
examples/qwen3_8/train_online_p2p_kv_allgather.yaml
examples/qwen3_8/train_text.py
hyper_parallel/components/checkpoint/config.py
hyper_parallel/components/functional/_ascendc_gdn.py
hyper_parallel/components/functional/_gdn_kkt.py
hyper_parallel/components/functional/_gdn_tril.py
hyper_parallel/components/functional/gated_delta_rule.py
hyper_parallel/components/functional/moe_token_unpermute.py
hyper_parallel/components/losses/chunked_cross_entropy.py
hyper_parallel/components/modules/__init__.py
hyper_parallel/components/modules/gdn_ascendc.py
hyper_parallel/components/modules/gdn_triton.py
hyper_parallel/core/distributed_checkpoint/api.py
hyper_parallel/core/dtensor/dtensor.py
hyper_parallel/core/dtensor/tensor_redistribution.py
hyper_parallel/core/fully_shard/hsdp_state.py
hyper_parallel/data/omni/README.md
hyper_parallel/data/omni/omni_transform.py
hyper_parallel/data/online/source.py
hyper_parallel/data/parallel/batch_sampler.py
hyper_parallel/data/text/chat_template.py
hyper_parallel/data/text/text_transform.py
hyper_parallel/distributed/_builder/default_templates.py
hyper_parallel/distributed/_builder/fsdp_adapter.py
hyper_parallel/distributed/_builder/planner.py
hyper_parallel/distributed/context_parallel/__init__.py
hyper_parallel/distributed/context_parallel/wrappers.py
hyper_parallel/distributed/encoder_parallel/README.md
hyper_parallel/distributed/expert_parallel/experts.py
hyper_parallel/models/_transformers/checkpoint_loader.py
hyper_parallel/models/build_options.py
hyper_parallel/models/qwen3_5/adapter/chunk_loss.py
hyper_parallel/models/qwen3_5/adapter/registration.py
hyper_parallel/models/qwen3_5/cp_batch.py
hyper_parallel/models/qwen3_5_moe/__init__.py
hyper_parallel/models/qwen3_5_moe/adapter/__init__.py
hyper_parallel/models/qwen3_5_moe/adapter/distributed/__init__.py
hyper_parallel/models/qwen3_5_moe/adapter/distributed/expert_parallel.py
hyper_parallel/models/qwen3_5_moe/adapter/encoder_parallel.py
hyper_parallel/models/qwen3_5_moe/adapter/registration.py
hyper_parallel/models/qwen3_5_moe/adapter/vlm_chunk_loss.py
hyper_parallel/models/qwen3_5_moe/vlm_cp.py
hyper_parallel/models/registry.py
hyper_parallel/trainer/base.py
hyper_parallel/trainer/callbacks/checkpoint_callback.py
hyper_parallel/trainer/config/data.py
hyper_parallel/trainer/text_trainer.py
hyper_parallel/trainer/vlm_trainer.py
tests/torch/context_parallel/_test_gdn_p2p_kv_allgather.py
tests/torch/context_parallel/_test_vlm_chunk_loss.py
tests/torch/context_parallel/test_gdn_p2p_kv_allgather.py
tests/torch/context_parallel/test_vlm_chunk_loss.py
tests/torch/encoder_parallel/_qwen_encoder_impl.py
tests/torch/encoder_parallel/_qwen_vlm_cp_impl.py
tests/torch/encoder_parallel/test_qwen_encoder_distributed.py
tests/torch/encoder_parallel/test_qwen_vlm_cp.py
tests/torch/shard/_test_uneven_full_tensor.py
tests/torch/shard/test_uneven_full_tensor.py
tests/ut/auto_models/_transformers/test_checkpoint_manager_save.py
tests/ut/auto_models/checkpoint/test_hf_checkpoint_callback.py
tests/ut/auto_models/components/datasets/llm/test_build_data_transform.py
tests/ut/auto_models/components/test_gated_delta_net.py
tests/ut/auto_models/components/test_gdn_cp_wrapper.py
tests/ut/auto_models/models/qwen3_5/test_cp_batch.py
tests/ut/auto_models/models/qwen3_5_moe/test_adapter_contracts.py
tests/ut/auto_models/test_build_options.py
tests/ut/auto_models/test_fsdp_root_wrap.py
tests/ut/auto_models/test_qwen3_5_chunk_loss.py
tests/ut/auto_models/test_qwen_vlm_chunk_loss.py
tests/ut/core/distributed_checkpoint/test_api.py
tests/ut/core/dtensor/test_dtensor_to.py
tests/ut/core/dtensor/test_tensor_redistribution.py
tests/ut/core/fully_shard/test_state.py
tests/ut/data/conftest.py
tests/ut/data/omni/test_build_dataset.py
tests/ut/data/omni/test_omni_data.py
tests/ut/data/omni/test_omni_transform.py
tests/ut/data/test_cyclic_batch_sampler.py
tests/ut/data/text/test_chat_template.py
tests/ut/data/text/test_validate_chat_templates.py
tests/ut/distributed/encoder_parallel/test_qwen_encoder.py
tests/ut/dual_mode_dtensor/test_ep_moe_local_region.py
tests/ut/dual_mode_dtensor/test_planner_golden_plans.py
tests/ut/trainer/test_checkpoint_cadence.py
tests/ut/trainer/test_checkpoint_callback.py
tests/ut/trainer/test_text_trainer.py
```
