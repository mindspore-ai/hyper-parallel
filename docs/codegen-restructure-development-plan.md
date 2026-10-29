# Codegen 重构渐进式开发计划

## 目标

最终训练链路应为：原生 `HyperAutoModel.from_config/from_pretrained` 完成模型构建、replacement、planner、apply、activation、FSDP、materialize/load，再记录 `CodegenSnapshot` 并旁路生成 bundle。训练阶段不再导入 generated modeling、不再从 generated metadata 决定并行化、不再重新构造 meta model、offline mesh 或 planner。

当前版本已经具备 HyperAutoModel 统一入口、共享 ShardingPlanner、custom/native 模型路径和 `from_config()` 配置传递，但仍处于旧 GEN 双路径稳定阶段，核心的“采集原生结果并旁路生成”尚未完成。

## 开发原则

1. 每个阶段独立提交、独立验证。
2. 当前阶段未通过，不进入下一阶段。
3. 先增加观测和旁路能力，再切换数据来源，最后删除旧路径。
4. codegen 失败不能改变原生训练的错误语义。
5. 不以已有 generated 产物替代 fresh generation 验证。
6. codegen 不得新增或长期保留 Trainer、`from_config()`、`from_pretrained()` 的公共参数；阶段 1～6 已存在的 `codegen_session`、`codegen_config` 等透传只视为过渡接入。
7. 原生模型构建层只允许暴露通用、只读且默认关闭的 observation/result seam，不得依赖 `CodegenBuildSession` 等 codegen 具体类型。
8. `codegen=False` 或未安装旁路观察器时，不得读取 artifact、保存/恢复额外 RNG、增加 collective、改变 planner/replacement 调用次数或引入生成模块。
9. 每个阶段必须单独审计 `hyper_parallel/codegen/` 之外的修改，说明必要性并验证 Trainer/AutoModel 公共 API 未漂移。

## 129 验证机信息

本节记录阶段 0～9 共用的多卡验证环境。固定机器能力和路径约定，但不把某一次查询得到的空闲卡状态、
进程号或临时工作树提交号当作长期事实。每次启动任务前仍需重新执行动态检查。

### 稳定配置

| 项目 | 配置 |
| --- | --- |
| 主机地址 | `8.92.7.129` |
| 登录用户 | `lxy` |
| 认证 | 专用 Ed25519 私钥；私钥只保存在本机受控目录，不写入仓库、日志或命令输出 |
| 加速卡 | 8 × Ascend 910B3，逻辑卡号 `0-7` |
| 单卡 HBM | 65536 MiB |
| NPU 工具 | `npu-smi 26.0.rc1`（2026-10-29 实测） |
| Python | `/home/lxy/miniforge3/envs/hyper_restructure_py311/bin/python` |
| torchrun | `/home/lxy/miniforge3/envs/hyper_restructure_py311/bin/torchrun` |
| 默认可见卡 | `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` |
| 默认进程数 | `8` |

Windows 侧不能把数字简称 `129` 直接交给 OpenSSH；它会被解析成 `0.0.0.129`。应使用完整地址
`8.92.7.129`。现有私钥的 Windows ACL 不满足 OpenSSH 的权限检查，因此当前已验证的连接方式是
Paramiko 加载专用 Ed25519 key。后续若统一 SSH 配置，应先收紧私钥 ACL，再增加明确的 Host alias，不能
通过复制私钥或放宽权限解决。

### 仓库与验证目录

| 路径 | 用途 | 使用约束 |
| --- | --- | --- |
| `/home/lxy/hyper-parallel-models-codegen` | 历史 `models_codegen` 工作树 | 只作历史对照；启动阶段 0 前核对 HEAD 和 dirty 状态 |
| `/home/lxy/hyper-parallel-validation-6b986fe8` | 历史 DeepSeek-V4.1 验证树 | 固定在旧提交语义，不作为新基线直接复用 |
| `/home/lxy/codegen_reval/val` | 双臂验证数据、渲染 YAML 和日志根目录 | 新阶段必须使用独立子目录，禁止覆盖历史日志 |
| `/home/lxy/codegen_regen_recorded_129_logs` | 历史全模型重新生成日志 | 只读对照 |
| `/home/lxy/codegen_regen_recorded_129_inputs` | 历史全模型生成输入 | 只读对照 |
| `/home/lxy/probe_ckpts` | 裁剪模型配置和 probe checkpoint 根目录 | 运行前检查目标子目录完整性 |

阶段 0 建议创建独立的一次性工作树和输出目录：

```text
/home/lxy/hyper-parallel-stage0-90c536d6
/home/lxy/codegen_reval/stage0-90c536d6/
├── rendered-yaml/
├── data/
├── logs/
├── bundles/
├── metadata/
└── environment/
```

工作树必须指向待验证 SHA，不允许在共享历史工作树上执行 `git reset --hard`、清理未确认目录或覆盖
tracked generated 产物。`CLEAN_GENERATED=1` 只能在上述一次性工作树中使用。

### 模型资产

| 模型 | 129 路径 | 阶段 0 用途 |
| --- | --- | --- |
| Qwen3-MoE | `/home/lxy/qwen3_30b_a3b_4L` | cropped config、tokenizer 和双臂 10-step 验证 |
| DeepSeek-V4.1 | `/home/lxy/DeepSeek-V4.1-Flash` | config、tokenizer、Engram 资产准备和双臂验证 |
| Kimi-K2.6 | `/home/lxy/probe_ckpts/kimi_k26` | cropped config 和 bundle 覆盖 |
| Qwen3-VL-MoE | `/home/lxy/probe_ckpts/qwen3_vl_moe` | cropped config 和 VLM bundle 覆盖 |

DeepSeek-V4.1 数据文件已经存在于
`/home/lxy/codegen_reval/val/data/deepseek_v41_online_1k.jsonl`，但阶段 0 仍应把数据文件大小、mtime 和
SHA256 记录到 `environment/manifest.txt`，防止后续运行无意中使用不同输入。其他模型同样执行该规则。

### 每次启动前的动态检查

以下检查全部通过后才能占用 8 卡：

1. `npu-smi info` 显示 8 张卡均为 `Health: OK`。
2. 同时检查 NPU 进程表和 `ps -ef` 中的 `torchrun`、训练 Python 进程；AICore 为 0 不等于卡可用。
3. 8 张卡必须全部无任务；不得在部分卡空闲时启动当前 8 卡验证。
4. 核对远端工作树 HEAD 等于本阶段冻结 SHA，且 `git status --short` 只包含已知验证产物。
5. 用目标 Python 打印 `hyper_parallel.__file__`，确认 editable install 指向本次一次性工作树。
6. 记录 Python、Torch、torch_npu、Transformers、CANN/npu-smi 版本以及关键环境变量。
7. 检查四类模型资产和本轮选择的数据文件可读。
8. 检查输出目录不存在，或为空且确认为本轮专用目录。

2026-10-29 的准备检查中，8 张卡健康，但卡 1、2、4、6 仍存在 Python 进程，因此当时不满足 8 卡任务
启动条件。该结果仅用于说明动态检查不能省略，不代表后续状态。

### 阶段 0 的 129 输出契约

每次运行至少保存以下信息：

- `environment/manifest.txt`：SHA、分支、dirty 状态、导入路径、软件版本、NPU 型号和数据 SHA256。
- `logs/<model>_{native,codegen}.log`：完整 stdout/stderr，不只保留 loss 摘要。
- `metadata/<model>/`：`codegen_meta.json`、signature、生成文件清单及每个文件的 SHA256。
- `bundles/<model>/`：fresh generation 后的完整 bundle 副本。
- `loss/<model>_{native,codegen}.txt`：按 step 排序的原始 loss；另存 abs diff 汇总。
- `processes/before.txt` 和 `processes/after.txt`：NPU 与训练进程快照，用于发现残留进程。
- `result.md`：每个模型的 PASS/FAIL、失败阶段和首个差异 step。

连接凭据、私钥路径和私钥内容不得进入上述输出。阶段 0 完成后，报告必须能从 manifest 唯一还原代码、
环境、数据和 bundle，才允许进入阶段 1。

## 阶段 0：冻结当前基线

固定当前提交和环境，保存 Qwen3-MoE、DeepSeek-V4.1、Qwen3-VL-MoE 的 native/codegen 10-step loss、bundle 文件集合、metadata schema 和 signature。

增加诊断计数器：native model build/planner/apply、codegen meta-model build/offline planner、generated model import/parallelize。

验收：基线可复现；计数器不改变行为；原生构建、旁路生成和训练错误可区分。

## 阶段 1：定义 Snapshot 和 BuildSession

新增 `CodegenStatus`：`not_required`、`collecting`、`derived`、`generating`、`generated`、`failed`、`skipped`。

新增不可变 `CodegenSnapshot`，包含 source/model identity、replacement、sharding、activation、boundaries、tied pairs、special handlers、diagnostics 和 failure/skip reason。新增 `CodegenBuildSession`，初期只收集内存 snapshot，不改变训练和生成行为。

Snapshot 只能包含 JSON、tuple、dict 和 primitive，不得保存 module、Tensor、DeviceMesh、ProcessGroup、闭包或 factory 实例。

验收：JSON round-trip、非法对象 fail-fast、多 rank snapshot 一致、`not_required` 不伪造空 plan、`failed` 包含稳定 failure reason。

## 阶段 2：采集原生 replacement

在 `compile_module_replacements()` 返回、`apply_module_replacements()` 前后采集声明规则、实际命中 FQN、source module type、exact type、replacement target、静态参数、应用顺序和未命中原因。

采集不能重新调用 factory、重新匹配或修改原生结果。增加 native replacement snapshot 与旧 codegen metadata 的对照工具。

验收覆盖 Qwen3-MoE、DeepSeek-V4.1、Qwen3-VL-MoE、Kimi；命中 FQN、source type、target、参数和顺序一致，factory 调用次数不增加。

## 阶段 3：采集原生 planner 和 apply 结果

拆分 `_plan_and_apply_sharding()`：先执行原生 `plan()` 并调用 `session.record_plan()`，再执行原生 `apply_sharding_plan()` 并调用 `session.record_plan_application()`。

记录 placements、boundaries、mesh dimensions、source shard info、resolved wrappers、tied parameters、special handlers 和 EP bindings。没有 planner 的场景显式记录 `not_required`，不得构造 offline mesh 或空 plan。

验收：planner/apply 调用次数不增加；loss、初始化 checksum、参数布局和 FSDP layout 不变；pure DP/FSDP 不调用 planner。

## 阶段 4：让 manager 消费 Snapshot

新增 `ensure_codegen_artifact_from_snapshot(snapshot, artifact_dir)`。新入口只消费 snapshot，禁止调用 `build_meta_model()`、`build_offline_mesh()`、`ShardingPlanner.plan()`、独立 replacement matching 和 `_prepare_codegen_build()`。

旧入口可以暂时保留兼容，但训练主链只能传 snapshot。

验收：同一 native build 生成的 snapshot bundle 与旧 bundle 逐项比较 placements、boundaries、injections、aliases、tied pairs、special handlers 和 replacement FQN；只允许 banner/path 差异。

## 阶段 5：将生成移动到 `finish()`

目标顺序：`from_config/from_pretrained` 创建 session，执行完整原生 infrastructure，在 materialize/load 完成后调用 `session.finish()`，再 `model.train()` 并返回原生模型。

`finish()` 不参与模型选择、不重新初始化参数、不重新 apply sharding、不改变 optimizer 参数列表、不导入 generated modeling。

生成失败只记录 `FAILED`，原生模型仍返回并继续训练。所有 rank 必须一致结束；bundle 使用临时目录加原子 rename，失败时清理临时产物。

本阶段允许 `codegen_session`、`codegen_config` 和 AutoModel 内部 sidecar 编排作为过渡实现，仅用于验证 finish 时序和失败隔离；不得把这些参数认定为稳定公共 API，也不得继续扩大其调用范围。

验收故障注入：snapshot 序列化、bundle 写入、emitter、signature、rank 0/非 rank 0 失败和超时；多 rank signature 不一致必须在文件轮询前同步快速失败，artifact barrier 异常不得静默吞掉；不得出现 rank hang、子进程或临时目录残留。

## 阶段 6：实现 `derived/not_required`

有计划时走 `native planner -> derived -> finish -> 完整 plan bundle`。无计划时走 `native guard -> not_required`，不调用 planner，不生成伪造 `param_plan`。

覆盖 pure DP、FSDP-only、TP、CP、EP、TP+CP、TP+EP、TP+CP+EP、HF/custom 和 VLM。

阶段 6 只完善状态与 bundle 契约，不得新增 Trainer、AutoModel、replacement 或 planner 的 codegen 专用公共参数。

## 阶段 6.5：收敛原生链路接入面

在删除训练侧 GEN backend 前，先把阶段 1～6 的过渡接入收敛为内部 observation seam：Trainer 不感知 `CodegenBuildSession`，AutoModel 公共 API 不接受 `codegen_session`，原生 builder 不导入 codegen 具体类型。旁路启用、输出目录和 session 生命周期由内部编排对象或上下文管理，原生 planner/replacement 只返回稳定、通用的只读构建事实。

验收：与阶段 0 基线比较 Trainer、`from_config()`、`from_pretrained()` 的公开签名；native-only 路径不读取 artifact、不操作额外 RNG、不增加 collective，planner/replacement/apply 调用次数不变；Qwen3-MoE、DeepSeek-V4.1 的 native-only checksum、loss 和 grad norm 不变。

## 阶段 7：删除训练侧 GEN backend

阶段 1～6.5 全部通过后执行：删除或废弃 Trainer `codegen` 开关；移除 AutoModel 的 `codegen`、`codegen_config`、`codegen_artifact_dir` 公共参数；删除训练侧 `ModelingBackend.GEN`、`_parallelize_from_generated()` 和 generated metadata sharding 决策；禁止训练导入 generated modeling。

验收：训练中 `generated_model_import_count`、`generated_parallelize_count`、`offline_planner_count` 均为零；DeepSeek-V4.1 step 3 之后不再发散，Qwen3-MoE/Qwen3-VL-MoE 继续与 native 对齐。

## 阶段 8：重构 CLI

CLI 只复用 HyperAutoModel 原生构建和 CodegenBuildSession：读取 YAML、原生构建、收集 snapshot、`finish()`、退出。CLI 不得创建 optimizer/dataloader，不得执行 train/forward/backward，不得独立 planner，不得导入 generated model。

验收：CLI 与 Trainer 生成相同 bundle；无训练组件和运行时残留；失败不伪造成功 bundle；相同 YAML signature 稳定。

## 阶段 9：删除旧路径

最后才清理只服务训练的 `codegen/inline`、独立 `patch_plan` 推导、offline mesh planner、HF-only 自动 EP 结构推断、generated `hp_parallelize` 训练入口和旧 `module_overrides` training ownership。

保留 bundle emitter、metadata schema、snapshot serializer、drift/preflight checker 和 CLI audit/export helper。

## 统一验收标准

### 单测

每个新模块都有 UT；snapshot/schema/error path 和 `not_required` 有覆盖；旧相关 UT 不退化。

### 训练正确性

至少覆盖 Qwen3-MoE、DeepSeek-V4.1、Qwen3-VL-MoE、一个 custom model 和一个 HF-only MoE，比较初始参数 checksum、1～10 步 loss、grad norm、optimizer 后参数 checksum、参数 shape/layout 以及 EP factory/forward 调用次数。

### 运行时隔离

必须证明训练不导入 generated modeling、不调用 generated parallelize、不构造 offline mesh、不重复 planner、不重复 replacement matching；codegen failure 不破坏原生训练。

### 性能

记录原生构建、snapshot、bundle 生成、首 step、稳态 step、峰值内存，以及冷生成、复用、失败场景的资源释放时间。

## 推荐提交拆分

1. `test(codegen): freeze current dual-path baseline`
2. `feat(codegen): add snapshot schema and build session`
3. `feat(codegen): record native replacement decisions`
4. `feat(codegen): record native sharding plan and application`
5. `feat(codegen): generate bundles from snapshots`
6. `feat(codegen): finish codegen after native infrastructure`
7. `feat(codegen): add derived and not-required states`
8. `feat(codegen): isolate codegen failures from native training`
9. `refactor(codegen): remove generated artifacts from training path`
10. `refactor(codegen): remove training codegen switch and GEN backend`
11. `refactor(codegen): migrate CLI to HyperAutoModel build session`
12. `cleanup(codegen): remove obsolete inline planner and EP inference`

每个提交都必须独立通过对应阶段的 UT、fresh generation、必要的 8 卡双臂验证和失败隔离检查。

## 文件级实施清单

本节是实际编码时的文件导航。标记为“新增”的路径是建议位置，可以根据现有包职责调整文件名，但不能改变数据所有权和调用时序。

### 阶段 0：基线与观测

| 文件 | 修改方向 |
| --- | --- |
| `hyper_parallel/codegen/diagnostics.py`（新增） | 定义构建、planner、apply、generated import/parallelize 的进程内计数器；默认关闭，不参与签名和训练语义。 |
| `hyper_parallel/models/_transformers/auto_model.py` | 仅在诊断开关启用时记录原生模型构建次数、backend 和最终模型类；不得改变分支判断。 |
| `hyper_parallel/models/_transformers/model_builder.py` | 在 `_plan_and_apply_sharding()` 和 `_parallelize_from_generated()` 入口记录调用次数。 |
| `hyper_parallel/codegen/manager.py` | 记录独立 meta model、offline planner 和 bundle generate/reuse 次数。 |
| `hyper_parallel/codegen/loader.py` | 记录 generated modeling import 次数。 |
| `hyper_parallel/codegen/runtime.py` | 记录 generated parallelize 次数。 |
| `examples/training_demo/codegen_validation/run_codegen_validation.sh` | 输出诊断计数、bundle 摘要、训练进程 import 证据和每步 loss。 |
| `examples/training_demo/codegen_validation/README.md` | 固化基线提交、环境、模型资产、命令和预期结果。 |
| `tests/ut/codegen/test_diagnostics.py`（新增） | 验证计数器默认关闭、线程安全、reset 和不进入 signature。 |

阶段 0 的最小命令：

```bash
pytest -q tests/ut/codegen/test_diagnostics.py
RUN_QWEN=1 RUN_DSV41=1 bash examples/training_demo/codegen_validation/run_codegen_validation.sh
```

### 阶段 1：Snapshot schema 与 BuildSession

| 文件 | 修改方向 |
| --- | --- |
| `hyper_parallel/codegen/snapshot.py`（新增） | 定义 `CodegenStatus`、`CodegenSnapshot` 及 replacement、plan、application、failure 子快照；所有字段不可变且可规范化序列化。 |
| `hyper_parallel/codegen/session.py`（新增） | 定义状态机和 `record_source()`、`record_replacements()`、`record_plan()`、`record_plan_application()`、`mark_not_required()`、`mark_failed()`、`finish()`。初期 `finish()` 只返回 snapshot。 |
| `hyper_parallel/codegen/hash.py` | 增加 snapshot canonical payload/hash；排除 build id、临时目录、rank 和随机对象地址。 |
| `hyper_parallel/codegen/meta.py` | 仅增加未来 snapshot schema 的版本兼容读取结构，暂不切换 emitter 输入。 |
| `hyper_parallel/codegen/__init__.py` | 只导出稳定的 session/snapshot 类型，不导出内部 recorder。 |
| `tests/ut/codegen/test_snapshot.py`（新增） | 覆盖 round-trip、canonical ordering、schema version、非法运行时对象和往返等价。 |
| `tests/ut/codegen/test_session.py`（新增） | 覆盖状态迁移、重复 record、缺失步骤、`not_required`、`failed` 和 finish 幂等性。 |

关键状态约束：

- `COLLECTING -> DERIVED -> GENERATED`。
- `COLLECTING -> NOT_REQUIRED`。
- 任意生成阶段可进入 `FAILED`，但不能把原生 build failure 标成 codegen failure。
- `finish()` 后禁止再 record。

### 阶段 2：原生 replacement 采集

| 文件 | 修改方向 |
| --- | --- |
| `hyper_parallel/models/replacement.py` | 为编译后的 replacement action 增加稳定 rule id、声明 target path 和静态参数视图；不得执行 factory。 |
| `hyper_parallel/trainer/config/replacement.py` | 在 YAML DTO 到 normalized spec 时保留稳定 declaration id，禁止用 `id(spec)` 关联。 |
| `hyper_parallel/trainer/config/parallelism.py` | `PlanOverride.replace_module` 转换时携带 declaration id；保持原有 public DTO 行为。 |
| `hyper_parallel/models/_transformers/model_builder.py` | 在 `_maybe_apply_module_replacements()` 中，将编译结果和实际命中 FQN 交给 session；apply 仍只执行一次。 |
| `hyper_parallel/codegen/snapshot.py` | 增加 replacement declaration、match、application 和 unsupported reason 数据结构。 |
| `hyper_parallel/codegen/session.py` | 实现 replacement 记录和重复 FQN/rule 冲突校验。 |
| `hyper_parallel/codegen/emit/replacement.py` | 增加从 snapshot 渲染的入口；旧的重新编译入口暂时保留用于对照。 |
| `hyper_parallel/codegen/patch_plan.py` | 过渡期仅提供 legacy 对照，不再作为新采集链的数据源。 |
| `tests/ut/auto_models/_transformers/test_codegen_replacement_binding.py` | 验证 factory 调用次数、命中 FQN、alias、exact type 和 apply 次数。 |
| `tests/ut/codegen/test_module_override.py` | 比较 snapshot 与 legacy module override 输出。 |
| `tests/ut/codegen/test_replacement_snapshot.py`（新增） | 覆盖闭包 target、不可序列化 factory、glob、无命中和同 FQN 冲突。 |

### 阶段 3：原生 plan 与 apply 采集

| 文件 | 修改方向 |
| --- | --- |
| `hyper_parallel/models/_transformers/model_builder.py` | 给 `_plan_and_apply_sharding()` 增加可选 `codegen_session=None`；将 guard、plan、apply 三个职责显式拆分并记录稳定 skip reason。 |
| `hyper_parallel/distributed/plan.py` | 如现有 plan 缺少只读规范化接口，增加不修改 plan 的导出方法；不得加入 codegen 依赖。 |
| `hyper_parallel/codegen/plan/freeze.py` | 支持直接冻结原生 plan 和 apply 前模型结构；停止假设输入来自 offline planner。 |
| `hyper_parallel/codegen/session.py` | 实现 `record_plan()`、`record_plan_application()` 和 FQN 关联检查。 |
| `hyper_parallel/codegen/snapshot.py` | 增加 `DERIVED`、`NOT_REQUIRED`、skip reason、resolved application 字段。 |
| `hyper_parallel/models/_transformers/auto_model.py` | 创建 session 后将其传给 `apply_model_infrastructure()`；此阶段仍保留旧 GEN 行为作对照。 |
| `tests/ut/auto_models/_transformers/test_init.py` | 验证 HF/custom、pure DP 和活动 TP/CP/EP 的 planner 调用次数及 skip reason。 |
| `tests/ut/codegen/test_plan_snapshot.py`（新增） | 验证 placements、I/O、injections、tied pairs、special handlers、boundary classes 的冻结结果。 |
| `tests/ut/codegen/test_spec_field_table.py` | 对照 snapshot plan 与现有 `CodegenMeta` 字段。 |
| `tests/ut/dual_mode_dtensor/test_planner_golden_plans.py` | 保证采集前后 golden plan 不变。 |

原生 guard 至少定义以下稳定 reason：

- `planner_unavailable`
- `mesh_unavailable`
- `pure_data_parallel`
- `device_mesh_unavailable`
- `derived`

未知原因不得伪装为 `not_required`。

### 阶段 4：Snapshot 驱动 manager/emitter

| 文件 | 修改方向 |
| --- | --- |
| `hyper_parallel/codegen/manager.py` | 新增 `ensure_codegen_artifact_from_snapshot()`；签名、锁、rank 协调和 reuse 都以 snapshot 为输入。新入口不得调用 `_fill_plan_fields()` 或 `_prepare_codegen_build()`。 |
| `hyper_parallel/codegen/meta.py` | 增加 snapshot status、schema version、failure/skip reason；`derived` 与 `not_required` 使用不同校验矩阵。 |
| `hyper_parallel/codegen/emit/bundle.py` | 接收 snapshot/meta，不接收 live model、mesh 或 planner。 |
| `hyper_parallel/codegen/emit/modeling.py` | 从 snapshot 中的 source/replacement/plan 声明生成；禁止生成期间重新匹配。 |
| `hyper_parallel/codegen/emit/parallel.py` | 只渲染已冻结的 plan/application；不推断 EP factory。 |
| `hyper_parallel/codegen/emit/recorded.py` | 从原生 resolved application 记录渲染，不读取 apply 后 live model 重建 plan。 |
| `hyper_parallel/codegen/emit/replacement.py` | 切换为 snapshot replacement 数据源。 |
| `hyper_parallel/codegen/check/preflight.py` | 根据 `derived/not_required/failed` 分别校验，拒绝状态与字段组合冲突。 |
| `hyper_parallel/codegen/check/drift.py` | snapshot source、replacement、plan 变化均使 cache 失效。 |
| `tests/ut/codegen/test_builder_driven_manager.py` | 改为验证 snapshot 驱动，保留 legacy 对照测试直到阶段 9。 |
| `tests/ut/codegen/test_preflight_boundary_forms.py` | 增加状态/字段合法矩阵。 |
| `tests/ut/codegen/test_snapshot_manager.py`（新增） | 断言新入口不调用 meta model、offline mesh、planner 和 replacement compiler。 |

### 阶段 5：接入 AutoModel finish 与失败隔离

| 文件 | 修改方向 |
| --- | --- |
| `hyper_parallel/models/_transformers/auto_model.py` | 在解析实际 `hf_config` 和 HF/custom 身份后创建 session；原生 infrastructure 返回后、`model.train()` 前调用 `finish()`。 |
| `hyper_parallel/models/_transformers/model_builder.py` | `apply_model_infrastructure(..., codegen_session=None)` 仅负责采集 seam，不负责编排生成；默认 `None` 保证 LlamaFactory/RL 等调用兼容。 |
| `hyper_parallel/codegen/session.py` | 实现同步 finish、rank owner 协议、超时、失败状态和资源清理。 |
| `hyper_parallel/codegen/artifact.py` | 临时目录、目录锁、原子 rename、失败清理和并发冲突处理。 |
| `hyper_parallel/codegen/manager.py` | 生成异常转换为 session failure result；原生 build 异常不得吞掉；生成/复用前校验全 rank snapshot signature 一致，artifact barrier 异常必须进入 sidecar failure。 |
| `hyper_parallel/trainer/base.py` | 此阶段只停止显式编排生成，仍可暂存旧字段用于 A/B 对照。 |
| `tests/ut/codegen/test_session_finish.py`（新增） | 覆盖 success/reuse/failure/timeout/cleanup。 |
| `tests/ut/codegen/test_artifact_layout.py` | 覆盖临时目录和原子发布。 |
| `tests/ut/auto_models/_transformers/test_init.py` | 验证 finish 时机在 infrastructure 后、`model.train()` 前。 |
| `tests/torch/codegen/` | 增加 rank 0 失败、非 owner 等待、signature mismatch 快速失败和超时的轻量 ST worker/launcher。 |

### 阶段 6：`derived/not_required` 完整契约

| 文件 | 修改方向 |
| --- | --- |
| `hyper_parallel/codegen/meta.py` | `not_required` 禁止非空 param plan/injections；`derived` 必须具有完整计划字段。 |
| `hyper_parallel/codegen/check/preflight.py` | 实现状态矩阵和错误编号，拒绝空 plan 冒充 derived。 |
| `hyper_parallel/codegen/emit/bundle.py` | `not_required` 只输出 source/replacement/audit 需要的内容，不生成并行入口。 |
| `hyper_parallel/codegen/emit/parallel.py` | 显式拒绝为 `not_required` 渲染 `hp_parallelize`。 |
| `hyper_parallel/codegen/loader.py`、`runtime.py` | 若兼容 helper 读取到 `not_required`，必须明确拒绝 parallelize，不回退 planner。 |
| `tests/ut/codegen/test_core_contracts.py` | 增加 derived/not_required/failed 往返和兼容矩阵。 |
| `tests/ut/codegen/test_preflight_boundary_forms.py` | 覆盖每种状态的字段组合。 |
| `tests/ut/codegen/test_not_required.py`（新增） | 覆盖 pure DP、FSDP-only、replacement-only 和未知 skip reason。 |

### 阶段 6.5：收敛 Trainer/AutoModel 接入面

| 文件 | 修改方向 |
| --- | --- |
| `hyper_parallel/trainer/base.py` | 不再向模型公共构建 API 透传 `codegen_session` 或完整 `codegen_config`；Trainer 不导入、不创建 `CodegenBuildSession`。 |
| `hyper_parallel/models/_transformers/auto_model.py` | 从 `from_config()`、`from_pretrained()` 公共签名移除 `codegen_session`；将 sidecar 生命周期移入内部 build observation 编排，native-only 路径不读取 artifact、不执行额外 RNG 保护。 |
| `hyper_parallel/models/_transformers/model_builder.py` | 用通用、默认 `None` 的只读 observer/result seam 替代 codegen 具体 session；不得导入 `hyper_parallel.codegen` 类型。 |
| `hyper_parallel/models/replacement.py` | observation callback 保持通用可选接口；原生 replacement DTO 不携带 codegen 专用状态，callback 为空时调用顺序和返回值与基线一致。 |
| `hyper_parallel/trainer/config/parallelism.py` | 稳定 replacement identity 作为通用构建事实保留；不得继续增加只为 codegen 服务的 Trainer DTO 字段。 |
| `tests/ut/auto_models/test_public_api.py` | 对比阶段 0 API manifest，断言 Trainer、`from_config()`、`from_pretrained()` 没有新增 codegen session 参数。 |
| `tests/ut/codegen/test_native_isolation.py`（新增） | 断言 native-only 不创建 session、不读取 artifact、不保存设备 RNG、不调用 barrier，planner/replacement/apply 次数不变。 |
| `examples/training_demo/codegen_validation/run_codegen_validation.sh` | 增加 native-only import/call 证据和 API signature 摘要。 |

阶段 6.5 是阶段 7 的硬前置；若仍需从 Trainer 或公共 AutoModel API 传入 codegen 具体对象，不得进入阶段 7。

### 阶段 7：删除训练消费路径

| 文件 | 修改方向 |
| --- | --- |
| `hyper_parallel/trainer/config/trainer.py` | 删除 `codegen: bool`；提供一次明确的旧 YAML 迁移错误或 warning，不能继续切 backend。 |
| `hyper_parallel/trainer/base.py` | 删除向 model builder 传递 `codegen`/`codegen_config`；改为传有限的 codegen output options，或由 AutoModel 内部创建 session。 |
| `hyper_parallel/models/_transformers/auto_model.py` | 删除公共 `codegen`、`codegen_config`、`codegen_artifact_dir` 参数和 GEN 分支。 |
| `hyper_parallel/codegen/modeling_backend.py` | 删除 `GEN` 和 `codegen_enabled` 分支；保留 native HF/custom backend 选择。 |
| `hyper_parallel/models/_transformers/model_builder.py` | 删除 `_parallelize_from_generated()`、generated replacement ownership 和 artifact-dir 分支。 |
| `hyper_parallel/codegen/loader.py` | 删除训练模型初始化入口；只保留 artifact inspection/兼容工具需要的读取。 |
| `hyper_parallel/codegen/runtime.py` | 删除训练可达的 `parallelize_from_generated()`；必要的纯数据解析 helper 迁入更小模块。 |
| `examples/training_demo/qwen3_moe/cropped_qwen3_moe.py` | 删除 `codegen`/`codegen_config` 转发，保留原生 Trainer 参数。 |
| `hyper_parallel/models/deepseek_v41/adapter/validation/cropped_model.py` | 删除 codegen backend 参数和分支。 |
| `hyper_parallel/models/kimi_k26/adapter/validation/cropped_model.py` | 删除 codegen backend 参数和分支。 |
| `examples/training_demo/codegen_validation/*.yaml` | 不再通过追加 `codegen: true` 构造双臂；改为“原生训练基线 vs 自动旁路采集训练”。 |
| `tests/ut/codegen/test_loader_model_class_ownership.py` | 改为断言训练永不加载 generated class。 |
| `tests/ut/codegen/test_runtime_boundary_wrap.py` | 删除训练消费断言，仅保留 artifact compatibility 测试。 |
| `tests/ut/auto_models/test_public_api.py` | 断言公开签名不再接受 `codegen`。 |

### 阶段 8：CLI 复用原生构建链

| 文件 | 修改方向 |
| --- | --- |
| `hyper_parallel/codegen/cli.py` | `generate --config` 调用内部 build-only orchestration，不再检查 YAML `codegen` 字段，不直接调用旧 manager 独立规划。 |
| `hyper_parallel/codegen/__main__.py` | 保持薄入口，只做退出码映射。 |
| `hyper_parallel/models/_transformers/auto_model.py` | 提供内部 build purpose/session options；不能重新增加 public bool 开关。 |
| `hyper_parallel/trainer/config/parser.py`、`resolver.py` | CLI 与 Trainer 复用同一 YAML 解析和 target 解析。 |
| `hyper_parallel/codegen/artifact.py` | 定义 CLI 默认输出目录和显式 output option 的解析。 |
| `tests/ut/codegen/test_cli.py` | 验证不构建 optimizer/dataloader、不执行 forward/backward、退出码、资源清理和多模型歧义。 |
| `tests/ut/codegen/test_cli_native_build.py`（新增） | 对比 CLI 与 Trainer snapshot/bundle。 |
| `docs/guide/codegen.md` | 更新为自动旁路生成和新 CLI 使用方式。 |

### 阶段 9：清理旧实现

删除前先用 `rg` 证明调用不可达，再逐文件删除或收缩：

| 文件/目录 | 清理方向 |
| --- | --- |
| `hyper_parallel/codegen/plan/derive.py` | 删除独立 meta model 构建和 planner 入口；若 source resolver 仍有用则迁移。 |
| `hyper_parallel/codegen/plan/offline_mesh.py` | 删除 codegen 专用 offline mesh。 |
| `hyper_parallel/codegen/patch_plan.py` | 删除生成侧 replacement/patch 重新观察；保留的数据 DTO 迁入 snapshot。 |
| `hyper_parallel/codegen/inline/` | 按实际依赖删除训练策略推断；仍用于纯源码 emitter 的 AST helper 可保留并改名。 |
| `hyper_parallel/codegen/loader.py` | 删除 generated model import 和注册逻辑。 |
| `hyper_parallel/codegen/runtime.py` | 删除 generated parallelize、fallback native apply 和训练状态恢复逻辑。 |
| `hyper_parallel/codegen/modeling_backend.py` | 若只剩 HF/custom 通用选择，迁回 transformers 层并删除此文件。 |
| `tests/ut/codegen/test_inline_*`、`test_toggle_boundary_lowering.py` | 删除旧策略推断测试，保留 emitter 语义测试。 |
| 所有 `generated/__init__.py` 和文档 | 删除“设置 `codegen: true` 重新生成”的旧说明。 |

## 依赖关系与可并行工作流

### 必须串行的关键路径

```text
阶段 0 基线
  -> 阶段 1 schema/session
  -> 阶段 2 replacement 采集
  -> 阶段 3 plan/apply 采集
  -> 阶段 4 snapshot 驱动生成
  -> 阶段 5 finish/失败隔离
  -> 阶段 6 状态契约完整化
  -> 阶段 6.5 原生链路接入面收敛
  -> 阶段 7 删除训练消费
  -> 阶段 8 CLI 切换
  -> 阶段 9 清理
```

阶段 7 之前不得删除旧 GEN 路径；阶段 6.5 完成前不得把过渡 `codegen_session`/`codegen_config` 参数视为稳定 API；阶段 4 之前不得停止 legacy bundle 对照；阶段 5 之前不得宣称生成失败与训练隔离。

### 可并行泳道

| 泳道 | 可并行任务 | 合并依赖 |
| --- | --- | --- |
| A：schema/session | 阶段 1 的 snapshot、状态机、canonical hash、UT | 最先完成，为 B/C/D 提供接口。 |
| B：replacement | 阶段 2 的 declaration id、命中记录、factory 调用计数 | 依赖 A 的 replacement DTO；可与 C 同时开发。 |
| C：plan/apply | 阶段 3 的 guard 拆分、plan freeze、application record | 依赖 A 的 plan DTO；可与 B 同时开发。 |
| D：emitter | 先为现有 emitter 增加“从纯数据输入”适配层，不切换 manager | 依赖 A 的 schema；可与 B/C 同时开发，最终在阶段 4 汇合。 |
| E：测试基建 | fixture、故障注入、调用计数、bundle diff 工具 | 从阶段 0 起可持续并行。 |
| F：多卡验证 | 129 脚本、资产准备、队列与报告模板 | 可与本地实现并行，但只验证已合并阶段。 |
| G：文档/迁移 | 配置废弃清单、CLI 文档草案、generated 说明盘点 | 可提前准备，阶段 7/8 后才能正式合入行为描述。 |

### 推荐并行批次

第一批：

- 主线：阶段 0 基线和诊断。
- 并行：测试 fixture、129 资产检查、bundle diff 工具。

第二批（阶段 1 schema 稳定后）：

- 开发者/分支 B：replacement 采集。
- 开发者/分支 C：planner/application 采集。
- 开发者/分支 D：emitter 纯数据适配。
- 测试分支 E：snapshot/schema/failure matrix。

第三批：

- 先合并 B、C，再由主线完成阶段 4 manager 切换。
- 同时由测试泳道跑 legacy bundle 与 snapshot bundle 差异报告。

第四批：

- 主线完成阶段 5 finish 和 rank 协议。
- 测试泳道并行做故障注入和 8 卡 ST。
- 文档泳道准备旧开关迁移说明。

第五批：

- 阶段 6 验证完整后，先串行完成阶段 6.5 的 Trainer/AutoModel 接入面收敛和 native-only 隔离验收。
- 阶段 6.5 通过后，阶段 7 的配置/API 删除与 loader/runtime 删除可以分文件并行，但必须在同一集成分支统一验收。
- CLI 阶段 8 可以提前开发 build-only orchestration 原型，但不能在阶段 7 完成前成为正式入口。

### 不建议并行的任务

- `auto_model.py` 构建时序与 `model_builder.py` plan/apply seam 不应由两个分支同时大改，冲突和时序误判风险高。
- `manager.py` snapshot 切换与 `meta.py` schema 版本切换应由同一集成负责人完成。
- 删除 `loader.py/runtime.py` 与保留旧 GEN 对照不能同时进行。
- 多个模型 bundle 的重新生成必须基于同一代码提交和同一 schema 版本，不能各自从不同中间分支生成后混合提交。

## 阶段完成检查表

每个阶段结束必须记录：

1. 实际修改文件和提交 SHA。
2. 新旧调用次数对比。
3. UT 命令、通过数和未覆盖项。
4. fresh generation 日志位置。
5. native/codegen 或 baseline/sidecar 的 10-step loss。
6. RNG、参数 checksum、grad norm 和 optimizer 后 checksum（适用阶段）。
7. generated import、generated parallelize、offline planner 是否为零。
8. 失败注入结果和资源残留检查。
9. bundle schema/source/plan/replacement 差异报告。
10. 是否满足进入下一阶段的门槛；不满足时必须停止推进。
