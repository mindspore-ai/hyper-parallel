# RL 三步重构实施与验收记录

日期：2026-11-04。远程工作树：`/home/zjy/Agentic_rl_20261104/hyper-parallel`；分支：`refactor/rl-three-step`。
实施基线：`a9fa47d64c46d61dd49fc8e56c97799a6aea4a88`。设计文档的较早基线与本次基线的 RL/UT 内容已核对一致。

## 第一步：收敛推理和权重发布

- 删除旧 consistency、Hyper 推理模型及 direct_reshard；配置显式拒绝已删除模式。
- native-vLLM 保留采样修复、工具证据、R3 和 IPC/HCCL worker hooks；插件入口迁移到 `rl.rollout.vllm_plugin`。
- 流式 full-gather 由 WeightPublisher 直接负责 bucket 生命周期与事务；删除只有一种实现的策略分派。
- 原始基线 1134 项 UT 通过；第一步 1046 项通过。数量下降对应已删除能力的测试，不能当作覆盖率提升。

## 第二步：拆分训练执行与同步编排

- 配置分为 schema/resolve/limits/validate；训练替换规则由 RL YAML 显式提供，不再读取其他训练 recipe。
- Actor/Critic 复用 execution 的模型、优化器、驻留和物化机制；input 保留动态 batching 与 DP 重平衡；targets 保留目标构建顺序。
- data 只持有数据合同与准备逻辑；trainer/build 组装资源，同步循环直接持有需要的依赖，无旧 runtime 基类或兼容转发。
- 角色规则采用整列表替换和独立 normalization；启动记录生效 HP 字段及激活的 module/method 匹配规则，写入 effective_config.json 和 checkpoint 配置。
- 必要性审查：执行层承担共享执行语义，targets 承担 RL 目标语义，lifecycle 承担失败关闭；不增加通用后端或预留 IsoExec 实现。

## 第三步：异步、恢复及发行收口

- 分离异步/共卡异步复用上述组件，各自保留循环与发布安全点；async_runtime 只保留协议、生产端和服务能力。
- CheckpointResources 替代 trainer 回引用；schema 2 拒绝旧格式；异步数据按显式记录名和字段编码，不做历史类路径重映射。
- 迁移 CLI、插件、Docker、示例、测试、文档和模块导航；删除过时的一致性启动脚本。
- 三种入口分派增加回归测试，覆盖同步资源仅构建一次以及异步模式选择冲突。
- 异步物理行预算现在显式检查迁移后的四个源文件且断言文件存在，避免空目录导致虚假通过。
  预算从 1100 调整到 1200，用于 schema 2 显式字段编码；注释、文档和空行仍计数，不通过压缩排版绕过限制。

## 验证范围与未完成的设备验收

最终回归：1060 passed（RL UT 加一项无需设备的结构检查），38 条依赖/设备探测警告，无失败。
命令：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q tests/ut/rl hyper_parallel/rl/tests/st/test_async_core_budget.py
```

详细日志位于远程工作根目录的 `final-ut.log`；文档检查覆盖 20 份 Markdown，0 项问题。

- 官方 ARM64 运行镜像的无 NPU 设备容器执行 CPU/逻辑回归；不修改宿主依赖，不占用训练卡。
- 独立 wheel 在单独 venv、`/tmp` 工作目录及隔离解释器中安装并导入；确认使用 wheel 内模块、新插件入口和 CLI。
  venv 复用镜像已有第三方依赖；不宣称在完全空白系统中验证了依赖安装。
- Python 导入目标、shell 语法、Git 空白、文档本地目标、AGENTS catalog 和 Markdown lint 分别检查。
- GRPO/PPO/GSPO、MoE/R3、Reward Model、Teacher、Agent、两种异步、故障关闭和 checkpoint 的 CPU 合同保留。
- NPU 短流程、分布式 IPC/HCCL、实际 vLLM worker 生命周期、设备恢复、长期学习质量全部待验收。
  CPU 测试与 YAML 结构检查不代表这些能力已通过设备或学习验收，历史 README 数据也不是本次重构结果。
- 未修改 HyperParallel 主框架实现；改动尚未 commit 或 push。

## 示例迁移账本

下表覆盖所有具有 model/train 的完整示例 YAML；其他任务或覆盖片段不当作独立训练配方。
结构检查包括可解析、显式 plan_overrides、native 模式及无旧 consistency 字段；模型文件/数据/设备可用性需在目标部署验证。
功能对应 UT 位于 `tests/ut/rl/{algorithm,policy,trainer,weight_sync,distillation,reward_model,agentic,fully_async,colocate_async}`；
设备入口位于 `hyper_parallel/rl/tests/st/`，不以 UT 替代。

| 原文件（原位迁移） | 模型 / 算法 / 模式 / 训练拓扑 | 推理模式迁移 | 验证状态 |
| --- | --- | --- | --- |
| [qwen3_4b_code_vllm](../examples/code/configs/qwen3_4b_code_vllm.yaml) | qwen3 / grpo / 同步; dp_replicate=1, dp_shard=2, tp=2, cp=1, pp=1 | native → native | YAML/显式规则通过；设备与学习待验收 |
| [qwen3_4b_gsm8k](../examples/colocate_async/configs/qwen3_4b_gsm8k.yaml) | qwen3 / grpo / 共卡异步; dp_replicate=1, dp_shard=2, tp=1, cp=1, pp=1 | hyper → native | YAML/显式规则通过；设备与学习待验收 |
| [qwen3_4b_search_r1](../examples/colocate_async/configs/qwen3_4b_search_r1.yaml) | qwen3 / grpo / 共卡异步; dp_replicate=1, dp_shard=8, tp=1, cp=1, pp=1 | hyper → native | YAML/显式规则通过；设备与学习待验收 |
| [qwen3_4b_grpo_opd](../examples/distillation/configs/qwen3_4b_grpo_opd.yaml) | qwen3 / grpo / 同步; dp_replicate=1, dp_shard=1, tp=1, cp=1, pp=1 | hyper → native | YAML/显式规则通过；设备与学习待验收 |
| [qwen3_4b_mopd](../examples/distillation/configs/qwen3_4b_mopd.yaml) | qwen3 / grpo / 同步; dp_replicate=1, dp_shard=1, tp=1, cp=1, pp=1 | hyper → native | YAML/显式规则通过；设备与学习待验收 |
| [qwen3_4b_opd](../examples/distillation/configs/qwen3_4b_opd.yaml) | qwen3 / grpo / 同步; dp_replicate=1, dp_shard=1, tp=1, cp=1, pp=1 | hyper → native | YAML/显式规则通过；设备与学习待验收 |
| [qwen3_4b_gsm8k_grpo](../examples/fully_async/configs/qwen3_4b_gsm8k_grpo.yaml) | qwen3 / grpo / 分离异步; dp_replicate=1, dp_shard=2, tp=1, cp=1, pp=1 | hyper → native | YAML/显式规则通过；设备与学习待验收 |
| [qwen3_4b_search_r1](../examples/fully_async/configs/qwen3_4b_search_r1.yaml) | qwen3 / grpo / 分离异步; dp_replicate=1, dp_shard=2, tp=1, cp=1, pp=1 | hyper → native | YAML/显式规则通过；设备与学习待验收 |
| [qwen3_4b_swe_smith](../examples/fully_async/configs/qwen3_4b_swe_smith.yaml) | qwen3 / grpo / 分离异步; dp_replicate=1, dp_shard=2, tp=1, cp=1, pp=1 | native → native | YAML/显式规则通过；设备与学习待验收 |
| [deepseek_multi_turn](../examples/gsm8k/configs/deepseek_multi_turn.yaml) | qwen3 / grpo / 同步; dp_replicate=1, dp_shard=2, tp=1, cp=1, pp=1 | native → native | YAML/显式规则通过；设备与学习待验收 |
| [qwen3_30b_a3b_codex_multi_turn](../examples/gsm8k/configs/qwen3_30b_a3b_codex_multi_turn.yaml) | qwen3_moe / grpo / 同步; dp_replicate=1, dp_shard=2, tp=2, ep=2, cp=1, pp=1 | native → native | YAML/显式规则通过；设备与学习待验收 |
| [qwen3_30b_a3b_gsm8k_vllm](../examples/gsm8k/configs/qwen3_30b_a3b_gsm8k_vllm.yaml) | qwen3_moe / grpo / 同步; dp_replicate=1, dp_shard=2, tp=2, ep=2, cp=1, pp=1 | native → native | YAML/显式规则通过；设备与学习待验收 |
| [qwen3_4b_gsm8k_base](../examples/gsm8k/configs/qwen3_4b_gsm8k_base.yaml) | qwen3 / grpo / 同步; dp_replicate=1, dp_shard=8, tp=1, cp=1, pp=1 | hyper → native | YAML/显式规则通过；设备与学习待验收 |
| [qwen3_4b_gsm8k_codex_50step](../examples/gsm8k/configs/qwen3_4b_gsm8k_codex_50step.yaml) | qwen3 / grpo / 同步; dp_replicate=1, dp_shard=8, tp=1, cp=1, pp=1 | native → native | YAML/显式规则通过；设备与学习待验收 |
| [qwen3_4b_gsm8k_gspo](../examples/gsm8k/configs/qwen3_4b_gsm8k_gspo.yaml) | qwen3 / gspo / 同步; dp_replicate=1, dp_shard=8, tp=1, cp=1, pp=1 | hyper → native | YAML/显式规则通过；设备与学习待验收 |
| [qwen3_4b_gsm8k_model_reward](../examples/gsm8k/configs/qwen3_4b_gsm8k_model_reward.yaml) | qwen3 / gspo / 同步; dp_replicate=1, dp_shard=8, tp=1, cp=1, pp=1 | hyper → native | YAML/显式规则通过；设备与学习待验收 |
| [qwen3_4b_gsm8k_ppo](../examples/gsm8k/configs/qwen3_4b_gsm8k_ppo.yaml) | qwen3 / ppo / 同步; dp_replicate=1, dp_shard=2, tp=1, cp=1, pp=1 | hyper → native | YAML/显式规则通过；设备与学习待验收 |
| [qwen3_4b_search_r1](../examples/search_r1/configs/qwen3_4b_search_r1.yaml) | qwen3 / grpo / 同步; dp_replicate=1, dp_shard=8, tp=1, cp=1, pp=1 | native → native | YAML/显式规则通过；设备与学习待验收 |
| [qwen3_5_9b](../examples/swe_smith/configs/qwen3_5_9b.yaml) | qwen3_5 / grpo / 同步; dp_replicate=1, dp_shard=2, tp=1, cp=1, pp=1 | native → native | YAML/显式规则通过；设备与学习待验收 |
| [qwen3_8_27b](../examples/swe_smith/configs/qwen3_8_27b.yaml) | qwen3_5 / grpo / 同步; dp_replicate=1, dp_shard=4, tp=1, cp=1, pp=1 | native → native | YAML/显式规则通过；设备与学习待验收 |
