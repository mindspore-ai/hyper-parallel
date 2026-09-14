# Hyper-RL Qwen3-4B ST 设计与运行

## 目标与边界

通过正式 `examples/train_rl.py` 启动真实 Transformer 模型、HyperParallel 分布式训练、
shared vLLM 和权重发布，验收模块组合后的系统行为。ST 没有 line/branch coverage 门槛。

测试框架从迁移来源恢复，并接入当前 master 的 RL 插件安装与生产入口。
UT、CPU 自验证、真实 NPU ST 的结果分别记录，不能相互替代。

## 场景

所有场景默认完成两步训练，使用 BF16、真实 checkpoint 和真实 GSM8K reward。
固定 seed、8 条 prompt、每次每个 Trainer DP rank 2 条 prompt、每条 4 个 response。
普通训练最多生成 512 token，外部 Agent 每次 completion 最多生成 256 token。
名称带 `consistency` 的场景启用训推一致性和 batch-invariant，其余场景关闭；不修改生产 YAML 默认值。

| pytest 参数 ID | NPU 数 | 场景 | 专项验收 |
| --- | ---: | --- | --- |
| dense-tp1-full | 2 | Qwen3 FSDP2 → rollout DP2/TP1 | packed full-gather、完整参数 bucket、8 条验证集评估（最多 512 token）、最终保存 |
| dense-tp2-direct | 4 | Qwen3 FSDP2/TP2 → DP2/TP2 | direct reshard |
| dense-disjoint-direct | 8 | Qwen3 Trainer 4 卡 + Rollout 4 卡 | 独立设备集合与 direct HCCL 发布 |
| dense-tp2-consistency-full | 4 | GRPO matched TP2，colocated | full-gather、更新前逐位相等、更新后旧策略负对照 |
| dense-tp2-consistency-direct | 4 | GRPO matched TP2，colocated | direct-reshard、更新前逐位相等、更新后旧策略负对照 |
| dense-disjoint-consistency-full | 8 | GRPO matched TP2，disjoint | packed HCCL、更新前逐位相等、更新后旧策略负对照 |
| dense-disjoint-consistency-direct | 8 | GRPO matched TP2，disjoint | direct HCCL、更新前逐位相等、更新后旧策略负对照 |
| checkpoint-resume | 2 | Qwen3 两个独立进程/容器 | 第一阶段保存 step1；第二阶段恢复、用 V1 生成并完成 step2；两阶段均检查 HF 配置、tokenizer 和完整权重分片 |
| codex-agent | 2 | Qwen3 + Native-vLLM + Codex | 真 Agent 工具调用、gateway、token evidence、session 释放 |
| deepseek-agent | 2 | Qwen3 + Native-vLLM + DeepSeek Harness | 真 SDK 工具调用、gateway、token evidence、session 释放 |
| ppo-tp1-full | 2 | PPO Actor/Reference/Critic + Hyper-vLLM DP2/TP1 | Actor/Critic 非零更新，full-gather 两步发布 |
| ppo-tp2-direct | 4 | PPO FSDP2/TP2 → Hyper-vLLM DP2/TP2 | Actor/Critic 非零更新，direct-reshard 两步发布 |
| ppo-checkpoint-resume | 2 | PPO 双角色 checkpoint | 保存 step1，新容器恢复后继续 step2/step3，再次保存双角色状态 |

上述场景均使用 Qwen3-4B。DeepSeek Agent 场景验证 Harness 接入，其示例、运行镜像与协议保留。
PPO 的配置与当前验收状态见 [PPO 文档](ppo.md)，历史 GRPO 记录不代表新增 PPO 已通过。

## 通用通过条件

- 正式训练进程正常退出；期望训练 step 和 policy/version 一致。
- 每步有效 action token、optimizer step、生成 token 均大于零；指标为有限值。
- 运行中至少一步 gradient norm > 0；Checkpoint 场景每个阶段均需非零更新证据。
- 每步记录的 `policy/version` 必须等于已提交训练 step。
- 最近完成策略必须等于启动时配置的策略。
- full-gather 桶全部 ACK/release，最大 in-flight 为 1；除单个超大参数外，packed bucket 不超过配置阈值。
- 一致性场景每步必须有有效 action token 比较，成功标志为 1，mismatch/max/mean 三项均为 0；
  更新后负对照必须有有效 token、合法差异计数，且运行中至少一步出现旧策略 bit-pattern 差异。
- PPO 每步必须有 Critic 有效 token 和 optimizer step，且运行中至少一步 Critic 梯度非零。
- evaluation 必须在最终 step 完成 8 条样本，correct/total 与 accuracy 一致，并有实际生成 token。
- 普通最终保存及 resume 均检查 checkpoint completion、rank-local runtime、HF config/tokenizer/权重分片；
  PPO resume 的 completion marker 必须声明 Critic。
- 外部 Agent 每个 session 至少两次 completion、模型发出 tool call、后续请求含 tool feedback，
  携带原始 token/logprob 和 policy version，token 与 logprob 数量相同、raw logprob 有限且非正，
  最后记录 session.released。
- 退出后本用例容器不存在，vLLM/gateway 端口可重新绑定。

真实 reward 全相同可能导致零优势，届时非零学习验收失败，不修改标签制造通过。
当前是两步系统验收，不推断收敛、长期稳定性或吞吐提升。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `test_rl_st.py` | 轻量 pytest 入口，参数化 13 个真实 NPU 场景，带仓库 ST marker |
| `tests/common/rl_st_cases.py` | UT/ST 共用的场景定义与生产 YAML 派生，支持测试包分开部署 |
| `st_runtime.py` | 资源检查、容器启动、超时清理、结果记录 |
| `_launch.py` | 结果挂载下的临时目录内调用公共 `torchrun_case`，将各 rank 日志转发到汇总日志后清理 |
| `_worker.py` | torchrun worker，通过 `runpy` 调用正式 `train_rl.py` |
| `st_evidence.py` | 解析正式日志、checkpoint 和 Agent session 记录并判定 |
| `test_st_support.py` | CPU 自验证：配置与命令、证据校验器、缺失/损坏证据拒绝、入口轻量导入 |
| `test_st_runtime.py` | CPU 自验证：required 资源门禁、失败报告、resume 阶段隔离、成功/失败/超时的精确容器清理 |
| `pytest.ini` | 独立 ST 收集和 marker |
| `.gitignore` | 忽略运行产物和缓存 |

生成的 YAML 直接传给正式入口；不修改 Trainer、算法、reward、模型或 vLLM 生产实现，
不注入替代 forward/optimizer/通信的 mock。CPU 自验证使用合成证据，只证明测试框架能够正确判定。

遵循仓库 [testing 规则](../../../.agent/rules/testing.md) 和
[代码风格](../../../.agent/rules/code-style.md)：pytest + `arg_mark`、
公共分布式启动器、非 `test_*.py` worker、父进程轻量导入、独立结果目录、明确 skip/fail。
真实 NPU 用例位于 `tests/torch/rl/`，可由仓库 Torch 测试入口收集；运行仍需配置 RL 专用模型、数据与设备。

## 环境

宿主需要 Python、pytest、PyYAML、Docker、Ascend 驱动，以及可访问的模型和 parquet 数据。
镜像必须预先存在，不自动 pull/build；容器内通过 `install_runtime.sh` 安装当前源码的
RL 包和 vLLM 注册入口，不下载依赖。外部 Agent 镜像还需包含指定版本 CLI/SDK：
Codex 0.152.1、deepseek-harness-sdk 0.1.1rc1。

显式选择可用设备，串行运行；调用方负责避免占用其他任务的卡和通信端口。
按已确认偏好，ST 不设置 `npu-smi Health=OK` 门禁。

```bash
cd /path/to/hyper-parallel

export RL_ST_MODEL=/home/mwl/ckpt/qwen3-4b
export RL_ST_DATA=/home/mwl/dataset/gsm8k/main
export RL_ST_DEVICES=4,5,6,7
export RL_ST_REQUIRED=1
export RL_ST_TIMEOUT=1800
export RL_ST_HCCL_BASE_PORT=62000

# 可选：覆盖默认的固定基础镜像
export RL_ST_INTERNAL_IMAGE=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-arm64

# 外部 Agent 场景单独指定镜像
export RL_ST_CODEX_IMAGE=hyper-parallel/hyper-rl-codex:v0.22.1rc1
export RL_ST_DEEPSEEK_IMAGE=hyper-parallel/hyper-rl-deepseek:v0.22.1rc1
```

模型必须含 `config.json` 且 model_type 对应场景；数据目录含 `train.parquet`、`test.parquet`。
未配置时本地显式 skip；`RL_ST_REQUIRED=1` 时缺资源直接失败，CI 不得靠 skip 通过。

## 执行

```bash
# 仅测试框架 CPU 自验证
python -m pytest -q tests/torch/rl/test_st_support.py tests/torch/rl/test_st_runtime.py

# 只收集真实 ST（不加载 Torch，不启动硬件作业）
python -m pytest --collect-only -q tests/torch/rl/test_rl_st.py

# 一个真实场景；参数 ID 见上表
python -m pytest -vs 'tests/torch/rl/test_rl_st.py::test_rl_system[dense-tp1-full]'

# 配齐全部资源及八张空闲卡后，串行执行全部真实 ST
python -m pytest -vs tests/torch/rl/test_rl_st.py
```

每次运行默认保存到 `hyper_parallel/rl/output/<case>-<唯一ID>/`，
可用 `RL_ST_RESULT_ROOT` 覆盖产物根目录：
`phase-N.yaml`、`phase-N.log` 和 `result.json`；Checkpoint 另有 `checkpoints/`，外部 Agent 另有 `sessions/`。
各 rank 的原始日志位于结果挂载内的 `rl-st-phase-N-*/` 临时子目录。
正常退出或可捕获异常时，内容汇总进 `phase-N.log` 后自动清理。
若进程被 SIGKILL、无法执行 finally，临时子目录保留供排查，容器删除后仍可读取原始日志。
第二阶段使用新容器复用同一结果目录。失败保留日志，结果只有完成全部校验后才为 passed。
资源检查发生在创建运行目录前，因此缺资源的 skip/fail 直接见 pytest 输出。

## 尚未声称覆盖的能力

本轮未覆盖多节点、PPO bit-exact、长时间压力/性能验收、Native/Hyper 同源权重全量对比，
也未把两步恢复声明为 optimizer/RNG 与不中断训练逐位相等。
这些应作为明确的后续 ST 场景，不由当前 CPU 自验证或 coverage 数值代替。

## 清理前版本验证记录（2026-09-08）

- CPU 测试框架自验证：17 passed；真实场景未配置启动参数时为 8 skipped。
- Pylint、compileall：通过；轻量入口不导入 Torch/NPU/HyperParallel。
- 6 个 Qwen3/Agent 场景的生成配置已通过 PR1354 正式配置校验。
- 真实 NPU 场景未运行：执行前复核时 0–7 卡均有其他训练进程，未占用这些设备。
  8 skipped 仅表示没有执行，不能解释为系统验收通过。
- 当时仓库文档目录检查被已删除的旧 `rl_tests/test_*.py` 引用阻塞；这些导航引用已在后续 Qwen3-4B 范围清理中更新。

## Qwen3-4B 范围清理后验证（2026-09-10）

- CPU 测试框架自验证：17 passed。
- 真实 NPU 场景：6 skipped；当前环境未提供所需运行资源，未执行真实模型训练。
- Codex / DeepSeek Harness 的场景及其配置逻辑保留。

## 当前 master 隔离适配验证（2026-09-12）

本轮使用 `/home/mwl/ckpt/qwen3-4b`、真实 GSM8K parquet 和文中固定镜像。
共享 HyperParallel 源码保持原样；训练加载经 RL 局部接口复用 `models/qwen3`。
下表只记录当前隔离版本，不引用此前已撤销的共享代码修改版本。

| 场景 / 历史结果目录（`rl_tests/st/results/` 下） | 结果 |
| --- | --- |
| `dense-tp1-full-a18df9fd3a` | 通过；两步非零梯度、V1/V2，每步 73 桶全部 ACK/release，评估 2 条及最终保存完成 |
| `checkpoint-resume-31a4809cb1` | 失败；第一阶段通过，第二阶段 optimizer 恢复触发 `_fused_adamw_` DTensor layout infer 错误 |
| `codex-agent-567b800b74` | 失败；第一步非零更新并发布 V1，第二步重复工具调用超过 `max_completions=2` |
| `codex-agent-3cab0991fe` | 复测仍未通过学习验收；两步执行、V1/V2 发布与 32 个 session 的完整工具交互通过，但两步梯度均为 0 |
| `deepseek-agent-57f57a7e48` | 通过；两步非零更新与发布，32 个 session、64 次 completion，真实工具往返与释放通过 |
| `dense-disjoint-direct-585d56e8c1` | 通过；八卡训推分离，两步非零更新、direct-reshard/HCCL 发布至 V2 |
| `dense-tp2-direct-55c001e3a7` | 通过；标准四卡 FSDP2/TP2 两步非零更新、direct-reshard 发布至 V2，启用完整重计算 |

TP1 首次尝试 `dense-tp1-full-2de25d5fc0` 继承了生产 YAML 的 batch-invariant，
在 vLLM 启动显存检查失败。仅在测试派生配置中关闭该选项后复测通过；生产默认值没有变，
本轮不声称 batch-invariant 或训推一致性已通过。

Codex 首次失败会话的工具反馈为进程仍运行，随后模型重复了相同计算工具调用。
ST 提示现明确要求 `exec_command` 使用 `login=false`，避免镜像登录 shell 启动延迟；
不增加 completion 上限，不修改真实 reward。复测的 32 个 session / 64 次 completion
全部完成工具往返并释放，V0/V1 会话版本及原始 token/logprob 单独校验通过。
两步均完成 73 个 full-gather bucket 的 ACK/release，训练进程正常退出。
但是每步 `reward/zero_std_groups=4`、advantage 全为 0、梯度范数为 0：
虽然跨 prompt 的总体准确率为 0.75，每个 prompt 内四个 response 的奖励相同，
GRPO 无组内优势，不能据此声称完成了有效学习。因此保留 `result.json` 的 failed，
不把不同运行的部分通过结果拼成一次完整通过。

断点恢复失败定位于 `rl/checkpoint.py` 调用共享 optimizer 的 `load_state_dict`：
PyTorch 在恢复空 optimizer state 时先执行一次 fused AdamW 初始化，进入了未注册的
DTensor 算子分发。模型 checkpoint 加载已经完成，但优化器恢复失败；
尚未实施修复。可进一步验证 RL 局部使用与保存路径相同的 `SkipDTensorDispatch`
保护是否足够，不修改共享 optimizer 源码。

CPU 复验：RL UT（排除 consistency）与 ST 自验证共 242 passed，原版共享 apply/planner
35 passed，合计 277 passed。CPU 结果不能替代上表未通过的真实场景。
日志、会话及 checkpoint 均保留；结果目录受 `.gitignore` 忽略，不应打包或提交。
全部六个计划场景均已尝试，按各场景最后一次运行结果统计为 **4 通过、2 未通过**。
测试结束后自建 `rl-st-*` 容器均已清理，8 张 NPU 均无运行进程；没有停止其他用户容器。

## 开启一致性的追加验证

历史上六个普通模式用例保持原配置不变，随后通过
`st_runtime.run_case(Case(..., tp=2, exact=True))` 派生开启一致性的独立测试配置，
仍走同一 Docker、公共分布式启动器和正式训练入口；`phase-1.yaml` 留存实际配置。
四卡 full-gather / direct-reshard 的逐 token 指标和结果目录统一记录在
[训推一致性文档](qwen3_training_inference_consistency.md)，不与上面的 4 通过 / 2 未通过统计混合。
当前已将 matched TP2 的 colocated/disjoint 两种同步方式注册为上表中的正式参数化 ST。
用例可收集、配置验证和 CPU 自检通过，不等于这些组合已完成当前版本真机验收。

## 当前 evaluation 验证（2026-09-12）

当前代码通过两卡 `dense-tp1-full` 场景验证 evaluation。数据为本地 GSM8K
`test.parquet`：共 1319 条，`prompt` / `extra_info` 等关键列无空值；全部样本可归一化为
严格数值 ground truth，tokenizer 后 prompt 长度为 46–211，没有触及 512-token 上限。

真机结果目录：`hyper_parallel/rl/output/dense-tp1-full-ec82295add/`。
Actor 完成两步训练并发布 V2 后，evaluation 使用确定性单响应配置评估前 8 条样本：

| 指标 | 结果 |
| --- | ---: |
| `validation/accuracy` | 0.25 |
| `validation/correct` / `validation/total` | 2 / 8 |
| `validation/generated_tokens` | 3907 |
| `validation/response_length_mean` | 488.375 |
| `validation/generation_seconds` | 153.949 |
| `validation/tokens_per_second` | 25.3786 |

ST、分布式汇总和 checkpoint 均通过，V2 checkpoint 位于该目录的
`checkpoints/step_2/`（约 32 GB）。Console backend 只记录汇总标量，忽略传入的
`validation/samples`；若需要逐样本 prompt/response/ground truth 表，当前应启用 W&B backend，
或后续为本地结果增加持久化 backend。0.25 只描述固定前 8 条样本，不代表完整 1319 条测试集准确率。
