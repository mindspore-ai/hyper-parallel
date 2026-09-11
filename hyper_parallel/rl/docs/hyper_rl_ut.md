# HyperParallel-RL UT 方案与验收结果

> 测试目录：`hyper_parallel/rl/rl_tests/ut`
>
> 覆盖范围：全部 `hyper_parallel/rl/rl/**/*.py`
>
> 验收门槛：Line coverage ≥ 80%，Branch coverage 暂不要求
>
> PR1354 实测结果：179 passed，Line coverage 80.41%（2026-09-08）

## 1. 总体设计

UT 按 RL 功能域组织，共 9 个模块、27 个测试文件（26 个 `test_*.py` 和 1 个 `agentic_ut.py`）：

| 模块 | 文件数 | 主要验收范围 |
| --- | ---: | --- |
| Trainer | 2 | 配置与模型组装、强同步训练编排、发布和清理 |
| Data | 2 | Prompt 处理及 Trajectory/Experience 数据合同 |
| Algorithm | 3 | 算法注册、advantage、reward 和 loss 数学 |
| Policy | 3 | Actor/Reference/Critic 前向计算及模型权重更新 |
| Rollout | 4 | shared vLLM、Qwen3、Qwen3-MoE、DeepSeek/Moonlight、TP/EP |
| Weight Sync | 8 | direct reshard、streaming full-gather、传输、worker 和 checkpoint |
| Consistency | 1 | Qwen3 dense 训练侧与推理侧一致性 |
| Agentic | 1 | 会话与工具/MCP、Codex/DeepSeek harness、gateway、protocol 和 trajectory |
| Utils/Observability | 3 | evaluation、指标聚合和 tracker |

测试在 CPU 上执行。vLLM server、HCCL、IPC、NPU 和 distributed 边界使用有限 fake/mock，
但算法公式、数据合同、TP/EP 布局、权重映射、bucket 计划和事务状态机执行真实生产函数。
Agentic 的 gateway 用例会启动本机回环 HTTP 服务，需要允许本地 socket 和线程事件循环通信；
测试不调用真实模型服务或外部 Agent SDK。

## 2. Trainer UT

目录：`trainer/`

- `test_config_runtime.py`
  - 验证完整配置解析、校验和 runtime 构建；
  - 验证 dense Qwen3、Qwen3-MoE、DeepSeek/Moonlight 的 checkpoint identity 和模型注册；
  - 验证四卡 DP2/TP2/EP4 topology，以及 direct/full-gather/fallback 配置传递。
- `test_trainer_orchestration.py`
  - 验证 tokenizer、Data、Actor、Reference、Critic、optimizer、scheduler、Rollout 和 tracker 组装；
  - 验证 rollout→Reference/Critic→target→Actor/Critic update→权重发布的强同步顺序；
  - 验证 colocated 训练状态释放、权重发布、rollout wake、evaluation、checkpoint 和清理。

## 3. Data UT

目录：`data/`

- `test_data_source.py`
  - 验证 Prompt 加载和归一化；
  - 验证 tokenize、padding、collate 和 `PromptRecord`；
  - 验证多 rank evaluation 输入分片和超出全局样本数的 padding row。
- `test_experience_preparer.py`
  - 验证 `Trajectory` 到 `ExperienceBatch` 的 padding、mask 和字段合同；
  - 验证按 GRPO/PPO requirements 组装 reference logprob、value、advantage 和 return target。

## 4. Algorithm UT

目录：`algorithm/`

- `test_algorithm_registry.py`
  - 验证 GRPO/PPO 注册、算法 requirements 和组件构建。
- `test_algorithm_advantage.py`
  - 验证 GRPO 分组 reward 归一化；
  - 验证 GAE、return 和 action mask 的反向递推；
  - 验证数值答案提取和 rule reward。
- `test_algorithm_loss.py`
  - 验证 clipped policy objective；
  - 验证 Actor loss、Reference KL、old-policy KL 和 masked aggregation；
  - 验证 PPO clipped value loss。

## 5. Policy UT

目录：`policy/`

- `test_actor_roles.py`
  - 验证可训练 Actor 与冻结 Reference 的模型、mode 和 optimizer 边界。
- `test_policy_compute.py`
  - 验证 Actor/Reference logprob 和 Critic value 的 micro-batch 切分、尾批、顺序和 mode 恢复。
- `test_policy_update.py`
  - 验证 Actor/Critic forward-backward、gradient sync/clip、optimizer/scheduler step 和更新指标。

## 6. Rollout UT

目录：`rollout/`

- `test_rollout_topology.py`
  - 验证 colocated/disjoint shared vLLM endpoint、owner、DP×TP rank、设备、host 和 port。
- `test_vllm_runtime.py`
  - 验证 generation 请求/结果、router capacity、返回顺序和 TP request owner；
  - 验证 HTTP client、shared server lifecycle、进程清理和当前 weight-control protocol；
  - 验证 expert-parallel CLI 参数和 Weight Sync 可观测字段透出。
- `test_qwen3_adapter.py`
  - 验证 dense Qwen3 Native/Hyper runtime 选择、TP mesh 和 sharding plan；
  - 验证 paged attention、模型 forward、logits、tied placement 和权重加载。
- `test_vllm_moe.py`
  - 验证 Qwen3-MoE 与 DeepSeek/Moonlight adapter 配置及完整 constructor；
  - 验证 Qwen3 dense/MoE 共用 paged-attention bridge，以及 DeepSeek absorbed MLA；
  - 验证 EP1/EP2/EP4 expert ownership、TP/EP plan 和 token padding；
  - 验证 Qwen3/DeepSeek routing、shared expert、本地 fused expert 正向执行和物理布局恢复；
  - 参数化验证 Qwen3-MoE 与 DeepSeek 的 packed/per-expert checkpoint 权重加载，保留全部四种组合；
  - 验证 DeepSeek public TP layout、router correction bias 和 packed-token forward。

## 7. Weight Sync UT

目录：`weight_sync/`

- `test_weight_sync_strategy.py`
  - 验证 TP1/TP2 下显式 direct-reshard、full-gather 和 fallback strategy 选择。
- `test_direct_reshard.py`
  - 验证 FSDP source metadata 到 TP destination region 的完整规划和重组；
  - 验证 direct-reshard 结果与 full-gather reference 一致，并兼容 legacy source metadata。
- `test_model_adapter.py`
  - 验证 dense/tied 参数映射；
  - 验证 Qwen3-MoE/DeepSeek packed expert 到 canonical gate/up/down region；
  - 验证 canonical fragment identity 和模型 family adapter 选择。
- `test_streaming_full_gather.py`
  - 验证有界 streaming bucket 计划、fragment gather/reconstruct 和 TP packing；
  - 验证 ACK 后 release、最大 in-flight buffer 和与 bucket 划分无关的 content identity。
- `test_weight_sync_transport.py`
  - 验证 Direct/Streaming HCCL route、packed bucket broadcast、worker ACK 和 close。
- `test_weight_sync_transaction.py`
  - 验证 colocated/disjoint direct publication 和 streaming full-gather publication；
  - 验证 plan→materialize→send→ACK→release→commit 强同步事务；
  - 验证 direct→streaming fallback、source identity、统计和 MoE TP/EP destination。
- `test_weight_sync_worker.py`
  - 验证 worker layout manifest、fingerprint、version、HCCL/IPC receive、commit、abort 和 wake；
  - 用 Qwen3-MoE EP1 canonical 与 DeepSeek EP4 physical 两组代表配置验证 TP/EP ownership；
  - 验证 Qwen3-MoE fused expert 物理布局恢复，以及 DeepSeek fused MoE/absorbed MLA 刷新。
- `test_checkpoint.py`
  - 验证 tied storage clone、optimizer 的 DTensor dispatch 边界；
  - 验证模型、optimizer、scheduler、dataloader、CPU/device RNG 和训练进度保存恢复。

## 8. Consistency UT

目录：`consistency/`

- `test_consistency.py`
  - 验证 Qwen3 dense consistency recipe、right-padding、packed forward、attention 和 RMSNorm；
  - 验证 partial-prefill RNG、模型身份/version、pre-update bit-exact logprob 和 post-update diagnostics。

当前 consistency 的 bit-exact 承诺仍仅针对 Qwen3 dense。MoE 和 DeepSeek/Moonlight 已覆盖模型组装、
Rollout、TP/EP、权重加载与同步，但没有在此 UT 中声明训练侧/推理侧 bit-exact 一致性。

## 9. Agentic UT

目录：`agentic/`

当前只有 `agentic/agentic_ut.py`，包含 20 个测试函数，参数化后为 25 个用例。
本次保留现有成功、错误和边界场景，只修正迁移后的收集与运行路径。

| 功能域 | 现有用例验证内容 |
| --- | --- |
| 公共类型与会话 | lazy exports、Action/Observation/TurnResult、单/多轮约束、token 对齐、turn/token 预算、policy identity 和关闭 |
| Chat template | 初始消息编码、action/observation 顺序、增量上下文编码与输入校验 |
| 工具与环境 | 注册、JSON/OpenAI tool protocol、schema、同步/异步 handler、调用数量、超时、错误、ToolEnvironment reward 和扩展模块加载 |
| MCP | tools/list、tools/call、initialize、stdio JSON-RPC 收发及入口参数 |
| Runner | 内部 AgentRunner 的参数/seed/response mask 校验；ProgramAgentRunner 的 owner/sibling 同步、序列化与 trajectory 身份校验 |
| Codex/DeepSeek harness | runtime 启停、episode identity、隔离配置、进程/SDK 边界、session 注册→执行→评分→trajectory 转换→删除 |
| Gateway 与协议 | 本地 HTTP health/session/错误路由；Codex Responses 与 Chat 消息/工具/推理内容转换；DeepSeek reasoning 参数与流式事件 |
| Trajectory | completion trace 的 token/logprob 对齐、工具调用间上下文、终止 token、预算与 trace 改写校验 |

这里的 DeepSeek harness 是 Agent 接入与协议测试，与 Rollout 的 DeepSeek/Moonlight 模型 adapter
测试是不同职责。模型 forward、MLA 和专家权重加载仍见第 6 节。

现有文件不会替代已经移走的 `test_agentic_runner.py`、`test_codex_agentic_runtime.py`、
`test_deepseek_agentic_runtime.py` 的全部测试场景。例如内部 `core/runner.py` 本轮只有
27.74% 行覆盖率，主要覆盖边界校验；不能把 Agentic 聚合达标解释为内部多轮 rollout 主链完整验收。
按本次要求暂不补充测试。

`conftest.py` 将 `agentic_ut.py` 加入 pytest 文件匹配规则，执行整个 `ut` 目录即可收集这些用例。
独立入口也已修正为当前源码目录，只运行本文件，不再引用未迁移的三个旧文件。

独立入口命令（在仓库根目录执行）：

```bash
HYPER_PARALLEL_PLATFORM=torch /home/mwl/envs/qwen_npu/bin/python \
  hyper_parallel/rl/rl_tests/ut/agentic/agentic_ut.py
```

该脚本保留原有的 Line/Branch 各 ≥80% 双门槛。本轮 25 个用例均通过，Line 为 82.00%、
Branch 为 74.90%，因此独立脚本退出码为 1。它与第 13 节仅要求全量 RL Line ≥80% 的命令不同；
本次未降低其门槛，也未新增用例补 Branch。

## 10. Utils/Observability UT

目录：`utils/`

- `test_evaluation.py`
  - 验证 evaluation 排除 padding row、分布式聚合和 rank0 结果输出。
- `test_monitoring_metrics.py`
  - 验证 Actor/global-token、Rollout/training diagnostics 和 learning gate；
  - 验证 Weight Sync attempted/completed strategy、fallback 和 streaming memory/ACK/release 指标。
- `test_monitoring_tracker.py`
  - 验证配置脱敏、rank0 backend 和多 backend fan-out。

## 11. MoE 与 DeepSeek/Moonlight 覆盖结论

PR 1348 新增的两个主要能力已被 UT 覆盖：

| 能力 | 主要测试文件 |
| --- | --- |
| 模型身份、配置和四卡 TP2/EP4 | `trainer/test_config_runtime.py` |
| Qwen3-MoE adapter、routing、expert 和权重加载 | `rollout/test_vllm_moe.py` |
| DeepSeek/Moonlight adapter、absorbed MLA 和权重加载 | `rollout/test_vllm_moe.py` |
| MoE canonical 权重映射 | `weight_sync/test_model_adapter.py` |
| MoE TP/EP destination 与发布事务 | `weight_sync/test_weight_sync_transaction.py` |
| Native/Hyper MoE ownership、物理布局和 MLA 刷新 | `weight_sync/test_weight_sync_worker.py` |

这些是 CPU UT 合同覆盖，不替代真实 NPU 上的 Qwen3-MoE/Moonlight、HCCL/IPC 和 TP2/EP4 ST。

## 12. 测试支撑文件

- `conftest.py`：设置 RL source root、Torch 平台，并收集 `agentic_ut.py`；
- `requirements-ut.txt`：UT 与 coverage 依赖；
- `.gitignore`：忽略 coverage 缓存和 HTML 报告；
- `coverage.json`、`coverage.xml`、`.coverage`：当前覆盖率结果。

## 13. 覆盖率验收

PR1354 全量实测结果（只运行本目录，没有合并 PR1348 的旧 coverage 数据）：

| 验收项 | 结果 |
| --- | ---: |
| 测试文件 | 27 |
| Pytest | 179 passed，0 failed，0 skipped |
| 全部 RL Line coverage | 80.41%（9,287/11,550） |
| Agentic 用例 | 25 passed |
| Agentic Line coverage（全量运行） | 82.00%（2,487/3,033） |
| Agentic Codex 子目录 | 85.79%（851/992） |
| Agentic DeepSeek 子目录 | 79.06%（555/702） |
| `transfer.py`、`vllm_worker.py`、`layout.py` | 80.82%（1,749/2,164） |

正式运行命令：

```bash
cd /home/mwl/project/hyper/hyper-parallel-pr1354

COVERAGE_FILE=hyper_parallel/rl/rl_tests/ut/.coverage \
HYPER_PARALLEL_PLATFORM=torch \
/home/mwl/envs/qwen_npu/bin/python -m pytest -q \
  hyper_parallel/rl/rl_tests/ut \
  --cov=rl \
  --cov-report=term-missing \
  --cov-report=xml:hyper_parallel/rl/rl_tests/ut/coverage.xml \
  --cov-report=json:hyper_parallel/rl/rl_tests/ut/coverage.json \
  --cov-fail-under=80
```

只运行精确的 `hyper_parallel/rl/rl_tests/ut`，不会收集同级旧 `rl_tests/test_*.py`。
如果仓库 CI 只执行 `pytest tests/ut`，还需要将上述命令接入 CI，否则不会自动发现本目录。
