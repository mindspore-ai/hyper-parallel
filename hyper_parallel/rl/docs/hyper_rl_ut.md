# Hyper-RL UT 方案与验收结果

> 测试目录：`tests/ut/rl`

master 接入增加 `trainer/test_qwen3_master.py`：验证共享模型真实 checkpoint 加载、padding logits/梯度、
TP 规划及原生 MLP 切分、适配对共享代码的隔离、full-gather、不同 TP/FSDP 分片到 Hyper/Native rollout 的 direct-reshard，
以及 HF 导出重载。NPU primitive 在这些 CPU 测试中用数学参考替代；不能以此替代真实 NPU ST。
>
> 覆盖范围：全部 `hyper_parallel/rl/rl/**/*.py`
>
> 验收门槛：Line coverage ≥ 80%，Branch coverage 暂不要求
>
> Qwen3-4B 与 Weight Sync 精简后的最新结果在本页“覆盖率验收”记录。

## 1. 总体设计

UT 按 RL 功能域组织，共 9 个模块、35 个测试文件（34 个 `test_*.py` 和 1 个 `agentic_ut.py`）：

| 模块 | 文件数 | 主要验收范围 |
| --- | ---: | --- |
| Trainer | 7 | 配置与模型组装、强同步训练编排、PPO targets、checkpoint、发布和清理 |
| Data | 3 | Prompt 处理及 Trajectory/Experience 数据合同 |
| Algorithm | 4 | 算法注册、advantage、reward 和 loss 数学 |
| Policy | 3 | Actor/Reference/Critic 前向计算及模型权重更新 |
| Rollout | 5 | shared vLLM、Qwen3 dense、Native/Hyper、插件、外部 Agent 版本绑定与 TP |
| Weight Sync | 7 | direct reshard、packed full-gather、传输和 worker |
| Consistency | 2 | Qwen3 dense 训练侧与推理侧一致性、跨 rank 错误传播 |
| Agentic | 1 | 会话与工具/MCP、Codex/DeepSeek harness、gateway、protocol 和 trajectory |
| Utils/Observability | 3 | evaluation、指标聚合和 tracker |

测试在 CPU 上执行。vLLM server、HCCL、IPC、NPU 和 distributed 边界使用有限 fake/mock，
但算法公式、数据合同、DP/TP 布局、权重映射、bucket 计划和事务状态机执行真实生产函数。
Agentic 的 gateway 用例会启动本机回环 HTTP 服务，需要允许本地 socket 和线程事件循环通信；
测试不调用真实模型服务或外部 Agent SDK。

## 2. Trainer UT

目录：`trainer/`

- `test_config_runtime.py`
  - 验证完整配置解析、校验和 runtime 构建；
  - 验证 Qwen3 dense 的 checkpoint identity 和 Native/Hyper 注册，拒绝已移除模型的 checkpoint；
  - 验证 EP1 边界并拒绝旧专家并行配置，保留 direct/full-gather 配置传递。
  - 从合法基线分别破坏 batch、并行、端口、设备隔离、容量和 evaluation 配置，要求正式校验器拒绝；
  - 对所有 ST 场景及 resume 阶段的派生 YAML 执行正式配置校验，mock 设备类型而不加载模型。
- `test_trainer_orchestration.py`
  - 验证 tokenizer、Data、Actor、Reference、Critic、optimizer、scheduler、Rollout 和 tracker 组装；
  - 验证 rollout→Reference/Critic→target→Actor/Critic update→权重发布的强同步顺序；
  - 验证 colocated 训练状态释放、权重发布、rollout wake、evaluation、checkpoint 和清理。
- `test_checkpoint.py`
  - 验证 tied storage clone、optimizer 的 DTensor dispatch 边界；
  - 验证模型、optimizer、scheduler、dataloader、CPU/device RNG 和训练进度保存恢复。
  - Native-core DCP 直接使用 Torch 通信接口；CPU round-trip 在该边界 mock 通信，不替换真实文件读写。
- `test_distributed.py`
  - 验证默认进程组销毁及 native-core P2P、mesh、layout、FSDP 和 redistribution 缓存清理；
  - 覆盖重复调用、未初始化状态及后端销毁异常；仅 mock 后端进程组，不创建硬件通信。
- `test_ppo_value_model.py`
  - 验证真实 tiny Qwen3 checkpoint 的 backbone 保持、scalar value head 学习与 TP 规划；
  - CPU SGD 显式关闭 foreach/fused 自动选择，避免安装了 torch-npu 时触发 NPU 初始化。

## 3. Data UT

目录：`data/`

- `test_data_source.py`
  - 验证 Prompt 加载和归一化；
  - 验证 tokenize、padding、collate 和 `PromptRecord`；
  - 验证多 rank evaluation 输入分片和超出全局样本数的 padding row。
- `test_experience_preparer.py`
  - 验证 `Trajectory` 到 `ExperienceBatch` 的 padding、mask 和字段合同；
  - 验证按 GRPO/PPO requirements 组装 reference logprob、value、advantage 和 return target。
- `test_contracts.py`
  - 拒绝陈旧 worker version、非法 action/padding mask、首 token action 和不对齐的 next-token 目标；
  - 验证 PPO bootstrap 每条 sequence 一个值，以及所有 trajectory 的版本与 Experience 一致。

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
- `test_advantage.py`
  - 验证交错 GRPO group 独立归一化、零方差组和 singleton group 拒绝；
  - 验证 GAE 空/单 action mask、非法 value/bootstrap shape；
  - 验证严格数值 reward 的最终答案、300 字符尾窗和批量精确匹配。

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
  - 拒绝布尔/非整数并行度、重复/非数字设备、非法 rank/world、非回环 host 和非法端口；
  - 保持非连续物理设备的顺序并规范化数字表示。
- `test_vllm_runtime.py`
  - 验证 generation 请求/结果、router capacity、返回顺序和 TP request owner；
  - 验证 HTTP client、shared server lifecycle、进程清理和当前 weight-control protocol；
  - 验证 Qwen3 启动参数和 Weight Sync 可观测字段透出。
- `test_qwen3_adapter.py`
  - 验证 dense Qwen3 Native/Hyper runtime 选择、TP mesh 和 sharding plan；
  - 验证 paged attention、模型 forward、logits、tied placement 和权重加载。
- `test_worker.py`
  - 验证 Codex/DeepSeek manager 的版本绑定、生成配置及结果传递；
  - 拒绝旧版本请求和执行期间策略漂移，保证 rollout 异常时清除 episode version。

## 7. Weight Sync UT

目录：`weight_sync/`

- `test_weight_sync_strategy.py`
  - 验证 TP1/TP2 下显式 direct-reshard 与 full-gather strategy 选择。
- `test_direct_reshard.py`
  - 验证 FSDP source metadata 到 TP destination region 的完整规划和重组；
  - 验证 direct-reshard 结果与 full-gather reference 一致，并拒绝缺少显式区域的 source metadata。
- `test_model_adapter.py`
  - 验证 tied/untied 参数映射；
  - 验证 Qwen3 模型 adapter 选择。
- `test_packed_weight.py`
  - 验证完整参数合桶、超大参数独占 bucket、tied 参数跳过和 metadata round trip；
  - 验证所有 rank 参与完整参数 materialize，只有 producer 生成 packed buffer。
- `test_weight_sync_transport.py`
  - 验证 Direct/Packed HCCL group，以及 IPC 单 endpoint、handle、worker ACK、失败 buffer 和 close；
  - 验证 direct HCCL 的 rank-0 RPC 协调者、非零 producer 和空闲 rank 分工；
  - 验证 packed HCCL RPC/broadcast 失败直接传播，IPC buffer 分配和 handle 导出失败先同步错误再停止。
- `test_weight_sync_transaction.py`
  - 参数化验证 colocated/disjoint × direct/full-gather 四种组合的实际张量写入；
  - 验证 plan→materialize→send→ACK→release→commit 强同步事务；
  - 验证传输、finish 和 resume 失败直接传播，且不进入恢复分支。
- `test_weight_sync_worker.py`
  - 验证 worker layout、整数 version、HCCL/IPC receive、commit 和 wake；
  - 验证 Qwen3 Native 融合 QKV/MLP 布局与 Hyper TP 目的端。
  - 验证 packed loader 失败时同步 IPC buffer，且不推进 pending/committed version。

## 8. Consistency UT

目录：`consistency/`

- `test_consistency.py`
  - 验证 Qwen3 dense consistency recipe、right-padding、packed forward、attention 和 RMSNorm；
  - 验证 partial-prefill RNG、模型身份/version、pre-update bit-exact logprob 和 post-update diagnostics。
- `test_gates.py`
  - 以 CPU FP32 bit pattern 验证正负零差异、padding 排除及 NaN/Inf 拒绝；
  - 验证版本、shape、dtype、空 mask、right-padding preflight，以及远端 rank 错误/差异使本地同步失败；
  - 负对照仅统计有效 action token，padding 差异或未变化模型不能证明学习。

当前 consistency 的 bit-exact 承诺仍仅针对指定的 Qwen3 dense 组合。

## 9. Agentic UT

目录：`agentic/`

当前只有 `agentic/agentic_ut.py`，包含 20 个测试函数，参数化后为 25 个用例。
本次保留现有成功、错误和边界场景，只修正迁移后的收集与运行路径。

| 功能域 | 现有用例验证内容 |
| --- | --- |
| 公共类型与会话 | lazy exports、Action/Observation/TurnResult、单/多轮约束、token 对齐、turn/token 预算、policy version 和关闭 |
| Chat template | 初始消息编码、action/observation 顺序、增量上下文编码与输入校验 |
| 工具与环境 | 注册、JSON/OpenAI tool protocol、schema、同步/异步 handler、调用数量、超时、错误、ToolEnvironment reward 和扩展模块加载 |
| MCP | tools/list、tools/call、initialize、stdio JSON-RPC 收发及入口参数 |
| Runner | 内部 AgentRunner 的参数/seed/response mask 校验；ProgramAgentRunner 的 owner/sibling 同步、序列化与 trajectory 身份校验 |
| Codex/DeepSeek harness | runtime 启停、episode version、隔离配置、进程/SDK 边界、session 注册→执行→评分→trajectory 转换→删除 |
| Gateway 与协议 | 本地 HTTP health/session/错误路由；Codex Responses 与 Chat 消息/工具/推理内容转换；DeepSeek reasoning 参数与流式事件 |
| Trajectory | completion trace 的 token/logprob 对齐、工具调用间上下文、终止 token、预算与 trace 改写校验 |

这里的 DeepSeek harness 是 Agent 接入与协议测试，配合 Qwen3-4B 保留；移除 DeepSeek/Moonlight 模型适配不改变这些测试。

迁移时的测试集合不会替代已经移走的 `test_agentic_runner.py`、`test_codex_agentic_runtime.py`、
`test_deepseek_agentic_runtime.py` 的全部测试场景。例如内部 `core/runner.py` 本轮只有
27.74% 行覆盖率，主要覆盖边界校验；不能把 Agentic 聚合达标解释为内部多轮 rollout 主链完整验收。
该段保留迁移时的覆盖缺口记录；不作为当前工作区的覆盖率数字。

`conftest.py` 将 `agentic_ut.py` 加入 pytest 文件匹配规则，执行整个 `ut` 目录即可收集这些用例。
独立入口也已修正为当前源码目录，只运行本文件，不再引用未迁移的三个旧文件。

独立入口命令（在仓库根目录执行）：

```bash
HYPER_PARALLEL_PLATFORM=torch /home/mwl/envs/qwen_npu/bin/python \
  tests/ut/rl/agentic/agentic_ut.py
```

该脚本保留原有的 Line/Branch 各 ≥80% 双门槛。本轮 25 个用例均通过，Line 为 82.00%、
Branch 为 74.90%，因此独立脚本退出码为 1。它与第 12 节仅要求全量 RL Line ≥80% 的命令不同；
本次未降低其门槛，也未新增用例补 Branch。

## 10. Utils/Observability UT

目录：`utils/`

- `test_evaluation.py`
  - 验证 evaluation 排除 padding row、分布式聚合和 rank0 结果输出。
- `test_monitoring_metrics.py`
  - 验证 Actor/global-token、Rollout/training diagnostics 和 learning gate；
  - 验证 Weight Sync configured/last strategy 和 packed bucket/ACK/release 指标。
- `test_monitoring_tracker.py`
  - 验证配置脱敏、rank0 backend 和多 backend fan-out。

## 11. 测试支撑文件

- `conftest.py`：设置 RL source root、Torch 平台，并收集 `agentic_ut.py`；
- `requirements-ut.txt`：UT 与 coverage 依赖；
- `.gitignore`：忽略 coverage 缓存和 HTML 报告；
- `hyper_parallel/rl/output/`：覆盖率、JUnit 和 ST 运行产物，均不提交。

## 12. 覆盖率验收

当前 `hyper-parallel_8581` 工作区、框架提交 `b7d72334` 的 CPU 联合回归：
449 passed，另有 46 个 subtest 通过；仅使用本工作区的 RL/Qwen3 代码，
共享框架文件保持原样。报告位于 `hyper_parallel/rl/output/local-master-adapt-20260914/`。

历史隔离验证（2026-09-14，原版 master `620130b2` 仅带入 RL/Qwen3 目录）：
RL UT 与 ST 自检查联合运行 449 passed（369 个 RL UT、80 个 ST 自检查），
另有 46 个 subtest 通过，行覆盖率为 86.47%（7,923 / 9,163）。
清理回归使用真实缓存对象和真实 Trainer 清理入口，仅 mock 后端通信；
DCP round-trip 仍读写真实模型及优化器文件。

2026-09-14 功能与边界测试补充后的固定镜像 CPU 验证（适配前工作区）：

| 验收项 | 结果 |
| --- | ---: |
| RL UT | 364 passed，另有 46 个 subtest 通过；无失败、无跳过 |
| 全部 RL Line coverage | 86.37%（7,894 / 9,140），达到 ≥80% |
| 全部 RL Branch coverage | 71.03%（2,199 / 3,096），记录但不作为全量 RL 门槛 |
| ST CPU 自验证 | 80 passed |
| 真实 ST 可收集场景 | 13；收集结果不代表 NPU 验收通过 |

环境使用运行镜像中的 Python 3.12、Torch 2.10.0+cpu、Transformers 5.5.4，
并安装 `requirements-ut.txt` 指定的测试依赖。CPU 容器未分配 NPU 设备。
开启 `--cov-branch` 时 coverage 表的 `Cover` 是行与分支合并值；
行门槛应读取 JSON 的 `totals.percent_statements_covered`，不可将合并值当作行覆盖率。
Agentic 独立脚本的双门槛保留，不能用全量 RL 行门槛替代。

Weight Sync 精简后的全量实测结果（2026-09-11）：

| 验收项 | 结果 |
| --- | ---: |
| Pytest | 198 passed，0 failed，0 skipped |
| 全部 RL Line coverage | 82.14% |
| Agentic 用例 | 25 passed |
| Agentic Line coverage（全量运行） | 81.98%（2,470/3,013） |
| Weight Sync Line coverage | 85.88%（1,563/1,820） |
| ST 支撑用例 | 16 passed |

正式运行命令：

```bash
cd /path/to/hyper-parallel
mkdir -p hyper_parallel/rl/output

COVERAGE_FILE=hyper_parallel/rl/output/.coverage \
HYPER_PARALLEL_PLATFORM=torch \
/home/mwl/envs/qwen_npu/bin/python -m pytest -q \
  tests/ut/rl \
  --cov=rl \
  --cov-report=term-missing \
  --cov-report=xml:hyper_parallel/rl/output/coverage.xml \
  --cov-report=json:hyper_parallel/rl/output/coverage.json \
  --cov-fail-under=80
```

单独复验 RL 时运行 `pytest tests/ut/rl`。
`tests/ut/rl/` 已纳入仓库 `tests/ut`，会随该入口一起收集；依赖要求见本页对应环境说明。
