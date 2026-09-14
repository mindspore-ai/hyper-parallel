# Hyper-RL 当前功能盘点

> 盘点日期：2026-09-13  
> 代码范围：`hyper_parallel/rl` 及其直接调用的 HyperParallel Qwen3 构建能力  
> 判定口径：以当前配置校验、运行调用链和测试为准；“代码已实现”与“已有真机验收”分别说明。

## 结论摘要

| 问题 | 当前结论 |
| --- | --- |
| 训练侧切分 | 支持 FSDP参数分片、TP2，以及 FSDP2 × TP2；纯 TP2 仍挂一个 size-one FSDP 域。当前不支持复制型 DP/HSDP、CP、PP、EP。 |
| 推理侧切分 | 共享 vLLM deployment 支持 DP × TP；当前公开并已验证的 Qwen3 范围是 TP1/TP2。PP、context parallel、expert parallel/EPLB 未开放。 |
| GRPO / PPO | 两者都有实际算法和 Trainer 闭环实现。GRPO 是当前完整支持主路径；PPO 已实现 Actor、Reference、Critic、GAE、双 loss、双角色 checkpoint，但只对部分拓扑和任务做过真机验收，因此项目状态标为“部分验证”。 |
| 权重同步 | 数据策略为 `full_gather`（默认）或 `direct_reshard`；colocated 使用 NPU IPC，disjoint 使用 HCCL，形成 2 × 2 组合。发布采用带 `policy_version` 的同步事务。 |
| Agentic | 支持 internal 单轮/多轮环境循环、Python 工具、自定义环境/奖励，以及 Codex CLI、DeepSeek Harness 两种程序化 Agent 接入；支持 MCP 工具桥接。 |
| Native / Hyper vLLM | 都支持。Native 使用 vLLM 原生 `Qwen3ForCausalLM`；Hyper 通过 vLLM plugin 注册 `HyperQwen3ForCausalLM`，仍复用 vLLM 的服务、Paged Attention、KV Cache 和 DP Router。 |
| Rollout 负载均衡 | 已接入，但 DP engine 选择由 vLLM upstream Router 负责。Hyper-RL 使用单一共享 endpoint，并实现按 child request 计数的有界并发 admission、滚动补充和 Trainer TP 请求去重；没有另写一套 per-engine 动态负载均衡器。 |

当前产品边界是单节点、Torch + Ascend NPU、同步在线 RL、Qwen3-4B dense。异步/off-policy、
多节点、多模态、MoE、动态扩缩容和动态专家重分配不属于当前已实现范围。

## 1. 训练侧和推理侧支持的切分方式

### 1.1 训练侧

训练拓扑由 `train.accelerator` 描述。当前配置层的硬限制见
[`rl/config.py::_trainer_topology`](../rl/config.py#L210-L237)：

| 维度 | 当前配置约束 | 实际含义 |
| --- | --- | --- |
| `dp_shard` | 正整数 | FSDP2 参数分片维度，也是当前训练侧数据并行域的一部分。 |
| `dp_replicate` | 必须为 1 | 没有开放 replicated DP，也没有形成 `dp_replicate > 1` 的 HSDP 拓扑。 |
| `tp` | >=1 | 支持不切 TP、TP、FSDP2× TP。 |
| `cp` | 必须为 1 | Context Parallel 未开放。 |
| `pp` | 必须为 1 | Pipeline Parallel 未开放。 |
| `ep` | 必须为 1 | 当前只有 dense Qwen3，Expert Parallel 未开放。 |

因此可用组合可以概括为：

| 组合 | 状态 | 说明 |
| --- | --- | --- |
| `dp_shard=N, tp=1` | 支持 | FSDP2 参数分片训练。示例包含 2 卡和 8 卡配置。 |
| `dp_shard=1, tp=2` | 支持 | 纯 TP2；Trainer 会补一个 size-one FSDP 域，以保留 checkpoint layout 和 TP-replicated 参数的梯度归约，见 [`SyncTrainer._setup_runtime`](../rl/trainer.py#L194-L219)。 |
| `dp_shard=N, tp=2` | 支持 | FSDP2 × TP2 混合切分；Qwen3-4B 四卡 `FSDP-shard2 × TP2` 已有 GRPO/PPO 真机记录。 |
| replicated DP / HSDP、CP、PP、EP | 不支持 | 在 RL 配置校验阶段直接拒绝，不能因为 HyperParallel 基础库有相应能力就视为 RL 已端到端支持。 |

Actor、Reference 和 PPO Critic 共用同一个训练 mesh；Critic 不创建独立并行拓扑。模型、优化器和 FSDP2
配置通过 [`build_runtime_config`](../rl/config.py#L1120-L1170) 交给 HyperAutoModel/HyperParallel 构建。

### 1.2 推理 / rollout 侧

rollout 只注册了 `vllm` engine，入口见
[`rl/roles/rollout/registry.py`](../rl/roles/rollout/registry.py) 和
[`rl/roles/rollout/vllm.py::build_vllm_engine`](../rl/roles/rollout/vllm.py#L1406-L1440)。

当前服务拓扑是：

```text
one shared vLLM endpoint
  -> vLLM DP Router
    -> data_parallel_size 个 engine replica
      -> 每个 replica 内 tensor_parallel_size 个 TP worker
```

支持情况如下：

| 维度/方式 | 当前状态 | 说明 |
| --- | --- | --- |
| vLLM DP | 支持 | `data_parallel_size` 为正整数，engine 数等于 DP size。每个 DP engine 是一个完整模型副本或一组 TP workers。 |
| vLLM TP | 支持 | schema 接受正整数，但当前公开支持和真机验收范围是 Qwen3 TP1/TP2；Hyper adapter 还要求 attention heads、KV heads 可被 TP 整除。 |
| DP × TP | 支持 | server 启动时同时传入 `--data-parallel-size` 与 `--tensor-parallel-size`。 |
| colocated | 支持 | Trainer 与 rollout 共用完整设备集，要求 CPU offload 和 `reshard_after_forward=true`，且 rollout DP × TP 必须等于 Trainer world size。 |
| disjoint | 支持 | Trainer 和 rollout 使用不重叠设备，rollout 设备数必须等于 DP × TP。 |
| vLLM PP / context parallel | 未开放 | RL launcher 没有对应配置；Hyper Qwen3 adapter 明确要求 PP=1、prefill/decode CP=1。 |
| EP / EPLB | 不支持 | `enable_expert_parallel`、`enable_eplb` 必须关闭。EPLB 是专家负载均衡，不等于本项目的请求路由负载均衡。 |

整个 rollout 当前也受单节点约束；`SyncTrainer._validate_runtime_topology()` 会拒绝
`LOCAL_WORLD_SIZE != world_size` 的共享 vLLM 场景，见 [`rl/trainer.py`](../rl/trainer.py#L525-L593)。

## 2. GRPO 和 PPO 的实现状态

### 2.1 GRPO

GRPO 已实现并打通同步训练闭环：

- 算法注册名为 `grpo`，需要可训练 Actor、冻结 Reference、rollout old logprobs 和分组 responses。
- 按 `group_id` 对组内 reward 做 mean/std 标准化，并广播到有效 action token。
- Actor 使用 clipped importance-ratio policy loss，可配置 dual clip。
- Actor loss 中加入相对冻结 Reference 的低方差 KL（k3）正则。
- 支持 response mini-batch、micro-batch、多个 policy update epoch、梯度裁剪和分布式 token 数归一化。
- 每一步执行 rollout → Reference logprobs/advantage → Actor update → 权重发布 → 指标/评估/checkpoint。

实现入口是 [`GRPOAlgorithm`](../rl/algorithm/loss.py#L250-L360)、
[`GroupRelativeAdvantageEstimator`](../rl/algorithm/advantage.py#L59-L107) 和
[`SyncTrainer._train_step`](../rl/trainer.py#L373-L460)。当前项目将 GRPO 标为完整支持。

### 2.2 PPO

PPO 不是只有接口或 TODO，实际实现已经存在：

- 算法注册名为 `ppo`，按 requirements 自动构建 Actor、冻结 Reference 和独立 Critic。
- Critic 复用 Qwen3 backbone，移除 LM head，增加零初始化的 FP32 scalar value head。
- 使用 terminal task reward 和 GAE，支持 `gamma`、`gae_lambda`、全局 action-token advantage normalization。
- Actor 使用 clipped policy objective + Reference KL。
- Critic 使用 clipped value regression loss。
- old logprobs、old values、advantages、returns 在更新前固定；Actor/Critic 有各自的 optimizer、scheduler、micro/mini batch 和 update epochs。
- checkpoint 同时保存/恢复 Actor、Critic、两套 optimizer/scheduler、RNG、dataloader 和训练进度；HF 导出只导出 Actor。

关键实现见 [`PPOAlgorithm`](../rl/algorithm/loss.py#L363-L486)、
[`GAEAdvantageEstimator`](../rl/algorithm/advantage.py#L110-L159)、
[`Critic`](../rl/roles/policy/critic.py) 和
[`SyncTrainer._build_models_and_optimizers`](../rl/trainer.py#L683-L754)。

结论是：**GRPO 和 PPO 都已经代码实现并能运行训练；但成熟度不同。**

- GRPO：当前主要完整支持路径。
- PPO：项目标记为“部分验证”。已有 Qwen3-4B 两卡 FSDP2/full-gather、四卡 FSDP2 × TP2/direct-reshard、
  两卡双角色 checkpoint 恢复真机记录。
- 尚不能外推为 PPO 多轮 Agent、PPO disjoint、PPO bit-exact 或长期收敛均已验收。详细记录见
  [PPO 文档](ppo.md#当前验证记录2026-09-12)。

## 3. 权重同步方式

权重同步是训练 Actor 到 vLLM rollout worker 的在线策略发布，不是 checkpoint 保存。实现由
[`ActorRolloutWeightSync`](../rl/roles/weight_sync/sync.py#L98-L297) 和
[`WeightPublisher`](../rl/roles/weight_sync/transfer.py#L296-L387) 编排。

### 3.1 两种数据策略

| 策略 | 状态 | 数据路径 |
| --- | --- | --- |
| `full_gather` | 支持，默认 | 从训练 FSDP/TP local state 逐参数聚合出完整 HF 权重，按 `bucket_size_mb` 打成 packed bucket；只有 Trainer rank 0 打包。worker 解包后调用 vLLM `load_weights()`，由 Native/Hyper 模型解释目标布局。 |
| `direct_reshard` | 支持 | 同时读取训练 source layout 和 rollout destination layout，计算物理区域交集，按 route/bucket 直接把所需 fragment 发往目标 TP rank，避免先构造所有完整参数。 |

两种策略都处理 Qwen3 的 tied embedding、Native-vLLM 的融合 QKV/gate-up 参数，以及 Hyper-vLLM 的 TP placement。
当前只支持 Qwen3 dense；配置只接受 `strategy` 与 `bucket_size_mb`，见
[`weight_sync/config.py`](../rl/roles/weight_sync/config.py)。

### 3.2 两种传输

| deployment | 传输 | 行为 |
| --- | --- | --- |
| `colocated` | NPU IPC | Trainer 与 rollout 共卡；通过 NPU IPC handle 共享 packed/direct buffer，并配合 vLLM sleep/wake 进行显存 residency 切换。 |
| `disjoint` | HCCL | Trainer 与 rollout 分卡；建立 Trainer producer 到 rollout workers 的 HCCL group，direct 定向发送，full-gather fan-out 到全部 workers。 |

因此当前实际支持四种组合：`full_gather + IPC`、`direct_reshard + IPC`、
`full_gather + HCCL`、`direct_reshard + HCCL`。

### 3.3 发布事务与一致性

一次成功发布的顺序是：

```text
关闭 rollout admission
  -> colocated 时唤醒 weights residency
  -> start_weight_update
  -> 按 bucket/fragment 传输并逐步 ACK
  -> finish_weight_update
  -> 校验所有 worker committed policy_version
  -> reset KV cache / resume admission
  -> controller 对 generation 暴露新版本
```

generation 前后也会检查 worker `policy_version`，防止一次请求跨越权重版本。任一 rank、worker 或传输阶段失败会
同步传播并终止本次运行；当前没有 direct→full fallback、透明 retry、abort/rollback 或同进程恢复。

## 4. Agentic 已实现功能

Agentic 与普通 RL 共用 `PromptRecord → Trajectory → ExperienceBatch` 的 token-first 合同。模型 action token
参与 policy loss，环境/tool observation 不参与；每条轨迹保留 raw token IDs、sampled-token FP32 raw logprobs、
action mask、reward、termination reason 和 worker policy version。

### 4.1 三种 runner

| `agentic.runner` | 状态 | 已实现能力 |
| --- | --- | --- |
| `internal` | 支持 | 框架内部 `AgentRunner + AgentSession`，支持 single-turn / multi-turn、批量活跃 session、稳定 row seed、最大轮次、observation/episode token 上限、终止原因和资源关闭。 |
| `codex` | 部分验证 | `ProgramAgentRunner` 启动固定版本 Codex CLI，通过本地 Responses gateway 转发到共享 vLLM；支持 shell、function/MCP tool、SSE 响应转换、轨迹捕获和自定义 reward callable。当前固定验证 `codex-cli 0.152.1`。 |
| `deepseek` | 部分验证 | `ProgramAgentRunner` 启动 DeepSeek Harness，通过本地 Chat gateway 转发到共享 vLLM；支持流式协议、工具记录、轨迹捕获和自定义 reward callable。当前固定验证 SDK `0.1.1rc1`。 |

三种 manager 的选择位于 [`SyncTrainer._new_rollout_manager`](../rl/trainer.py#L772-L845)，完整接口说明见
[Agentic RL](agentic_rl.md)。配置只接受这三种 runner；任意第四种 program runner 还不能只通过 YAML 注册。

### 4.2 环境、工具与协议

当前已提供：

- `ENVIRONMENTS` registry 和 `agentic.module_path` 动态加载，可注册用户自己的 Environment builder。
- Environment 生命周期：`reset()`、每轮 `step()`、`close()`，并显式管理 observation、action、reward、done 和 metadata。
- `ToolEnvironment`，把 action parser、工具执行、observation 格式化和终局 reward 组合起来。
- episode-local `ToolRegistry`，支持 Python 同步或异步 handler。
- `json_function_call`、`openai_tool_call` 两种 internal interaction protocol。
- 工具参数的 JSON Schema object 子集校验、单次调用超时（包含排队时间）、并发上限、每轮调用数上限、
  超时同步任务的有界保留，以及将可恢复错误转为 model-visible result。
- MCP stdio bridge，把同一 `ToolRegistry` 暴露成 `tools/list` 和 `tools/call`，供 Codex runner 使用。
- GSM8K 单轮环境、多轮 calculator 工具环境和规则奖励示例。

代码入口是 [`agentic/envs/environment.py`](../rl/agentic/envs/environment.py)、
[`agentic/tools/executor.py`](../rl/agentic/tools/executor.py)、
[`agentic/core/runner.py`](../rl/agentic/core/runner.py)、
[`agentic/core/program_runner.py`](../rl/agentic/core/program_runner.py) 和
[`agentic/mcp_server.py`](../rl/agentic/mcp_server.py)。

### 4.3 当前边界

- Agentic 运行仍属于同步、单节点训练，不等于已经实现通用异步 Agent RL。
- 外部 harness 必须复用唯一共享 vLLM endpoint，不能创建第二个 rollout Router。
- Trainer TP > 1 时只有 TP group 的 request-owner rank 执行外部 program，轨迹再同步给 sibling ranks。
- 还没有多模态 Environment/trajectory 的端到端实现。
- DeepSeek Agentic 已有 Native-vLLM 两步训练真机记录；Codex 的流程、工具往返和权重发布通过过真机测试，
  但当次 reward 组内相同导致零 advantage/零梯度，不能算严格非零学习验收。

## 5. Rollout 是否同时支持原生 vLLM 和 Hyper 注册

**支持，两者都运行在 vLLM 内，不是两个不同的 generation backend。**

| `rollout.vllm.model_implementation` | vLLM architecture | 加载方式 |
| --- | --- | --- |
| `native` | `Qwen3ForCausalLM` | 使用 checkpoint 自带 HF architecture，由 vLLM 原生 Qwen3 实现加载；server command 不添加 Hyper architecture override。 |
| `hyper` | `HyperQwen3ForCausalLM` | RL 包通过 `vllm.general_plugins` entry point 注册模型，启动 server 时用 `--hf-overrides` 选择该 architecture。 |

选择与身份校验见 [`rl/roles/model.py`](../rl/roles/model.py#L27-L134)，插件入口见
[`pyproject.toml`](../pyproject.toml#L11-L12) 和
[`rl/roles/rollout/vllm_plugin.py`](../rl/roles/rollout/vllm_plugin.py)。Hyper adapter：

- 基于 Transformers Qwen3 模型语义；
- 将 attention leaf 替换成 vLLM `Attention`，因此仍使用 vLLM Paged Attention 和 KV Cache；
- 复用 vLLM 已建立的 TP process group 构造 HyperParallel mesh 并应用 TP plan；
- 支持 `load_weights()` 和在线 full/direct 权重同步；
- 当前只支持 BF16、非量化 Qwen3 dense，PP=1、CP=1、非多模态。

Hyper model registration 对版本有严格保护：当前只在 `vllm==0.22.1`、
`vllm-ascend==0.22.1rc1` 时注册私有生命周期适配。Native 和 Hyper 都复用同一个
`VLLMGenerationEngine`、OpenAI-compatible HTTP 接口、策略版本校验、DP Router、Prefix Cache/Chunked Prefill
配置透传和权重发布控制面。

## 6. Rollout 路由与负载均衡

### 6.1 已实现的请求路径

当前是一个 coordinator-owned shared endpoint：

- Trainer rank 0 启动、检查并关闭一组 vLLM server processes；其他 Trainer ranks 连接同一 host/port。
- server 以 `--data-parallel-size DP --tensor-parallel-size TP` 启动。
- 所有 generation 请求都发到同一 `/v1/completions`（外部 Agent 使用 gateway 转换后的
  `/v1/chat/completions`）。
- endpoint 内部由 **vLLM upstream DP Router** 选择 DP engine；Hyper-RL 不设置生产用固定 DP-rank header，
  也不维护 per-replica port 或 rank-local server。
- `api_server_count` 由 vLLM upstream 管理，Hyper-RL 配置层会拒绝用户覆盖。

所以，如果“支持 rollout 路由负载均衡”指多个 rollout DP replicas 之间的请求分发，答案是 **支持**；
但实现归属是 vLLM Router，而不是 Hyper-RL 自己实现的 least-load/round-robin 算法。

### 6.2 Hyper-RL 自己补充的 admission 与去重

Hyper-RL 在 Router 前增加了请求侧容量控制，代码见
[`_VLLMHTTPClient._dispatch_completion_requests`](../rl/roles/rollout/vllm.py#L496-L592) 和
[`VLLMGenerationEngine._local_child_capacity`](../rl/roles/rollout/vllm.py#L1113-L1147)：

- 使用长期 `aiohttp.ClientSession` 和独立 asyncio loop，并发提交 HTTP 请求。
- admission 按 parent request 展开的 child 数计费，而不是只数 HTTP request 数；`n` choices 也计入容量。
- 使用 `asyncio.FIRST_COMPLETED` 滚动补充 pending work，避免等待整批最慢请求后才提交下一批。
- local quota 从 `rollout DP × (max_num_seqs × 2)` 的全局有界容量按 Trainer logical request ranks 分配；
  `max_num_seqs` 也可按工作量、KV cache 容量和 batch-token 上限自动计算。
- completion 可以乱序完成，但按原始 row slot 重排输出，并保留稳定 row seed。
- Trainer TP > 1 时，每个训练 TP group 只有 `tp_rank=0` 发一次 HTTP 请求，随后广播 sequences、mask、
  raw logprobs、耗时和 policy version，避免 TP siblings 重复采样。

这套逻辑解决的是客户端背压、并发利用率、结果顺序和重复请求问题；**它不读取每个 vLLM DP engine 的实时负载，
也不替代 vLLM Router 的 engine 选择。** 相关单测位于
[`test_vllm_runtime.py`](../../../tests/ut/rl/rollout/test_vllm_runtime.py)，覆盖容量上限、滚动补充、乱序恢复、共享
endpoint owner 和 TP request owner。

### 6.3 未实现的路由能力

- 没有 Hyper-RL 自定义的 per-engine least-load、weighted routing 或固定 rank affinity。
- 没有多节点 rollout Router、动态扩缩容或故障 engine 自动摘除/重试合同。
- 没有透明 generation retry；请求失败会跨 Trainer ranks 传播并终止当前同步运行。
- 没有 MoE EPLB；当前 Qwen3 dense 强制关闭 expert parallel 和 EPLB。

## 7. 当前能力边界汇总

| 能力 | 状态 |
| --- | --- |
| 单节点同步 GRPO | 已支持 |
| 单节点同步 PPO | 已实现，部分拓扑/任务已验证 |
| Qwen3-4B dense | 已支持 |
| Trainer FSDP2、TP1/TP2、FSDP2 × TP2 | 已支持 |
| Rollout DP × TP、colocated/disjoint | 已支持；公开验证范围以 TP1/TP2 为主 |
| Native-vLLM / Hyper-vLLM | 均支持 |
| full-gather / direct-reshard，IPC / HCCL | 均支持 |
| vLLM upstream DP Router + 客户端有界并发 | 已支持 |
| Internal 单轮/多轮 Agent、工具、自定义环境/奖励 | 已支持 |
| Codex / DeepSeek 程序化 Agent | 已接入，部分验证 |
| Checkpoint/resume、评估、console/W&B、可选 bit-exact gate | 已支持，组合范围受文档约束 |
| 复制型 DP/HSDP、CP、PP、EP/EPLB | 当前 RL 不支持 |
| 多节点、异步/off-policy、动态扩缩容、透明 retry | 未实现 |
| 多模态、MoE、大模型多节点 RL | 规划中/当前适配已移除 |

### 7.1 四卡真实测试数据对比总表

| 测试方向 | 拓扑 | 指标 | 基准方案 | 对比方案 | 实测变化 | 统计窗口与状态 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 训推 logprob | Trainer FSDP4 → Hyper-vLLM DP4×TP1 | mean absolute `logprob_diff` | 一致性关闭：`0.00807577` | 一致性开启：`0` | 降为 0，6144 token bit-exact | 对齐 step 1–3；开启 arm 三步 mismatch 均为 0 |
| 训推 logprob | Trainer FSDP4 → Hyper-vLLM DP4×TP1 | max absolute `logprob_diff` | 一致性关闭：`0.500000` | 一致性开启：`0` | 最大差异由 0.5 降为 0 | 对齐 step 1–3，共 6144 个有效 action token |
| Rollout 推理 | Trainer FSDP4 → Hyper-vLLM DP4×TP1 | 稳态 generation latency | 一致性关闭：`13.0041s` | 一致性开启：`13.4983s` | **增加 3.80%** | step 2–3；每 step 16 sequences / 2048 generated tokens |
| Rollout 推理 | Trainer FSDP4 → Hyper-vLLM DP4×TP1 | 稳态吞吐 | 一致性关闭：`157.49 tok/s` | 一致性开启：`151.72 tok/s` | **下降 3.66%** | 两步合计 4096 generated tokens，吞吐按总 token/总时间计算 |
| 完整训练步 | Trainer FSDP4 → Hyper-vLLM DP4×TP1 | `timing_s/step` | 一致性关闭：`45.7556s` | 一致性开启：`57.3987s` | **增加 25.45%** | step 2–3；包含额外 bit-exact gate 和 post-update logprob replay |
| Rollout 实现 | Trainer FSDP4 → rollout DP4×TP1 | 5-step mean absolute `logprob_diff` | Hyper-vLLM：`0.00869173` | Native-vLLM：`0.00870777` | Native **增加 0.18%** | 普通模式 step 1–5，各 10240 个有效 action token；两者均非 bit-exact |
| Rollout 实现 | Trainer FSDP4 → rollout DP4×TP1 | 5-step max absolute `logprob_diff` | Hyper-vLLM：`0.500000` | Native-vLLM：`0.685040` | Native **增加 37.01%** | 普通模式 step 1–5 的全局最大值 |
| Rollout 实现 | Trainer FSDP4 → rollout DP4×TP1 | generation latency | Hyper-vLLM：`12.5480s` | Native-vLLM：`13.1498s` | Native **增加 4.80%** | 主口径 step 2–5；Native step 2 有一次 20.4405s 抖动 |
| Rollout 实现 | Trainer FSDP4 → rollout DP4×TP1 | generation throughput | Hyper-vLLM：`163.21 tok/s` | Native-vLLM：`155.74 tok/s` | Native **下降 4.58%** | 主口径 step 2–5，共 8192 generated tokens/arm |
| 权重同步 | Trainer FSDP4 → Hyper-vLLM DP2×TP2 | 稳态 `timing_s/weight_sync` | full-gather：`14.27390s` | direct-reshard：`6.41118s` | **降低 55.08%，direct 快 2.23×** | step 2–4；两组均完成 V1–V4 发布 |
| 完整训练步 | Trainer FSDP4 → Hyper-vLLM DP2×TP2 | `timing_s/step` | full-gather：`32.7551s` | direct-reshard：`25.6580s` | **降低 21.67%** | step 2–4；比较仅切换 weight-sync strategy |
| PPO 闭环 | Trainer FSDP-shard2×TP2 → Hyper-vLLM DP2×TP2 | Actor / Critic gradient norm | Step 1：`2.46875 / 1.24728` | Step 2：`0.691406 / 0.908462` | 两步均非零 | 真实 GSM8K reward；direct-reshard 发布 V1/V2；进程正常退出 |

一致性开启 arm 在完成三个 bit-exact step 后，于 step 4 切回训练 residency 时发生 NPU OOM，因此上表的一致性
性能只使用两个共同完成的稳态 step，不代表长期稳定性能。权重同步两组均完整运行 4 steps。Native-vLLM
普通模式完整运行 5 steps；其现有配置不能启用 `qwen3_ascend_consistency_v1`，因此 Native 数据是训推差异诊断，
不是 bit-exact profile 验收。逐 step 数据和计算口径见下节。

进一步的运行合同和历史验收记录见 [vLLM Rollout](vllm_rollout.md)、[PPO](ppo.md)、
[Agentic RL](agentic_rl.md)、[系统测试记录](hyper-rl-st.md) 和
[训练-推理一致性](qwen3_training_inference_consistency.md)。

### 7.2 Native-vLLM 四卡实测明细

Native-vLLM arm 与上表的 Hyper-vLLM consistency-off arm 使用相同设备、模型、数据、seed、请求量、cache、
admission、full-gather 和训练配置，只修改：

```text
rollout.vllm.model_implementation: hyper -> native
```

共同拓扑与工作量：Trainer `FSDP-shard4 × TP1`，rollout `DP4 × TP1`，每 step 4 prompts × 4 responses，
每条最多 128 tokens，实际每 step 2048 generated/action tokens。日志确认 Native arm 使用原生
`Qwen3ForCausalLM`，完整运行 5 steps，并完成 V1–V5 发布。

#### Native-vLLM 训推 logprob 差异

| Step | 有效 token | mean logprob diff | max logprob diff |
| ---: | ---: | ---: | ---: |
| 1 | 2048 | 0.00788266 | 0.685040 |
| 2 | 2048 | 0.00821808 | 0.560932 |
| 3 | 2048 | 0.00869796 | 0.618365 |
| 4 | 2048 | 0.00847277 | 0.473377 |
| 5 | 2048 | 0.01026740 | 0.450287 |
| **全部 5 steps** | **10240** | **0.00870777** | **0.685040** |

Native 普通模式的 mean/max diff 均非 0，因此当前 Trainer 与 Native-vLLM **不是 bit-exact**。现有
`consistency.enabled=true` 会在配置校验阶段要求 `model_implementation=hyper`，不能用于 Native-vLLM；
所以没有“Native 开启现有 profile”的合法对照数据。

#### Native-vLLM 与 Hyper-vLLM 普通模式推理性能

| Step | Hyper generation / tok/s | Native generation / tok/s | Native latency 相对 Hyper |
| ---: | ---: | ---: | ---: |
| 1，首次请求 | 37.1519 s / 55.125 | 18.5404 s / 110.461 | -50.10% |
| 2 | 14.0727 s / 145.530 | 20.4405 s / 100.193 | +45.25% |
| 3 | 11.9355 s / 171.588 | 10.6167 s / 192.904 | -11.05% |
| 4 | 12.1164 s / 169.028 | 10.8860 s / 188.132 | -10.16% |
| 5 | 12.0675 s / 169.713 | 10.6560 s / 192.192 | -11.70% |
| **主口径：Step 2–5** | **12.5480 s / 163.21** | **13.1498 s / 155.74** | **+4.80%，吞吐 -4.58%** |
| **中位数：Step 2–5** | **12.0920 s** | **10.7710 s** | **-10.92%** |
| **敏感性：Step 3–5** | **12.0398 s / 170.10** | **10.7196 s / 191.05** | **-10.97%，吞吐 +12.32%** |

主口径沿用测试前确定的 step 2–5，不因结果改变窗口。Native step 2 的 `20.4405s` 明显高于其 step 3–5，
使平均值和中位数得出不同方向。step 3–5 的 generation 样本标准差为 Native `0.1455s`、Hyper `0.0936s`，
尾部三个 step 内 Native 更快且相对稳定；但这次单次顺序运行无法判断 step 2 抖动是 Native 第二次请求/首次发布后
的必然冷态，还是系统噪声。

本轮结论：

- Native-vLLM 五步平均 logprob diff 为 `0.00870777`、最大为 `0.68504`，未达到 bit-exact。
- 相比 Hyper-vLLM 普通模式，mean diff 几乎相同（Native 高 `0.18%`），但观察到的 max diff 高 `37.01%`。
- 按预定 step 2–5 窗口，Native 平均 latency 慢 `4.80%`、吞吐低 `4.58%`；按中位数或 step 3–5，
  Native 反而快约 `10.9%`。当前样本不足以声称任一实现有稳定性能优势，需要交替顺序、多次重复运行确认。

原始日志：[Native-vLLM DP4，5 steps 完成](../output/feature-bench-20260913/4c-fsdp4-rollout-native-dp4-consistency-off.log)。

### 7.3 PPO 四卡跑通验证

本轮使用项目自带 `qwen3_4b_gsm8k_ppo.yaml`，通过 Docker 运行真实 Qwen3-4B、GSM8K 和非零学习率：

```text
Algorithm: PPO
Trainer: FSDP-shard2 × TP2，4 ranks
Rollout: Hyper-vLLM DP2 × TP2，colocated
Weight sync: direct_reshard + NPU IPC
Actor LR: 1e-6
Critic LR: 1e-5
Workload: 每 step 4 prompts × 4 responses = 16 logical trajectories，max_new_tokens=512
Steps: 2
```

| 指标 | Step 1 | Step 2 | 跑通判定 |
| --- | ---: | ---: | --- |
| Reward mean | 0.3750 | 0.3125 | 两步均有 0/1 混合 reward |
| Actor gradient norm | 2.46875 | 0.691406 | 两步均非零 |
| Actor total loss | 0.000294601 | 0.0000146838 | 两步均为有限非零值 |
| Critic gradient norm | 1.24728 | 0.908462 | 两步均非零 |
| Critic value loss | 0.00425813 | 0.00321206 | 两步均为有限非零值 |
| Critic value mean | 0 | 0.00882578 | value head 在首步更新后产生非零预测 |
| Actor/Critic valid tokens | 7226 / 7226 | 7669 / 7669 | policy/value target 对齐并完成优化 |
| Weight sync | 8.90231 s | 5.59834 s | direct-reshard 成功完成 |
| Published policy version | V1 | V2 | 两次发布均完成 worker version 校验 |
| 完整 step 时间 | 337.258 s | 150.535 s | Step 1 含 lazy vLLM 启动与首次请求冷态 |

结论：**PPO 在该四卡 `FSDP-shard2×TP2 → DP2×TP2` 组合上可以跑通。** Actor 和 Critic 都执行了
非零更新，第二步使用已发布 V1 完成新 rollout，并再次更新和发布 V2；容器退出码为 0，日志未发现 Traceback、
ERROR 或 OOM，结束后测试容器和 NPU process 均已清理。

本轮关闭了 evaluation、定期 checkpoint 和最终 checkpoint，因此只证明两步 PPO 训练与在线策略发布闭环可运行，
不重复验证 checkpoint resume、HF 导出或长期收敛。Console rollout 汇总会把 Trainer TP sibling 上广播的相同
trajectory 再次计入，因而日志中的 `rollout/sequence_count=32`，配置对应的逻辑轨迹数是 16；Actor/Critic 优化指标
使用各 DP coordinate 的有效 token 数 7226/7669。

原始日志：[四卡 PPO 两步跑通](../output/feature-bench-20260913/4c-ppo-fsdp2tp2-rollout-dp2tp2-direct.log)。
