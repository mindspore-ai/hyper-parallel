# Agentic RL：任务环境与外部 Agent 如何接入 Hyper-RL

Agentic 模块负责让模型在一个 episode 内反复执行“生成动作—调用工具—接收观察—继续生成”，并把完整过程收敛为
Hyper-RL 认识的 `Trajectory`。Hyper-RL 仍然负责数据加载、批处理、优势估计、Actor 更新、权重发布和检查点。

Agentic 决定模型如何与任务环境交互；Hyper-RL 决定这些交互数据如何用于强化学习。

## 整体关系

![Agentic RL 与 Hyper-RL 的交互架构](images/agentic_rl_architecture.svg)

三种 runner 共享 `PromptRecord → Trajectory → ExperienceBatch` 训练契约，由同一个 `SyncTrainer` 编排。
一个 episode 可以包含多条逐调用 `Trajectory`；episode 是奖励单位，训练行保留各次真实生成上下文。

## 职责边界

| 层次 | 主要职责 | 不负责的内容 |
| --- | --- | --- |
| Agentic | episode 生命周期、多轮动作与观察、工具执行、终止判断、任务奖励、轨迹构造 | 优势估计、梯度更新、权重发布 |
| Rollout | 提供共享 vLLM 推理端点，返回 token ID、raw logprob 和策略身份 | 任务语义和奖励规则 |
| Hyper-RL Trainer | 构建 prompt、选择 runner、准备训练 batch、更新 Actor、发布策略并记录指标 | 具体工具如何实现 |

`SyncTrainer` 是唯一的顶层编排者。它在 `rl/trainer.py` 中创建 rollout engine，再根据 `agentic.runner` 选择对应的
rollout manager。Agentic 不绕过 Trainer，也不直接更新模型参数。

## 三种执行路径

| `agentic.runner` | 实际入口 | episode 控制者 | 适用方式 |
| --- | --- | --- | --- |
| `internal`（默认） | `RolloutManager` → `AgentRunner` | Hyper-RL 内部循环 | 自定义环境、协议和 Python 工具 |
| `codex` | `CodexRolloutManager` → `ProgramAgentRunner` | Codex CLI program | Codex Responses 协议、shell 或 MCP 工具 |
| `deepseek` | `DeepSeekRolloutManager` → `ProgramAgentRunner` | DeepSeek Harness program | DeepSeek Chat 协议和 Harness 工具 |

配置校验只接受以上三个值。当前不能在 YAML 中填写任意第四种 runner。`ProgramAgentRunner` 虽然是公开的 Python
接口，但把新的外部 harness 接入 `SyncTrainer` 仍需要增加对应的 runtime、factory、manager 和配置校验。
DeepSeek 的实现目录是 `rl/agentic/ds_harness/`，YAML 中的 runner 值和子配置键仍分别是 `deepseek` 与
`agentic.deepseek`。

### Internal：框架内多轮交互

`AgentRunner` 为每个 `PromptRecord` 创建 `AgentSession` 和已注册的 Environment。每一轮按以下顺序运行：

1. Environment 的 `reset()` 返回初始 `Observation`。
2. `AgentRunner` 将活跃 session 左填充成 batch，并调用共享 `GenerationEngine`。
3. 生成结果作为 `Action` 交给 Environment 的 `step()`。
4. `ToolEnvironment` 使用 `InteractionProtocol` 解析最终答案或工具调用。
5. 工具调用由 `ToolExecutor` 在 `ToolRegistry` 中查找并执行，结果编码成下一轮 observation。
6. 最终答案由环境奖励函数评分；达到终止条件后，`AgentSession.build()` 生成 `Trajectory`。

已完成的 session 仍保留一个 dummy batch row，以保证所有分布式 rank 执行相同数量的 generation collective；
dummy 输出会被丢弃，不进入训练。

### 单轮 code 环境

`examples.code.agent` 注册 `code_stdio`，使用同一个 internal runner 完成一次 Python stdio 生成与远程判题。
`data.row_adapter: examples.code.prepare_data:adapt_row` 保留多消息、稳定任务 ID 和私有结构化测试；
测试不进入 prompt，代码提取不改写训练使用的采样 token/logprob。训练 TP 组仅 request owner 调用环境，
其他 rank 重放相同 observation、reward 和终止状态，避免重复判题。

全部测试按空白分词精确比较后给二值奖励；HTTP、协议或沙箱服务故障上抛，不能混成候选代码的零分。
`finish_reason=length` 随 action 传递并记录截断。数据审核、固定镜像部署与运行命令见
[单轮 code 示例](../examples/code/README.md)。

### Codex 与 DeepSeek：外部 harness 交互

两条外部路径使用相同的数据面结构：

1. manager 启动本节点的 Gateway，并读取 vLLM 当前整数 `policy_version`。
2. runtime 将该策略身份绑定到本次 episode。
3. `ProgramAgentRunner` 只在 Trainer TP 组的 request-owner rank 启动外部 program。
4. Gateway 把 Codex Responses 或 DeepSeek Chat 请求转换为 vLLM 的 `/v1/chat/completions` 请求。
5. Gateway 记录模型返回的 token ID、raw sampled-token logprob、工具调用和策略身份。
6. harness 的 trajectory builder 将记录转换为标准 `Trajectory`。
7. request-owner 将 trajectory 序列化，并通过 `synchronize_agent_payload()` 同步给同 TP 组的 sibling ranks。
8. manager 再次读取策略身份；episode 执行期间发生版本变化会直接报错。

Codex Gateway 按 `agentic.max_turns` 预留模型调用额度，成功、失败和取消路径均释放预留；
`agentic.codex.max_inflight_requests` 限制共享后端的同时请求数，默认 1。
最后一次调用仍保留原工具选择，额度用尽不会改写模型动作来强制答案。
一般调用额度耗尽仍按失败拒绝训练，配置应为最终答案预留调用额度。
终态读取 session 先封闭新请求，再等待在途请求排空；释放 session 同样排空。
整组 program 收尾后再同步跨 rank 错误。
标准 `tools` 与 input 中 `additional_tools` 共用工具名称映射，未知声明明确报错。

Codex 可按配置启动 MCP stdio server。`rl/agentic/mcp_server.py` 只是把现有 `ToolRegistry` 暴露为 MCP
`tools/list` 和 `tools/call`，不会复制或改写工具实现。

## Hyper-RL 如何消费 Agentic 结果

一次训练 step 的真实顺序位于 `SyncTrainer._train_step()`：

1. `build_prompt_records()` 把 dataloader batch 转成 `PromptRecord`。
2. 选定的 rollout manager 调用 `generate()`，返回 `ExperienceBatch`。
3. rollout engine 从 rollout residency 切回 training residency。
4. 逐调用 batch 先按 DP 最大行数补齐，再执行角色前向；`ExperiencePreparer` 根据算法需求准备训练目标。
5. Actor 使用准备好的 experience 执行更新。
6. `_publish_policy()` 把 `PolicySnapshot(version=next_step)` 发布到 vLLM。
7. vLLM 恢复 rollout residency，下一 step 使用新策略生成。

Agentic 输出与训练侧之间最重要的边界是 token-first 契约：

| 字段 | 作用 | 约束 |
| --- | --- | --- |
| `token_ids` | 完整 prompt、动作和观察序列 | rollout 返回的 ID 是训练依据，不能 decode 后再 encode |
| `attention_mask` | 标记有效 token | 必须与 `token_ids` 等长 |
| `action_mask` | 标记参与策略损失的模型动作 | 不能选中 padding；环境和工具 observation 不参与策略损失 |
| `rollout_log_probs` | rollout 策略的 next-token logprob | 使用 FP32 raw logprob，并与 next-token 位置对齐 |
| `reward` | episode 最终奖励 | 每条 trajectory 一个标量 |
| `worker_policy_version` | 记录生成使用的策略版本 | batch 内必须一致，并匹配请求的策略版本 |

`build_experience_batch()` 会验证这些 trajectory 的对齐关系，并将其填充成 `ExperienceBatch`；随后算法和 Actor
只处理标准 batch，不需要知道 episode 来自 internal、Codex 还是 DeepSeek。

## 逐调用轨迹与 episode GRPO

外部 Codex 与 DeepSeek program 每次模型调用返回一条训练行，tokens 严格等于本次真实 prompt 加 sampled action；
历史被 harness 改写或裁剪也不会伪造连续上下文。`episode_id`、`call_index` 和 `call_count`
描述完整 episode，缺失、重复调用以及 prompt/group/策略版本/奖励不一致都会报错。
旧连续轨迹构造器（含 DeepSeek）只有在后续 prompt 精确延续全部已采样 token 时才允许拼接。
分段路径的 `max_episode_tokens` 校验每次调用的真实 prompt 加 action 长度；调用总数由 `max_turns`
限制，不把重复上下文累加成单条虚构轨迹的长度。

例如同一道题的两个候选分别调用 2 次和 5 次，最终奖励为 1 和 0：GRPO 对两个 episode 计算优势，
再映射到七条调用行；准确率为 1/2，不能按调用行算成 2/7。
每条调用的动作仍参与既有有效 token 均值损失；episode 计奖并不等于每个 episode 的梯度权重相同。
评估和样例按 episode 汇总，响应长度为其所有调用的动作 token 总数。

DP rank 的调用行数不同会追加 `dp_padding` 行，保留合法上下文供前向执行，但 action mask、奖励和优势为零。
它们不参与奖励、episode 数、响应长度或损失的有效 token 分母。
分段轨迹当前只支持 GRPO；Codex/DeepSeek+PPO 在启动前拒绝，现有非分段 PPO 仍按原合同运行。

## 工具失败归因与收尾

固定版本 vLLM/Hermes 只读采集原始 token、engine text、parser 输入和结果，不修改采样动作。
归因规则为：

- 原始证据一致且模型 JSON 或工具调用封装结构不合法：模型失败，保留该次动作及概率；在剩余调用额度内可重新生成。
- 原始工具 JSON 合法但解析结果不一致：基础设施失败，不进入训练。
- 工具调用缺少原始证据或证据互相矛盾：未知失败，不进入训练。

无法训练的失败拒绝整组更新，不能悄悄丢弃候选；同一 session 的后续模型失败不能覆盖已有基础设施失败。
Codex 格式重采样也计入调用额度和完整 episode；已确认的模型格式错误耗尽额度时形成零奖励终止结果。
DeepSeek 保留自己的终止方式：已记录证据的模型格式错误或明确调用额度耗尽可形成零奖励终止，
未知 SDK `error/aborted`、服务错误和解析证据不一致均拒绝训练。
外部程序结束、超时或取消后清理本程序组的工具子进程；vLLM 先优雅退出，超时才强制终止。
最终 checkpoint 在评估结束后先释放 vLLM 及其 IPC 消费者，减少保存阶段的资源竞争。

## 接入自定义 Environment

现有的自定义入口是 `agentic.module_path` 和 `ENVIRONMENTS`：

1. 在扩展模块中实现接收 `EpisodeContext` 的 environment builder。
2. 返回实现 `reset()`、`step()`、`close()` 的 Environment；需要工具时可组合 `ToolEnvironment`、
   `InteractionProtocol`、`ToolExecutor` 和 `ToolRegistry`。
3. 使用 `ENVIRONMENTS.register("name")(builder)` 注册。
4. 在 YAML 中配置同一模块路径和环境名称。

```yaml
agentic:
  runner: internal
  module_path: examples.gsm8k.agent
  environment: gsm8k_tools
  interaction_mode: multi_turn
  protocol: json_function_call
  max_turns: 2
  max_observation_tokens: 0
  max_episode_tokens: 1024
```

仓库中的 [GSM8K 环境](../examples/gsm8k/agent.py)提供单轮问答和组合 calculator 的多轮任务，
两条路径都通过同一个 `ENVIRONMENTS` registry 接入 Trainer。上面的 YAML 是 `agentic` 片段；
完整训练配置见 [单轮配置](../examples/gsm8k/configs/single_turn.yaml)和
[多轮配置](../examples/gsm8k/configs/multi_turn.yaml)，训练入口为 [train_rl.py](../train_rl.py)。

## 外部 runner 的必要配置

Codex 和 DeepSeek 除各自子配置外，还必须满足以下共享约束：

- `rollout.engine` 必须是 `vllm`。
- `rollout.vllm.logprobs_mode` 必须是 `raw_logprobs`。
- `rollout.vllm.enable_auto_tool_choice` 必须为 `true`。
- `rollout.vllm.tool_call_parser` 必须是 `hermes`，原始解析证据仅适配固定镜像中的该解析器。
- Gateway 端口不能与 vLLM 端口相同。
- Codex CLI 当前固定验证版本为 `0.152.1`。
- DeepSeek Harness SDK 当前固定验证版本为 `0.1.1rc1`。

完整配置以以下文件为准：

- [Codex 多轮配置](../examples/gsm8k/configs/codex_multi_turn.yaml)
- [DeepSeek 多轮配置](../examples/gsm8k/configs/deepseek_multi_turn.yaml)

## 代码导航

| 关注点 | 文件 |
| --- | --- |
| 配置与 runner 约束 | `rl/config.py` |
| 唯一训练编排入口 | `rl/trainer.py` |
| 三种 rollout manager | `rl/roles/rollout/worker.py` |
| Internal episode 循环 | `rl/agentic/core/runner.py`、`session.py` |
| 外部 program 数据面 | `rl/agentic/core/program_runner.py` |
| 环境与工具组合 | `rl/agentic/envs/environment.py`、`rl/agentic/tools/` |
| Codex 适配 | `rl/agentic/codex/` |
| DeepSeek 适配 | `rl/agentic/ds_harness/` |
| MCP 工具桥接 | `rl/agentic/mcp_server.py` |
| 标准训练数据契约与 episode 分组 | `rl/dataset/contracts.py`、`batch_builder.py`、`episodes.py` |
| 原始工具证据与可训练性 | `rl/tool_protocol.py`、`rl/roles/rollout/vllm_plugin.py` |
| 共享 vLLM 与策略身份 | `rl/roles/rollout/vllm.py` |

## 当前实现边界

- 当前支持的训练算法是 GRPO 和 PPO；实际验证范围以[特性清单](current_feature_inventory.md)及
  [系统测试指南](hyper-rl-st.md)为准。多节点和异步 rollout 不在本文描述的实现范围内。
- 外部 runner 复用唯一共享 vLLM endpoint，不创建第二个 rollout Router 或 rank-local server。
- External harness 的语义循环可以不同，但必须返回标准、token 对齐且策略身份一致的 `Trajectory`。
- Agentic UT 可从仓库根目录运行 `pytest -q tests/ut/rl/agentic/agentic_ut.py`；
  独立脚本的 coverage 门槛及已有缺口见 [UT 指南](hyper_rl_ut.md)。

## 验证入口

逐调用轨迹、episode GRPO、DP 补齐及工具失败归因的 CPU 合同位于 `tests/ut/rl/` 的
`data/`、`agentic/` 和 `trainer/` 模块；入口清单见[功能验证说明](moe_code_agent.md#单元测试)。
`hyper_parallel/rl/tests/st/test_feature_st.py` 提供独立两进程 Gloo 梯度验证及真实 agent 训练入口。
Gloo 验证不需要模型；真实训练须显式配置 CLI/SDK、模型、数据、设备与具有不等调用的任务，
并保留真实参数更新、策略版本和零损失补齐断言。CPU 用例及资源缺失时的 skip 不能代替真机验收。
