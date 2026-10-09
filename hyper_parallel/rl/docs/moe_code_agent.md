# Qwen3-MoE、单轮 Code 与 Agent 功能及验证

本文概述 Qwen3-MoE、单轮 code 和外部 agent 三项扩展及其验证入口。
实现合同与使用方法归属下列已有文档；不包含本机路径、实验流水账或 Code Agent 后续开发计划。

## 功能与支持边界

| 扩展 | 支持内容 | 当前边界与使用文档 |
| --- | --- | --- |
| Qwen3-30B-A3B | 公共模型构建、EP/EDP、专家权重 full_gather/direct_reshard 发布 | GRPO、colocated native vLLM、consistency off、EPLB off；见 [rollout](vllm_rollout.md#qwen3-30b-a3b) |
| 单轮 Python code | 结构化私有测试、TP owner 唯一执行、SandboxFusion 判题、评估指标 | Python stdio、逐测试空白分词比较、全测二值奖励；见 [code 示例](../examples/code/README.md) |
| 外部 agent | 每次模型调用的真实 P/A、episode GRPO、DP 零损失补齐、工具证据与收尾 | 分段 PPO 不支持；无前缀相同调用的拼接优化；见 [Agent 合同](agentic_rl.md) |

模型格式错误、服务故障和未知失败必须区分；后两者不能悄悄转为零奖励或丢弃部分候选。
保留真实 token、logprob、action mask、策略版本和发布失败传播，不通过重建上下文或缩小判题集合通过验证。

## 单元测试

UT 位于 `tests/ut/rl/`，使用现有 pytest runner；CPU 逻辑用例不要求设备或真实分布式初始化。

| 模块 | 代表性测试 |
| --- | --- |
| MoE 配置与同步 | `trainer/test_moe_config.py`、`weight_sync/test_moe_weight_sync.py` |
| code 数据与判题 | `data/test_code_data.py`、`agentic/test_code_judge.py`、`agentic/test_code_runner.py` |
| agent 轨迹与编排 | `data/test_episodes.py`、`agentic/test_agent_program.py`、`agentic/test_deepseek_segments.py` |
| 工具归因与收尾 | `agentic/test_agent_protocol.py`、`trainer/test_agent_integration.py` |

```bash
python -m pytest -q tests/ut/rl
```

## 系统测试

ST 保留在 `hyper_parallel/rl/tests/st/`，由轻量 `test_feature_st.py` 显式启动；收集阶段不导入训练框架。
`_moe_train.py`、`_code_train.py`、`_agent_train.py` 和 `_agent_dp.py` 仅为子进程 worker。
这些用例与原 `test_rl_st.py` 的 dense/一致性/PPO 配方互补，不替换原门禁。

两进程 CPU/Gloo 验证比较真实不等长调用补齐前后的梯度与更新参数，不需要 NPU：

```bash
python -m pytest -q hyper_parallel/rl/tests/st/test_feature_st.py::test_agent_dp_padding
```

真实训练须先完成当前 checkout 的 editable 安装，准备模型、数据、设备和对应服务。
在配置好的训练环境中选择一个入口，配置必须与设备数一致，结果目录须为空：

```bash
export RL_ST_MOE_CONFIG=/path/to/resolved-moe.yaml
export RL_ST_WORLD_SIZE=4
export RL_ST_RESULT_DIR=/path/to/new-results
export RL_ST_REQUIRED=1
python -m pytest -q 'hyper_parallel/rl/tests/st/test_feature_st.py::test_feature_training[moe]'
```

code 和 agent 分别使用 `RL_ST_CODE_CONFIG`、`RL_ST_AGENT_CONFIG` 与参数 `[code]`、`[agent]`。
MoE 两种发布策略应分别运行；code 的 dense/MoE 配方也分别验收；agent 的 Codex/DeepSeek 配方分别运行。
worker 保留两步更新、真实参数变化与策略版本检查；agent 额外要求真实多调用及 DP 不等行数补齐。
agent 所需动态调用模式应在配方中明确配置，不能把没有观察到补齐的运行记为该项通过。
输出记录只写入结果目录，不纳入版本管理。

SandboxFusion 的独立真实服务测试无需模型或训练设备，配置见 [code 示例](../examples/code/README.md)。
未设置真实资源时对应测试标记 skip；设置 `RL_ST_REQUIRED=1` 后缺少配置会失败，skip 不算验收通过。

## 验证结果的口径

迁移阶段曾分别验证 MoE 四卡两种权重发布、dense/MoE 单轮 code、Codex/DeepSeek 分段训练、
真实不等调用补齐及 dense 一致性/PPO 回归。每项结果只适用于当时指定镜像、模型、数据和拓扑。
本次测试归位后，RL UT、启动器合同及两进程 Gloo 验证合计 580 项通过；
另有 3 项真实 NPU 训练与 1 项真实 SandboxFusion 因未配置资源而跳过，未算作通过。
本次 PR 整理不把历史真机结果称作新测试入口或更新后共享核心的重新验收。

不宣称完整 benchmark、MoE PPO/disjoint、任意模型一致性或仓库级 Code Agent 训练通过。
临时开发约定、原始日志、权重、缓存和本机实验产物不属于 PR 内容。
