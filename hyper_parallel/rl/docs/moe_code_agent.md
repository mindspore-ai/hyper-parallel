# MoE R3、Code 与 Agent 使用及验证

本指南汇总 MoE 路由重放、单轮代码任务和外部 Agent 的支持范围、配置及验证入口。
训练环境见 [Docker 运行说明](../docker/README.md)，各任务的数据准备和启动方法见对应示例。

## 功能与支持边界

| 功能 | 支持范围 | 使用入口 |
| --- | --- | --- |
| Qwen3-30B-A3B | GRPO、colocated native vLLM、训练 TP1/TP2 与 EP/EDP；关闭 consistency 和 EPLB | [MoE rollout](vllm_rollout.md#qwen3-30b-a3b) |
| MoE R3 | Qwen3-MoE Actor 的路由重放；internal 单轮/连续多轮、Codex 逐调用轨迹 | [R3 配置](#启用-r3)、[GSM8K 示例](../examples/gsm8k/README.md) |
| 单轮 Python code | 私有 stdin/stdout 测试、TP owner 唯一判题、独立 SandboxFusion 服务 | [code 示例](../examples/code/README.md) |
| 外部 Agent | 每次模型调用保留真实 prompt/action，episode GRPO，DP 零损失补齐 | [Agent 合同](agentic_rl.md)、[Search-R1](search_r1.md) |
| 仓库 Code Agent | 隔离候选工作区、冻结产物、独立 grader；本地或远端执行 | [SWE-Smith 示例](../examples/swe_smith/README.md) |

MoE 不支持 PPO、disjoint rollout 或 Hyper-vLLM。R3 尚未适配 DeepSeek runner，
不支持与启用的 distillation/OPD 组合。模型格式错误或已确认的任务预算耗尽可形成任务失败；
通信、执行服务和未知错误必须向上传播，不能静默记零分或丢弃候选。

## MoE R3

Rollout Router Replay（R3）记录采样时的 expert 选择，在 Actor 的概率计算、训练前向和
activation checkpoint 反向重算时重放。重放的是离散 expert ID；权重仍根据当前 router 概率
计算，router 和 expert 梯度正常参与更新。Reference 保持原生路由，`old_log_probs` 保留 rollout 原始值。

R3 的数据传递与微批作用域属于通用 RL 功能，与数据集、奖励规则和 Agent 调用预算独立。
当前只有 Qwen3-MoE 的 gate 接口适配，并不表示任意 MoE 模型都可直接启用。
适配位于 RL 内，不要求修改共享模型、FSDP/HSDP、TP/EP 或优化器代码。

### 启用 R3

默认配置不改变原有行为：

```yaml
train:
  router_replay:
    mode: disabled  # R3 同时开启 rollout capture 和 Actor replay
  model_init_cpu_threads: null
rollout:
  vllm:
    sleep_level: 1
```

推荐从 [Qwen3-30B-A3B Codex 四卡配置](../examples/gsm8k/configs/qwen3_30b_a3b_codex_multi_turn.yaml)
开始。在已安装当前 checkout 的训练容器中，挂载配置要求的权重与数据后执行：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  hyper_parallel/rl/train_rl.py \
  hyper_parallel/rl/examples/gsm8k/configs/qwen3_30b_a3b_codex_multi_turn.yaml \
  --train.router_replay.mode=R3
```

该配方预设训练 TP2/EP2/EDP2、rollout TP4/DP1，R3 默认仍关闭。
开启 R3 不会调整候选数、工具调用次数、上下文、生成预算、评测或保存设置。
无需单独配置 `enable_return_routed_experts`；若手动设置，必须与 R3 开关一致。

固定运行环境为 Transformers 5.5.4、vLLM 0.23.0 / vLLM-Ascend 0.23.0.post1。
适配器区分 Transformers 5.6 前的概率返回值与 5.6 起的 logits 返回值，保留原生 gate 输出契约。
更换 Transformers 版本仍须验证适配器及完整训练流程，不能仅依赖同名类；固定真机验证环境不变。
当前要求：

- `model_implementation: native`、`deployment: colocated`、`enforce_eager: true`、`logprobs_mode: raw_logprobs`。
- 关闭 EPLB 和 consistency，训练 CP=PP=1。
- rollout `data_parallel_size: 1`，TP 覆盖共用设备；此限制来自该 Ascend capture 的 DP padding 布局，
  不限制训练 DP。若从单轮 MoE 配方启用 R3，须同时将其 rollout DP2/TP2 改为 DP1/TP4。

初始化线程与睡眠级别可独立选择：`--train.model_init_cpu_threads=8` 仅临时调整模型构建的
intra-op 线程；`--rollout.vllm.sleep_level=2` 丢弃旧推理权重以降低 CPU 内存占用，
恢复采样前必须完整发布权重。默认仍是 `null` 和 `1`，不会因 R3 自动改变。
完整条件见 [初始化线程](../README.md#模型初始化-cpu-线程)和
[推理生命周期](vllm_rollout.md#policy-生命周期)。`full_gather` 的设备侧聚合内置生效，没有新增开关。

### 数据与执行流程

```text
vLLM capture → GenerationResult → Trajectory → ExperienceBatch
             → Actor 微批 RouterReplay.use → gate hook → expert / EP
             → backward 完成后清理 replay 状态
```

`RouterTrace` 保存 CPU 整数 `expert_ids[S,L,K]`、布尔 `valid_mask[S]` 和稳定 `layer_ids`。
路由按模型输入 token 位置对齐，不使用 next-token logprob 的 shift。
例如 `[p0,p1,a0,a1]` 通常只有前三个输入位置执行过 forward，最后采样的 `a1` 没有路由。
最后有效 action 位于 `m` 时，`0..m-1` 的因果前缀均需有效，不能只采集 action mask 对应位置。

最后未执行的 token、终态 observation 和 padding 保持无效；无效位置使用原生路由，不能伪造 expert 0。
trace 随训练行保存在 CPU，微批执行时才补齐并传到设备。传输使用紧凑二进制，NPY 解码禁用 pickle。
shape、层映射、专家范围、重复 expert 或因果覆盖不合法时明确失败，不回退到普通训练。

`router_adapter.py` 仅在已构建 Actor 的 gate 实例安装可移除 hook，
以 rollout ID gather 当前概率，并按模型原有规则归一化；不改变参数名称、state_dict 或优化器引用。
微批作用域覆盖 forward 和 backward，包括 checkpoint 重算，正常或异常退出均清理状态。

### 多轮交互

| 轨迹方式 | trace 处理 | 原因 |
| --- | --- | --- |
| internal 连续上下文 | 新一轮完整 trace 替换旧快照；重叠有效前缀的有序 expert ID 必须一致 | 新 prompt 已包含历史 action 和 observation，直接拼接 trace 会重复前缀 |
| Codex 逐调用训练 | 每次真实 prompt/action 保留独立 trace，不删除早期调用 | harness 可以改写或裁剪上下文，最后一次调用不能替代历史调用 |

Codex Gateway 从模型响应提取路由，Agent 只接收正常响应。路由不重复写入审计 metadata，
调用身份、策略版本、episode 奖励及 DP 零损失补齐仍沿用 [Agent 合同](agentic_rl.md)。
Search-R1 引用修正等追加调用同样经过该路径，无需按数据集增加重放逻辑。

接入新任务时，重点检查是否改变原始 token、调用记录或轨迹拼接。仅改变题目和奖励无需复制 R3 实现；
引入新的上下文截断、调用合并或传输边界时，应补对应验证。

### 实现位置与参考

下列路径相对 `hyper_parallel/rl/`：

| 职责 | 文件 |
| --- | --- |
| 数据合同与训练行绑定 | `rl/dataset/contracts.py`、`rl/dataset/batch_builder.py` |
| capture 解码与 TP 传输 | `rl/roles/rollout/router_capture.py`、`rl/roles/rollout/vllm.py` |
| 模型无关 replay 作用域 | `rl/router_replay.py` |
| Qwen3-MoE gate 适配与 Actor 接线 | `rl/roles/policy/router_adapter.py`、`rl/roles/policy/actor.py` |
| 多轮调用记录与轨迹 | `rl/agentic/codex/gateway.py`、`harness.py`、`rl/agentic/core/` |
| 配置与装配 | `rl/config.py`、`rl/trainer.py` |

实现参考 [R3 论文](https://arxiv.org/html/2510.11370v1)及以下固定版本：

- [verl](https://github.com/verl-project/verl/tree/f95be0226a65d3f335901ee6703f974fe82f840d)：
  replay indices、当前概率计算权重及 forward/backward 作用域；router 适配留在 RL 层。
- [Agent Lightning](https://github.com/microsoft/agent-lightning/tree/69f5a40b36d9bff646b8e7a5532e8b78bd41eaf9)：
  Proxy 提取路由与逐调用训练记录；连续轨迹聚合时才能删除被覆盖的旧 trace。
- [Uni-Agent](https://github.com/verl-project/uni-agent/tree/13f14f361912bf0c4cd53701bfe1e9d09510a3df)：
  连续 token chain 使用最新完整路由。该优化不能直接用于本仓 Codex 的独立调用行。

## 单元测试

UT 位于 `tests/ut/rl/`，以下为核心验证入口，覆盖 CPU 逻辑和必要的 mock：

| 模块 | 代表性测试（相对 UT 根目录） |
| --- | --- |
| MoE 配置与同步 | `trainer/test_moe_config.py`、`weight_sync/test_moe_weight_sync.py` |
| R3 位置、梯度、重算和 capture | `policy/test_router_replay.py`、`rollout/test_router_capture.py` |
| 设备聚合、桶释放和 sleep 2 | `weight_sync/test_packed_weight.py`、`weight_sync/test_weight_sync_transaction.py` |
| 初始化 CPU 线程恢复 | `policy/test_actor_roles.py`、`trainer/test_config_runtime.py` |
| code 数据、判题和轨迹 | `data/test_code_data.py`、`agentic/test_code_judge.py`、`agentic/test_code_runner.py` |
| Agent 调用、预算和补齐 | `data/test_episodes.py`、`agentic/test_agent_program.py`、`agentic/codex/test_gateway_transport.py` |
| 仓库任务与启动流程 | `agentic/swe_smith/` |

```bash
python -m pytest -q tests/ut/rl
```

## 系统测试

ST 位于 `hyper_parallel/rl/tests/st/`，由 `test_feature_st.py` 显式启动。
`_moe_train.py`、`_code_train.py`、`_agent_train.py` 和 `_agent_dp.py` 为子进程 worker；
原 `test_rl_st.py` 的 dense、一致性和 PPO 用例继续独立使用。

两进程 CPU/Gloo 验证比较不等长调用补齐前后的梯度与更新参数，不需要 NPU：

```bash
python -m pytest -q hyper_parallel/rl/tests/st/test_feature_st.py::test_agent_dp_padding
```

真实 MoE/R3 训练使用已有 MoE 配方，准备模型、数据、设备并按需开启 R3：

```bash
export RL_ST_MOE_CONFIG=/path/to/resolved-moe.yaml
export RL_ST_WORLD_SIZE=4
export RL_ST_RESULT_DIR=/path/to/new-results
export RL_ST_REQUIRED=1
python -m pytest -q 'hyper_parallel/rl/tests/st/test_feature_st.py::test_feature_training[moe]'
```

配置必须与设备数一致，结果目录须为空。code 和 agent 分别使用 `RL_ST_CODE_CONFIG`、
`RL_ST_AGENT_CONFIG` 与参数 `[code]`、`[agent]`。测试保留两步参数更新与策略版本检查；
R3 observer 额外验证真实 gate 的 expert ID、当前权重、checkpoint 重算和 router 梯度。
Agent 用例要求真实多调用及不等行数补齐，不能将未触发的场景记为通过。

SandboxFusion 的独立真实服务测试不需要模型或 NPU，入口与资源配置见 [code 示例](../examples/code/README.md)。
缺少资源时 skip 不算通过；设置 `RL_ST_REQUIRED=1` 后，缺失配置会失败。
所有运行结果写入独立输出目录，不纳入版本管理。

## 验证范围与限制

| 路径 | 已有功能覆盖 | 结论限制 |
| --- | --- | --- |
| 单轮 MoE R3 | 路由、gate 权重、反向重算、两步更新和版本发布 | 限已适配模型与固定运行时 |
| GSM8K Codex R3 | 工具上下文、逐调用 trace、预算终止及两步闭环 | 不代表训练收敛 |
| Search-R1 Codex R3 | 搜索上下文、不等调用 DP 补齐、首步重放与更新发布；引用修正调用的采集 | 未完成完整两步闭环，不能把后续采集记为训练重放通过 |
| internal 多轮 / SWE-Smith R3 | 连续前缀检查 / 逐调用路由交接的 CPU 合同 | 未单独完成对应 R3 多轮真机闭环 |
| 独立代码沙箱 | 隔离合同及 dense code 两步训练、评测和保存 | 不代表全部数据标签正确或 checkpoint 恢复通过 |

精确重放 expert 选择不等于训练与推理 logprob 一致；当前 MoE 同权重更新前仍存在概率差异，
未声明 bit-exact 或学习收益。通用训练与 Agent 功能通过，也不自动证明其与 R3 的所有组合已验证。
更换模型、capture 后端、并行布局或轨迹结构后，应重新执行受影响的合同和真实运行验证。
