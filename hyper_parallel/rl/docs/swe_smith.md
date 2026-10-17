# SWE-Smith Code Agent：实现与限制

使用步骤见[示例 README](../examples/swe_smith/README.md)。本文维护实现边界、评分合同和验收要求。

## 代码组织

任务通过 `program_callable` 接入公共 `SyncTrainer`；任务数据、仓库编排和业务评分留在示例。

| 位置 | 职责 |
| --- | --- |
| `examples/swe_smith/agent.py` | 仓库准备、Codex 生命周期、调用预算、冻结提交、评分和逐调用轨迹 |
| `workspace.py`、`container_execution.py` | 固定问题分支、可信测试恢复和容器内会话中继 |
| `artifacts.py`、`reward.py` | 源码补丁投影、保护测试/配置、独立 pytest 评分与失败归因 |
| `prepare_data.py`、`prepare_images.py` | 子集/全量转换、任务镜像映射、架构和身份核验 |
| `infer.py`、`configs/` | 无训练的推理入口、模型配方及固定九例的镜像/任务身份 |
| `launch.py`、`run_config.py` | 本地或 SSH 沙盒选择、路径和数据/镜像身份核验、可信 controller 生命周期 |
| `rl/agentic/codex/`、`rl/agentic/envs/docker_workspace.py` | 通用 Codex 协议、原始采样证据、本地/远端工作区 |
| `models/qwen3_5/adapter/selected_log_probs.py` | 模型侧动作位置概率、分块词表投影与重计算 |
| `rl/roles/`、`rl/weight_sync/`、`rl/checkpoint.py` | 训练构造、流式发布、图刷新及完整恢复 |

## 任务与评分合同

候选检出实例的问题分支，核验“Bug Patch → Remove F2P Tests”的结构并解析固定 commit。
随后移除 Git 历史，保留 controller 捕获的可信问题基线。
容器无网络、Docker socket 和 NPU；管理凭据不进入候选，relay 仅绑定其会话。
候选、relay 和 grader 使用同一显式 Docker endpoint，远端连接失败不会切换本地执行。

Responses 中同一回复的说明文字和函数调用合并为一个 assistant Chat 消息。
来源校验允许解码串与 engine text 仅相差一个精确的末尾 EOS；原始采样 token 保留不变。
未闭合 XML 属于不可执行的模型格式错误，其他来源不一致继续拒绝会话。
预算终止响应确认交付后 relay 停止接收新请求，随后由控制器关闭；CLI 重试不能重新打开会话。

Codex 正常结束或经网关证实调用预算耗尽后，停止候选并导出真实代码。
`artifacts.py` 在 controller 上生成补丁，允许普通 Python / Python stub 源码修改，
保护已有测试、项目配置、特殊文件和执行权限；不信任候选 Git 元数据。
新增非 Python 的复现输入、备份等中间文件不进入提交补丁；新增 Python 测试/hook 仍检查受保护路径。
评分版本为 `swe-smith-pytest-v3`，冻结工件可以独立重评，无需重新生成模型轨迹。
工件 byte/file 上限由任务配置控制。

grader 从新容器启动，校验相同问题/测试 commit，应用冻结补丁并恢复可信测试。
默认 `f2p_only=true`：全部 F2P 与相同测试文件内的 P2P；设为 false 可检查完整 P2P。
所有选中测试均有可信状态且全部为 PASSED/XFAIL 才得到奖励 1；完整判题失败为 0。
已独立确认的提交或语法错误可形成零奖励。候选源码导致测试无法收集时，
须通过 pytest 钩子确认异常来自修改源码，且原问题基线能收集全部选中测试，才记为零奖励。
保留真实异常和已报告状态，不为未运行节点伪造 FAILED。
采样、容器、传输、无法归因的收集失败及其他不完整判题错误均拒绝更新。

## 训练配置与资源

模型范围为 Qwen3.5-9B / Qwen3.8-27B 的文本 GRPO、训练 TP=CP=1、
colocated native vLLM、流式 `full_gather`、consistency off。不声明 BF16 bit-exact。
训练和新进程恢复命令见[示例 README](../examples/swe_smith/README.md#训练一步与新进程恢复)。
恢复须保持模型、配方及数据/镜像清单身份一致；恢复后先发布恢复版本，再继续采样和更新。
支持完整训练 checkpoint 恢复，不支持恢复进行中的 Codex 会话或未完成的 optimizer step。

短预算配方每 rank 一个提示、每提示两个会话，每会话最多两次模型调用、单次最多 512 输出 token。
`train.response_mini_batch_size` 的单位是逐调用训练行。增加调用预算会增加重计算上下文，
须同时检查训练批大小、显存和耗时。真实奖励相同时可能零梯度；学习验收需真实混合奖励，
可开启 `train.learning_gate` 检查非零更新，默认评分不注入人工奖励。

统一入口的训练端默认使用 `HCCL_IF_BASE_PORT=63200`、`HCCL_NPU_SOCKET_PORT_RANGE=63200-63299`；
rollout 使用 `63400-63499`。覆盖训练端口须成对设置两个环境变量，范围在 `1024..65520` 内，
包含基础端口且不与 rollout 重叠。9B 短配方显式配置 8 GiB KV cache，为重复权重发布保留空间。

### 采样组与训练 DP

公共 Trainer 按训练 DP rank 分配提示，`rollout.num_return_sequences` 是每个提示的会话数；
组内 GRPO 优势在该 rank 内计算。目前不支持将同一提示组的会话分发到多个训练 DP rank。
推理 TP 与训练采样组独立：vLLM DP1 / TP2 是一个两卡推理副本，不会自动重分配采样组。

双卡 FSDP2 需要两个可分配提示。使用 `prepare_data --repeats 2` 创建同题副本后，
每 rank 四会话得到两个独立四样本组、共八会话；每提示改为八会话则全局为十六会话。
跨训练 DP 的采样组重分配应由公共 RL 层实现，保留唯一 episode 身份、真实奖励、
policy version 和零损失 padding。

## 参考预算推理

[qwen3_5_9b_infer.yaml](../examples/swe_smith/configs/qwen3_5_9b_infer.yaml)
提供较长推理预算：100 次调用、单次最多 12,288 输出 token、上下文 81,920、
5,400 秒、温度 1.0、thinking 关闭。工具协议为 Codex/Qwen3Coder。
`max_observation_chars=6000` 对模型收到的观察保留首尾并按字符截断，原始工具证据不变。
自动压缩阈值为 65,536，压缩请求也计入调用预算；CLI 自身的输出限制仍然有效。
该配方不声明与其他 Agent 的提示、观察包装、压缩策略或随机轨迹完全一致。
独立 grader 资源和超时由配方控制，外部终止不计作求解失败。

## 图执行配置

固定 vLLM 0.23.0 / vLLM-Ascend 0.23.0.post1、rollout DP1 支持显式 FULL_DECODE_ONLY：
设置 `enforce_eager=false`，在 `rollout.vllm` 下加入以下配置，并保持 `max_num_seqs=4`。

```yaml
compilation_config:
  mode: 3
  cudagraph_mode: FULL_DECODE_ONLY
  cudagraph_capture_sizes: [1, 2, 4]
  max_cudagraph_capture_size: 4
```

该模式包含 `torch.compile` 和图重放。流式 full-gather 完整发布并唤醒 KV 后，
在开放采样前重新捕获 decode 图，刷新失败不得提交版本。

## 验证范围与回归

9B 短预算 eager 覆盖了 ARM64 Ascend controller 搭配本地 ARM64 或远端 amd64 沙盒，
包括推理、训练、完整保存和新进程恢复。任务评分覆盖 schema、boltons、oauthlib 九例，
通过问题基线失败/可信修复成功对照检查独立 grader。零奖励、零梯度的闭环只证明流程，
不能作为有效学习的证据。

9B 图推理已运行；图模式重复发布刷新、长预算真实混合奖励更新/恢复、27B 完整回归和
全数据集验收仍需单独执行。更多仓库须检查架构、分支布局、测试依赖和资源需求。
当前源码补丁范围只含普通 Python / Python stub；其他语言及需要修改构建配置的任务不在验收范围内。

从仓库根目录运行 CPU 合同检查：

```bash
python -m pytest -q tests/ut/rl/agentic/swe_smith \
  tests/ut/rl/agentic/codex tests/ut/rl/agentic/test_docker_workspace.py
```

模型数学合同在 `tests/ut/auto_models/models/qwen3_5/`，训练、发布和恢复在 `tests/ut/rl/` 对应模块。
共享代码变更还须覆盖 Search-R1、文本网关、Hermes 证据和完整 checkpoint 回归。
CPU 通过不等同于新硬件组合、长预算学习或全数据集通过；真实运行前须检查设备、镜像空间、
独立的输出/checkpoint 目录和预算。
