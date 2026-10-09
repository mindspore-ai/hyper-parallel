# HotpotQA 文件搜索

See [prepare_data.py](../examples/search_r1/prepare_data.py), [training config](../examples/search_r1/configs/qwen3_4b_search_r1.yaml), and [run_search_r1.sh](../examples/search_r1/scripts/run_search_r1.sh).

在宿主机运行 `launcher.py`，由宿主机启动可信训练容器及同级、无特权 Codex 容器。
不使用 Docker-in-Docker。Codex 只能读取该题候选文章，通过 shell 搜索；
不挂载数据集标签、其他会话、NPU 或 Docker socket。每个会话启动前实际探测隔离边界。

## 数据与训练

训练题从官方 train 的 90,447 题抽取，留出题来自独立 validation 源。
准备程序检查跨源 ID/规范化问题重叠，记录源文件和产物 SHA256，禁止覆盖已有数据。
启动前重新核验数据哈希；不将旧 validation 训练实验的 checkpoint 当作独立评测基线。

在 `rl/` 下，确认设备空闲并获准使用后执行：

```bash
python3 examples/search_r1/launcher.py --devices 2,3 --steps 5 \
  --data /absolute/path/to/prepared-hotpotqa --backend-privileged \
  --output output/search_r1/search_r1_expanded_check
```

`--backend-privileged` 仅用于可信训练后端解决本机 Ascend 驱动权限问题，绝不用于 Codex。
`jobs/` 内的运行目录使用内部会话 UUID；题目 ID、训练/评测阶段保留在轨迹元数据中，
不直接用于 Docker 名称或挂载路径（评测 ID 可能包含冒号）。
本项目仍是每题候选文章搜索，不是全维基检索。

Trainer 使用 HCCL `62800-62900`，vLLM 使用 `62600-62700`，避免同卡多进程争用默认端口。
这不替代设备预留；启动前仍须确认两张卡没有其他用户任务。
固定评测 8 道留出题，每 50 步及最终保存 checkpoint 并重载模型验证。
`--resume /原输出目录/checkpoints/step_N` 要求完整 checkpoint 标记及相同数据哈希。

## 正确性与故障边界

- 每轮分别训练真实 `P_i + A_i`；`training-audit.json` 核对 token、mask、采样 logprob。
  不要求相邻调用前缀连续，不用重新编码后的历史替换训练输入。
- 启用 Qwen3/Ascend 严格数值一致性 profile，更新前动作 logprob 不一致则中止。
  `model_implementation: hyper` 用于匹配推理/训练内核，不改变 shell 搜索方式。
- 训练 reward 和评测按完整 episode 统计，不按模型调用次数重复加权。
- 最后一个常规模型调用禁用工具，另预留一次真实引用修正调用；全部动作保留。
- 已确认的模型预算用尽、重复未命中/命令错误、无效引用是任务失败，保留真实动作并记零奖励。
  没有任何采样动作则报错，不能制造空训练样本。
- `failure_origin`、`failure_reason`、`trainable` 分别记录责任、原因和训练资格。
  环境故障、缺少执行工具、HTTP/网络错误、未经归因的上下文超限及未知失败中止整次更新；
  不补零奖励、不丢弃组成员、不只屏蔽 loss，避免污染 GRPO 优势。
- `rg` 的退出码 1 表示未命中；同一工作目录、同一命令重复失败会收到反馈，三次后终止。
  无法安全判断的复杂 shell 非零退出不猜测归因，按不可训练处理。
- Hermes 证据由固定版本 vLLM 插件记录：真实生成 token、解码文本、引擎文本、parser 输入和输出。
  证据一致的非法工具 JSON 会收到明确格式反馈并重新采样，计入总模型调用预算；不执行、不修复原命令。
  恢复成功仍按最终任务结果评分。合法原文解析失败或证据不一致则中止更新。
- 格式恢复调用也占用原有预算，因此可能用掉原计划预留的引用修正调用；不会额外扩充预算。
- `training-audit.json` 保留原始前缀不匹配数量与最多三个 token 差异样本；训练仍使用真实逐调用 token。
- `training-acceptance.json` 只有在指定优化步数、权重同步及全部会话审计通过后才产生。
  引用存在/可见不等于语义正确；短程通过也不保证任意长程训练无故障。

## 文件职责

- `workspace.py`：只导出当前题目的候选文章。
- `agent.py`：任务结果、引用检查与修正、逐调用训练轨迹。
- `container_execution.py`：容器启动/清理、隔离探针、会话中继与搜索预算反馈；仅依赖标准库。
- `launcher.py`：宿主机训练入口、数据校验与训练验收。
- 通用 token/mask/logprob 审计位于 `rl/agentic/core/program_runner.py`。

隔离检查仍在每个正式会话启动前执行，独立冒烟入口放在测试目录。

在宿主机的 `rl/` 目录执行无模型隔离检查（不使用 NPU）：

```bash
python3 tests/st/search_r1_isolation.py \
  --output output/search_r1/search_r1_isolation_check
```

输出目录必须是新目录。该检查验证容器边界与 Codex 启动，不代替真实模型训练测试。
历史实验详情以各输出目录的日志为准。
