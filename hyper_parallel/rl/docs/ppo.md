# Qwen3-4B PPO

PPO 复用现有 SyncTrainer、vLLM 和权重发布流程，使用独立 Actor、Reference 和 Critic。
Actor/Critic 各自更新，Reference 冻结，仅 Actor 发布到 vLLM。
模型与并行能力由现有 HyperParallel 承担，代码改动限定在 RL 与 models/qwen3。

## 模型边界

- `models/qwen3` 负责标准 Qwen3 加载、模块替换、tied 参数和并行适配。
  `build_causal_lm` 接受可选 `model_transform`，在标准 HF 模型创建后、TP/FSDP 规划前调用。
- `rl/roles/policy/critic.py` 定义 ValueHead、价值 forward 和 `build_value_model`。
  它向标准构建接口传入价值头转换及复制参数声明，不实现 checkpoint 读取或参数切分算法。
- Critic 移除词表头，使用独立标量头输出 `values[B,L]`，按 next-token 对齐到 `[B,L-1]`。
  初始标量头为零基线，确保不同 rank 一致；head 在构建阶段进入 FSDP、optimizer 和 checkpoint。
- Actor/Reference 的普通融合与一致性分支保持各自数值路径。
  Qwen3 专属构建实现统一位于 `models/qwen3/runtime.py`。

## 算法与轨迹

PPO 使用任务终点奖励、GAE、clipped policy loss 和 clipped value loss。
Reference KL 保持在 Actor loss 中，不重复加入 token reward。
采样 old logprobs、old values、advantages 和 returns 在每轮优化前固定并 detach。

GAE 沿有效 action 递推；工具观察和 padding 不成为动作，也不截断相邻 action 的回报链。
returns 使用未归一化的优势构造，只有 Actor 使用的 advantages 在 DP 全局有效 token 上归一化；
TP 副本不重复计入统计。

首个 recipe 使用已终止的单轮 GSM8K 轨迹。真正终止（`done=true`）的 bootstrap 为零。
非终止截断必须显式携带 `metadata.bootstrap_context_complete=true`，表示 token 序列包含完整的
后续状态上下文，才使用最后一个有效状态的 value；没有此证据时所有训练 rank 同步报错。
标记必须由任务/轨迹生产者依据真实上下文提供，不能仅为通过检查而设置。
现有多轮 GRPO 逻辑不变，PPO 多轮与外部 Agent 组合须单独验收。

## 配置

示例：[qwen3_4b_gsm8k_ppo.yaml](../examples/configs/qwen3_4b_gsm8k_ppo.yaml)。
默认两卡 FSDP2 / rollout DP2、两步训练、每个 rank 两个 prompt、每个 prompt 四个 response。
PPO 允许每个 prompt 只有一个 response，仍须满足 mini-batch 不超过本 rank 样本数。

`algorithm.name=ppo` 启用 Critic；`train.critic` 支持：

| 配置 | 含义 / 默认 |
| --- | --- |
| `weights_path` | 初始标准 Qwen3 checkpoint；默认 Actor 的初始权重路径 |
| `optimizer` | 覆盖 Actor optimizer 配置；生成独立 optimizer/scheduler |
| `micro_batch_size` | 默认继承 train.micro_batch_size |
| `response_mini_batch_size` | 默认继承 train.response_mini_batch_size |
| `update_epochs` | 默认继承 critic_update_epochs 或 policy_update_epochs |

两种角色共用训练 mesh；Critic 不创建独立进程组。
两卡示例可在已安装当前 RL 的对应 NPU 镜像内运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=2,3 HYPER_PARALLEL_PLATFORM=torch \
PYTHONPATH=/repo/hyper_parallel/rl:/repo \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  /repo/hyper_parallel/rl/examples/train_rl.py \
  /repo/hyper_parallel/rl/examples/configs/qwen3_4b_gsm8k_ppo.yaml
```

其中 /repo 为当前 checkout 的挂载路径，模型和数据挂载按示例 YAML 配置。
现有 TP shell 脚本会覆盖自己的若干参数；PPO 验证使用下面的正式 ST 入口。

## 保存与恢复

保持旧 Actor checkpoint 的 `model`、`optimizer`、`scheduler` 键。
PPO 增加 collective `critic` 模型状态、rank-local `critic_optimizer` /
`critic_scheduler`，并在完成标记中声明 Critic 所有权。
缺失角色状态或角色类型不匹配时拒绝续训。
两套模型/optimizer/scheduler 完成恢复后最后恢复 RNG 和数据位置，再发布恢复的 Actor 版本。

HF 导出仍只导出 Actor；Critic 是 DCP 训练状态，不导出为可采样策略。
optimizer 恢复通过 RL 局部 SkipDTensorDispatch 保护惰性 fused AdamW 初始化，
共享 optimizer 源码不修改。

## 验收

CPU 覆盖：真实小模型的 backbone 保真、价值头梯度、TP 复制布局，GAE 手算、观察间隔、
终止/截断 bootstrap、DP 归一化、双角色独立状态恢复，以及既有 RL 回归。

正式真机用例位于 `tests/torch/rl/test_rl_st.py`：

- `ppo-tp1-full`：两步 Actor/Critic 非零更新、full-gather 发布至 V2。
- `ppo-checkpoint-resume`：第一容器保存 V1，新容器恢复后继续 step2/step3，并再次保存。

沿用 ST 文档的镜像、模型、数据和设备配置；默认统一写入 `hyper_parallel/rl/output`，
也可设置 `RL_ST_RESULT_ROOT` 覆盖根目录。每个用例保留 resolved YAML、rank 日志及 result.json。
完成状态以实际验收为准；两步运行不代表收敛或长期稳定性。

在仓库根目录使用本地已有镜像执行两卡验证：

```bash
export RL_ST_MODEL=/home/mwl/ckpt/qwen3-4b
export RL_ST_DATA=/home/mwl/dataset/gsm8k/main
export RL_ST_DEVICES=2,3
export RL_ST_RESULT_ROOT=/home/mwl/project/hyper/hyper-parallel-master/hyper_parallel/rl/output
export RL_ST_REQUIRED=1
python -m pytest -v 'tests/torch/rl/test_rl_st.py::test_rl_system[ppo-tp1-full]'
```

恢复验证将参数 ID 改为 `ppo-checkpoint-resume`。测试派生自同一个 PPO 示例 YAML，
只调整测试规模、设备和输出位置等运行参数。

### 当前验证记录（2026-09-12）

固定镜像：`swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-arm64`。
模型为本地 Qwen3-4B，数据为真实 GSM8K；所有 NPU 用例串行运行。

| 用例 | 结果 |
| --- | --- |
| 两卡 PPO full-gather | 两步通过；Actor 梯度 1.67188 / 0.761719，Critic 梯度 1.03888 / 0.729663，V1/V2 发布完成 |
| 两卡 PPO 保存/恢复 | 通过；第一容器保存 step1，新容器恢复后继续 step2/step3，完成再次保存 |
| 四卡 PPO FSDP2×TP2/direct-reshard | 两步 Actor/Critic 非零更新与 V1/V2 发布通过 |
| GRPO 四卡一致性回归 | 两步通过；7296 / 7674 个有效 token 的 mismatch/max abs/mean abs 均为 0 |
| CPU 回归 | RL、ST 自测与原版共享 apply/planner 合计 295 passed |
| 静态检查 | 修改代码完整 pylint 通过 |

本轮验证时产物根目录为 `/tmp/hyper-ppo-st`，对应目录为
`ppo-tp1-full-bc74c8223c`、`ppo-checkpoint-resume-9c157631d9` 和 `ppo-tp2-direct-2b0af7eee3`。
GRPO 回归目录为 `grpo-tp2-consistency-regression-6de951f437`。
恢复用例的两个 checkpoint 都包含 Critic 所有权标记和 Actor HF 导出；
这些历史测试 checkpoint 已按要求删除，验证日志和 `result.json` 保留。
后续 ST 和 Qwen3 Docker 脚本默认写入 `hyper_parallel/rl/output`。
本轮所有测试容器均已清理，NPU 已释放。共享 HyperParallel 已跟踪代码保持 Git HEAD 原样。
目前不声明 PPO 多轮 Agent、训推分离或 PPO bit-exact 已完成真机验收。
