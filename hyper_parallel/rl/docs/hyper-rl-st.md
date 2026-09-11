# hyperparallel-RL PR1354 ST 设计与运行

## 目标与边界

通过正式 `examples/train_rl.py` 启动真实 Transformer 模型、HyperParallel 分布式训练、
shared vLLM 和权重发布，验收模块组合后的系统行为。ST 没有 line/branch coverage 门槛。

本目录是根据 PR1354 生产接口新建的实现，不依赖旧 ST 或 `rl_tests` 中已删除的测试。
UT、CPU 自验证、真实 NPU ST 的结果分别记录，不能相互替代。

## 场景

所有场景默认完成两步训练，使用 BF16、真实 checkpoint 和真实 GSM8K reward。
固定 seed、8 条 prompt、每次每个 Trainer DP rank 2 条 prompt、每条 4 个 response，最多生成 256 token。

| pytest 参数 ID | NPU 数 | 场景 | 专项验收 |
| --- | ---: | --- | --- |
| dense-tp1-full | 2 | Qwen3 FSDP2 → rollout DP2/TP1 | streaming full-gather、128 MiB 桶、训练前 bit-exact、evaluation |
| dense-tp2-direct | 4 | Qwen3 FSDP2/TP2 → DP2/TP2 | direct reshard、训练前 bit-exact |
| dense-disjoint-direct | 8 | Qwen3 Trainer 4 卡 + Rollout 4 卡 | 独立设备集合与 direct HCCL 发布 |
| checkpoint-resume | 2 | Qwen3 两个独立进程/容器 | 第一阶段保存 step1；第二阶段恢复、用 V1 生成并完成 step2 |
| qwen3-moe-ep4 | 4 | Qwen3-30B-A3B FSDP2/TP2/EP4 | Hyper-vLLM expert 权重、EP4 worker、两步 direct 发布和后续生成 |
| moonlight-ep4 | 4 | Moonlight FSDP2/TP2/EP4 | Hyper-vLLM、expert/absorbed MLA、两步 direct 发布和后续生成 |
| codex-agent | 2 | Qwen3 + Native-vLLM + Codex | 真 Agent 工具调用、gateway、token evidence、session 释放 |
| deepseek-agent | 2 | Qwen3 + Native-vLLM + DeepSeek Harness | 真 SDK 工具调用、gateway、token evidence、session 释放 |

MoE EP 与 DP/TP 共用设备，不把 EP4 再乘到设备数。MoE 场景不声明 bit-exact。
DeepSeek Agent 场景使用 Qwen3 模型；它与 Moonlight 模型场景是不同功能。

## 通用通过条件

- 正式训练进程正常退出；期望训练 step 和 policy/version 一致。
- 每步有效 action token、optimizer step、生成 token 均大于零；指标为有限值。
- 运行中至少一步 gradient norm > 0 且 policy fingerprint 改变；V1/V2 worker 权重摘要发生变化。
  Checkpoint 场景每个阶段均需非零更新证据。
- 每个 Trainer rank 保存 V0/V1 的 rollout；token、mask、logprob 长度和 SHA256 可核对，
  padding 不进入 action mask，生成使用正确策略版本，各 rank fingerprint 一致。
- 每个 rollout DP/TP worker 保存 V1/V2 完整参数 manifest；检查 rank、模型 family、版本、
  参数数量、字节数和摘要；MoE 必须带 EP4 与专家参数，Moonlight 必须带 absorbed MLA 证据。
- 完成策略必须等于配置策略，不允许成功场景静默 fallback。
- full-gather 桶全部 ACK/release，最大 in-flight 为 1，gather/pack 单缓冲 ≤128 MiB。
- bit-exact 场景有效比较 token >0，mismatch/max-abs/mean-abs 均为零。
- 外部 Agent 每个 session 至少两次 completion、模型发出 tool call、后续请求含 tool feedback，
  携带原始 token/logprob 和 policy identity，最后记录 session.released。
- 退出后本用例容器不存在，vLLM/gateway 端口可重新绑定。

真实 reward 全相同可能导致零优势，届时非零学习验收失败，不修改标签制造通过。
当前是两步系统验收，不推断收敛、长期稳定性或吞吐提升。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `test_rl_st.py` | 轻量 pytest 入口，参数化 8 个真实 NPU 场景，带仓库 ST marker |
| `st_runtime.py` | 场景定义、生产 YAML 派生、资源检查、容器启动、超时清理、结果记录 |
| `_launch.py` | 容器内调用仓库公共 `torchrun_case`，转发各 rank 日志 |
| `_worker.py` | torchrun worker，通过 `runpy` 调用正式 `train_rl.py` |
| `st_evidence.py` | 解析正式日志/rollout/manifest/checkpoint/Agent trace 并判定 |
| `test_st_support.py` | CPU 自验证：配置与命令、证据校验器、缺失/损坏证据拒绝、入口轻量导入 |
| `pytest.ini` | 独立 ST 收集和 marker |
| `.gitignore` | 忽略运行产物和缓存 |

生成的 YAML 直接传给正式入口；不修改 Trainer、算法、reward、模型或 vLLM 生产实现，
不注入替代 forward/optimizer/通信的 mock。CPU 自验证使用合成证据，只证明测试框架能够正确判定。

遵循仓库 [testing 规则](../../../.agent/rules/testing.md) 和
[代码风格](../../../.agent/rules/code-style.md)：pytest + `arg_mark`、
公共分布式启动器、非 `test_*.py` worker、父进程轻量导入、独立结果目录、明确 skip/fail。
本轮目录遵循用户指定的 `rl_tests/st`，不会被 `pytest tests/torch` 自动收集，CI 需显式加入本入口。

## 环境

宿主需要 Python、pytest、PyYAML、Docker、Ascend 驱动，以及可访问的模型和 parquet 数据。
镜像必须预先存在，不自动 pull/build/install。外部 Agent 镜像还需包含指定版本 CLI/SDK：
Codex 0.152.1、deepseek-harness-sdk 0.1.1rc1。

显式选择可用设备，串行运行；调用方负责避免占用其他任务的卡和通信端口。
按已确认偏好，ST 不设置 `npu-smi Health=OK` 门禁。

```bash
cd /home/mwl/project/hyper/hyper-parallel-pr1354

export RL_ST_MODEL=/home/mwl/ckpt/qwen3-4b
export RL_ST_DATA=/home/mwl/dataset/gsm8k/main
export RL_ST_DEVICES=4,5,6,7
export RL_ST_REQUIRED=1
export RL_ST_TIMEOUT=1800
export RL_ST_HCCL_BASE_PORT=62000

# 可选：覆盖默认的固定基础镜像
export RL_ST_INTERNAL_IMAGE=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.22.1rc1-arm64

# MoE 场景分别要求其对应完整 checkpoint
export RL_ST_MOE_MODEL=/path/to/Qwen3-30B-A3B
export RL_ST_MOONLIGHT_MODEL=/path/to/Moonlight-16B-A3B-Instruct

# 外部 Agent 场景单独指定镜像
export RL_ST_CODEX_IMAGE=hyper-parallel/hyper-rl-codex:v0.22.1rc1
export RL_ST_DEEPSEEK_IMAGE=hyper-parallel/hyper-rl-deepseek:v0.22.1rc1
```

模型必须含 `config.json` 且 model_type 对应场景；数据目录含 `train.parquet`、`test.parquet`。
未配置时本地显式 skip；`RL_ST_REQUIRED=1` 时缺资源直接失败，CI 不得靠 skip 通过。

## 执行

```bash
# 仅测试框架 CPU 自验证
python -m pytest -q hyper_parallel/rl/rl_tests/st/test_st_support.py

# 只收集真实 ST（不加载 Torch，不启动硬件作业）
python -m pytest --collect-only -q hyper_parallel/rl/rl_tests/st/test_rl_st.py

# 一个真实场景；参数 ID 见上表
python -m pytest -vs 'hyper_parallel/rl/rl_tests/st/test_rl_st.py::test_rl_system[dense-tp1-full]'

# 配齐全部资源及八张空闲卡后，串行执行全部真实 ST
python -m pytest -vs hyper_parallel/rl/rl_tests/st/test_rl_st.py
```

每次运行保存到本目录的 `results/<case>-<唯一ID>/`：
`phase-N.yaml`、`phase-N.log`、`result.json`、`rollouts/`、`manifests/`；
Checkpoint 另有 `checkpoints/`，外部 Agent 另有 `sessions/`。
第二阶段使用新容器复用同一结果目录。失败保留日志，结果只有完成全部校验后才为 passed。
资源检查发生在创建运行目录前，因此缺资源的 skip/fail 直接见 pytest 输出。

## 尚未声称覆盖的能力

本轮未实现故障注入 fallback、多节点、PPO/Critic、长时间压力/性能验收、Native/Hyper
同源权重全量对比，也未把两步恢复声明为 optimizer/RNG 与不中断训练逐位相等。
这些应作为明确的后续 ST 场景，不由当前 CPU 自验证或 coverage 数值代替。

## 本轮验证状态（2026-09-08）

- CPU 测试框架自验证：17 passed；真实场景未配置启动参数时为 8 skipped。
- Pylint、compileall：通过；轻量入口不导入 Torch/NPU/HyperParallel。
- 6 个 Qwen3/Agent 场景的生成配置已通过 PR1354 正式配置校验。
- 真实 NPU 场景未运行：执行前复核时 0–7 卡均有其他训练进程，未占用这些设备。
  8 skipped 仅表示没有执行，不能解释为系统验收通过。
- 仓库文档目录检查仍被已删除的旧 `rl_tests/test_*.py` 引用阻塞；
  本轮只在 `rl_tests/st` 内落地，没有修改其他导航文档。
