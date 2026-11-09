# Hyper-RL 功能导航

[English](rl-navigation.md) | **简体中文**

路径相对 `hyper_parallel/rl/`；`tests/ut/rl/` 相对仓库根目录。

| 功能 | 配置/入口 | 实现 | 验证 |
| --- | --- | --- | --- |
| 构建与支持矩阵 | `train_rl.py`、本次 YAML | `rl/config/loader.py`、`rl/config/`、`rl/trainer/build.py` | `tests/ut/rl/trainer/test_config_runtime.py` |
| 显式替换规则 | `train.plan_overrides`、角色 overrides | `rl/config/resolve.py`、`rl/workers/adapters/qwen3.py` | `test_qwen3_master.py`、`test_moe_config.py` |
| 同步训练 | 默认执行模式 | `rl/trainer/sync.py` | `test_trainer_orchestration.py` |
| 分离异步 | `async_training` | `rl/trainer/fully_async.py`、`rl/async_runtime/producer.py` | `tests/ut/rl/fully_async/` |
| 共卡异步 | `colocate_async` | `rl/trainer/colocate_async.py`、`rl/async_runtime/colocated.py` | `tests/ut/rl/colocate_async/` |
| GRPO/PPO/GSPO | `algorithm.name` | `rl/algorithm/`、`rl/workers/actor.py`、`critic.py` | `tests/ut/rl/algorithm/`、`policy/` |
| 概率、目标、bootstrap | `old_log_prob_source`、算法需求 | `rl/workers/targets.py`、`rl/data/experience.py` | 目标顺序、概率身份及 PPO 测试 |
| 共享优化器与驻留 | optimizer、accelerator | `rl/workers/engine/` | optimizer precision、phase offload、checkpoint 测试 |
| 动态 batching / DP balance | 三类 token budget、`balance_dp_tokens` | `rl/workers/input/` | 分批预算、梯度、行顺序测试 |
| Agent 及真实轨迹 | `agentic`、examples | `rl/agentic/`、`rl/data/` | `tests/ut/rl/agentic/`、`data/` |
| native-vLLM | `model_implementation: native` | `rl/rollout/vllm.py`、`vllm_plugin.py` | rollout runtime、plugin、sampler 测试 |
| full_gather IPC/HCCL | weight_sync、deployment | `rl/weight_sync/` | transaction、transport、worker 和 MoE 映射测试 |
| RM | `reward_model` | `rl/reward_model/runtime.py` | `tests/ut/rl/reward_model/` |
| Teacher / OPD / MOPD | `distillation` | `rl/distillation/`、`rl/algorithm/distillation.py` | `tests/ut/rl/distillation/` |
| MoE R3 | `train.router_replay.mode` | `rl/workers/adapters/router_*`、`rl/rollout/router_capture.py` | router replay 与 MoE 测试 |
| 保存与恢复 | `train.checkpoint` | `rl/trainer/checkpoint.py`、`rl/async_runtime/protocol.py` | schema、精度、拓扑、版本、RNG 与 buffer 恢复测试 |

设备测试入口位于 `hyper_parallel/rl/tests/st/`；使用前按[系统测试说明](../hyper_parallel/rl/README.zh.md#系统测试)分配设备。
详见[运行时架构](../hyper_parallel/rl/docs/architecture.zh.md)和[native-vLLM 合同](../hyper_parallel/rl/docs/vllm_rollout.zh.md)。
