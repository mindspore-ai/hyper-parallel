# vLLM Rollout

## 适用范围

本文定义 Hyper-RL 的 shared vLLM rollout 合同，覆盖 Qwen3 Hyper/Native TP1/TP2、colocated NPU IPC、disjoint HCCL、
在线权重发布和同步失败语义。Qwen3 dense 保留原有路径；Qwen3-30B-A3B 的 MoE 接入范围见下文。
Agentic 继续通过同一生成与权重发布接口工作。

公共运行时与 family 模型适配分开维护：代码目录及插件加载路径见
[模型注册与构建](../rl/roles/model_setup.py)。目录职责不改变下述运行合同或配置。

## Qwen3-30B-A3B

MoE 使用公共 HyperAutoModel 和 Qwen3-MoE recipe 构造训练模型，rollout 使用 native vLLM。
入口示例为 [Qwen3-30B-A3B GSM8K 配置](../examples/gsm8k/configs/qwen3_30b_a3b_gsm8k_vllm.yaml)。
首批实现限于 GRPO、colocated、consistency-off、EPLB 关闭；不支持 MoE PPO 或 disjoint。
功能边界与验证入口见 [MoE 使用及验证指南](moe_code_agent.md)。

- `train.accelerator.ep` 控制训练专家并行，`edp_shard` 控制专家数据并行分片。
  EP 必须整除专家数，EP×EDP 必须整除训练 world size；dense 模型两者均为 1。
- `rollout.vllm.enable_expert_parallel` 控制推理专家并行；推理设备数仍为 DP×TP，
  EP 不额外增加设备。示例训练 TP2/EP2/EDP2、推理 DP2/TP2 使用四张卡。
- `full_gather` 将 GroupedExperts 的 gate/up/down 切片还原为逐专家 checkpoint 权重，
  通过 native layerwise reload 更新推理存储。接收方持有独立 tensor，避免跨 bucket 引用失效。
- `direct_reshard` 使用实际 expert map 和物理 worker 路由，直接复制源 EP/EDP 区域与目标专家存储的交集。
  专家 worker 的回执必须覆盖全部 DP×TP 设备；不能将一个 DP engine 的结果视为全体结果。
- 两条路径均保留版本确认和失败传播，无隐式回退。dense 的 builder、TP 传输和一致性路径保持独立。

训练前明确选择策略和设备，按当前 checkout 安装主包及 RL 包并核对导入位置。
该 MoE 路径使用固定统一 Ascend 镜像的专家布局和加载接口，其他 vLLM 版本需重新验证。

### MoE R3 路由重放

`train.router_replay.mode: R3` 同时启用 expert capture 和 Actor replay，默认 `disabled`。
当前适配 Qwen3-MoE、internal/Codex、共卡 native vLLM；固定 Ascend 运行时要求 eager、
raw logprobs 和 rollout DP1，推理 TP 覆盖共用设备。R3 关闭时保留既有推理拓扑。

完整配置、逐调用 trace、模型适配与验证范围统一见 [MoE R3 指南](moe_code_agent.md#moe-r3)。
该功能不改变 Agent 预算、初始化线程或睡眠级别，也不保证训练与推理概率一致。

## Ownership

```text
Trainer ranks
    -> one coordinator-owned endpoint
        -> vLLM DP router
            -> DP engines
                -> TP workers
```

- Trainer rank 0 启动、检查并关闭 server process group；其他 ranks 连接同一 endpoint。
- Owner 启动前拒绝已占用端口，health 成功后再次确认 owned process 存活。
- 所有 Trainer ranks 必须解析出相同 endpoint 和 physical worker mapping。
- vLLM upstream 管理 DP routing 和 frontend 数量；Hyper-RL 不设置生产用固定 DP rank header。
- `rank_local`、per-rank server、per-replica port 和 `api_server_count` 已删除。

Colocated 使用完整 Trainer 设备集，不接受 `visible_devices`。Disjoint 必须配置互不重复、与 Trainer 不重叠且数量等于
rollout DP×TP 的 `visible_devices`。

## Request 与 Admission

- Completion token IDs 是唯一权威输出。
- 训练请求保存 sampled-token FP32 raw logprobs。
- Row seed 由稳定 prompt identity 和 response index 派生，不依赖 Trainer rank、endpoint 或 DP engine。
- Completion 可以乱序返回，但必须写回稳定 row slot；choice indices 必须完整且无重复。

HTTP client 在独立 asyncio loop 中复用长期 `aiohttp.ClientSession`。Admission 按 child 数计费，使用
`asyncio.FIRST_COMPLETED` 持续补充 pending work。首个请求失败后停止 admission、取消其余 tasks，并通过 rank-synchronized
路径传播原始错误。

当前不提供透明 generation retry。未来若引入 retry，必须保持 token、seed、policy version 和 row order。

## Policy 生命周期

Generation 前后都验证所有 workers 的整数 `policy_version`。

Colocated：

```text
rollout(V)
-> sleep(level=sleep_level)
-> Trainer update
-> wake weights
-> NPU IPC transaction
-> verify workers
-> wake/reset KV cache
-> resume
-> publish V+1
```

`rollout.vllm.sleep_level` 默认 `1`，仅接受整数 `1` 或 `2`：

- `1`：将推理权重备份到 CPU，允许未更新权重时恢复推理。
- `2`：丢弃推理旧权重；唤醒 weights 仅重新分配存储，必须完整发布 Actor 权重并确认版本后
  才能恢复采样。vLLM 单独保存必要的模型 buffers，不等于备份整份参数。

CPU 内存紧张时由用户显式选择 level 2，仅用于 colocated；不自动切换睡眠级别。当前会在无发布时恢复推理的中间 checkpoint
（`0 < save_steps < max_steps`）以及启用 Reward Model 的评估组合，在启动前拒绝。
最终 checkpoint、checkpoint reload 校验及不使用 Reward Model 的普通评估允许。
发布调用必须提供完整 Actor 快照；本选项不提供增量权重发布能力。

Disjoint：

```text
rollout remains resident
-> Trainer update
-> pause admission
-> HCCL transaction
-> verify workers
-> reset cache and resume
-> publish V+1
```

Pause、sleep/wake、transaction、cache reset、resume 和 close 只由 coordinator 对共享 endpoint 执行一次；每个 worker
仍需独立提交并验证版本。

## 权重同步

- 默认策略为 `full_gather`，`bucket_size_mb` 默认 `128`；可显式选择 `direct_reshard`，运行中不切换策略。
- 支持模型的 TP1/TP2 均可选择 `full_gather` 与 `direct_reshard`，TP1 不再自动改写显式策略。
- `full_gather` 按训练侧完整参数 gather，并将多个完整参数组成 packed bucket；只有 Trainer rank 0 打包。
  对共享 Qwen3 的融合参数，worker 根据模型提供的行区间元数据还原 HF 权重，再调用 vLLM `load_weights()` 完成 Native 融合或 Hyper TP 切分。
- Colocated packed buffer 经 Trainer group 复制后由每个同设备 Trainer rank 导出 NPU IPC handle；disjoint 使用一个 rank-0-to-all-workers HCCL group。
- `full_gather` 内置先将当前参数的本地分片临时搬到通信设备，再执行完整参数聚合与打包，无额外开关。
  CPU offload 的源模型参数保持在 CPU；聚合使用设备侧 collective，避免先在 CPU 重建整参数。
  传输仍检查 packed buffer 的通信设备，IPC 与 disjoint HCCL 的设备合同不变。
- Direct planner 同时解释 Trainer FSDP/TP source layout 与 rollout DP/TP destination layout，普通模式允许
  已支持的 train/inference degree mismatch。
- Colocated full/direct 使用 NPU IPC；disjoint 使用 HCCL fan-out。
- Direct 显式计算 Trainer source 与 Native/Hyper rollout destination 的物理交集；full-gather 不读取 rollout destination layout。
- `bucket_size_mb` 是 full-gather 的合桶阈值。单个完整参数超过阈值时独占一个超限 bucket；direct 仍将它作为 fragment 硬上限。
- 任一同步阶段失败都会跨 Trainer rank 传播并终止当前运行，不执行 abort、fallback 或同进程恢复。
- IPC 在 packed buffer 分配后、handle 导出后先同步本地错误，成功才进入后续 broadcast 或 handle all-gather；已导出且接收状态未知的 buffer 由 transport 持有到 close。

设备聚合与睡眠级别的核心测试见 [验证入口](moe_code_agent.md#单元测试)。

## 配置

```yaml
rollout:
  engine: vllm
  vllm:
    deployment: colocated  # or disjoint
    sleep_level: 1  # 2 discards old weights and requires complete publication
    model_implementation: hyper  # or native
    data_parallel_size: 2
    tensor_parallel_size: 2
    host: 127.0.0.1
    port: 8422
    weight_sync:
      strategy: full_gather  # default; or direct_reshard
      bucket_size_mb: 128
train:
  accelerator:
    dp_shard: 2
    tp: 2
consistency:
  enabled: false
```

公开配置值沿用 `hyper/native`，vLLM 插件入口沿用 `register_hyper_models`，
模型注册名称沿用 `HyperQwen3ForCausalLM`。这些名称属于已有配置及运行环境的兼容接口；
RL 内部标识符的整改不要求更换镜像、调整依赖版本或迁移已有配置。

HCCL base port 和 socket range 必须成对配置，位于 CANN 支持的 `[1024, 65520]` 内，且 base port 必须包含在
`START-END` range 中。正式 launcher 会在创建 Docker 容器前验证。

以下字段已删除并显式拒绝：

```text
rollout.vllm.topology
rollout.vllm.request_concurrency
rollout.vllm.api_server_count
HYPER_QWEN3_VLLM_TOPOLOGY
HYPER_QWEN3_TP_TOPOLOGY
```

## 验证状态

Colocated 与 disjoint 复用同一份配置 schema、`train_rl.py`、rollout controller 和权重事务接口；deployment 只选择
设备所有权、residency 和传输实现。

2026-09-12 的四卡 colocated GRPO 已有 direct-reshard 与 packed full-gather 两步 bit-exact
验收，具体数据及后续回归统一见[一致性验证记录](qwen3_training_inference_consistency.md)。
模型接入及 PPO 目标计算的逻辑覆盖见
[模型接入测试](../../../tests/ut/rl/trainer/test_qwen3_master.py)与
[PPO 训练目标测试](../../../tests/ut/rl/trainer/test_ppo_targets.py)；逻辑测试不代替真机闭环证据。
迁移来源的 disjoint/cache/resume 历史结果不能替代当前版本对应组合的 bit-exact 验收，
也不提供多节点或性能承诺。目录重组本身不扩大这些验证范围。

Dense 模型仍要求 `train.accelerator.ep=1`，推理专家并行关闭。MoE 的 EP/EDP 和验收范围见
[Qwen3-30B-A3B](#qwen3-30b-a3b)；两类模型均不开放 EPLB。

## 修改门禁

修改 request、scheduler、ownership、publication 或 cache lifecycle 时至少验证：

```text
authoritative token IDs and stable row order
response mask/logprob alignment
all-worker policy-version agreement
sync failures terminate every Trainer rank
clean server and NPU process shutdown
matched TP consistency mismatch/max/mean = 0/0/0
```

Bit-exact 定义见 [Qwen3 训练-推理一致性](qwen3_training_inference_consistency.md)。
