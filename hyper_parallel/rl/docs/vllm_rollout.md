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
当前迁移的实际验收结果记录在 [M1 执行记录](moe_code_agent.md#功能与支持边界)，不能沿用其他分支的历史结果。

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
-> sleep(level=1)
-> Trainer update
-> wake weights
-> NPU IPC transaction
-> verify workers
-> wake/reset KV cache
-> resume
-> publish V+1
```

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

- 默认策略为 `full_gather`；启动时显式选择 `full_gather` 或 `direct_reshard`，运行中不切换策略。
- 支持模型的 TP1/TP2 均可选择 `full_gather` 与 `direct_reshard`，TP1 不再自动改写显式策略。
- `full_gather` 按训练侧完整参数 gather，并将多个完整参数组成 packed bucket；只有 Trainer rank 0 打包。
  对共享 Qwen3 的融合参数，worker 根据模型提供的行区间元数据还原 HF 权重，再调用 vLLM `load_weights()` 完成 Native 融合或 Hyper TP 切分。
- Colocated packed buffer 经 Trainer group 复制后由每个同设备 Trainer rank 导出 NPU IPC handle；disjoint 使用一个 rank-0-to-all-workers HCCL group。
- 开启训练侧 CPU offload 时，聚合结果可能位于 CPU；传输前将当前 packed bucket 搬到通信设备，
  确保所有 IPC 广播参与者使用 NPU/HCCL，并让 disjoint HCCL 的发送 buffer 与通信组设备一致。
- Direct planner 同时解释 Trainer FSDP/TP source layout 与 rollout DP/TP destination layout，普通模式允许
  已支持的 train/inference degree mismatch。
- Colocated full/direct 使用 NPU IPC；disjoint 使用 HCCL fan-out。
- Direct 显式计算 Trainer source 与 Native/Hyper rollout destination 的物理交集；full-gather 不读取 rollout destination layout。
- `bucket_size_mb` 是 full-gather 的合桶阈值。单个完整参数超过阈值时独占一个超限 bucket；direct 仍将它作为 fragment 硬上限。
- 任一同步阶段失败都会跨 Trainer rank 传播并终止当前运行，不执行 abort、fallback 或同进程恢复。
- IPC 在 packed buffer 分配后、handle 导出后先同步本地错误，成功才进入后续 broadcast 或 handle all-gather；已导出且接收状态未知的 buffer 由 transport 持有到 close。

## 配置

```yaml
rollout:
  engine: vllm
  vllm:
    deployment: colocated  # or disjoint
    model_implementation: hyper  # or native
    data_parallel_size: 2
    tensor_parallel_size: 2
    host: 127.0.0.1
    port: 8422
    weight_sync:
      strategy: direct_reshard  # or full_gather
      bucket_size_mb: 128
train:
  accelerator:
    dp_shard: 2
    tp: 2
consistency:
  enabled: false
```

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
