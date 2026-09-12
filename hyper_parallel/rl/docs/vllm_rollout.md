# vLLM Rollout

## 适用范围

本文定义 HyperParallel-RL 的 shared vLLM rollout 合同，覆盖 Qwen3 Hyper/Native TP1/TP2、colocated NPU IPC、disjoint HCCL、
在线权重发布和同步失败语义；[MoE 模型](moe_models.md) 复用这些合同，仅扩展静态专家归属及模型特有布局。

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
- vLLM upstream 管理 DP routing 和 frontend 数量；HyperParallel-RL 不设置生产用固定 DP rank header。
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

当前不提供透明 generation retry。未来若引入 retry，必须保持 token、seed、policy identity 和 row order。

## Policy 生命周期

Generation 前后都验证所有 workers 的 version/fingerprint。

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
仍需独立 commit 和验证 identity。

## 权重同步

- 默认策略为 `full_gather`，fallback 默认 `none`；显式启用 `direct_reshard` 不会隐式开启 fallback。
- 支持模型的 TP1/TP2 均可选择 `full_gather` 与 `direct_reshard`，TP1 不再自动改写显式策略。
- `full_gather` 使用逐 fragment/bucket gather、传输、ACK 后释放的有界实现；colocated 走 NPU IPC，disjoint 走按 TP
  目标划分的 HCCL broadcast，不再提供 whole-model 实现选择器。
- Direct planner 同时解释 Trainer FSDP/TP/EP source layout 与 rollout DP/TP/EP destination layout，普通模式允许
  已支持的 train/inference degree mismatch；MoE 当前只开放 colocated。
- Colocated full/direct 使用 NPU IPC；disjoint 使用 HCCL fan-out。
- Acceptance 使用 Trainer source-derived expected manifest，不用 full/direct 两个待测路径互相证明正确。

仅当显式配置 `fallback_strategy=full_gather` 时，Direct 失败后执行：

```text
keep admission paused
-> abort pending transactions
-> restore committed identity V
-> full-gather overwrite all V+1 parameters
-> verify workers
-> reset cache, resume and publish V+1
```

Abort 只恢复 transaction identity，不承诺回滚已经写入的 bytes。Fallback 也失败时 controller version 不前进，admission
不恢复，相关 buffers 保留到安全释放或 server shutdown。

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
      fallback_strategy: full_gather  # or none
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

## 已验证拓扑

Colocated 与 disjoint 复用同一份配置 schema、`train_rl.py`、rollout controller 和权重事务接口；deployment 只选择
设备所有权、residency 和传输实现。下表的 bit-exact 结果仅针对 Qwen3 dense，不能推广到 MoE。

| Deployment | Trainer                        | Rollout                  | 传输    | 验证结果                                   |
| ---------- | ------------------------------ | ------------------------ | ------- | ------------------------------------------ |
| Colocated  | `FSDP-shard2×TP2`，4 NPU    | `DP2×TP2`，共享 4 NPU | NPU IPC | full/direct/fallback 两步 RL 均为`0/0/0` |
| Disjoint   | `FSDP-shard2×TP2`，NPU 0–3 | `DP2×TP2`，NPU 4–7   | HCCL    | full/direct/fallback 两步 RL 均为`0/0/0` |

Disjoint 还验证了 direct partial receive 后 fallback、direct 与 fallback 双失败不发布、Prefix Cache、Chunked Prefill，
以及四 Trainer rank DCP destroy/resume/refit。以上是单节点 Ascend 910B3 功能与正确性结论，不包含多节点和性能承诺。

MoE 的四卡 TP2/EP4 发布矩阵、完整内容和有界缓冲验收见 [MoE 权重发布验收](moe_models.md#权重发布验收)。

## 修改门禁

修改 request、scheduler、ownership、publication 或 cache lifecycle 时至少验证：

```text
authoritative token IDs and stable row order
response mask/logprob alignment
all-worker version/fingerprint agreement
transaction abort/fallback ordering
clean server and NPU process shutdown
matched TP consistency mismatch/max/mean = 0/0/0
```

Bit-exact 定义见 [Qwen3 训练-推理一致性](qwen3_training_inference_consistency.md)。
