# Weight Sync 精简方案

> 状态：已实施。当前方案以 Qwen3-4B、同步训练和失败即退出为边界。

## 目标

Weight Sync 保留现有框架需要的两种同步算法和两种部署方式，同时缩短主流程、明确所有权，并删除正常训练不需要的恢复和内容诊断体系。

最终结构是：

- 一套发布事务：`WeightPublisher.publish()`。
- 两种数据策略：`DirectReshardStrategy`、`FullGatherStrategy`。
- 两种传输：`IPCWeightTransport`、`HCCLWeightTransport`。
- 一个共享 vLLM endpoint。
- 一个运行时身份字段：整数 `policy_version`。

四种组合由“策略 × 传输”装配得到，不分别实现四个 publisher，也不通过多层继承复用代码。

## 明确删除的能力

- direct 失败后切换 full-gather 的 fallback 配置和执行分支。
- worker abort RPC、pending identity 恢复和发布层补偿逻辑。
- 内容 fingerprint、fragment/content identity 和完整参数摘要比较。
- 在线 parameter manifest、layout/transaction trace 和内存诊断文件。
- 权重同步故障注入环境变量、注入器和测试场景。
- rollout artifact SHA 证据与 fingerprint learning gate。
- 旧接口、旧配置字段、旧 payload schema 和兼容别名。

失败后的合同很简单：异常跨 Trainer rank 传播，当前运行退出。系统不承诺在同一进程中将已经部分写入的 worker 参数恢复到旧字节，也不会重新开放采样。

## 保留的正确性边界

- `policy_version` 必须单调提交，worker 必须全部报告同一目标版本。
- generation 请求前后读取的 worker 版本必须相同。
- direct plan 必须完整覆盖每个 Qwen3 目标物理区域；full-gather 发送完整 HF 参数并由 vLLM `load_weights()` 解释目标布局。
- full-gather 必须保持逐 bucket materialize、发送、ACK、释放顺序；除单个超大参数外，bucket 受配置阈值约束。
- IPC/HCCL 的 route、worker 坐标、字节数和 ACK 仍做协议级校验。
- Qwen3 Native 融合参数、Hyper TP placement 和 tied embedding 映射保留。
- colocated sleep/wake、cache reset、rollout pause/resume 的成功路径保留。
- checkpoint 保存恢复是训练持久化能力，与在线参数 manifest 无关，继续保留。

## 文件职责与精简结果

### `__init__.py`

只导出上层实际使用的 controller、snapshot、client、publisher 和构造函数。删除旧组合类与旧 transfer/refit 名称。

### `config.py`

只解析：

- `strategy`: `direct_reshard` 或 `full_gather`
- `bucket_size_mb`: 正整数

删除 fallback 常量、字段和组合校验。未知字段直接报错，不静默兼容旧配置。

### `sync.py`

负责策略生命周期，不负责 HTTP 细节和数据搬运：

```text
rollout -> prepare_for_training -> update_weights -> prepare_for_rollout -> rollout
```

保留 `PolicySnapshot`、跨 rank 错误同步、pending version、sleep/wake 和 admission。删除 fingerprint 状态、generation identity 二元组和 resume 失败后的补偿 pause。对外使用 `generation_version()`。

### `vllm_client.py`

集中 pause/sleep/wake/start/finish/resume、collective RPC、单 endpoint 和 `committed_policy_version()`。Direct worker-layout RPC 的拓扑校验和 DP replica 收口也由该边界负责。删除 fingerprint、内容校验和 abort 请求。

### `transfer.py`

这是唯一发布编排位置。publisher 在构建时固定一个 strategy 和一个 transport，一次调用只尝试一次。执行成功后仅校验 worker 已提交目标整数版本。

`WeightSource` 只负责把 Actor snapshot 映射为两种策略使用的权重名称。Direct strategy 收集 source/destination descriptions；full-gather 缓存完整参数 bucket且不查询 rollout layout。Publisher 只编排一次 prepare/start/transfer/finish/version-check。删除 fallback 状态、尝试序列、失败原因、内容 accumulator、manifest 和 fault hooks。

### `ipc.py`

负责 colocated 设备映射、IPC handle 导出、请求投递、ACK 和 buffer 生命周期。正常 ACK 后释放 buffer；未知传输状态下保留到 transport `close()`，不提供公开的失败恢复入口。

Direct 与 packed 共用 handle 导出/收集、rank-0 序列化投递和 ACK 后同步。Packed buffer 分配失败在 broadcast 前同步；handle 导出失败在 handle all-gather 前同步，异常直接结束本次运行。两条路径各自决定目标设备和 payload，保留 direct 定向发送与 packed 每设备导出语义。

### `hccl.py`

负责 disjoint 建组、后台 worker RPC、broadcast、ACK 和关闭。Direct 保留 source/target-TP route；full-gather 使用一个 producer-to-all-workers group。删除 fault hook 和失败恢复 API。

四处“后台 RPC + 本地 collective + 错误同步 + RPC 结果收集”共用 `_rpc_with_collective()`，endpoint 检查集中在 `_resolve_endpoint()`。Rank 0 负责 RPC；direct 由 route 的 source rank 建组和发送，packed 由 rank 0 建组和发送。参与 rank、group 成员和 ACK 校验仍由具体路径表达。

### `layout.py`

负责 direct 所需的源/目标物理区域、交集、完整覆盖、切块和打包。原 `tensor_ops.py` 的 `local_tensor()` 与 `pack_direct_bucket()` 已合并到这里，避免只有两个强相关函数的薄文件。

后续收口将 IPC 物理 worker 映射移入 `ipc.py`，删除 content identity 遗留的 canonical metadata，并把 region/bucket helper 限定为 direct 内部实现。`SourceTensorLayout.source_name/source_starts` 等通用物理来源字段继续保留，不把布局合同绑定到 Qwen3-only 假设。文件由 843 行降到 795 行。

### `model_adapter.py`

只保留 Qwen3-4B 参数语义：Actor 名称、tied embedding、Native QKV/gate-up 融合目标和 Hyper TP placement。模型差异不会进入 publisher 或 transport。

名称转换直接调用模型注册项，删除转发包装；Native QKV 使用明确的 shard/axis-0 布局，删除重复常量数组和未使用的参数。保留 tied/untied 映射、embedding alias 以及通用 source/目标区域字段。

### `packed_weight.py`

负责按完整参数构建 bucket、所有 Trainer rank 参与 `full_tensor()`、仅 rank 0 打包，以及 worker 侧重建 `(name, tensor)`。不包含 transport、目标 TP 布局或发布事务。

### `vllm_worker.py`

按“拓扑/布局 RPC → 接收 → 写入 → 版本提交”组织。Direct 使用显式 scatter；full-gather 调用模型 `load_weights()`。worker 只维护 pending/committed version；删除 pending content、fingerprint、manifest、abort 和内存诊断。

Direct 与 packed 共用 HCCL 建组实现、IPC 设备坐标校验和 handle 导入、版本前置条件以及 ACK 构造。各 RPC 仍分别表达目标选择、scatter 或 `load_weights()`，保留各自的同步和 buffer 释放顺序。Worker 持有接收侧 HCCL group，Trainer transport 持有发送侧 group；两者没有合并为跨进程状态对象。

### `rl/checkpoint.py`

负责 Actor、optimizer、scheduler、RNG、dataloader 和 checkpoint 完成标记。它不属于在线权重同步，已从 `roles/weight_sync/` 移到 RL 顶层；格式、保存时机和恢复语义不变。

### 已删除文件

- `tensor_ops.py`：函数并入 `layout.py`。
- `streaming_full_gather.py`：目标 fragment 规划被完整参数 packed 流替代。
- `identity.py`：在线内容身份体系已删除。
- `diagnostics.py`：在线 trace、manifest、内存采样和故障注入已删除。

## 依赖方向

```text
rollout engine
  -> ActorRolloutWeightSync
    -> WeightPublisher
      -> DirectReshardStrategy | FullGatherStrategy
      -> IPCWeightTransport | HCCLWeightTransport
        -> vLLM worker RPC
```

`layout.py` 只提供 direct 数据规划，`packed_weight.py` 只提供 full-gather 完整参数合桶。transport 不拥有 controller 版本，strategy 不拥有通信 group，worker 不解释 Trainer phase。

## 成功与失败流程

成功：

```text
Actor V+1 ready
  -> rollout admission closed
  -> worker update started
  -> selected strategy transfers every bucket
  -> worker commits V+1
  -> controller verifies all worker versions
  -> cache/reset lifecycle completes
  -> rollout admission opens
  -> controller exposes V+1
```

失败：

```text
any exception
  -> synchronize the error across Trainer ranks
  -> propagate
  -> terminate the current run
```

没有同进程 recovery、fallback、abort 或部分提交后的继续训练。

## 配置迁移

有效配置：

```yaml
weight_sync:
  strategy: direct_reshard
  bucket_size_mb: 128
```

若希望更保守地使用 vLLM checkpoint-format loader，应直接配置 `strategy: full_gather`。旧 `fallback_strategy` 和故障注入环境变量已删除。

## 验收

Weight Sync 目录 Python 源码由基线的 8,294 行降到 3,934 行，其中 322 行来自 checkpoint 的职责迁移，其余缩减来自删除重复实现、恢复和诊断代码。此次 transport/adapter 收口减少 118 行：`hccl.py` 567 → 488、`ipc.py` 344 → 320、`model_adapter.py` 372 → 357。`transfer.py` 为 392 行，`vllm_worker.py` 为 747 行；IPC、HTTP client 与 packed-weight contract 分别持有自己的协议和资源。

- 配置测试确认只接受两个当前字段并拒绝旧字段。
- 四种策略/部署组合执行实际 tensor 写入并提交整数版本。
- direct 的布局计划、route 和 fragment 硬上限，以及 full-gather 的完整参数合桶、超限单参数、ACK 和 producer 所有权受测试覆盖。
- transfer、finish、resume 错误均直接传播，测试确认不执行恢复动作。
- rollout、ExperienceBatch、Agentic session/gateway/program 只传递整数版本。
- 全量 UT、覆盖率、ST 支撑用例与 Pylint 在变更后运行；设备侧 IPC/HCCL 集成仍以真实 NPU ST 为准。

后续若重新引入恢复能力，应作为独立设计处理：先定义字节级回滚或进程重启语义，再添加配置与测试，不能恢复为隐式 fallback。
