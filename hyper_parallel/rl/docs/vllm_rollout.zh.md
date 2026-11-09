# vLLM Rollout

[English](vllm_rollout.md) | **简体中文**

## 适用范围

仅保留 native-vLLM。训练模型由 HP 构建，RL 插件不注册替代 Qwen3 模型。
插件入口为 `rl.rollout.vllm_plugin:install_runtime_hooks`，安装权重同步、工具证据、partial service 和必要采样修复。
版本敏感的私有生命周期钩子必须匹配支持的 vLLM/vLLM-Ascend 组合，未知版本明确失败。

## Qwen3-30B-A3B

MoE 使用本次 YAML 声明的 HP 替换规则和 native rollout。保持 GRPO、colocated、EPLB 关闭等既有约束，不支持 MoE PPO 或 disjoint。
full_gather 将 GroupedExperts 切片还原为逐专家 checkpoint 权重，通过 native layerwise reload 更新目标存储。
接收方持有独立 tensor，避免后续 bucket 释放造成悬空引用；回执必须覆盖全部 DP×TP worker。

### MoE R3 路由重放

`train.router_replay.mode: R3` 启用 expert capture 和 Actor replay，默认 `disabled`。
继续限制为现有 eager/raw-logprob/拓扑组合；不改变 Agent 预算或睡眠级别。
详细范围见 [MoE 指南](moe_code_agent.zh.md#moe-r3)。

## Ownership

所有训练 rank 参与参数 gather；协调 rank 管理共享 HTTP 控制端点。
worker 在实际加载完成后确认版本，DP engine 返回值不能代替所有物理 worker 的确认。
配置校验区分共卡设备映射和分离设备集合，并检查每个训练 rank 的 endpoint 一致性。

## Request 与 Admission

请求必须携带已提交策略版本。暂停 admission 后等待活动请求结束或按 partial 协议暂停，再更新参数。
token IDs、raw logprobs、worker version 和可选路由 trace 一起返回；工具/观察 token 不成为模型 action。
采样器修复恢复 partial prefill 中被丢弃请求的 RNG 偏移，不重建已删除的一致性模型路径。

## Policy 生命周期

训练释放分片和 optimizer 驻留后才唤醒 rollout。发布失败不唤醒 KV cache、不提交版本。
加载完成、缓存处理和 committed-version 检查全部成功后才恢复 admission。
重启恢复从新进程实际状态出发，重新发布 Actor；不认为旧 checkpoint 的版本代表新 worker 已加载。

## 权重同步

`WeightPublisher` 直接实现流式 full_gather；没有单实现 strategy 选择层。
colocated 使用 IPC，disjoint 使用 HCCL。每次一个 bucket，确认后释放；超大单参数独占 bucket。
桶计划只缓存名称和布局，不缓存完整 gather 后参数；tied 权重和 MoE 映射保持各自的 native 加载语义。
旧 `direct_reshard` 配置和 RPC 已删除。

## 配置

```yaml
rollout:
  engine: vllm
  vllm:
    model_implementation: native
    weight_sync:
      strategy: full_gather
      bucket_size_mb: 128
```

这是字段示意；启动时使用包含模型、数据、算法和设备声明的完整 examples 配置。
运行前安装主项目和 `hyper_parallel/rl`，核对 `rl.__file__` 指向当前 checkout，并检查插件 metadata 的新入口。

## 验证状态

CPU UT 验证事务顺序、故障传播、参数内容、单 bucket 内存生命周期和恢复协议。
NPU 训练、真实 IPC/HCCL、MoE、vLLM 图捕获及长期学习效果必须在明确分配设备后分别验收。
历史设备结果不自动证明本次重构后的设备路径已通过。

## 修改门禁

不得用禁用概率记录、放宽版本检查或跳过 worker 回执换取测试通过。
入口、配置、插件和接收端必须同步迁移；完整合同见[架构](architecture.zh.md)。
