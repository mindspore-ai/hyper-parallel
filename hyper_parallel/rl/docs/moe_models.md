# MoE 模型

hyperparallel-RL 支持 Qwen3-30B-A3B 和 DeepSeek-V3 family 的 Moonlight-16B-A3B-Instruct，使用固定运行镜像中的
Transformers 5.5.4、vLLM/vLLM-Ascend。模型识别来自 checkpoint 的 `config.json`，不依赖本地目录名称。
环境准备和镜像挂载见 [README](../README.md)，共享运行时与事务合同见 [vLLM Rollout](vllm_rollout.md)。

## 配置与运行

从仓库根目录运行，模型与 GSM8K 数据需已经挂载到运行容器。Moonlight 示例默认使用 Native-vLLM、FSDP4、
rollout DP4/TP1，并显式选择 direct-shard；全局未指定策略时的默认值仍是 full-gather。

```bash
python -m torch.distributed.run --nproc_per_node=4 \
  hyper_parallel/rl/examples/train_rl.py \
  hyper_parallel/rl/examples/configs/moonlight_16b_a3b_gsm8k_native_vllm.yaml
```

`model.weights_path`、`model.tokenizer_path` 和 `data.train_path` 可通过同名 `--配置键=值` 覆盖。
使用 Hyper-vLLM 时增加 `--rollout.vllm.model_implementation=hyper`。Moonlight 训练必须显式设置
`model.attention_implementation=transformers_builtin`：该 checkpoint 没有 Q-LoRA，不能使用要求 Q-LoRA 的 Hyper MLA。

Qwen3-30B-A3B 可复用同一示例，覆盖模型路径和 attention 选择，无需复制另一份完整 YAML：

```bash
python -m torch.distributed.run --nproc_per_node=4 \
  hyper_parallel/rl/examples/train_rl.py \
  hyper_parallel/rl/examples/configs/moonlight_16b_a3b_gsm8k_native_vllm.yaml \
  --model.registry_name=qwen3_30b_a3b --model.name=qwen3_moe \
  --model.weights_path=/models/Qwen3-30B-A3B \
  --model.tokenizer_path=/models/Qwen3-30B-A3B \
  --model.attention_implementation=null \
  --rollout.vllm.enable_expert_parallel=true \
  --train.micro_batch_size=2
```

两种模型的四卡 TP2/EP4 组合在上述命令后增加：

```text
--train.accelerator.dp_shard=2 --train.accelerator.tp=2 --train.accelerator.ep=4
--rollout.vllm.data_parallel_size=2 --rollout.vllm.tensor_parallel_size=2
--rollout.vllm.enable_expert_parallel=true
--train.prompt_batch_size=2 --train.response_mini_batch_size=4 --train.micro_batch_size=4
```

Trainer 的 expert mesh 来自 HyperParallel 的 TP-extend-EP 规则；rollout 的 DP2×TP2 展平为 effective EP4。
EP 与 TP/FSDP 共用四张卡，不将 DP×TP×EP 相乘计算设备数。Hyper EP 要求 `enforce_eager=true`，EPLB 必须关闭。
显存不足时减小采样长度或 batch；不要改变专家归属来绕过容量限制。

示例保留真实学习门禁：GSM8K 按既有规则提取答案，组内相同奖励导致 GRPO 零优势属于数学预期。
`train.learning_gate.enabled=true` 会对缺少混合奖励/非零梯度的验收运行报错；它不是保证收敛的生产配置。
使用者应根据数据与验收目的显式配置该开关，不应修改奖励标签来让门禁通过。

## 组件归属

| 组件 | Trainer + Hyper-vLLM | 保留推理组件的理由 |
| --- | --- | --- |
| 外层 decoder、norm、router、shared experts | Transformers 定义，公共 HyperParallel 计划 | 无需另一套模型或路由算法 |
| Qwen3 attention | dense/MoE 共用适配器及 TP 规则 | vLLM attention 管理 paged KV 与 decode |
| Moonlight MLA | 公共计划确定投影布局，adapter 校验实际本地切片 | absorbed MLA 持有投影、latent paged cache 与派生权重刷新状态 |
| Routed experts | 公共 EP placement、dispatch/combine；本地 FusedMoE 叶子 | Ascend 融合/分组 GEMM 与物理权重布局，叶子不重复执行 EP 通信 |
| 在线更新 | 共用 canonical adapter、布局、流式同步及事务 | MLA 派生权重在发布前逐层刷新，专家物理布局保持可执行 |

Native-vLLM 保留原生模型、融合布局及并行算子，不为结构一致而替换原生实现。两种推理方式共享控制面和权重语义，
不要求物理参数或 kernel 相同；性能优势需在相同 workload 下另行测量。

## 权重发布验收

使用完整 checkpoint 和真实 FSDP/TP/EP 训练源，按以下顺序验证精简后的同步路径：

1. 同一源策略先通过 direct-shard 发布 V1，再通过流式 full-gather 发布 V2。
2. 修改真实 norm/expert 分片后，用相同的新源策略分别发布 V3/V4；Moonlight 同时修改 MLA 的 KV-B 投影。
3. 再次修改参数及 `lm_head`，在 direct 的首桶已写入并确认后注入故障，由流式 full-gather 完整覆盖并发布 V5。

每次发布都执行源派生的内容身份校验，并保存全部 worker 参数的 SHA256 manifest。同源 direct/full 的完整源摘要、
目标 manifest 和发布身份必须分别一致；变更参数必须到达对应目标分片，所有 worker 的版本必须一致。
每个版本发布后都执行生成，检查 token/mask、版本及有限的原始 logprob，避免只证明权重可写入而不能运行。

验收使用 128 MiB bucket：单个 gather/pack/IPC buffer 不超过 128 MiB，两个传输缓冲合计不超过 256 MiB，
未确认 bucket 数为 1，所有 bucket 均确认后释放。这是同步临时缓冲上限，不包含常驻模型、KV cache 或分配器缓存。

上述实验冻结训练源、显式修改真实参数，不执行 optimizer；其结论是权重发布和恢复正确，不替代真实 RL 学习验收，
也不要求 Native/Hyper 生成结果 bit-exact。实际学习结果和未覆盖能力见下节。

精简后的实现已使用四张 Ascend 910B3、BF16，完成 Trainer FSDP2/TP2/EP4 → rollout DP2/TP2/EP4 的以下回归。
同一模型两种后端的五个版本均使用完全相同的训练源权重摘要：

| 完整模型 | Hyper-vLLM | Native-vLLM | 每次 full/fallback 桶数 |
| --- | --- | --- | --- |
| Moonlight-16B-A3B-Instruct | V1–V5 通过 | V1–V5 通过 | 336 |
| Qwen3-30B-A3B | V1–V5 通过 | V1–V5 通过 | 600 |

四组均通过完整内容、专家归属、真实参数变更、部分写入恢复及发布后生成检查；全部桶确认并释放，
传输缓冲满足上述上限，正常退出时未出现 NPU IPC producer 提前结束告警。

## 验证范围与限制

- 使用完整 30B/16B checkpoint 验证四卡训练源及 Native/Hyper 发布；局部前向/梯度检查不替代真实模型验证。
- Qwen3-A3B 的 Native/Hyper 两步功能闭环、受控非零更新及 Trainer TP1/EP4→Hyper TP2/EP4 已验证。
- Moonlight Hyper 的 FSDP2/TP2/EP4→DP2/TP2/EP4 两步真实非零学习已验证。
- Moonlight Native 同拓扑的训练/发布环节已执行，真实 RL 因全零奖励在学习门禁终止；不声明该实验的连续两步或
  非零学习通过。两个后端各自的同源 direct/full、真实 norm/expert/KV-B 变更、部分写入 fallback 和后续生成均通过。
- EP1/TP1 原有路径保留；Trainer TP2 要求 EP4。EP 关闭时 packed experts 内部 TP、TP>2、MoE disjoint、
  CP/PP、多节点、量化、动态 EPLB 和长期稳定性尚未完成支持或验收。
- MoE 不属于 [Qwen3 dense bit-exact](qwen3_training_inference_consistency.md) 的结论，结构复用不等于数值一致。
