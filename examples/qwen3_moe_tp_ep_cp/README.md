# Qwen3-MoE TP + EP + CP demos

这两个配置复用 `examples/training_demo/tiny_qwen3_moe` 和 indexed mock
数据，目标是用 8 张 Ascend NPU 验证 TP、EP 与 CP 的组合拓扑。EP 不依赖
隐式注入，`*.mlp` 明确指定
`qwen3moe_ep_compute_fn`，由该工厂完成 router、EP dispatch/combine 和本地
expert 计算。

| 配置 | 拓扑 | CP wrapper |
| --- | --- | --- |
| `train_tp2_ep2_cp2_ulysses.yaml` | TP=2，EP=2，CP=2 | `sdpa_hf_ulysses_cp_wrapper` |
| `train_tp2_ep2_cp4_hybrid.yaml` | TP=2，EP=2，CP=4 | `sdpa_hf_hybrid_cp_wrapper`，`ulysses_degree=2` |

Hybrid 使用 CP4 是实现约束的必然结果：当前 Hybrid 要求
`1 < ulysses_degree < cp_size` 且 `cp_size % ulysses_degree == 0`，所以 CP2、
degree=2 会被正确拒绝为 Pure Ulysses 配置。

## 运行

```bash
cd /path/to/hyper-parallel
export HYPER_PARALLEL_PLATFORM=torch
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /home/wyd/env.sh

torchrun --nproc_per_node=8 --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29531 --rdzv_id=qwen3_tp_ep_cp_ulysses \
  --module examples.training_demo.train_text \
  examples/qwen3_moe_tp_ep_cp/train_tp2_ep2_cp2_ulysses.yaml

torchrun --nproc_per_node=8 --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29532 --rdzv_id=qwen3_tp_ep_cp_hybrid \
  --module examples.training_demo.train_text \
  examples/qwen3_moe_tp_ep_cp/train_tp2_ep2_cp4_hybrid.yaml
```

两份配置均为 1 个 optimizer step、固定种子和本地 mock 数据，不依赖外部
数据集或模型下载。tiny 模型使用 8 个 Q head 和 4 个 K/V head；TP2 后每个
rank 为 4/2 个 head，满足 Ulysses 的 head 维整除约束。运行时使用 HCCL；
不要将 backend 改为 Gloo。
