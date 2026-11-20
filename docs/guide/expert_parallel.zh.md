# EP 专家并行指南

[English](expert_parallel.md) | **简体中文**

## 使用 EP 训练

通过模型 recipe 配置专家 mesh，并选择路由专家计算入口。该入口使用模型自带的 router 和分配给本 rank 的专家参数。参考 [Qwen3-MoE 训练 recipe](../../hyper_parallel/models/qwen3_moe/recipes/train.yaml)。

模型 recipe 通过 EP 条件选择专家并行计算入口。EP>1 时，先交换本轮的 count，再分别发送隐状态和专家 ID，最后将专家输出发回源 rank。EP=1 时使用模型原有的非 EP 前向。

## 专家容量

在通用 EP 入口的 `compute` 下引用模型原有的计算工厂，并在同级配置 `capacity_factor`。每个源 rank 按专家限制槽位数。两层工厂同时指定时以外层为准。超出容量时优先保留高分槽位。未设置上限时保留所有槽位。

```yaml
plan_overrides:
  - match: "*.mlp.forward"
    when: ep
    replace:
      kind: method
      implementation:
        _target_: hyper_parallel.distributed.expert_parallel.recipes.ep_compute_with_dispatch_options
        compute:
          _target_: hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel.qwen3moe_ep_compute_fn
          use_grouped_gemm: true
        capacity_factor: 1.25
```

## 自定义 dispatch 和 combine

调用方已有专家 ID 和路由权重时，可在一维专家 mesh 上使用 `EPContext`。调用方提供以 `Shard(0)` 切分的专家参数，对派发后的槽位执行计算并加权，然后将本次 dispatch 的 receipt 传给 `combine` 一次。

[EP 和 PP+EP 示例](../../examples/README.md#pytorch-moe-examples) 使用示例目录中的原有 EP 实现。切换到主训练 EP 路径列入重构第二阶段。旧 hook 式 EP style 和独立 `components.modules.moe` 导入已删除。参见 [API 参考](../api/api_reference.zh.md#ep-专家并行)。
