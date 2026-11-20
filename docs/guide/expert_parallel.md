# EP Expert Parallelism Guide

**English** | [简体中文](expert_parallel.zh.md)

## Training with EP

Use a model recipe to configure the expert mesh and select a routed-expert compute factory. The factory uses the model's router and the parameters assigned to the local experts. See the [Qwen3-MoE training recipe](../../hyper_parallel/models/qwen3_moe/recipes/train.yaml).

Model recipes select the EP compute factory with the EP condition. With EP>1, EP exchanges the current counts, dispatches hidden states and expert IDs in separate exchanges, and sends expert outputs back to the source ranks. With EP=1, the model uses its existing non-EP forward.

## Expert capacity

Use the shared EP factory to apply `capacity_factor` to the existing model factory under `compute`. It limits slots per expert on each source rank. If both factories specify it, the outer value takes precedence. The highest-weight slots are kept when the cap is exceeded.

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

## Custom dispatch and combine

Use `EPContext` with a one-dimensional expert mesh if the caller already has expert IDs and routing weights. The caller supplies experts sharded with `Shard(0)`, computes and weights the dispatched slots, then passes the dispatch receipt to `combine` once.

The [EP and PP+EP examples](../../examples/README.md#pytorch-moe-examples) use a local copy of the original EP style. Their migration to the training EP path is tracked in stage two of the refactor. The old hook-based EP styles and the standalone `components.modules.moe` import have been removed. See the [API reference](../api/api_reference.md#ep-expert-parallelism).
