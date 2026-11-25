# Parallelism-invariant operators

## PIK row-parallel projection

PIK fixes the partition of the global input feature axis and the order in which partial products are added.
For bitwise comparisons across TP sizes, use the same input, weights, global leaf count, leaf layout and
TorchNPU Level 3 policy. Each TP rank owns consecutive complete leaves. The global input width must be
divisible by the leaf count, which must be a power of two and divisible by each compared TP size.

```python
from hyper_parallel.components.parallelism_invariant.row_parallel_linear import PikPlan, pik_linear, prepare_pik

# After selecting the NPU, prepare once on every rank in the caller-owned HCCL group.
# Use tp_group=None for a single-rank projection.
context = prepare_pik(global_k, output_features, plan=PikPlan(num_leaves=4), group=tp_group)
output = pik_linear(local_input, local_weight, context=context)
```

With BF16 input `[..., K/TP]` and weight `[N, K/TP]`, each rank receives `[..., N]`.
For sequence-parallel output, pass `sequence_parallel=True` and the token axis as `sequence_dim` to
`prepare_pik`. The operator returns that rank's contiguous slice of the axis.
Preparation initializes the process-wide Level 3 policy before graph capture. The forward supports
first-order autograd. Backward across TP sizes is checked for correctness without a bitwise guarantee.

For inference, `pik_linear_inference` uses a BF16 all-reduce with gradients disabled, TP=2, two strided
leaves and the `all_gather` plan. Other configurations and gradient-enabled calls use `pik_linear`.
