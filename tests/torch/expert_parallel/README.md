# Expert Parallelism Combination Tests

This directory contains distributed tests that validate the interaction between
basic Expert Parallelism (EP) and other parallel strategies (DP, TP, CP).

## Test Objective

Verify that ExpertParallel and ExpertTensorParallel work correctly when combined
with Data Parallel (FSDP), Tensor Parallel, and Context Parallel dimensions.

## Validation Method

All tests follow the same pattern:

1. Build a standalone MoE model (no parallelism) as the reference
2. Build a parallel MoE model with the target strategy applied
3. Run forward + backward on both models with the same input
4. Compare outputs and gradients within tolerance

## Test Coverage

| Template Name | dp | ep | tp | cp | Min Cards | Validation Scope |
|---------------|----|----|----|----|-----------|------------------|
| ep-only       | 1  | 2  | 1  | 1  | 2         | Basic EP functionality |
| tp-only       | 1  | 1  | 2  | 1  | 2         | Expert-internal TP |
| dp-ep         | 2  | 2  | 1  | 1  | 4         | DP + EP interaction |
| ep-tp         | 1  | 2  | 2  | 1  | 4         | EP + TP interaction |
| dp-ep-tp      | 2  | 2  | 2  | 1  | 8         | DP + EP + TP 3D interaction |
| dp-ep-cp      | 2  | 2  | 1  | 2  | 8         | EP + CP dimension compatibility (only validates EP works with a CP-included mesh, not CP communication itself) |
| dp-ep-cp-with-attention | 2 | 2 | 1 | 2 | 8 | **Real CP communication** with Self-Attention + MoE block |

### 2-Card Runnable Cases (32 total)

| Test Group | Count | Parameter Variations |
|------------|-------|----------------------|
| EP Base | 6 | num_experts: 2/4/8, top_k: 1/2 |
| EP + grouped_mm | 6 | num_experts: 2/4/8, top_k: 1/2 |
| EP + Shared Expert | 6 | num_experts: 2/4/8, top_k: 1/2 |
| TP-only | 8 | num_experts: 2/4, top_k: 1/2, hidden_dim: 64/128 |
| Validation | 6 | Valid/invalid mesh configurations |

## Running Tests

### 2-Card Environment

```bash
pytest tests/torch/expert_parallel/test_combinations.py -v -k "test_2card_group"
```

## Constraints

| Constraint | Description |
|------------|-------------|
| num_experts % ep == 0 | Experts must be evenly divisible across EP ranks |
| hidden_dim % tp == 0 | Hidden dimension must be divisible by TP degree |
| dp % ep == 0 (when dp > 1) | Data parallel degree must be divisible by EP degree |
| dp * ep * tp * cp == world_size | Product of all dimensions must equal total devices |
| sequence_length % cp == 0 | CP attention tests require sequence length divisible by cp |
| CP support | Only dimension compatibility is validated; full CP communication requires attention module (covered by cp-with-attention test) |
