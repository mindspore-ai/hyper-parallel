# MoE Correctness Invariants

**Formal invariants** that must hold for correct MoE routing.

## Invariant 1: Token Conservation

**Statement:** The number of tokens entering MoE equals the number exiting.

### Mathematical Formulation

```
∀ layer ∈ MoE_layers:
  |input_tokens| = |output_tokens|
```

Where:
- `input_tokens`: Tokens entering MoE layer
- `output_tokens`: Tokens exiting MoE layer

### Implementation Check

```python
def check_token_conservation(input_tokens, output_tokens):
    assert input_tokens.shape[0] == output_tokens.shape[0], \
        f"Token count mismatch: {input_tokens.shape[0]} → {output_tokens.shape[0]}"
```

### What Can Go Wrong

1. **Buffer overflow**: Expert capacity exceeded, tokens dropped
2. **All-to-All bug**: Incorrect split sizes
3. **Routing bug**: Some tokens not assigned to any expert

---

## Invariant 2: Expert Assignment Completeness

**Statement:** Every token is assigned to exactly `top_k` experts.

### Mathematical Formulation

```
∀ token ∈ batch:
  |assigned_experts(token)| = top_k
```

And:
```
sum(tokens_per_expert) = |batch| × top_k
```

### Implementation Check

```python
def check_assignment_completeness(tokens_per_expert, batch_size, top_k):
    total_assignments = sum(tokens_per_expert)
    expected = batch_size * top_k

    assert total_assignments == expected, \
        f"Assignment mismatch: {total_assignments} != {expected}"
```

### What Can Go Wrong

1. **Capacity limit**: capacity_factor too small, tokens dropped
2. **Routing scores**: NaN/Inf in routing probabilities
3. **Top-k selection**: Incorrect implementation

---

## Invariant 3: Expert-Major Ordering

**Statement:** After dispatch All-to-All, tokens are grouped by expert.

### Mathematical Formulation

```
∀ i < j:
  if expert_id[i] < expert_id[j]:
    then position[i] < position[j]
```

Tokens are ordered: [Expert 0 tokens...][Expert 1 tokens...]...[Expert N tokens...]

### Implementation Check

```python
def check_expert_major_order(tokens, expert_ids, tokens_per_expert):
    current_pos = 0

    for expert_id in range(num_experts):
        count = tokens_per_expert[expert_id]

        # All tokens in [current_pos:current_pos+count] should belong to expert_id
        for i in range(current_pos, current_pos + count):
            assert expert_ids[i] == expert_id, \
                f"Token {i} assigned to expert {expert_ids[i]}, expected {expert_id}"

        current_pos += count
```

### What Can Go Wrong

1. **Permutation indices**: Incorrect computation
2. **All-to-All**: Wrong send/recv split sizes
3. **Sorting**: Expert assignment not sorted correctly

---

## Invariant 4: Gradient Connectivity

**Statement:** All activated experts must receive gradients.

### Mathematical Formulation

```
∀ expert ∈ experts:
  if tokens_assigned(expert) > 0:
    then ∀ param ∈ expert.parameters():
      param.grad ≠ None
```

### Implementation Check

```python
def check_gradient_connectivity(model, tokens_per_expert):
    for expert_id, num_tokens in enumerate(tokens_per_expert):
        if num_tokens > 0:
            expert = model.experts[expert_id]

            for param in expert.parameters():
                if param.requires_grad:
                    assert param.grad is not None, \
                        f"Expert {expert_id} missing gradient (received {num_tokens} tokens)"

                    assert param.grad.abs().sum() > 0, \
                        f"Expert {expert_id} has zero gradient"
```

### What Can Go Wrong

1. **Backward bug**: Gradient not flowing through All-to-All
2. **Detached tensors**: `.detach()` called incorrectly
3. **In-place operations**: Breaking gradient computation

---

## Invariant 5: Capacity Bounds

**Statement:** Tokens assigned to expert ≤ capacity.

### Mathematical Formulation

```
∀ expert ∈ experts:
  tokens_assigned(expert) ≤ capacity(expert)

where:
  capacity(expert) = (total_tokens / num_experts) × capacity_factor × top_k
```

### Implementation Check

```python
def check_capacity_bounds(tokens_per_expert, total_tokens, num_experts,
                         capacity_factor, top_k):
    capacity = int((total_tokens / num_experts) * capacity_factor * top_k)

    for expert_id, num_tokens in enumerate(tokens_per_expert):
        assert num_tokens <= capacity, \
            f"Expert {expert_id} exceeded capacity: {num_tokens} > {capacity}"
```

### What Can Go Wrong

1. **Capacity too small**: Many tokens dropped
2. **Capacity too large**: High memory usage
3. **Dynamic capacity**: Not implemented correctly

---

## Invariant 6: Routing Score Validity

**Statement:** All routing scores must be valid (not NaN/Inf).

### Mathematical Formulation

```
∀ token ∈ batch:
  ∀ expert ∈ experts:
    isfinite(routing_score[token][expert]) = True
```

And:
```
∀ token ∈ batch:
  sum(routing_probabilities[token]) ≈ 1.0  (if softmax normalized)
```

### Implementation Check

```python
def check_routing_scores(routing_scores, routing_probs):
    # Check for NaN/Inf
    assert torch.isfinite(routing_scores).all(), \
        "Routing scores contain NaN or Inf"

    # Check normalization (if using softmax)
    if routing_probs is not None:
        sums = routing_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            "Routing probabilities not normalized"
```

### What Can Go Wrong

1. **Numerical instability**: Overflow in exp()
2. **Zero logits**: All routing scores identical
3. **Loss computation**: Auxiliary loss NaN

---

## Invariant 7: Load Balance Auxiliary Loss

**Statement:** Load balance loss must be computed correctly (if enabled).

### Mathematical Formulation

```
load_balance_loss = α × sum(f_i × P_i)

where:
  f_i = fraction of tokens routed to expert i
  P_i = fraction of routing probability mass for expert i
  α = load_balance_coefficient
```

### Implementation Check

```python
def check_load_balance_loss(tokens_per_expert, routing_probs,
                           total_tokens, num_experts):
    # Compute f_i (fraction routed)
    f = tokens_per_expert / total_tokens

    # Compute P_i (routing probability mass)
    P = routing_probs.mean(dim=0)

    # Loss should be sum(f × P)
    loss = torch.sum(f * P) * num_experts

    # Loss should be in reasonable range
    assert 0 <= loss <= num_experts, \
        f"Load balance loss out of range: {loss}"
```

### What Can Go Wrong

1. **Double counting**: Both router loss and manually computed
2. **Wrong normalization**: Not scaled by num_experts
3. **Gradient issues**: Loss not backpropagated

---

## Invariant 8: Memory Bounds

**Statement:** Peak memory should not exceed device capacity.

### Mathematical Formulation

```
peak_memory ≤ device_memory × 0.95  (5% safety margin)

where:
  peak_memory = model_memory + optimizer_memory +
                activation_memory + communication_buffer
```

### Implementation Check

```python
def check_memory_bounds(device_memory_gb, peak_allocated_gb):
    threshold = device_memory_gb * 0.95

    assert peak_allocated_gb <= threshold, \
        f"Memory exceeded: {peak_allocated_gb:.1f}GB > {threshold:.1f}GB"
```

### What Can Go Wrong

1. **Capacity overflow**: Too many active experts
2. **Gradient accumulation**: Not clearing buffers
3. **All-to-All buffers**: Temporary buffers too large

---

## Summary Table

| Invariant | Check Type | Criticality | When to Check |
|-----------|-----------|-------------|---------------|
| Token Conservation | Exact | CRITICAL | Every forward pass |
| Assignment Completeness | Exact | CRITICAL | Every forward pass |
| Expert-Major Order | Exact | CRITICAL | After dispatch All-to-All |
| Gradient Connectivity | Exact | CRITICAL | After backward pass |
| Capacity Bounds | Exact | HIGH | During routing |
| Routing Score Validity | Approximate | HIGH | During routing |
| Load Balance Loss | Approximate | MEDIUM | If loss enabled |
| Memory Bounds | Approximate | MEDIUM | Periodically |

**Violation Handling:**
- CRITICAL: Abort training, fix bug
- HIGH: Log warning, investigate
- MEDIUM: Monitor, adjust if persistent

---

**See also:** [gradient-flow-patterns.md](gradient-flow-patterns.md) for expected gradient patterns.
