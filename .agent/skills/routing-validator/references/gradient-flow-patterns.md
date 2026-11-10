# Gradient Flow Patterns

**Expected gradient patterns** for correct MoE training.

## Pattern 1: All Experts Active (Ideal)

**Characteristics:**
- All experts activated in every step
- All expert parameters receive gradients
- Gradient norms roughly uniform across experts

**Example:**
```python
experts_activated = [1, 1, 1, 1, 1, 1, 1, 1]  # All 8 experts
gradient_norms = [0.023, 0.021, 0.024, 0.022, 0.023, 0.020, 0.025, 0.021]
```

**When observed:**
- Balanced training data
- Uniform routing (early training)
- Proper capacity_factor (no drops)

---

## Pattern 2: Sparse Activation (Common)

**Characteristics:**
- Some experts activated, others dormant
- Only activated experts receive gradients
- Gradient norms vary by activation frequency

**Example:**
```python
experts_activated = [1, 0, 1, 0, 1, 1, 0, 1]  # 5/8 experts
gradient_norms = [0.023, 0.000, 0.024, 0.000, 0.023, 0.025, 0.000, 0.021]
                  #  ✓    dead   ✓    dead   ✓     ✓    dead   ✓
```

**When observed:**
- Specialized routing (late training)
- Load imbalance (some experts preferred)
- Normal behavior for < top_k activation

**Correctness:** ✅ PASS (dormant experts are OK if intentional)

---

## Pattern 3: Dead Experts (Warning)

**Characteristics:**
- Some experts never activated (across all steps)
- Permanently zero gradients
- Parameters don't update

**Example:**
```python
# Across 100 steps
expert_activation_count = [100, 98, 95, 0, 102, 97, 0, 99]
                           #        ↑ dead      ↑ dead

gradient_norms_step_100 = [0.023, 0.021, 0.024, 0.000, 0.023, 0.020, 0.000, 0.021]
```

**When observed:**
- Severe load imbalance
- Too many experts (over-capacity)
- Poor initialization

**Correctness:** ⚠️ WARN (functional but suboptimal)

---

## Pattern 4: Missing Gradients (Error)

**Characteristics:**
- Expert activated but no gradient
- Backward pass broken
- Parameters frozen unintentionally

**Example:**
```python
# Expert 2 received tokens but no gradient!
tokens_per_expert = [245, 312, 289, 256, 234, 267, 298, 303]
                     #        ↑ activated (289 tokens)

gradient_norms = [0.023, 0.021, 0.000, 0.022, 0.023, 0.020, 0.025, 0.021]
                  #               ↑ but gradient is zero!
```

**When observed:**
- Backward implementation bug
- Detached tensors in forward
- Gradient checkpointing issue

**Correctness:** ❌ FAIL (critical bug)

---

## Pattern 5: Gradient Explosion

**Characteristics:**
- Some expert gradients extremely large (> 100×)
- Numerical instability
- Training may diverge

**Example:**
```python
gradient_norms = [0.023, 0.021, 15.8, 0.022, 0.023, 0.020, 0.025, 0.021]
                  #               ↑ exploded (700× normal)
```

**When observed:**
- Routing scores unstable (NaN/Inf)
- Learning rate too high
- Auxiliary loss weight too large

**Correctness:** ⚠️ WARN (functional but unstable)

---

## Pattern 6: Gradient Vanishing

**Characteristics:**
- All gradients extremely small (< 1e-10)
- Learning stalled
- No parameter updates

**Example:**
```python
gradient_norms = [3e-12, 2e-12, 5e-12, 1e-12, 4e-12, 2e-12, 6e-12, 3e-12]
                  # All vanished
```

**When observed:**
- Too many layers (deep model)
- Activation functions saturated
- Gradient scaling issue

**Correctness:** ⚠️ WARN (functional but not learning)

---

## Pattern 7: Inconsistent Gradients

**Characteristics:**
- Gradients present sometimes, absent other times
- Non-deterministic
- May indicate race condition

**Example:**
```python
# Step 100
gradient_norms_step_100 = [0.023, 0.021, 0.024, 0.022, 0.023, 0.020, 0.025, 0.021]

# Step 101 (same expert usage)
gradient_norms_step_101 = [0.023, 0.021, 0.000, 0.022, 0.023, 0.020, 0.025, 0.021]
                           #               ↑ disappeared!
```

**When observed:**
- Distributed training bug
- Async gradient computation
- Memory corruption

**Correctness:** ❌ FAIL (critical bug)

---

## Diagnostic Checks

### Check 1: Gradient Presence

```python
def check_gradient_presence(model, tokens_per_expert):
    for expert_id, num_tokens in enumerate(tokens_per_expert):
        if num_tokens > 0:
            expert = model.experts[expert_id]
            for param in expert.parameters():
                if param.requires_grad:
                    if param.grad is None:
                        return f"FAIL: Expert {expert_id} missing gradient (had {num_tokens} tokens)"
    return "PASS"
```

### Check 2: Gradient Magnitude

```python
def check_gradient_magnitude(model, tokens_per_expert, threshold=1e-8):
    for expert_id, num_tokens in enumerate(tokens_per_expert):
        if num_tokens > 0:
            expert = model.experts[expert_id]
            for name, param in expert.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()

                    if grad_norm < threshold:
                        return f"WARN: Expert {expert_id}.{name} gradient vanished ({grad_norm:.2e})"

                    if grad_norm > 10.0:
                        return f"WARN: Expert {expert_id}.{name} gradient exploded ({grad_norm:.2e})"

    return "PASS"
```

### Check 3: Gradient Consistency

```python
def check_gradient_consistency(model_t1, model_t2, tokens_t1, tokens_t2):
    """Compare gradients across consecutive steps"""

    for expert_id in range(num_experts):
        activated_t1 = tokens_t1[expert_id] > 0
        activated_t2 = tokens_t2[expert_id] > 0

        if activated_t1 == activated_t2:  # Same activation pattern
            expert_t1 = model_t1.experts[expert_id]
            expert_t2 = model_t2.experts[expert_id]

            for (name1, param1), (name2, param2) in zip(
                expert_t1.named_parameters(),
                expert_t2.named_parameters()
            ):
                has_grad_t1 = param1.grad is not None
                has_grad_t2 = param2.grad is not None

                if has_grad_t1 != has_grad_t2:
                    return f"FAIL: Inconsistent gradient for expert {expert_id}.{name1}"

    return "PASS"
```

---

## Expected Gradient Norms by Component

### MoE Router

```python
# Gate/router parameters
router.weight.grad_norm: 0.01 - 0.1  (typical)
```

### Expert FFN Layers

```python
# Expert feed-forward parameters
expert.w1.weight.grad_norm: 0.02 - 0.05  (typical)
expert.w2.weight.grad_norm: 0.02 - 0.05  (typical)
expert.w3.weight.grad_norm: 0.02 - 0.05  (for SwiGLU)
```

### Load Balance Loss Gradients

```python
# Auxiliary loss contribution
load_balance_grad_norm: 0.001 - 0.01  (much smaller than main loss)
```

---

## Troubleshooting

### Issue: All gradients zero

**Possible causes:**
1. Loss not calling `.backward()`
2. Optimizer in eval mode (`.eval()` called)
3. Parameters frozen (`.requires_grad = False`)

**Fix:**
```python
# Ensure model in train mode
model.train()

# Ensure backward called
loss.backward()

# Check requires_grad
for param in model.experts.parameters():
    assert param.requires_grad, "Parameters frozen!"
```

---

### Issue: Some experts have no gradient

**Possible causes:**
1. Experts not activated (normal if dormant)
2. Backward broken (critical if activated)
3. Detached tensors

**Fix:**
```python
# Check if activated
if tokens_per_expert[expert_id] > 0:
    # Expert was used, gradient should exist
    assert expert.weight.grad is not None
else:
    # Expert not used, no gradient expected
    pass
```

---

### Issue: Gradient explosion

**Possible causes:**
1. Learning rate too high
2. Routing scores unstable
3. Auxiliary loss weight too high

**Fix:**
```python
# Clip gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Reduce learning rate
optimizer.param_groups[0]['lr'] *= 0.5

# Reduce auxiliary loss
load_balance_loss_weight *= 0.1
```

---

**See also:** [correctness-invariants.md](correctness-invariants.md) for formal invariants.
