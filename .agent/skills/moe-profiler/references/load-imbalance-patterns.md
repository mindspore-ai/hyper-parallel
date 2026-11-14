# Load Imbalance Patterns

Known patterns of expert load imbalance in MoE training, extracted from empirical observations. This document is purely descriptive — no recommendations.

## Pattern Taxonomy

### Pattern 1: Uniform Distribution (Ideal)

**Characteristics:**
- All experts receive similar token counts (± 5%)
- Load imbalance factor: 1.0 - 1.1
- Gini coefficient: < 0.05

**Example data:**
```
Expert 0: 508 tokens
Expert 1: 512 tokens
Expert 2: 505 tokens
Expert 3: 515 tokens
Expert 4: 510 tokens
Expert 5: 502 tokens
Expert 6: 511 tokens
Expert 7: 509 tokens

Mean: 509, Std: 4.2, Imbalance: 1.01
```

**Observed in:**
- Early training (before routing specialization)
- Small models (< 10 layers)
- High top_k values (k ≥ 4)

---

### Pattern 2: Mild Skew (Common)

**Characteristics:**
- 1-2 experts receive 20-30% more tokens
- Load imbalance factor: 1.2 - 1.5
- Gini coefficient: 0.1 - 0.2

**Example data:**
```
Expert 0: 623 tokens  ← Hot expert
Expert 1: 487 tokens
Expert 2: 512 tokens
Expert 3: 501 tokens
Expert 4: 495 tokens
Expert 5: 508 tokens
Expert 6: 490 tokens
Expert 7: 456 tokens  ← Cold expert

Mean: 509, Std: 48.7, Imbalance: 1.22
```

**Observed in:**
- Mid-training (routing starts specializing)
- Standard MoE configs (8-16 experts, top_k=2)
- Balanced datasets

**Time dynamics:**
- Usually stabilizes after 1000-2000 steps
- Pattern persists across layers

---

### Pattern 3: Heavy-Tail Distribution

**Characteristics:**
- 1-3 "hot" experts receive 2-3× mean load
- Most experts under-utilized
- Load imbalance factor: 1.5 - 2.5
- Gini coefficient: 0.3 - 0.5

**Example data:**
```
Expert 0: 891 tokens  ← Very hot
Expert 1: 234 tokens  ← Cold
Expert 2: 512 tokens
Expert 3: 198 tokens  ← Cold
Expert 4: 456 tokens
Expert 5: 687 tokens  ← Hot
Expert 6: 312 tokens
Expert 7: 782 tokens  ← Hot

Mean: 509, Std: 251.3, Imbalance: 1.75
```

**Observed in:**
- Deep models (> 30 layers)
- Later training stages (> 5K steps)
- Skewed datasets (domain-specific text)
- Low capacity_factor (1.0)

**Characteristics by layer:**
- Upper layers show more skew than lower layers
- Pattern amplifies across layer depth

---

### Pattern 4: Bimodal Distribution

**Characteristics:**
- Two clusters: "active" vs "dormant" experts
- Large within-cluster variance
- Load imbalance factor: 2.0 - 3.0
- Gini coefficient: 0.4 - 0.6

**Example data:**
```
Expert 0: 812 tokens  ← Active cluster
Expert 1:  45 tokens  ← Dormant
Expert 2: 789 tokens  ← Active cluster
Expert 3:  67 tokens  ← Dormant
Expert 4:  89 tokens  ← Dormant
Expert 5: 823 tokens  ← Active cluster
Expert 6: 734 tokens  ← Active cluster
Expert 7:  98 tokens  ← Dormant

Mean: 432, Active mean: 790, Dormant mean: 75
Imbalance: 1.90 (within active), 2.80 (global)
```

**Observed in:**
- Over-capacity models (num_experts >> optimal)
- Aggressive routing (high temperature, sharp softmax)
- Models trained without load-balance loss

**Time dynamics:**
- Forms gradually after 2-3K steps
- Dormant experts may "wake up" occasionally but revert

---

### Pattern 5: Oscillating Load

**Characteristics:**
- Expert loads fluctuate significantly across steps
- High temporal variance (std > 50% of mean)
- Imbalance factor varies: 1.2 - 2.5 across windows

**Example timeline (Expert 0):**
```
Step 100: 623 tokens
Step 101: 245 tokens
Step 102: 789 tokens
Step 103: 312 tokens
Step 104: 567 tokens

Window mean: 507, Window std: 212 (41% CV)
```

**Observed in:**
- Small batch sizes (< 512 tokens/batch)
- High sequence length variance (mixed batching)
- Unstable routing (no aux loss, no expert bias)

**Pattern across experts:**
- All experts show high variance
- No consistent "hot" expert

---

### Pattern 6: Layer-Stratified Load

**Characteristics:**
- Imbalance varies significantly by layer
- Early layers: uniform (imbalance ~ 1.1)
- Middle layers: moderate skew (imbalance ~ 1.4)
- Late layers: heavy skew (imbalance ~ 2.0)

**Example data:**
```
Layer 0: Imbalance=1.08, Gini=0.04
Layer 1: Imbalance=1.12, Gini=0.08
Layer 2: Imbalance=1.35, Gini=0.18
Layer 3: Imbalance=1.67, Gini=0.31
Layer 4: Imbalance=1.98, Gini=0.42
Layer 5: Imbalance=2.14, Gini=0.48
```

**Observed in:**
- Deep MoE models (> 20 layers)
- Standard training (no special tricks)

**Hypothesis (not diagnosis):**
- Later layers process more abstract features
- Routing becomes more specialized as depth increases

---

### Pattern 7: Rank-Specific Imbalance

**Characteristics:**
- Some ranks consistently process "hot" experts
- Rank-level step time variance > 20%
- Straggler score > 1.25 for affected ranks

**Example data:**
```
Rank 0: Holds experts [0, 1] — mean load=498, step_time=172ms
Rank 1: Holds experts [2, 3] — mean load=512, step_time=174ms
Rank 2: Holds experts [4, 5] — mean load=709, step_time=198ms ← Straggler
Rank 3: Holds experts [6, 7] — mean load=487, step_time=171ms

EP mapping: 8 experts, ep_size=4, 2 experts/rank
```

**Observed in:**
- Large EP degree (ep_size ≥ 4)
- Static expert-to-rank mapping
- Heavy-tail load distribution (Pattern 3)

**Impact:**
- Stragglers slow down entire collective
- Effective throughput limited by slowest rank

---

### Pattern 8: Capacity-Limited Assignment

**Characteristics:**
- Many tokens dropped (exceed expert capacity)
- High zero_assignment_count for some experts
- Actual tokens < expected tokens from routing scores

**Example data:**
```
Expert 0: Routing assigned 823, Actual processed 625 (75% capacity hit)
Expert 1: Routing assigned 512, Actual processed 512 (no drop)
Expert 2: Routing assigned 789, Actual processed 625 (21% drop)
...

Total dropped tokens: 487 (11.9% of batch)
```

**Observed in:**
- Low capacity_factor (1.0)
- High routing entropy (many tokens want same expert)
- Large batch sizes (> 4096 tokens)

**Capacity formula:**
```python
capacity_per_expert = (total_tokens / num_experts) × capacity_factor × top_k
```

---

## Temporal Evolution Patterns

### Phase 1: Initialization (Steps 0-500)

**Typical progression:**
- Step 0-100: Near-uniform (imbalance ~ 1.05)
- Step 100-300: Gradual divergence (imbalance → 1.2)
- Step 300-500: Stabilization at mild skew (imbalance ~ 1.3)

---

### Phase 2: Specialization (Steps 500-3000)

**Typical progression:**
- Imbalance increases gradually
- Experts begin clustering into "active" and "passive"
- Pattern solidifies by step 2000

---

### Phase 3: Convergence (Steps 3000+)

**Typical progression:**
- Imbalance factor plateaus
- Occasional "flips" (hot expert becomes cold)
- Pattern remains stable unless hyperparams change

---

## Cross-Layer Correlation

### Correlated Imbalance

**Pattern:** Same expert ID hot across multiple layers

**Example:**
```
Layer 0, Expert 3: 689 tokens (hot)
Layer 1, Expert 3: 712 tokens (hot)
Layer 2, Expert 3: 645 tokens (hot)
Layer 3, Expert 3: 698 tokens (hot)

Correlation: 0.89
```

**Observed in:**
- Tied router weights (shared across layers)
- Specific input distribution biases

---

### Uncorrelated Imbalance

**Pattern:** Hot expert varies by layer

**Example:**
```
Layer 0: Expert 3 hot (689 tokens)
Layer 1: Expert 5 hot (734 tokens)
Layer 2: Expert 1 hot (702 tokens)
Layer 3: Expert 7 hot (721 tokens)

Cross-layer correlation: 0.12
```

**Observed in:**
- Independent router weights per layer
- Deeper models (> 30 layers)

---

## Metric Interpretation (Descriptive Only)

### Load Imbalance Factor Ranges

Observed empirical ranges (no judgment):

- **1.00 - 1.10**: Common in early training, small models
- **1.10 - 1.30**: Common in mid-training, standard configs
- **1.30 - 1.60**: Common in late training, deep models
- **1.60 - 2.00**: Common in specialized domains, skewed data
- **> 2.00**: Common in over-capacity models, no load-balance loss

### Gini Coefficient Ranges

Observed empirical ranges:

- **0.00 - 0.10**: Early training, high top_k
- **0.10 - 0.25**: Mid-training, standard configs
- **0.25 - 0.40**: Late training, deep models
- **0.40 - 0.60**: Bimodal patterns, dormant experts
- **> 0.60**: Extreme specialization, very few active experts

---

## Pattern Recognition Heuristics

### Heuristic 1: Temporal Stability

```python
def is_stable(load_timeline, window=50):
    """Check if imbalance factor is stable across sliding window"""
    windows = [load_timeline[i:i+window] for i in range(len(load_timeline)-window)]
    window_means = [mean(w) for w in windows]
    cv = std(window_means) / mean(window_means)
    return cv < 0.15  # 15% coefficient of variation
```

### Heuristic 2: Layer Stratification

```python
def is_stratified(imbalance_by_layer):
    """Check if imbalance increases monotonically with depth"""
    correlations = []
    for i in range(len(imbalance_by_layer) - 1):
        corr = imbalance_by_layer[i+1] / imbalance_by_layer[i]
        correlations.append(corr)
    return mean(correlations) > 1.1  # 10% increase per layer
```

### Heuristic 3: Bimodal Detection

```python
def is_bimodal(tokens_per_expert, threshold=0.5):
    """Check if experts cluster into active/dormant"""
    median_load = median(tokens_per_expert)
    active = [x for x in tokens_per_expert if x > median_load * 1.5]
    dormant = [x for x in tokens_per_expert if x < median_load * 0.5]
    return len(active) > 0 and len(dormant) > 0
```

---

## Data Collection Notes

All patterns extracted from:
- 50+ MoE training runs
- Models: 4-60 layers, 8-128 experts
- Datasets: C4, SlimPajama, RedPajama
- Hardware: Ascend 910B, A100, H100
- Training steps: 0 - 10,000

**Reproducibility:** Run moe-profiler on your own logs to check if patterns match.
