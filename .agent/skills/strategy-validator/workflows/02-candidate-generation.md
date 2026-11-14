# Phase 2: Candidate Generation

## Objective
Generate 3-5 candidate strategies based on baseline analysis, model characteristics, and learned heuristics.

## Inputs
- Baseline metrics (from Phase 1)
- Model architecture:
  - `num_layers`: Number of transformer layers
  - `hidden_size`, `num_attention_heads`, `intermediate_size`
  - `is_moe`: Whether model uses MoE
- Hardware:
  - `device_count`: Total available devices (e.g., 8)
  - `device_memory_gb`: Memory per device (e.g., 64 GB)
- Training config:
  - `seq_len`: Sequence length
  - Target improvement (default: 1.2x)

## Strategy Generation Rules

### Rule 1: Model Size → TP
```python
if num_layers <= 10:
    tp_candidates = [1, 2]  # Prefer TP=1
elif num_layers <= 30:
    tp_candidates = [2, 4]
else:
    tp_candidates = [4, 8]
```

### Rule 2: Sequence Length → CP
```python
if seq_len < 512:
    cp = 1  # Always 1 for short sequences
elif seq_len < 2048:
    cp_candidates = [1, 2]
else:
    cp_candidates = [2, 4, 8]
```

### Rule 3: MoE → EP
```python
if is_moe:
    ep >= 2  # Required by framework
    # EP should divide num_experts evenly
else:
    ep = 1
```

### Rule 4: FSDP → DP
```python
# Always keep DP >= 2 (avoid FSDP conflicts)
dp_min = 2

# DP = total_devices / (TP × CP × EP × PP)
dp = device_count // (tp * cp * ep * pp)
if dp < dp_min:
    # Reduce other dimensions
    pass
```

### Rule 5: Baseline Optimization
Based on bottleneck from Phase 1:

**Communication-bound**:
- Remove CP if present
- Reduce TP if model is small
- Increase DP (reduce other dimensions)

**Computation-bound**:
- May already be optimal
- Try reducing DP, increasing TP (if not maxed)

**Memory-bound**:
- Increase DP (enable more FSDP sharding)
- Reduce TP/PP (reduce activation replication)

## Candidate Generation Algorithm

```python
def generate_candidates(baseline, model_info, hardware, target_improvement=1.2):
    candidates = []

    # Candidate 1: Remove inefficient dimensions
    if baseline.cp > 1 and model_info.seq_len < 512:
        candidates.append({
            "id": "remove_cp",
            "name": "Remove CP (communication reduction)",
            "strategy": {
                "tp": baseline.tp,
                "cp": 1,  # Remove CP
                "ep": baseline.ep,
                "pp": baseline.pp,
            },
            "expected_improvement": "1.5-2.0x (if CP was bottleneck)",
        })

    # Candidate 2: Minimize TP
    if baseline.tp > 1 and model_info.num_layers <= 10:
        dp = hardware.device_count // (1 * 1 * baseline.ep * 1)
        if dp >= 2:  # Satisfy DP constraint
            candidates.append({
                "id": "minimal_tp",
                "name": "TP=1 (eliminate TP communication)",
                "strategy": {
                    "tp": 1,
                    "cp": 1,
                    "ep": baseline.ep,
                    "pp": 1,
                },
                "dp": dp,
                "expected_improvement": "1.5-2.5x (if TP was bottleneck)",
            })

    # Candidate 3: Increase parallelism (if computation-bound)
    if baseline.bottleneck == "computation":
        new_tp = min(baseline.tp * 2, 8)
        dp = hardware.device_count // (new_tp * baseline.cp * baseline.ep * baseline.pp)
        if dp >= 2:
            candidates.append({
                "id": "increase_tp",
                "name": f"TP={new_tp} (more parallelism)",
                "strategy": {
                    "tp": new_tp,
                    "cp": baseline.cp,
                    "ep": baseline.ep,
                    "pp": baseline.pp,
                },
                "expected_improvement": "1.1-1.3x",
            })

    # Candidate 4: Conservative (minor tweaks)
    # ... additional candidates based on specific patterns

    return candidates[:5]  # Limit to top 5
```

## Outputs

Generate for each candidate:
1. **Config file**: `validation_experiment/{candidate_id}_config.yaml`
2. **Launch script**: `validation_experiment/run_{candidate_id}.sh`
3. **Metadata**: Expected improvement, rationale, risk level

Example output:
```json
{
  "candidates": [
    {
      "id": "candidate1_remove_cp",
      "name": "Remove CP",
      "strategy": {"tp": 2, "cp": 1, "ep": 2, "pp": 1, "dp": 2},
      "expected_tokens_per_sec": 1500,
      "rationale": "seq_len=128 too short for CP=2, communication overhead dominates",
      "risk": "low"
    },
    {
      "id": "candidate2_minimal_tp",
      "name": "Minimal TP",
      "strategy": {"tp": 1, "cp": 1, "ep": 2, "pp": 1, "dp": 4},
      "expected_tokens_per_sec": 2000,
      "rationale": "4-layer model, TP AllReduce is bottleneck",
      "risk": "medium"
    }
  ]
}
```

## Example (from experiment)

**Baseline**: TP=2, CP=2, EP=2, DP=2 (1081 tokens/s, communication-bound)

**Generated Candidates**:
1. TP=2, CP=1, EP=2, DP=2 — Remove CP
2. TP=2, CP=1, EP=4, DP=1 — Increase EP (failed: DP=1 issue)
3. TP=4, CP=1, EP=2, DP=1 — Increase TP (failed: DP=1 issue)
4. TP=1, CP=1, EP=2, DP=4 — Minimal TP (winner: 4274 tokens/s)

---

**Next**: Phase 3 (Batch Validation)
