# Phase 3: Batch Validation

## Objective
Execute all candidate strategies in sequence, monitor for errors, and collect performance metrics.

## Inputs
- List of candidates (from Phase 2)
- Generated config files and launch scripts
- Target hardware (one node from the authorized pool)

## Execution Strategy

### Sequential Validation (Recommended)
Run candidates one at a time:
```bash
for candidate in candidate1 candidate2 candidate3; do
    echo "Running $candidate..."
    bash validation_experiment/run_${candidate}.sh

    # Check for errors
    if grep -q "Error\|OOM\|Traceback" validation_experiment/${candidate}/validation_run.log; then
        echo "✗ $candidate failed"
        # Save error for analysis
    else
        echo "✓ $candidate completed"
    fi

    # Short delay between runs
    sleep 5
done
```

**Pros**: Simple, easy to debug
**Cons**: Sequential (3-5 candidates × 7 min = 20-35 min)

### Parallel Validation (Advanced)
If multiple nodes available:
```bash
# Node 1
bash validation_experiment/run_candidate1.sh &

# Node 2
bash validation_experiment/run_candidate2.sh &

wait
```

**Pros**: Faster (all candidates in ~7 min)
**Cons**: Requires multiple nodes, harder to debug

## Monitoring

### Real-time Progress
```bash
# Terminal 1: Run validation
bash validation_experiment/run_candidate1.sh

# Terminal 2: Monitor progress
tail -f validation_experiment/candidate1/validation_run.log | grep "Training:"
```

### Error Detection
Watch for these patterns:

**1. FSDP Conflict (DP=1 issue)**:
```
RuntimeError: A view was created in no_grad mode...
```
→ Immediate rejection, skip candidate

**2. OOM**:
```
NPU out of memory
torch.cuda.OutOfMemoryError
```
→ Note memory limit, suggest reduced parallelism

**3. MoE Weight Error**:
```
ValueError: Detected fused MoE expert weights incompatible with TP sharding
```
→ EP=1 not allowed, increase EP

**4. Hang (no progress)**:
```bash
# No log output for >60 seconds
timeout 600 bash validation_experiment/run_candidate.sh
```
→ Communication deadlock, abort and analyze

**5. NaN/Inf Loss**:
```
training/foundation_loss=nan
```
→ Gradient instability, check learning rate / grad clipping

## Success Criteria

A candidate is considered **successful** if:
1. ✅ Completes 50 training steps without errors
2. ✅ Throughput > baseline (any improvement)
3. ✅ Memory within device limits (< 90% capacity)
4. ✅ Stable performance (std < 10% of mean)

## Metrics Extraction

After each successful run, extract metrics:
```python
import re, statistics

log_path = f"validation_experiment/{candidate_id}/validation_run.log"
pattern = re.compile(r'tokens/s=([\d.]+),?\s+step_time=([\d.]+)s.*memory/device_max_allocated_gb=([\d.]+)')

tokens_per_sec = []
step_times_ms = []
memory_gb = []

with open(log_path) as f:
    for line in f:
        match = pattern.search(line)
        if match:
            tps, step_time, mem = match.groups()
            tokens_per_sec.append(float(tps))
            step_times_ms.append(float(step_time) * 1000)
            memory_gb.append(float(mem))

# Use last 30 steps (exclude warmup)
tokens_per_sec = tokens_per_sec[-30:]
step_times_ms = step_times_ms[-30:]

metrics = {
    "throughput_tokens_per_sec": statistics.mean(tokens_per_sec),
    "std_tokens_per_sec": statistics.stdev(tokens_per_sec) if len(tokens_per_sec) > 1 else 0,
    "step_time_ms": statistics.mean(step_times_ms),
    "memory_per_device_gb": max(memory_gb),
    "num_samples": len(tokens_per_sec),
}

# Save
import json
with open(f"validation_experiment/{candidate_id}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
```

## Output

For each candidate, generate:
- `validation_experiment/{candidate_id}/validation_run.log` — Full training log
- `validation_experiment/{candidate_id}/metrics.json` — Extracted metrics
- Status: ✅ Success / ❌ Failed (with error type)

## Example Results

```
Candidate 1 (TP=2, CP=1, EP=2, DP=2):
  ✓ Completed 50 steps
  Throughput: 2305 tokens/s (2.13x vs baseline)
  Memory: 6.83 GB
  Status: ✅ SUCCESS

Candidate 2 (TP=2, CP=1, EP=4, DP=1):
  ✗ Failed at initialization
  Error: RuntimeError: A view was created in no_grad mode
  Status: ❌ FSDP CONFLICT

Candidate 3 (TP=4, CP=1, EP=2, DP=1):
  ✗ Failed at initialization (same error)
  Status: ❌ FSDP CONFLICT

Candidate 4 (TP=1, CP=1, EP=2, DP=4):
  ✓ Completed 50 steps
  Throughput: 4274 tokens/s (3.95x vs baseline)
  Memory: 6.18 GB
  Status: ✅ SUCCESS ⭐
```

---

**Next**: Phase 4 (Comparison & Iteration)
