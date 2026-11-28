# Phase 1: Baseline Analysis

## Objective
Extract performance metrics from existing baseline training to establish comparison reference.

## Inputs
- Baseline training log (e.g., `output/training_demo/run_parallel_online.log`)
- Baseline configuration (e.g., `train_parallel_online.yaml`)

## Tasks

### 1. Locate baseline artifacts
```bash
# Find most recent training log
find output/training_demo -name "*.log" -type f -mtime -7

# Confirm config file
ls -lh examples/training_demo/qwen3_moe/train_parallel_online.yaml
```

### 2. Extract configuration
Parse YAML to identify:
- `accelerator`: tp_size, cp_size, ep_size, pp_size
- `fsdp_config`: dp_shard_size, edp_shard_size
- `training`: global_batch_size, micro_batch_size, train_iters
- `data_transform`: max_seq_len

Calculate derived metrics:
- Total devices = TP × CP × EP × PP × DP
- DP size = total_devices / (TP × CP × EP × PP)

### 3. Extract performance metrics
Parse training log using regex:
```python
pattern = r'tokens/s=([\d.]+),?\s+step_time=([\d.]+)s.*memory/device_max_allocated_gb=([\d.]+)'
```

Extract:
- `throughput_tokens_per_sec`: Mean of last 30-50 steps (after warmup)
- `std_tokens_per_sec`: Standard deviation (for stability check)
- `step_time_ms`: Mean step time in milliseconds
- `memory_per_device_gb`: Peak memory usage

### 4. Identify bottlenecks
Analyze step time components:
- **Communication-bound**: Step time >> expected compute time
  - Indicators: High TP/CP degree, short sequences, small models
  - Solution: Reduce TP/CP

- **Computation-bound**: Step time ≈ expected compute time
  - Indicators: Large models, long sequences, low TP/CP
  - Solution: May already be optimal, or increase parallelism

- **Memory-bound**: Near OOM, frequent cache clearing
  - Indicators: memory_allocated > 90% of device capacity
  - Solution: Increase DP (FSDP sharding), reduce TP/PP

### 5. Generate baseline report
```json
{
  "baseline": {
    "config": {
      "tp": 2, "cp": 2, "ep": 2, "pp": 1, "dp": 2,
      "seq_len": 128,
      "global_batch_size": 8,
      "model_layers": 4
    },
    "metrics": {
      "throughput_tokens_per_sec": 1081.4,
      "std_tokens_per_sec": 31.4,
      "step_time_ms": 722.8,
      "memory_per_device_gb": 6.81
    },
    "bottleneck": "communication",
    "evidence": "CP=2 for seq_len=128 introduces ~400ms overhead"
  }
}
```

## Outputs
- `validation_experiment/baseline_metrics.json`
- Bottleneck diagnosis (communication / computation / memory)
- Recommended optimization directions

## Example (from Qwen3-30B-A3B experiment)

**Baseline**: TP=2, CP=2, EP=2, DP=2
- Throughput: 1081.4 tokens/s
- Step time: 722.8 ms
- Memory: 6.81 GB

**Bottleneck**: Communication (CP=2 for short seq_len=128)

**Recommendation**: Remove CP, potentially reduce TP

---

**Next**: Phase 2 (Candidate Generation)
