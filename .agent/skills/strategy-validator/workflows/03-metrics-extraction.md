# Phase 3: Metrics Extraction

Parse training logs and profiler traces to extract standardized performance metrics.

## 3.1 Primary Metrics

Target metrics matching PSA estimates:

| Metric | Unit | Source | Priority |
|--------|------|--------|----------|
| **Throughput** | tokens/s, samples/s | Training log | P0 |
| **Memory peak** | GB per device | Training log + profiler | P0 |
| **MFU** | % | Computed from FLOPs + time | P0 |
| **Bubble ratio** | % (PP only) | Profiler trace | P1 |
| **Step time** | ms | Training log | P1 |
| **Communication time** | ms | Profiler trace | P2 |

## 3.2 Log Parsing

### 3.2.1 Extract Throughput

Target log lines (after warmup, steps 11-50):

```
[Step 11/50] loss=2.298, throughput=28.5 samples/s, tokens/s=116K, step_time=1.23s
[Step 12/50] loss=2.287, throughput=28.7 samples/s, tokens/s=117K, step_time=1.21s
...
[Step 50/50] loss=2.103, throughput=29.2 samples/s, tokens/s=119K, step_time=1.19s
```

**Extraction script:**

```python
import re
import numpy as np

LOG_FILE = "validation_run_20261016_143052.log"
WARMUP_STEPS = 10

def parse_throughput(log_path, warmup=10):
    pattern = re.compile(
        r'\[Step (\d+)/\d+\] '
        r'.*throughput=([\d.]+) samples/s, '
        r'tokens/s=([\d.]+)([KM]?)'
    )

    samples_per_sec = []
    tokens_per_sec = []

    with open(log_path) as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue

            step, samples_s, tokens, unit = match.groups()
            step = int(step)

            if step <= warmup:
                continue  # Skip warmup

            samples_per_sec.append(float(samples_s))

            # Parse tokens/s with K/M suffix
            tokens_val = float(tokens)
            if unit == 'K':
                tokens_val *= 1000
            elif unit == 'M':
                tokens_val *= 1_000_000
            tokens_per_sec.append(tokens_val)

    return {
        "mean_samples_per_sec": np.mean(samples_per_sec),
        "std_samples_per_sec": np.std(samples_per_sec),
        "mean_tokens_per_sec": np.mean(tokens_per_sec),
        "std_tokens_per_sec": np.std(tokens_per_sec),
        "num_samples": len(samples_per_sec)
    }

# Result:
# {
#   "mean_samples_per_sec": 28.9,
#   "std_samples_per_sec": 0.4,
#   "mean_tokens_per_sec": 118234,
#   "std_tokens_per_sec": 1632,
#   "num_samples": 40
# }
```

### 3.2.2 Extract Memory

Target log lines (logged every N steps or at peak):

```
[Memory] Rank 0: allocated=58.2GB, reserved=61.4GB, max_allocated=61.1GB
[Memory] Rank 1: allocated=58.3GB, reserved=61.5GB, max_allocated=61.2GB
```

**Extraction:**

```python
def parse_memory(log_path):
    pattern = re.compile(
        r'\[Memory\] Rank (\d+): '
        r'allocated=([\d.]+)GB, '
        r'reserved=([\d.]+)GB, '
        r'max_allocated=([\d.]+)GB'
    )

    max_allocated_per_rank = {}

    with open(log_path) as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue

            rank, allocated, reserved, max_alloc = match.groups()
            rank = int(rank)
            max_alloc = float(max_alloc)

            # Track peak across all logs for this rank
            if rank not in max_allocated_per_rank:
                max_allocated_per_rank[rank] = max_alloc
            else:
                max_allocated_per_rank[rank] = max(
                    max_allocated_per_rank[rank], max_alloc
                )

    # Average across ranks (should be similar for data parallel)
    peak_memory_gb = np.mean(list(max_allocated_per_rank.values()))
    std_memory_gb = np.std(list(max_allocated_per_rank.values()))

    return {
        "peak_memory_gb": peak_memory_gb,
        "std_memory_gb": std_memory_gb,
        "per_rank": max_allocated_per_rank
    }

# Result:
# {
#   "peak_memory_gb": 61.3,
#   "std_memory_gb": 0.2,
#   "per_rank": {0: 61.1, 1: 61.2, ...}
# }
```

### 3.2.3 Extract Step Time

```python
def parse_step_time(log_path, warmup=10):
    pattern = re.compile(r'\[Step (\d+)/\d+\].*step_time=([\d.]+)s')

    step_times_ms = []

    with open(log_path) as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue

            step, time_s = match.groups()
            step = int(step)

            if step <= warmup:
                continue

            step_times_ms.append(float(time_s) * 1000)  # Convert to ms

    return {
        "mean_step_time_ms": np.mean(step_times_ms),
        "std_step_time_ms": np.std(step_times_ms)
    }

# Result: {"mean_step_time_ms": 1203.5, "std_step_time_ms": 12.4}
```

## 3.3 Profiler Trace Analysis

Use Torch profiler JSON output (`step_15_rank_0.pt.trace.json`).

### 3.3.1 Communication Time Breakdown

Extract collective ops (AllReduce, AllGather, ReduceScatter, etc.):

```python
import json

def parse_profiler_trace(trace_path):
    with open(trace_path) as f:
        trace = json.load(f)

    # Trace format: {"traceEvents": [...]}
    events = trace.get("traceEvents", [])

    comm_ops = {
        "AllReduce": [],
        "AllGather": [],
        "ReduceScatter": [],
        "Send": [],
        "Recv": []
    }

    for event in events:
        name = event.get("name", "")
        dur = event.get("dur", 0)  # Duration in microseconds

        for op_name in comm_ops.keys():
            if op_name.lower() in name.lower():
                comm_ops[op_name].append(dur / 1000)  # Convert to ms
                break

    # Aggregate
    total_comm_time_ms = sum(sum(times) for times in comm_ops.values())

    breakdown = {
        op: {"count": len(times), "total_ms": sum(times)}
        for op, times in comm_ops.items()
        if times
    }

    return {
        "total_comm_time_ms": total_comm_time_ms,
        "breakdown": breakdown
    }

# Result:
# {
#   "total_comm_time_ms": 142.3,
#   "breakdown": {
#     "AllReduce": {"count": 96, "total_ms": 85.2},
#     "AllGather": {"count": 48, "total_ms": 32.1},
#     "ReduceScatter": {"count": 48, "total_ms": 25.0}
#   }
# }
```

### 3.3.2 Bubble Ratio (Pipeline Parallel)

If `pp > 1`, compute bubble time:

```python
def compute_bubble_ratio(trace_path, pp_degree):
    if pp_degree == 1:
        return {"bubble_ratio": 0.0, "note": "No PP"}

    with open(trace_path) as f:
        trace = json.load(f)

    events = trace.get("traceEvents", [])

    # Find compute vs idle time on each PP rank
    compute_time = 0
    idle_time = 0

    for event in events:
        cat = event.get("cat", "")
        dur = event.get("dur", 0) / 1000  # ms

        if "forward" in cat.lower() or "backward" in cat.lower():
            compute_time += dur
        elif "wait" in cat.lower() or "idle" in cat.lower():
            idle_time += dur

    total_time = compute_time + idle_time
    bubble_ratio = idle_time / total_time if total_time > 0 else 0

    return {
        "bubble_ratio": bubble_ratio,
        "idle_time_ms": idle_time,
        "compute_time_ms": compute_time
    }

# Result: {"bubble_ratio": 0.061, "idle_time_ms": 73.5, "compute_time_ms": 1130.0}
```

## 3.4 MFU from framework telemetry

Prefer the framework's measured output. Enable `training.mfu` with
`enabled` and `peak_tflops_per_device`. The framework publishes
`performance/mfu_percent`, `performance/model_tflops_per_device`, and
`performance/tokens_per_s_per_device` using its PaLM-style model-FLOP
convention. See
[MFU accounting and targets](../../perf-playbook/references/mfu-targets.md)
for MFU, executed-work HFU, and target definitions.

If telemetry is unavailable, recompute offline using the same convention.
For the standard dense-attention approximation:

```python
def model_flops_per_token(active_matmul_params, layers, attention_width, seq_len):
    """Include forward/backward work; attention_width is num_heads * head_dim."""
    return 6 * active_matmul_params + 12 * layers * attention_width * seq_len
```

- `6 × N_active` already includes forward work of approximately `2N`
  and backward work of approximately `4N`. Do not multiply it by two again.
- Keep the sequence-dependent attention term; it can be substantial for
  long sequences. Specialized attention architectures require their
  corresponding model-FLOP implementation.
- Exclude embedding lookups from `N_active`. Count routed expert matrix
  parameters in proportion to `top_k / num_experts` for the supported
  uniform expert architecture.

Always report MFU together with its accounting convention and device peak.
Different conventions can produce substantially different values for the
same run.

## 3.5 Consolidated Output

Merge all metrics into `validation_metrics.json`:

```json
{
  "strategy": {
    "dp": 4, "tp": 8, "pp": 2, "cp": 1, "ep": 1,
    "fsdp_level": "level2"
  },
  "model": {
    "name": "Qwen3-30B-A3B",
    "params": 30e9,
    "layers": 24,
    "seq_length": 4096
  },
  "hardware": {
    "device_type": "Ascend 910B",
    "device_count": 64,
    "peak_tflops": 320
  },
  "metrics": {
    "throughput": {
      "samples_per_sec": 28.9,
      "tokens_per_sec": 118234,
      "std_tokens_per_sec": 1632
    },
    "memory": {
      "peak_gb": 61.3,
      "std_gb": 0.2
    },
    "mfu": {
      "percent": 42.1,
      "actual_tflops": 134.7
    },
    "step_time": {
      "mean_ms": 1203.5,
      "std_ms": 12.4
    },
    "communication": {
      "total_ms": 142.3,
      "breakdown": {
        "AllReduce": {"count": 96, "total_ms": 85.2},
        "AllGather": {"count": 48, "total_ms": 32.1},
        "ReduceScatter": {"count": 48, "total_ms": 25.0}
      }
    },
    "bubble": {
      "ratio": 0.061,
      "idle_ms": 73.5
    }
  },
  "run_metadata": {
    "total_steps": 50,
    "warmup_steps": 10,
    "runtime_sec": 512,
    "timestamp": "2026-10-16T14:30:52Z",
    "log_file": "validation_run_20261016_143052.log"
  }
}
```

## 3.6 Error Handling

**Missing metrics:**

```python
def validate_metrics(metrics):
    required = ["throughput", "memory", "mfu", "step_time"]
    missing = [k for k in required if k not in metrics]

    if missing:
        return {
            "status": "INCOMPLETE",
            "missing_metrics": missing,
            "note": "Check log format or profiler output"
        }

    return {"status": "COMPLETE"}
```

**Outlier detection:**

```python
# Flag if std_dev > 20% of mean (unstable run)
if metrics["throughput"]["std_tokens_per_sec"] / metrics["throughput"]["tokens_per_sec"] > 0.2:
    warnings.append("High throughput variance (>20%), consider re-running")
```

## 3.7 Output for Phase 4

**Success:**
```
✓ Metrics extracted from validation_run_20261016_143052.log
✓ Profiler traces analyzed (2 files)
✓ Output: validation_metrics.json
✓ All P0 metrics available
✓ No stability warnings

Ready to execute Phase 4: Comparison Report
```

**With warnings:**
```
⚠ Metrics extracted with warnings:
  - Communication breakdown incomplete (no profiler trace for rank 0)
  - High memory variance across ranks (std=2.1GB, investigate load imbalance)
✓ P0 metrics available, proceeding to Phase 4 with caveats
```
