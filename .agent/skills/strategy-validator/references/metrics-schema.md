# Metrics Schema

Standardized metric definitions and extraction rules for strategy validation.

## Metric Taxonomy

### P0: Core Performance Metrics

Required for all validation runs, directly comparable to PSA estimates.

#### 1. Throughput

**Definition:** Training speed measured in tokens or samples processed per second.

```python
{
  "throughput": {
    "tokens_per_sec": float,      # Primary metric: tokens/s
    "samples_per_sec": float,      # Samples/s (= tokens_per_sec / seq_length)
    "mean": float,                 # Average over steps (excluding warmup)
    "std": float,                  # Standard deviation
    "min": float,                  # Minimum observed
    "max": float,                  # Maximum observed
    "num_samples": int,            # Number of steps measured
    "unit": "tokens/s"
  }
}
```

**Extraction rule:**
- Parse from log lines: `tokens/s=(\d+\.?\d*)([KM]?)`
- Exclude first N warmup steps (default: 10)
- Compute statistics over remaining steps
- Convert K/M suffixes: K=1e3, M=1e6

**Tolerance:** ±15% vs PSA estimate

**Example log line:**
```
[Step 11/50] loss=2.298, throughput=28.5 samples/s, tokens/s=116.7K, step_time=1.23s
```

#### 2. Memory

**Definition:** Peak GPU/NPU memory allocated per device.

```python
{
  "memory": {
    "peak_allocated_gb": float,    # Max memory allocated (primary)
    "peak_reserved_gb": float,     # Max memory reserved by allocator
    "mean_allocated_gb": float,    # Average across ranks
    "std_gb": float,               # Cross-rank standard deviation
    "per_rank": {                  # Per-rank breakdown
      "0": float,
      "1": float,
      ...
    },
    "utilization_percent": float,  # peak / device_capacity × 100
    "headroom_gb": float,          # device_capacity - peak
    "unit": "GB"
  }
}
```

**Extraction rule:**
- Parse from log: `max_allocated=(\d+\.?\d*)GB`
- Track maximum across all logged steps per rank
- Average across ranks (expect <5% variance for homogeneous workloads)

**Tolerance:** ±20% vs PSA estimate

**Example log line:**
```
[Memory] Rank 0: allocated=58.2GB, reserved=61.4GB, max_allocated=61.1GB
```

#### 3. Model FLOPs Utilization (MFU)

**Definition:** Ratio of actual compute FLOPs to theoretical peak hardware FLOPs.

```python
{
  "mfu": {
    "percent": float,              # Primary metric: MFU %
    "actual_tflops": float,        # Actual achieved TFLOPS
    "peak_tflops": float,          # Hardware peak (from known_hardware.md)
    "flops_per_step": float,       # Computed FLOPs per training step
    "step_time_sec": float,        # Mean step time used for calculation
    "unit": "percent"
  }
}
```

**Calculation:** Prefer the framework's built-in `training.mfu` telemetry,
which publishes `performance/mfu_percent` using its PaLM-style model-FLOP
convention. See
[MFU accounting and targets](../../perf-playbook/references/mfu-targets.md).

An offline recomputation must use the same convention.
`6 × N_active` already includes forward and backward work; multiplying it
by two again doubles the reported MFU incorrectly. Include the supported
attention term `12 × L × H × d_head × S` rather than dropping it.

**Prediction uncertainty:** A PSA estimate is not an acceptance threshold.
Calibrate prediction error for the model family and operating region, and
report the measured result with its declared accounting convention.

**Comparison:** Use a target relative to a measured baseline or a documented
roofline analysis. Do not impose a universal MFU range on a model family.
Sparse MoE with a matrix optimizer and dense training can have very
different useful-compute fractions.

#### 4. Step Time

**Definition:** Wall-clock time per training step (forward + backward + optimizer).

```python
{
  "step_time": {
    "mean_ms": float,              # Average step time (primary)
    "std_ms": float,               # Standard deviation
    "min_ms": float,
    "max_ms": float,
    "median_ms": float,
    "p95_ms": float,               # 95th percentile
    "unit": "milliseconds"
  }
}
```

**Extraction rule:**
- Parse from log: `step_time=(\d+\.?\d*)s`
- Convert to milliseconds
- Exclude warmup steps
- Flag if std > 10% of mean (unstable)

**Tolerance:** ±15% vs PSA estimate

### P1: Communication & Pipeline Metrics

Required when parallel dimensions active (TP/CP/EP/PP > 1).

#### 5. Communication Breakdown

**Definition:** Time spent in collective operations, extracted from profiler traces.

```python
{
  "communication": {
    "total_ms": float,             # Total comm time per step
    "percent_of_step": float,      # comm_time / step_time × 100
    "breakdown": {
      "AllReduce": {
        "count": int,              # Number of ops per step
        "total_ms": float,
        "mean_ms": float
      },
      "AllGather": {...},
      "ReduceScatter": {...},
      "Send": {...},
      "Recv": {...},
      "AllToAll": {...}            # EP only
    },
    "by_dimension": {              # Attribution
      "TP": float,                 # TP communication time
      "CP": float,
      "EP": float,
      "DP_FSDP": float
    },
    "overlap_estimate": float,     # % of comm hidden by compute
    "exposed_ms": float,           # Exposed (non-overlapped) comm time
    "unit": "milliseconds"
  }
}
```

**Extraction rule:**
- Parse Torch profiler JSON (`*.pt.trace.json`)
- Filter events by name (case-insensitive match on collective names)
- Sum durations per operation type
- Attribute to dimension based on operation pattern:
  - AllReduce × 4 per layer → TP
  - AllGather + ReduceScatter → FSDP (DP)
  - Send/Recv → PP
  - AllToAll → EP

**Tolerance:** ±30% vs PSA estimate (communication highly sensitive to network variance)

#### 6. Pipeline Bubble

**Definition:** Idle time ratio in pipeline parallel (PP > 1 only).

```python
{
  "bubble": {
    "ratio": float,                # Idle / total time (primary)
    "percent": float,              # ratio × 100
    "idle_time_ms": float,
    "compute_time_ms": float,
    "schedule": str,               # "1f1b" or "interleaved_1f1b"
    "num_microbatches": int,
    "unit": "ratio"
  }
}
```

**Extraction rule:**
- From profiler trace: categorize events as "compute" (forward/backward kernels) vs "idle" (wait/recv)
- `bubble_ratio = idle_time / (idle_time + compute_time)`
- If profiler unavailable, estimate from theoretical formula:
  ```python
  bubble_1f1b = (pp - 1) / (num_microbatches + pp - 1)
  ```

**Tolerance:** ±5 percentage points vs PSA estimate

**Assessment:**
- ≤5%: Excellent
- 5-10%: Good
- 10-20%: Acceptable (recommend interleaved schedule)
- >20%: Poor (increase microbatches or reduce PP)

### P2: Diagnostic Metrics

Optional, for deep analysis or debugging.

#### 7. Loss Trajectory

```python
{
  "loss": {
    "final": float,
    "initial": float,
    "mean": float,
    "trend": str,                  # "decreasing" | "stable" | "diverging"
    "nan_or_inf": bool             # True if any NaN/Inf detected
  }
}
```

**Use case:** Detect numerical instability (NaN/Inf indicates config issue).

#### 8. Gradient Norm

```python
{
  "grad_norm": {
    "mean": float,
    "max": float,
    "clipped_steps": int,          # Steps where grad norm exceeded max_grad_norm
    "overflow_steps": int          # FP16/BF16 overflow events
  }
}
```

**Use case:** Diagnose mixed-precision issues or exploding gradients.

#### 9. Device Utilization (from profiler)

```python
{
  "utilization": {
    "compute_percent": float,      # % time in CUDA/NPU kernels
    "memory_bandwidth_percent": float,
    "kernel_efficiency": float     # Achieved / theoretical kernel FLOPs
  }
}
```

**Use case:** Identify compute vs memory-bound workloads.

---

## Extraction Pipeline

### Step 1: Parse Training Log

```python
import re
from typing import Dict, List

def parse_training_log(log_path: str, warmup_steps: int = 10) -> Dict:
    """Extract metrics from training log."""

    metrics = {
        "throughput": {"samples": [], "tokens": []},
        "step_time": [],
        "loss": [],
        "memory": {}
    }

    patterns = {
        "throughput": re.compile(
            r'throughput=([\d.]+) samples/s, tokens/s=([\d.]+)([KM]?)'
        ),
        "step_time": re.compile(r'step_time=([\d.]+)s'),
        "loss": re.compile(r'loss=([\d.]+|nan|inf)'),
        "memory": re.compile(
            r'Rank (\d+):.*max_allocated=([\d.]+)GB'
        )
    }

    with open(log_path) as f:
        for line in f:
            # Extract step number
            step_match = re.search(r'Step (\d+)/', line)
            if step_match:
                step = int(step_match.group(1))
                if step <= warmup_steps:
                    continue  # Skip warmup

            # Throughput
            match = patterns["throughput"].search(line)
            if match:
                samples_s, tokens, unit = match.groups()
                tokens_val = float(tokens)
                if unit == 'K':
                    tokens_val *= 1000
                elif unit == 'M':
                    tokens_val *= 1_000_000

                metrics["throughput"]["samples"].append(float(samples_s))
                metrics["throughput"]["tokens"].append(tokens_val)

            # Step time
            match = patterns["step_time"].search(line)
            if match:
                metrics["step_time"].append(float(match.group(1)) * 1000)  # to ms

            # Loss
            match = patterns["loss"].search(line)
            if match:
                loss_str = match.group(1)
                if loss_str not in ["nan", "inf"]:
                    metrics["loss"].append(float(loss_str))
                else:
                    metrics["loss"].append(loss_str)

            # Memory
            match = patterns["memory"].search(line)
            if match:
                rank, mem_gb = match.groups()
                rank = int(rank)
                mem_gb = float(mem_gb)

                if rank not in metrics["memory"]:
                    metrics["memory"][rank] = []
                metrics["memory"][rank].append(mem_gb)

    return metrics
```

### Step 2: Compute Statistics

```python
import numpy as np

def compute_statistics(metrics: Dict) -> Dict:
    """Compute aggregated statistics."""

    result = {}

    # Throughput
    tokens_per_sec = metrics["throughput"]["tokens"]
    samples_per_sec = metrics["throughput"]["samples"]

    result["throughput"] = {
        "tokens_per_sec": float(np.mean(tokens_per_sec)),
        "samples_per_sec": float(np.mean(samples_per_sec)),
        "std_tokens_per_sec": float(np.std(tokens_per_sec)),
        "min_tokens_per_sec": float(np.min(tokens_per_sec)),
        "max_tokens_per_sec": float(np.max(tokens_per_sec)),
        "num_samples": len(tokens_per_sec)
    }

    # Step time
    step_times = metrics["step_time"]
    result["step_time"] = {
        "mean_ms": float(np.mean(step_times)),
        "std_ms": float(np.std(step_times)),
        "min_ms": float(np.min(step_times)),
        "max_ms": float(np.max(step_times)),
        "median_ms": float(np.median(step_times)),
        "p95_ms": float(np.percentile(step_times, 95))
    }

    # Memory (across ranks)
    memory_per_rank = {
        rank: max(mem_list) for rank, mem_list in metrics["memory"].items()
    }
    peak_memory_values = list(memory_per_rank.values())

    result["memory"] = {
        "peak_allocated_gb": float(np.mean(peak_memory_values)),
        "std_gb": float(np.std(peak_memory_values)),
        "per_rank": memory_per_rank
    }

    # Loss
    losses = [l for l in metrics["loss"] if isinstance(l, float)]
    result["loss"] = {
        "final": float(losses[-1]) if losses else None,
        "initial": float(losses[0]) if losses else None,
        "mean": float(np.mean(losses)) if losses else None,
        "nan_or_inf": any(isinstance(l, str) for l in metrics["loss"])
    }

    return result
```

### Step 3: Parse Profiler Trace

```python
import json

def parse_profiler_trace(trace_path: str) -> Dict:
    """Extract communication metrics from profiler JSON."""

    with open(trace_path) as f:
        trace = json.load(f)

    events = trace.get("traceEvents", [])

    comm_ops = {
        "AllReduce": [],
        "AllGather": [],
        "ReduceScatter": [],
        "Send": [],
        "Recv": [],
        "AllToAll": []
    }

    for event in events:
        name = event.get("name", "").lower()
        dur_us = event.get("dur", 0)

        for op_name, op_list in comm_ops.items():
            if op_name.lower() in name:
                op_list.append(dur_us / 1000)  # Convert to ms
                break

    # Aggregate
    breakdown = {}
    total_ms = 0

    for op_name, durations in comm_ops.items():
        if durations:
            breakdown[op_name] = {
                "count": len(durations),
                "total_ms": sum(durations),
                "mean_ms": sum(durations) / len(durations)
            }
            total_ms += sum(durations)

    return {
        "total_ms": total_ms,
        "breakdown": breakdown
    }
```

### Step 4: Consolidate Output

```python
def generate_metrics_json(log_path, profiler_path, model_config, hardware_config):
    """Generate complete metrics JSON."""

    # Parse raw data
    raw_metrics = parse_training_log(log_path)
    stats = compute_statistics(raw_metrics)

    if profiler_path:
        comm_metrics = parse_profiler_trace(profiler_path)
        stats["communication"] = comm_metrics

    # Compute MFU
    mfu = compute_mfu(
        model_params=model_config["params"],
        seq_length=model_config["seq_length"],
        global_batch_size=model_config["batch_size"],
        step_time_ms=stats["step_time"]["mean_ms"],
        peak_tflops=hardware_config["peak_tflops"]
    )
    stats["mfu"] = mfu

    # Add metadata
    output = {
        "metrics": stats,
        "model": model_config,
        "hardware": hardware_config,
        "run_metadata": {
            "log_file": log_path,
            "profiler_trace": profiler_path,
            "timestamp": datetime.utcnow().isoformat()
        }
    }

    return output
```

---

## Validation Rules

### Metric Completeness Check

```python
def validate_metrics_completeness(metrics: Dict) -> Dict:
    """Check if all P0 metrics are present."""

    required_p0 = ["throughput", "memory", "mfu", "step_time"]
    missing = [m for m in required_p0 if m not in metrics]

    warnings = []

    if missing:
        return {
            "status": "INCOMPLETE",
            "missing_metrics": missing
        }

    # Check data quality
    if metrics["throughput"]["num_samples"] < 10:
        warnings.append("Low sample count for throughput (<10 steps)")

    if metrics["step_time"]["std_ms"] / metrics["step_time"]["mean_ms"] > 0.2:
        warnings.append("High step time variance (>20%), unstable run")

    if metrics["loss"].get("nan_or_inf"):
        warnings.append("NaN/Inf detected in loss, numerical instability")

    return {
        "status": "COMPLETE" if not warnings else "COMPLETE_WITH_WARNINGS",
        "warnings": warnings
    }
```

### Tolerance Assessment

```python
def assess_tolerance(actual: float, estimate: float, tolerance_pct: float) -> Dict:
    """Check if metric is within tolerance."""

    delta_pct = (actual - estimate) / estimate * 100
    within_tolerance = abs(delta_pct) <= tolerance_pct

    return {
        "actual": actual,
        "estimate": estimate,
        "delta_percent": round(delta_pct, 1),
        "within_tolerance": within_tolerance,
        "assessment": "✓ PASS" if within_tolerance else "⚠ DEVIATION"
    }
```

---

## Output Format

Final `validation_metrics.json`:

```json
{
  "metrics": {
    "throughput": {...},
    "memory": {...},
    "mfu": {...},
    "step_time": {...},
    "communication": {...},
    "bubble": {...}
  },
  "model": {...},
  "hardware": {...},
  "run_metadata": {...},
  "validation": {
    "status": "COMPLETE",
    "warnings": []
  }
}
```


---

## MoE-Specific Metrics (MoE Models Only)

**Load when `model.is_moe == True`**

For detailed definitions, formulas, and collection methods, see [moe-metrics.md](moe-metrics.md).

### Expert Load Metrics

```json
{
  "moe_metrics": {
    "expert_load": {
      "mean_load_imbalance_factor": 1.42,
      "max_load_imbalance_factor": 1.68,
      "mean_load_std": 0.24,
      "interpretation": "Good (factor 1.42)"
    }
  }
}
```

### Communication Metrics

```json
{
  "moe_metrics": {
    "communication": {
      "mean_alltoall_ratio": 0.22,
      "max_alltoall_ratio": 0.28,
      "alltoall_efficiency": 0.68,
      "interpretation": "Moderate overhead (22%)"
    }
  }
}
```

### Routing Efficiency

```json
{
  "moe_metrics": {
    "routing": {
      "mean_routing_entropy": 0.87,
      "mean_expert_coverage": 0.96,
      "dropped_token_ratio": 0.003,
      "interpretation": "Excellent"
    }
  }
}
```

**Complete specification:** [moe-metrics.md](moe-metrics.md)
