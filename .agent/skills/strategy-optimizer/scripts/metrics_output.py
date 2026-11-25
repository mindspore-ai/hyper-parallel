#!/usr/bin/env python3
"""
Standardized metrics output for strategy validation.

Outputs JSON metrics to stdout that can be parsed by agents/tools.
"""
import json
import sys
from typing import Dict, Any, Optional


def output_metrics(
    step: int,
    throughput_tokens_per_sec: float,
    memory_allocated_gb: float,
    mfu_percent: Optional[float] = None,
    bubble_ratio: Optional[float] = None,
    step_time_ms: Optional[float] = None,
    **kwargs
) -> None:
    """
    Output standardized metrics in JSON format.

    Args:
        step: Current training step
        throughput_tokens_per_sec: Throughput in tokens/s
        memory_allocated_gb: Peak memory allocated in GB
        mfu_percent: Model FLOPs Utilization (0-100)
        bubble_ratio: Pipeline bubble ratio (0-1)
        step_time_ms: Step time in milliseconds
        **kwargs: Additional custom metrics

    Output format (JSON to stdout):
        {
            "step": 50,
            "throughput_tokens_per_sec": 118234.5,
            "memory_allocated_gb": 61.2,
            "mfu_percent": 42.1,
            "bubble_ratio": 0.053,
            "step_time_ms": 687.3,
            ...
        }
    """
    metrics = {
        "step": step,
        "throughput_tokens_per_sec": throughput_tokens_per_sec,
        "memory_allocated_gb": memory_allocated_gb,
    }

    # Optional metrics
    if mfu_percent is not None:
        metrics["mfu_percent"] = mfu_percent

    if bubble_ratio is not None:
        metrics["bubble_ratio"] = bubble_ratio

    if step_time_ms is not None:
        metrics["step_time_ms"] = step_time_ms

    # Additional custom metrics
    metrics.update(kwargs)

    # Output to stdout
    print(json.dumps(metrics), flush=True)


def parse_metrics_from_log(log_file: str) -> Dict[str, Any]:
    """
    Parse standardized JSON metrics from log file.

    Args:
        log_file: Path to log file

    Returns:
        Dict with aggregated metrics
    """
    import statistics

    metrics_by_step = []

    with open(log_file) as f:
        for line in f:
            # Skip non-JSON lines
            if not line.strip().startswith('{'):
                continue

            try:
                metric = json.loads(line.strip())
                if "step" in metric and "throughput_tokens_per_sec" in metric:
                    metrics_by_step.append(metric)
            except json.JSONDecodeError:
                continue

    if not metrics_by_step:
        return {"error": "No metrics found"}

    # Skip warmup (first 10 steps)
    warmup_steps = 10
    if len(metrics_by_step) > warmup_steps:
        valid_metrics = metrics_by_step[warmup_steps:]
    else:
        valid_metrics = metrics_by_step

    # Aggregate
    throughputs = [m["throughput_tokens_per_sec"] for m in valid_metrics]
    memories = [m["memory_allocated_gb"] for m in valid_metrics]

    result = {
        "throughput_tokens_per_sec": statistics.mean(throughputs),
        "throughput_std": statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
        "memory_peak_gb": max(memories),
        "total_steps": len(metrics_by_step),
        "measured_steps": len(valid_metrics),
        "warmup_steps": warmup_steps,
    }

    # Optional metrics (if present)
    if "step_time_ms" in valid_metrics[0]:
        step_times = [m["step_time_ms"] for m in valid_metrics]
        result["step_time_ms"] = statistics.mean(step_times)
        result["step_time_std_ms"] = statistics.stdev(step_times) if len(step_times) > 1 else 0.0

    if "mfu_percent" in valid_metrics[0]:
        mfus = [m["mfu_percent"] for m in valid_metrics]
        result["mfu_percent"] = statistics.mean(mfus)

    if "bubble_ratio" in valid_metrics[0]:
        bubbles = [m["bubble_ratio"] for m in valid_metrics]
        result["bubble_ratio"] = statistics.mean(bubbles)

    return result


# Example usage in training script
if __name__ == "__main__":
    # Simulate training loop
    for step in range(1, 51):
        # Your training code here
        # ...

        # Calculate metrics
        throughput = 118234.5 + step * 10
        memory = 61.2 + step * 0.01
        mfu = 42.1
        bubble = 0.053
        step_time = 687.3

        # Output standardized metrics
        output_metrics(
            step=step,
            throughput_tokens_per_sec=throughput,
            memory_allocated_gb=memory,
            mfu_percent=mfu,
            bubble_ratio=bubble,
            step_time_ms=step_time,
        )

    # Parse metrics from log
    # metrics = parse_metrics_from_log("validation_run.log")
    # print(f"\nAggregated: {metrics}")
