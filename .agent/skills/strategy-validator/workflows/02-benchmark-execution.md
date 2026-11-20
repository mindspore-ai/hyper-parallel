# Phase 2: Benchmark Execution

Launch distributed training job and monitor for completion, OOM, hangs, or errors.

## 2.1 Device Availability Check

Before launching, verify hardware state:

```bash
# For Ascend NPU
npu-smi info | grep "Health Status" | wc -l
# Expected: {{ device_count }} healthy devices

# For NVIDIA GPU
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | wc -l
# Expected: {{ device_count }} devices

# Check for existing processes
pgrep -af "train_lm.py|torchrun" && echo "WARNING: Training processes detected"
```

**Abort if:**
- Available devices < required `device_count`
- Existing training processes detected (resource conflict)
- Device health check fails

## 2.2 Launch Distributed Job

Execute generated script from Phase 1:

```bash
# Set monitoring environment
export VALIDATION_START_TIME=$(date +%s)
export VALIDATION_LOG="validation_run_$(date +%Y%m%d_%H%M%S).log"

# Launch with timeout protection (15 min max for 50 steps)
timeout 900 bash scripts/run_validation_TIMESTAMP.sh 2>&1 | tee "$VALIDATION_LOG"

EXIT_CODE=${PIPESTATUS[0]}
echo "Exit code: $EXIT_CODE" >> "$VALIDATION_LOG"
```

### 2.2.1 Monitoring Signals

During execution, watch for:

**Progress indicators (expected every 5-10s):**
```
[Step 10/50] loss=2.345, throughput=28.3 samples/s, tokens/s=115K
[Step 11/50] loss=2.298, throughput=28.5 samples/s, tokens/s=116K
```

**Memory allocation (first 5 steps):**
```
[Memory] Rank 0: allocated=58.2GB, reserved=61.4GB, max_allocated=61.1GB
```

**Profiler checkpoints (steps 15, 45):**
```
[Profiler] Step 15 trace saved to: ./profiler_traces/step_15_rank_*.pt.trace.json
```

### 2.2.2 Failure Modes

**OOM Detection:**
```bash
grep -E "CUDA out of memory|NPU out of memory|torch.OutOfMemoryError" "$VALIDATION_LOG"
# If found → immediate abort, report OOM at step X
```

**Hang Detection:**
```bash
# No progress for >60s
tail -F "$VALIDATION_LOG" | while read line; do
  if [[ $line =~ "Step" ]]; then
    LAST_PROGRESS=$(date +%s)
  fi
  NOW=$(date +%s)
  if (( NOW - LAST_PROGRESS > 60 )); then
    echo "HANG detected: no progress for 60s"
    pkill -9 -f train_lm.py
    exit 1
  fi
done
```

**Loss Divergence:**
```bash
grep -E "loss=nan|loss=inf" "$VALIDATION_LOG"
# If found → gradient instability, report with LR/batch size
```

**Communication Timeout:**
```bash
grep -E "HCCL timeout|NCCL timeout|ProcessGroupNCCL" "$VALIDATION_LOG"
# Common in CP/TP with misconfigured mesh
```

## 2.3 Live Metrics Streaming (Optional)

For real-time monitoring, parse log and emit structured events:

```python
# monitor_validation.py
import re
import sys

LOG_PATTERN = re.compile(
    r'\[Step (\d+)/(\d+)\] loss=([\d.]+), '
    r'throughput=([\d.]+) samples/s, tokens/s=([\d.]+[KM]?)'
)

for line in sys.stdin:
    match = LOG_PATTERN.search(line)
    if match:
        step, total, loss, samples_s, tokens_s = match.groups()
        print(json.dumps({
            "step": int(step),
            "total_steps": int(total),
            "loss": float(loss),
            "samples_per_sec": float(samples_s),
            "tokens_per_sec": parse_throughput(tokens_s),
            "timestamp": time.time()
        }))
```

Run alongside training:
```bash
tail -F "$VALIDATION_LOG" | python monitor_validation.py > live_metrics.jsonl
```

## 2.4 Completion Criteria

### Success:
```
✓ All 50 steps completed
✓ Exit code 0
✓ Final checkpoint saved
✓ Profiler traces written
✓ No OOM/hang/timeout errors
```

### Partial Success (Recoverable):
```
⚠ Completed 50 steps but with warnings:
  - Gradient overflow (AMP): expected with mixed precision
  - Minor HCCL retries: transient network issue
→ Proceed to metrics extraction with caveat
```

### Failure (Non-Recoverable):
```
✗ OOM at step 12 → Strategy rejected (memory estimate too optimistic)
✗ Hang at step 8 → Communication deadlock (likely CP/PP boundary issue)
✗ Loss NaN at step 3 → Gradient instability (LR too high or FP16 underflow)
✗ Timeout (>15 min) → Throughput far below estimate, abort
```

## 2.5 Cleanup

After run (success or failure):

```bash
# Kill any lingering processes
pkill -9 -f "train_lm.py|torchrun"

# Archive logs
mkdir -p validation_logs/$(date +%Y%m%d)
mv validation_run_*.log validation_logs/$(date +%Y%m%d)/

# Clear HCCL/NCCL cache (if needed for next run)
rm -rf /tmp/hccl_* /tmp/nccl_*

# Device reset (Ascend only, if OOM)
npu-smi reset -d 0-63
```

## 2.6 Output for Phase 3

**Success artifacts:**
```
validation_run_20261016_143052.log          # Full training log
live_metrics.jsonl                          # Timestamped step metrics
profiler_traces/step_15_rank_0.pt.trace.json  # Torch profiler data
profiler_traces/step_45_rank_0.pt.trace.json
checkpoint_step_50/                         # Final model state (optional)
```

**Failure report (if applicable):**
```json
{
  "status": "FAILED",
  "failure_mode": "OOM",
  "failed_at_step": 12,
  "error_message": "torch.cuda.OutOfMemoryError: Tried to allocate 8.00 GiB...",
  "peak_memory_gb": 68.4,
  "estimated_memory_gb": 58.0,
  "delta_percent": 17.9
}
```

**Summary to user:**

```
✓ Benchmark execution completed successfully
✓ Runtime: 8m 32s (50 steps)
✓ No OOM/hang/timeout errors
✓ Logs: validation_run_20261016_143052.log
✓ Profiler traces: 2 files generated

Ready to execute Phase 3: Metrics Extraction
```

## 2.7 Multi-Candidate Execution

For Top-3 validation, run sequentially with inter-run cleanup:

```bash
for candidate in candidate1 candidate2 candidate3; do
  echo "=== Running $candidate ==="

  # Pre-flight check
  ./check_devices.sh || { echo "Devices not ready"; exit 1; }

  # Execute
  timeout 900 bash scripts/run_validation_${candidate}.sh 2>&1 | \
    tee validation_${candidate}.log

  EXIT_CODE=${PIPESTATUS[0]}

  # Record outcome
  echo "$candidate: exit_code=$EXIT_CODE" >> validation_summary.txt

  # Cleanup before next run
  pkill -9 -f train_lm.py
  sleep 10  # Allow devices to release

  # Early abort on catastrophic failure
  if grep -q "CUDA error: device-side assert" validation_${candidate}.log; then
    echo "Device error detected, aborting remaining candidates"
    break
  fi
done
```

**Result:**
```
candidate1: exit_code=0 ✓
candidate2: exit_code=137 (OOM) ✗
candidate3: exit_code=0 ✓
```

Proceed to Phase 3 only for successful runs.
