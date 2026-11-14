---
name: moe-profiler
description: >
  Extract verifiable MoE training facts: expert load distribution,
  All-to-All communication overhead, routing integrity, gradient flow.
  NO diagnosis, NO attribution, NO optimization suggestions.
  Pure fact extraction tool for downstream analysis.
---

Executable evidence contract: [supported inputs and missing evidence](references/input-contract.md).
Missing evidence is null/UNVERIFIED; extraction completion does not imply correctness PASS.


# MoE Profiler

**Pure fact extraction tool** — extracts quantitative MoE training metrics without interpretation.

## When to use

- User provides MoE training logs, profiler output, or checkpoints
- Need baseline MoE metrics before optimization
- Debugging MoE load imbalance, communication overhead, or routing issues
- **Not for**: diagnosis, root-cause analysis, or optimization recommendations (→ `moe-optimizer`)

## Trigger patterns

- "MoE profiling"
- "Inspect expert-load distribution"
- "Measure All-to-All communication overhead"
- "Inspect MoE routing"
- "Summarize expert utilization"

## Prerequisites

At least one of:
- **Training logs** with MoE metrics (tokens_per_expert, routing scores)
- **Profiler output** (e.g., Ascend Profiler, PyTorch Profiler)
- **Checkpoint directory** with MoE module state_dict

Optional but recommended:
- Environment variable `HP_MOE_PROFILING=1` during training (enables detailed hooks)
- MoE profiling output directory: `HP_MOE_PROFILING_DIR` (default: `./moe_profile`)

## Execution flow

```text
Phase 1: Input Discovery
  - Locate training logs (*.log, stdout)
  - Locate profiler output (ASCEND_PROFILER_OUTPUT, torch_profiler/)
  - Locate checkpoints (*.pt, checkpoint-*/)
  → Validate at least one input source exists

Phase 2: Extract MoE Facts
  - Run extraction script with discovered inputs
  - Parse logs for MoE-specific metrics
  - Analyze profiler traces for All-to-All timing
  - Check checkpoint for routing/gradient integrity

Phase 3: Validate Output
  - Verify all 15 fixed artifacts generated
  - Check moe_facts.json structure
  - Ensure numerical consistency

Phase 4: Deliver
  - Output directory + moe_facts.json index
  - Print summary to stdout (for parsing)
  - Hand off to human expert or moe-optimizer
```

## Command

```bash
python <skill-dir>/scripts/extract_moe_facts.py \
  --log-dir <training-logs> \
  --profile-dir <profiler-output> \
  --checkpoint-dir <checkpoint> \
  --output-dir <facts-output>
```

**Arguments:**
- `--log-dir`: Directory containing training logs (searches recursively for *.log)
- `--profile-dir`: Profiler output directory (optional)
- `--checkpoint-dir`: Model checkpoint directory (optional)
- `--output-dir`: Where to write 15 fixed artifacts (default: `./moe_facts_output`)
- `--steps`: Number of steps to analyze (default: all available)
- `--ranks`: Comma-separated rank IDs to analyze (default: all)

**Minimal example:**
```bash
python scripts/extract_moe_facts.py \
  --log-dir /path/to/training_logs \
  --output-dir ./moe_analysis
```

## Fixed Output Artifacts (15 files)

### Index
1. **`moe_facts.json`** — Global metrics index (structured, machine-readable)
2. **`moe_facts.md`** — Human-readable summary for experts

### Load Distribution (3 files)
3. **`expert_load_per_layer.csv`** — Per-layer expert load distribution
   ```csv
   layer_id,expert_id,mean_tokens,std_tokens,min_tokens,max_tokens,utilization_ratio
   0,0,245.3,12.1,198,289,0.82
   0,1,312.7,18.9,267,354,1.05
   ```

4. **`expert_load_timeline.csv`** — Load over time (sliding window)
   ```csv
   step,layer_id,expert_id,tokens_assigned,routing_score_mean
   10,0,0,251,0.0823
   10,0,1,308,0.1045
   ```

5. **`token_distribution_summary.csv`** — Statistical summary
   ```csv
   layer_id,total_tokens,num_experts,mean_per_expert,std_per_expert,load_imbalance_factor,gini_coefficient
   0,4096,8,512.0,87.3,1.68,0.24
   ```

### Communication Analysis (3 files)
6. **`alltoall_timing.csv`** — All-to-All operation timing
   ```csv
   step,layer_id,direction,duration_ms,volume_mb,bandwidth_gbps,ep_size
   10,0,dispatch,12.3,4.2,2.73,4
   10,0,combine,11.8,4.2,2.85,4
   ```

7. **`alltoall_volume.csv`** — Communication volume statistics
   ```csv
   layer_id,total_dispatch_mb,total_combine_mb,tokens_exchanged,avg_token_size_bytes
   0,168.4,168.4,4096,42
   ```

8. **`communication_breakdown.csv`** — Time breakdown
   ```csv
   step,compute_ms,alltoall_ms,other_comm_ms,alltoall_ratio,total_step_time_ms
   10,142.5,24.1,8.3,0.138,175.0
   ```

### Correctness Checks (3 files)
9. **`routing_integrity.csv`** — Token conservation check
   ```csv
   layer_id,step,input_tokens,output_tokens,tokens_matched,integrity_status
   0,10,4096,4096,4096,PASS
   ```

10. **`gradient_flow_check.csv`** — Gradient presence verification
    ```csv
    layer_id,expert_id,param_name,has_grad,grad_norm,grad_zero_ratio
    0,0,w1.weight,true,0.0234,0.0
    0,1,w1.weight,true,0.0189,0.0
    ```

11. **`expert_coverage.csv`** — Expert activation coverage
    ```csv
    layer_id,expert_id,steps_activated,activation_ratio,zero_assignment_count
    0,0,50,1.0,0
    0,7,23,0.46,27
    ```

### Straggler Detection (2 files)
12. **`rank_timing_distribution.csv`** — Per-rank timing
    ```csv
    rank,mean_step_time_ms,std_ms,min_ms,max_ms,straggler_score
    0,174.2,8.3,162.1,198.5,1.02
    1,175.8,9.1,163.4,203.7,1.03
    7,212.4,15.6,189.2,251.3,1.24
    ```

13. **`straggler_analysis.csv`** — Straggler identification
    ```csv
    rank,is_straggler,slowdown_ratio,suspected_cause,expert_load_variance
    7,true,1.22,high_load_variance,2.34
    ```

### Raw Data (2 files)
14. **`raw_metrics.json`** — Complete raw data (for reproducibility)
    ```json
    {
      "steps": [...],
      "layers": [...],
      "experts": [...],
      "ranks": [...]
    }
    ```

15. **`profiler_metadata.json`** — Provenance information
    ```json
    {
      "extraction_time": "2024-01-20T14:30:22",
      "input_sources": {
        "log_dir": "/path/to/logs",
        "log_files": ["rank0.log", "rank1.log"],
        "profile_dir": null,
        "checkpoint_dir": null
      },
      "model_config": {
        "num_layers": 4,
        "num_experts": 8,
        "top_k": 2
      },
      "hardware": {
        "world_size": 8,
        "ep_size": 4,
        "device_type": "Ascend910B"
      },
      "extractor_version": "1.0.0"
    }
    ```

## References (on-demand)

Load these only when interpreting specific metrics or patterns:

- [references/moe-metrics.md](references/moe-metrics.md) — Detailed metric definitions
- [references/load-imbalance-patterns.md](references/load-imbalance-patterns.md) — Known imbalance patterns
- [references/communication-overhead-analysis.md](references/communication-overhead-analysis.md) — Comm cost analysis

## Responsibility Boundary

### ✅ This skill DOES:
- Extract quantitative facts from logs/profiles/checkpoints
- Compute statistics (mean, std, min, max, ratios)
- Check basic invariants (token conservation, gradient presence)
- Generate standardized CSV/JSON outputs

### ❌ This skill DOES NOT:
- Make threshold judgments (e.g., "load imbalance is severe")
- Diagnose root causes (e.g., "bottleneck is All-to-All")
- Attribute problems (e.g., "caused by ep_size too large")
- Provide optimization recommendations (e.g., "reduce ep_size to 4")
- Compare against baselines or targets

**Handoff:** All 15 artifacts → performance expert or `moe-optimizer` skill for interpretation.

## Output Format (stdout)

After successful extraction, print structured summary:

```json
{
  "status": "SUCCESS",
  "output_dir": "./moe_analysis",
  "artifacts_generated": 15,
  "summary": {
    "steps_analyzed": 50,
    "layers_analyzed": 4,
    "experts_per_layer": 8,
    "ranks_analyzed": 8,
    "key_metrics": {
      "mean_load_imbalance_factor": 1.68,
      "mean_alltoall_ratio": 0.138,
      "max_straggler_slowdown": 1.22,
      "routing_integrity": "PASS",
      "gradient_flow": "PASS"
    }
  }
}
```

On failure:
```json
{
  "status": "FAILED",
  "error": "No MoE metrics found in logs",
  "input_sources_checked": [
    "/path/to/logs/*.log",
    "/path/to/profile"
  ],
  "suggestion": "Ensure HP_MOE_PROFILING=1 during training"
}
```

## Safety & Limitations

- **Read-only**: Never modifies input files
- **Idempotent**: Can re-run on same inputs
- **No network**: Purely local file processing
- **Memory-bounded**: Streams large log files (no full load)
- **Time complexity**: O(steps × layers × experts) for log parsing
- **Typical runtime**: ~30 seconds for 100 steps, 8 layers, 64 experts

**Abort conditions:**
- No valid input sources found
- Corrupted log files (partial lines, binary data)
- Inconsistent metadata (mismatched world_size, layer count)

**Warnings (non-blocking):**
- Missing profiler output (skips timing analysis)
- Missing checkpoint (skips gradient checks)
- Sparse MoE metrics in logs (HP_MOE_PROFILING not enabled)
