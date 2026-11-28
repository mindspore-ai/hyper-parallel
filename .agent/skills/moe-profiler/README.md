# MoE Profiler

**Pure fact extraction tool for MoE training analysis.**

## Overview

The `moe-profiler` skill extracts quantitative MoE (Mixture of Experts) training metrics from logs, profiler output, and checkpoints. It generates 15 standardized artifacts for downstream analysis.

**Key principle:** This skill extracts facts only — no diagnosis, no attribution, no recommendations.

## Quick Start

```bash
# Basic usage (logs only)
python scripts/extract_moe_facts.py \
  --log-dir /path/to/training_logs \
  --output-dir ./moe_analysis

# With profiler output
python scripts/extract_moe_facts.py \
  --log-dir /path/to/logs \
  --profile-dir /path/to/profiler_output \
  --output-dir ./moe_analysis

# Full analysis (logs + profiler + checkpoint)
python scripts/extract_moe_facts.py \
  --log-dir /path/to/logs \
  --profile-dir /path/to/profiler_output \
  --checkpoint-dir /path/to/checkpoint \
  --output-dir ./moe_analysis
```

## Prerequisites

### For detailed metrics (recommended):
Set environment variables during training:
```bash
export HP_MOE_PROFILING=1
export HP_MOE_PROFILING_DIR=./moe_profile
```

This enables lightweight instrumentation hooks in `hyper_parallel/distributed/expert_parallel/`.

### Minimal requirements:
At least one of:
- Training logs with MoE metrics
- Profiler output (Ascend Profiler, PyTorch Profiler)
- Model checkpoint

## Output Artifacts (15 files)

### Index (2 files)
1. `moe_facts.json` — Machine-readable metrics index
2. `moe_facts.md` — Human-readable summary

### Load Distribution (3 files)
3. `expert_load_per_layer.csv` — Per-layer, per-expert statistics
4. `expert_load_timeline.csv` — Load over time
5. `token_distribution_summary.csv` — Aggregate statistics (imbalance factor, Gini coefficient)

### Communication (3 files)
6. `alltoall_timing.csv` — All-to-All operation timing
7. `alltoall_volume.csv` — Communication volume per layer
8. `communication_breakdown.csv` — Time decomposition (compute vs communication)

### Correctness (3 files)
9. `routing_integrity.csv` — Token conservation check
10. `gradient_flow_check.csv` — Gradient presence verification
11. `expert_coverage.csv` — Expert activation coverage

### Stragglers (2 files)
12. `rank_timing_distribution.csv` — Per-rank timing statistics
13. `straggler_analysis.csv` — Straggler identification

### Raw Data (2 files)
14. `raw_metrics.json` — Complete raw data
15. `profiler_metadata.json` — Provenance information

## Example Output

After successful extraction:
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
      "routing_integrity": "PASS",
      "gradient_flow": "PASS"
    }
  }
}
```

## Typical Workflow

1. **Run training with profiling enabled**
   ```bash
   HP_MOE_PROFILING=1 python train_moe.py
   ```

2. **Extract facts after training**
   ```bash
   python .agent/skills/moe-profiler/scripts/extract_moe_facts.py \
     --log-dir ./logs \
     --output-dir ./moe_facts
   ```

3. **Inspect artifacts**
   ```bash
   cat ./moe_facts/moe_facts.md
   less ./moe_facts/token_distribution_summary.csv
   ```

4. **Hand off to analysis tools**
   - Human expert review
   - `moe-optimizer` skill (automated optimization)
   - Custom analysis scripts

## Integration with Other Skills

### With `moe-optimizer`:
```bash
# moe-optimizer automatically invokes moe-profiler
# User just provides logs directory
```

### With `strategy-optimizer`:
```bash
# Extract MoE facts first
python scripts/extract_moe_facts.py --log-dir ./logs --output-dir ./facts

# Then use facts to guide strategy optimization
# (moe_facts.json feeds into strategy constraints)
```

## Metrics Reference

See [references/moe-metrics.md](references/moe-metrics.md) for detailed metric definitions.

### Key Metrics

**Load Imbalance Factor:**
```
Ratio of max expert load to mean expert load.
1.0 = perfect balance, >2.0 = severe imbalance
```

**All-to-All Overhead Ratio:**
```
Fraction of step time spent in All-to-All communication.
<0.1 = compute-bound, >0.5 = communication-bound
```

**Gini Coefficient:**
```
Measure of load inequality.
0.0 = uniform, 1.0 = maximal skew
```

## Limitations

- **Read-only:** Never modifies input files
- **Memory-bounded:** Streams large log files
- **No real-time monitoring:** Post-training analysis only
- **Checkpoint limitations:** Most checkpoints don't save gradients
- **Profiler dependency:** Detailed timing requires profiler output

## Troubleshooting

### No MoE metrics found
```
Error: No MoE metrics found in logs
```
**Solution:** Ensure `HP_MOE_PROFILING=1` during training.

### Missing profiler files
```
Warning: No profiler files found, skipping timing analysis
```
**Solution:** This is non-blocking. Timing artifacts will be empty but other artifacts still generated.

### Corrupted checkpoint
```
Error: Failed to load checkpoint
```
**Solution:** Provide `--log-dir` only; checkpoint validation is optional.

## Performance

- **Typical runtime:** ~30 seconds for 100 steps, 8 layers, 64 experts
- **Memory usage:** <1 GB (streams large files)
- **Disk usage:** ~10 MB output for 100-step analysis

## Development

### Adding new metrics

1. Update relevant analyzer (`_moe_load_analysis.py`, etc.)
2. Add CSV column to artifact generation
3. Update `moe-metrics.md` reference
4. Update this README

### Testing

```bash
# Run with sample data
python scripts/extract_moe_facts.py \
  --log-dir tests/fixtures/moe_logs \
  --output-dir /tmp/test_output

# Verify 15 artifacts generated
ls /tmp/test_output/*.csv | wc -l  # Should be 11
ls /tmp/test_output/*.json | wc -l  # Should be 2
ls /tmp/test_output/*.md | wc -l    # Should be 1
```

## License

Apache 2.0 (same as HyperParallel project)
