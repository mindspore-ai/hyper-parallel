# Benchmark Recipes

Standard workloads for validating parallel strategies.

## Design Principles

1. **Reproducibility**: Fixed seed, synthetic data, deterministic operations
2. **Speed**: 50 steps (~5-10 min) sufficient for throughput/memory measurement
3. **Coverage**: Representative of real training (forward + backward + optimizer step)
4. **Profiling**: Enable trace collection at specific checkpoints

## Standard Recipe: Language Model Pre-Training

### Target Models

| Model | Params | Layers | Hidden | Heads | KV Heads | FFN Dim | Validation Time |
|-------|--------|--------|--------|-------|----------|---------|-----------------|
| Qwen3-0.5B | 0.5B | 24 | 896 | 14 | 2 | 4864 | ~2 min |
| Qwen3-1.8B | 1.8B | 24 | 2048 | 16 | 2 | 11008 | ~3 min |
| Qwen3-7B | 7B | 28 | 4096 | 32 | 4 | 11008 | ~5 min |
| Qwen3-14B | 14B | 40 | 5120 | 40 | 8 | 13824 | ~7 min |
| Qwen3-30B-A3B (cropped) | ~12B | 24 | 8192 | 64 | 8 | 29568 | ~8 min |
| Qwen3-30B-A3B (full) | 30B | 60 | 8192 | 64 | 8 | 29568 | ~15 min |

**Cropping for speed:** Reduce layers to 24-40% of original to get 2-3x faster runs while preserving memory/comm characteristics.

### Workload Parameters

```yaml
benchmark_recipe: lm_pretraining

model:
  model_name_or_path: /path/to/Qwen3-30B-A3B
  crop_layers: 24  # Reduce from 60 for fast validation

data:
  dataset_type: synthetic
  seed: 42

  # Synthetic data config
  vocab_size: 152064  # Qwen3 vocab
  sequence_length: 4096
  num_samples: 1000  # Infinite iterator for benchmark

training:
  max_steps: 50
  warmup_steps: 10

  # Batch sizes
  per_device_train_batch_size: 1  # Micro-batch
  global_batch_size: 128  # Adjust based on dp × num_microbatches

  # Optimizer (full fidelity)
  optimizer_type: adamw
  learning_rate: 1e-4
  lr_scheduler_type: constant_with_warmup
  weight_decay: 0.01

  # No checkpointing (benchmark only)
  save_strategy: no

profiling:
  profile: true
  profile_steps: [15, 45]  # After warmup, near end
  profile_memory: true
  profile_ranks: [0]  # Only rank 0 to reduce overhead

logging:
  logging_steps: 1
  log_memory_steps: 5
  log_level: info
```

### Synthetic Data Generator

```python
import torch
from torch.utils.data import IterableDataset, DataLoader

class SyntheticLMDataset(IterableDataset):
    """Infinite synthetic data for benchmarking."""

    def __init__(self, vocab_size, seq_length, seed=42):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.seed = seed

    def __iter__(self):
        # Each worker gets different seed
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed + (worker_info.id if worker_info else 0)
        generator = torch.Generator().manual_seed(seed)

        while True:
            # Generate random tokens
            input_ids = torch.randint(
                0, self.vocab_size,
                (self.seq_length,),
                generator=generator
            )

            # Labels = shifted input_ids (causal LM)
            labels = input_ids.clone()

            yield {
                "input_ids": input_ids,
                "labels": labels
            }

# Usage in training script
def create_benchmark_dataloader(config, parallel_config):
    dataset = SyntheticLMDataset(
        vocab_size=config.vocab_size,
        seq_length=config.sequence_length,
        seed=config.seed
    )

    # Micro-batch size per device
    batch_size = config.per_device_train_batch_size

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # No multiprocessing for synthetic data
        pin_memory=True
    )

    return dataloader
```

## Recipe Variants

### 1. Memory-Limited Validation (Fast)

For quick checks (OOM detection, basic throughput):

```yaml
training:
  max_steps: 20  # Shorter run
  warmup_steps: 5
  per_device_train_batch_size: 1

profiling:
  profile: false  # Skip profiling for speed

logging:
  logging_steps: 1
```

**Use case:** Initial screening of Top-3 candidates (2-3 min each)

### 2. Full Profiling Validation

For detailed communication/compute analysis:

```yaml
training:
  max_steps: 100
  warmup_steps: 20

profiling:
  profile: true
  profile_steps: [25, 50, 75]  # Multiple checkpoints
  profile_memory: true
  profile_with_stack: true  # Capture kernel stacks
  profile_ranks: [0, 1]  # Multiple ranks for comm analysis

  # Advanced profiling
  record_shapes: true
  with_flops: true
  with_modules: true
```

**Use case:** Production-readiness validation, PSA calibration

### 3. Long-Context Validation

For CP/sequence-parallel strategies:

```yaml
model:
  crop_layers: 12  # Fewer layers to fit memory with long seq

data:
  sequence_length: 32768  # 32K context

training:
  max_steps: 30  # Shorter due to longer per-step time
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4  # Maintain global batch
```

**Use case:** CP > 1 strategies, long-sequence workloads

### 4. MoE Validation

For EP (expert parallel):

```yaml
model:
  model_name_or_path: /path/to/mixtral-8x7b  # or Qwen2.5-MoE

training:
  max_steps: 40
  load_balancing_loss_weight: 0.01  # Enable auxiliary loss

profiling:
  profile_steps: [15, 35]
  # Monitor AllToAll communication for EP
```

**Use case:** EP > 1, MoE models

## Expected Outputs

### Success Metrics (per recipe)

**Derive targets from a measured baseline for this workload.** Absolute
throughput, memory usage, and MFU depend on the architecture, sequence
length, parallel layout, optimizer, and software/hardware versions.
A range copied from another configuration is not an acceptance criterion.
For example, a sparse MoE with a matrix optimizer can report PaLM MFU near
12% without implying the same bottleneck as a dense model.

| Criterion | Definition |
|---|---|
| Throughput / step time | Improvement relative to the measured baseline for this run. |
| Memory | Fraction of device capacity, with a declared OOM margin. |
| MFU / HFU | Declare each accounting convention, then use a baseline-relative target or a documented roofline analysis. |
| Candidate comparison | Same nodes, device count, and measured step count; exclude the agreed warmup and run back-to-back. |

Set targets through `training.mfu.target_*` and inspect the final
acceptance output. See
[MFU targets](../../perf-playbook/references/mfu-targets.md).

When a result differs from the baseline expectation, verify accounting and
configuration equivalence before investigating hardware or implementation
faults.

### Failure Modes

| Symptom | Likely Cause | Recipe Adjustment |
|---------|--------------|-------------------|
| OOM at step 1-5 | Model states too large | Enable FSDP level3, reduce micro-batch |
| OOM at step 15+ | Activation accumulation | Enable activation checkpoint |
| Throughput < 50% expected | Communication bottleneck | Reduce TP/CP, check network |
| Loss = NaN | Numerical instability | Lower LR, check mixed precision |
| Hang at init | Mesh config mismatch | Verify dp×tp×pp×cp×ep = device_count |

## Recipe Selection Guide

```python
def select_recipe(model_size_b, device_count, validation_depth):
    """
    Args:
        model_size_b: Model size in billions
        device_count: Total devices
        validation_depth: "fast" | "standard" | "full"
    """

    # Determine crop ratio
    if model_size_b > 30:
        crop_ratio = 0.25  # Crop to 25% layers
    elif model_size_b > 10:
        crop_ratio = 0.40
    else:
        crop_ratio = 1.0  # No cropping for small models

    # Select step count
    steps_map = {
        "fast": 20,
        "standard": 50,
        "full": 100
    }
    max_steps = steps_map[validation_depth]

    # Enable profiling?
    profile = validation_depth in ["standard", "full"]

    return {
        "crop_ratio": crop_ratio,
        "max_steps": max_steps,
        "warmup_steps": max_steps // 5,
        "profile": profile,
        "profile_steps": [max_steps // 3, max_steps * 2 // 3] if profile else []
    }
```

## Multi-Candidate Recipe

When validating Top-3 strategies sequentially:

```bash
#!/bin/bash
# Run 3 candidates with fast recipe (20 steps each, ~15 min total)

CANDIDATES=("candidate1" "candidate2" "candidate3")
RECIPE="fast"  # Use fast recipe for multi-candidate

for candidate in "${CANDIDATES[@]}"; do
    echo "=== Validating $candidate ==="

    # Generate config with fast recipe
    python generate_config.py \
        --psa_output psa_${candidate}.json \
        --recipe $RECIPE \
        --output config_${candidate}.yaml

    # Launch
    timeout 300 torchrun \
        --nproc_per_node=64 \
        scripts/train_lm.py \
        --config config_${candidate}.yaml \
        2>&1 | tee log_${candidate}.txt

    # Extract quick metrics
    python extract_metrics.py log_${candidate}.txt > metrics_${candidate}.json

    # Cleanup for next run
    pkill -9 -f train_lm.py
    sleep 5
done

# Compare results
python compare_candidates.py metrics_candidate*.json
```

## Validation Checklist

Before running benchmark:

- [ ] Model checkpoint accessible
- [ ] Devices available (via `npu-smi` / `nvidia-smi`)
- [ ] Config matches PSA strategy (dp×tp×pp×cp×ep = device_count)
- [ ] Sufficient disk space (~100 MB for logs)
- [ ] No conflicting processes
- [ ] Recipe parameters match validation goal (fast/standard/full)

After benchmark:

- [ ] All steps completed (no OOM/hang)
- [ ] Logs contain throughput lines
- [ ] Memory metrics logged
- [ ] Profiler traces written (if enabled)
- [ ] Exit code 0
