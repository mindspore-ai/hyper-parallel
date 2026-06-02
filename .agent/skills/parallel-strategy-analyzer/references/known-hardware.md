# Known Hardware Specifications

Reference table for auto-filling hardware parameters in Phase 1.

## Key Concept

**Parallel sharding is per die, not per chip.** For multi-die chips, each die is an independent compute unit. `dies_per_node = dies_per_chip × chips_per_node` determines the effective device count.

## Hardware Table

| Device | dies/chip | chips/node | dies/node | Mem/die | bw_intra/die | bw_inter/die | BF16 TFLOPS/die |
|--------|-----------|------------|-----------|---------|--------------|--------------|-----------------|
| Ascend A2 | 1 | 8 | 8 | 64GB | 400 GB/s | 50 GB/s | 256 |
| Ascend A3 | 2 | 8 | 16 | 64GB | 400 GB/s | 50 GB/s | 200 |
| Ascend 950DT | 2 | 8 | 16 | 72GB | 1000 GB/s | 100 GB/s | 250 |
| A100 40GB | 1 | 8 | 8 | 40GB | 600 GB/s | 50 GB/s | 312 |
| A100 80GB | 1 | 8 | 8 | 80GB | 600 GB/s | 50 GB/s | 312 |
| H100 80GB | 1 | 8 | 8 | 80GB | 900 GB/s | 100 GB/s | 990 |
| H200 141GB | 1 | 8 | 8 | 141GB | 900 GB/s | 100 GB/s | 990 |

> All bandwidth and memory values are **per die**. For multi-die chips, chip-level bandwidth is shared between dies.

## Hardware-Specific Strategy Guidelines

### Ascend A2 (1 die/chip, 64GB/die, 8 dies/node)

- HCCS intra-node (~400 GB/s), inter-node ~50 GB/s
- tp_size: 1, 2, 4, 8
- Prefer tp=8 for models > 13B
- Memory-constrained (64GB): aggressive FSDP or PP needed for large models

### Ascend A3 (2 dies/chip, 64GB/die, 16 dies/node)

- Each chip has 2 dies, each die is an independent compute unit (~200 TFLOPS BF16)
- Per-die bandwidth: ~400 GB/s intra-node, ~50 GB/s inter-node (chip bandwidth shared by 2 dies)
- Intra-chip die-to-die bandwidth is higher than inter-chip HCCS
- tp_size: up to 16 (all dies in node), commonly 8 or 16
- Same per-die memory as A2 (64GB), but 2× dies per node enables finer-grained sharding

### Ascend 950DT (2 dies/chip, 72GB/die, 16 dies/node) — Atlas 950 Supernode

- Each chip has 2 dies, each die: 72GB HBM, ~250 TFLOPS BF16
- Per-die bandwidth: ~1000 GB/s intra-node, ~100 GB/s inter-node (chip bandwidth shared by 2 dies)
- Atlas 950 supernode: 8192 chips × 2 dies = 16384 dies, 160 cabinets, 16 PB/s total interconnect
- Per-chip memory bandwidth: 4 TB/s
- FP8: ~976 TFLOPS/chip (~488/die), FP4: ~1.95 PFLOPS/chip
- Supports BF16, FP8, MXFP8, HiF8, MXFP4, HiF4
- **Strategy implications**:
  - Ultra-high intra-node bandwidth makes TP overhead very low → TP=16 viable
  - 72GB/die + high bandwidth → can handle larger micro-batches, reducing PP bubble
  - Supernode scale (16384 dies) → large DP groups viable, FSDP communication well-overlapped
  - Consider FP8 training to leverage ~2× compute vs BF16
  - At supernode scale: prefer TP intra-node, PP/DP inter-node

### NVIDIA A100 (1 die/chip, 80GB/die, 8 dies/node)

- NVLink intra-node (~600 GB/s), inter-node ~50 GB/s
- tp_size: 1, 2, 4, 8
- More memory headroom, can run larger micro-batch

### NVIDIA H100 (1 die/chip, 80GB/die, 8 dies/node)

- NVSwitch intra-node (~900 GB/s), inter-node ~100 GB/s
- Higher compute, may need more TP to keep GPUs fed
- NVLink across nodes possible (DGX SuperPOD)
