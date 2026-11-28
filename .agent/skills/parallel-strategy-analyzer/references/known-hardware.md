# Known Hardware Specifications

Reference configurations for Phase 1. Confirm the deployment before using a
row; for other hardware, collect the job's configuration as described below.

## Key Concept

**Parallel sharding is per die, not per chip.** For multi-die chips, each die is an independent compute unit. `dies_per_node = dies_per_chip × chips_per_node` determines the effective device count.

## Hardware Table

| Device | dies/chip | chips/node | dies/node | Mem/die | bw_intra/die | bw_inter/die | BF16 TFLOPS/die |
|--------|-----------|------------|-----------|---------|--------------|--------------|-----------------|
| Ascend A2 | 1 | 8 | 8 | 64GB | 400 GB/s | 50 GB/s | 256 |
| Ascend A3(384超节点) | 2 | 8 | 16 | 64GB | 400 GB/s | 50 GB/s | 200 |
| Ascend A3(64超节点) | 2 | 4 | 8 | 64GB | 196 GB/s | 196 GB/s | 400 |
| A100 40GB | 1 | 8 | 8 | 40GB | 600 GB/s | 50 GB/s | 312 |
| A100 80GB | 1 | 8 | 8 | 80GB | 600 GB/s | 50 GB/s | 312 |
| H100 80GB | 1 | 8 | 8 | 80GB | 900 GB/s | 100 GB/s | 990 |
| H200 141GB | 1 | 8 | 8 | 141GB | 900 GB/s | 100 GB/s | 990 |

> All bandwidth and memory values are **per die**. For multi-die chips, chip-level bandwidth is shared between dies.

For these A3 deployments, one physical card contains one two-die chip:

| A3 specification | Nodes/supernode | Physical cards/node | Physical cards/supernode | Dies/supernode |
|---|---|---|---|---|
| Ascend A3(384超节点) | 48 | 8 | 384 | 768 |
| Ascend A3(64超节点) | 16 | 4 | 64 | 128 |

When the request only says A3, query the target nodes' physical card inventory:
8 cards/node selects the 384-card specification; 4 cards/node selects the
64-card specification. Confirm two dies per card and a homogeneous node group.
Logical device counts, container-visible devices, and training ranks alone do
not establish the physical card count. If inventory is unavailable, ask the
user. Only if the specification remains unspecified after that inquiry, use
8 cards / 16 dies per node and label it as an assumption.

These are reference values for the two specified deployments, not new
measurements. Confirm actual supernode membership and cross-supernode links
before pricing communication; the inter-node column does not establish that
the same bandwidth holds between supernodes.

## Hardware-Specific Strategy Guidelines

### Ascend A2 (1 die/chip, 64GB/die, 8 dies/node)

- HCCS intra-node (~400 GB/s), inter-node ~50 GB/s
- tp_size: 1, 2, 4, 8
- Prefer tp=8 for models > 13B
- Memory-constrained (64GB): aggressive FSDP or PP needed for large models

### Ascend A3(384超节点) (2 dies/card, 8 cards/node, 16 dies/node)

- One supernode: 48 nodes, 384 physical cards, 768 dies.
- Each die: 64GB memory and 200 TFLOPS BF16 reference peak.
- Per-die bandwidth: 400 GB/s intra-node and 50 GB/s inter-node.
- Node-local tp_size candidates: 1, 2, 4, 8, 16. Check model divisibility and measured communication cost.
- Do not reuse the 64-card deployment's timing calibration or allocator reserve measurements.

### Ascend A3(64超节点) (2 dies/card, 4 cards/node, 8 dies/node)

- One supernode: 16 nodes, 64 physical cards, 128 dies.
- Each chip has 2 dies, each die is an independent compute unit (~400 TFLOPS BF16)
- Per-die bandwidth: 196 GB/s intra-node and 196 GB/s inter-node (chip bandwidth shared by 2 dies)
- tp_size: up to 8 (all dies in the node), commonly 2, 4, or 8
- Same per-die memory as A2 (64GB) and the same 8 dies per node; this limit applies only to this A3 specification.

### Other hardware: use the job's confirmed configuration

Obtain the following from the user's hardware configuration or environment
inspection, recording the source and units for each value:

- Physical cards and dies per card, logical devices visible to the job per
  node, and the actual allocated node/device count. Reconcile the physical
  inventory with container visibility and the launcher's device mapping.
- Memory capacity per logical device, currently available memory, and the
  task's memory budget. Capacity alone does not establish usable headroom.
- Node-local and cross-node link topology and effective bandwidth for the
  relevant collectives and message sizes. Use supplied measurements or an
  authorized communication probe; link type or advertised peak alone does
  not establish effective bandwidth.
- Supported dtypes and kernels from the installed runtime. If a utilization
  estimate needs peak compute, record the user's applicable per-device,
  per-dtype peak and its source separately from measured throughput.

Derive node-local TP/CP candidates from confirmed logical device counts and
placement, then check model divisibility and communication cost. Choose
micro-batches from the memory estimate and validate them on the target job.
Do not infer a fixed TP degree, communication overlap, or low-precision gain
from a product name or the size of an entire installation. Request missing
inputs before making estimates that depend on them; keep unmeasured bandwidth
and performance unverified.

### NVIDIA A100 (1 die/chip, 80GB/die, 8 dies/node)

- NVLink intra-node (~600 GB/s), inter-node ~50 GB/s
- tp_size: 1, 2, 4, 8
- More memory headroom, can run larger micro-batch

### NVIDIA H100 (1 die/chip, 80GB/die, 8 dies/node)

- NVSwitch intra-node (~900 GB/s), inter-node ~100 GB/s
- Higher compute, may need more TP to keep GPUs fed
- NVLink across nodes possible (DGX SuperPOD)
