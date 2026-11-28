# Phase 1: Collect Model & Hardware Info

## Goal

Gather all model architecture and hardware parameters needed for analysis. Auto-fill known models, infer missing values, validate consistency.

## Process

### 1.1 Parse User Input

Extract from user's request:
- Model name or total parameter count
- Device count and type
- Constraints: sequence length, batch size, memory budget, existing config

### 1.2 Auto-Fill Known Models

See `references/known-models.md` for full model architecture tables (LLaMA, GPT-3, Mixtral, DeepSeek-V2/V3/V3.2, Qwen/Qwen3/Qwen3.5) and parameter inference formulas for unknown models.

### 1.3 Collect Hardware Info

See [hardware reference configurations](../references/known-hardware.md) for
deployment-specific values and the collection procedure for other hardware.
Obtain the job-visible logical devices per node, device mapping, per-device
memory and memory budget from the user configuration or environment inspection.
Obtain effective communication bandwidth from applicable supplied measurements
or an authorized probe, with the collective and message size recorded. Record
sources and units; do not substitute installation-wide scale or advertised peak
bandwidth for the resources allocated to the job. Request missing inputs before
making estimates that depend on them.

For A3, query the target node inventory (for example the physical card IDs in
`npu-smi info` together with the host's device mapping). Verify whether there
are **8 physical cards/node** (384-card supernode, 48 nodes, 16 dies/node) or
**4 physical cards/node** (64-card supernode, 16 nodes, 8 dies/node). Both have
two dies per card. Record the inventory source and check that the requested
node group has the same specification. A restricted container's logical device
count is insufficient. If querying cannot establish the specification, ask the
user; only if it remains unspecified, assume 8 cards/node and record that assumption.

Pass the confirmed physical count to SAPP as `--cards-per-node 8` or
`--cards-per-node 4`, or select `--cluster a3_384` / `--cluster a3_64` explicitly.

### 1.3 Validate & Request Missing Info

**Required** (cannot proceed without):
- Total parameter count or model name
- Device count
- Device type or (memory + devices_per_node)

**Defaults if not provided:**
- `seq_len = 4096`
- `global_batch_size = 1024`
- `micro_batch_size = 1`
- `precision = "bf16"`
- `optimizer = "adam"`

## Output

Complete parameter set:

Hardware fields below are placeholders filled from the selected specification
or environment evidence. They do not establish default device counts or capacity.

```
Model:    LLaMA-2-70B
  params: 70B, L=80, h=8192, n_h=64, n_kv=8 (GQA), d_ff=28672, V=32000
  seq_len=4096, global_batch=1024, micro_batch=1

Hardware: <total_devices>× <device_type> (<dies_per_card> dies/card, if applicable)
  logical_devices/node=<devices_per_node>, total_devices=<total_devices>, nodes=<node_count>
  memory/device=<memory_per_device_gb> GB
  effective_bw_intra/device=<bw_intra_gbps> GB/s, effective_bw_inter/device=<bw_inter_gbps> GB/s
  peak/device=<peak_tflops> TFLOPS, dtype=<compute_dtype>, source=<hardware_evidence>
```
