# Llama3-style Tensor Parallel Example (HyperParallel)

This directory shows how to run **tensor parallel + sequence parallel (SP)** on a **Llama3-style decoder** on **Ascend NPU** using HyperParallel: `parallelize_module` with `ParallelStyle` helpers such as `ColwiseParallel`, `RowwiseParallel`, `SequenceParallel`, and `PrepareModuleInput`.

For Tensor Parallel usage and APIs, see the in-repo module:

- `hyper_parallel.core.tensor_parallel`: `parallelize_module` and the concrete `ParallelStyle` classes (details in source and project docs).

## Contents

| File | Description |
|------|-------------|
| `model.py` | Small Llama3-style stack: `tok_embeddings`, `layers.*`, `attention` (wq/wk/wv/wo), `feed_forward` (w1/w2/w3), RMSNorm, `output`. |
| `parallelize.py` | `parallelize_llama3()`: applies TP+SP (row-wise embedding, sequence `Shard`, Colwise/Rowwise patterns for attention and SwiGLU). |
| `tensor_parallel_example.py` | Entry point: distributed init, `broadcast_state_dict_from_rank0`, `parallelize_llama3`, short training loop. |
| `__init__.py` | Re-exports main symbols (add this directory to `PYTHONPATH` or rely on the example script path setup). |

## Requirements

Match the frontend stack and **HyperParallel** setup described in `examples/README.md` at the repo root.

| Component | Notes |
|-----------|-------|
| Python | >= 3.9 |
| CANN / drivers | Compatible with your Ascend setup |
| HyperParallel | Installed and importable as `hyper_parallel` |

Source your Ascend or virtualenv activation script before running so NPU and HyperParallel are available.

## How to run

Run from this directory or the repo root. The number of processes must match the TP plan; the example builds a 1D `DeviceMesh` with **world size = TP degree**.

```bash
cd examples/torch/llama3

# Example: 2-way tensor parallel (use a multi-process launcher that supports distributed jobs)
torchrun --nnodes=1 --nproc_per_node=2 tensor_parallel_example.py
```

From the repo root:

```bash
torchrun --nnodes=1 --nproc_per_node=2 examples/torch/llama3/tensor_parallel_example.py
```

## Constraints and notes

1. **Sequence parallel**: activations use `Shard(1)` on the sequence dimension, so **sequence length must divide the TP world size**.
2. **Attention heads**: Colwise sharding splits the last dimension; **`n_heads` and `n_kv_heads` must both divide the TP world size**.
3. **`parallelize_llama3`** currently implements only the path with **sequence parallel enabled**; `enable_sequence_parallel=False` is not implemented.
4. **Training loop**: wrap only **`loss.backward()` and `optimizer.step()`** with `SkipDTensorDispatch`. Do **not** wrap the entire forward pass in it, or TP hooks may not see the expected `DTensor` behavior for linear layers.
5. The sample uses a **small teaching config** (`Llama3DemoConfig`), not a full production-scale model; numerics and performance follow this repository’s HyperParallel implementation.

## Relation to other examples in this repo

| Example | Role |
|---------|------|
| `examples/torch/fully_shard/` | Demonstrates `fully_shard` (sharded data parallel style); this directory focuses on **declarative tensor parallel** without stacking `fully_shard`. |

Combining **data parallel / fully sharded** execution with **TP** requires a multi-dimensional `DeviceMesh`, sub-mesh slicing, and the corresponding HyperParallel APIs; that is out of scope here.
