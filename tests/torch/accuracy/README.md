# Llama3 Numerical Tests for Parallel Configurations

**English** | [简体中文](README.zh.md)

This directory runs the Llama3-style demo from `examples/torch/llama3` in parallel configurations on the **HyperParallel Torch backend**
and compares training loss step by step with a **single-device baseline** in the same process, providing numerical regression coverage for the parallel stack.

Each case reuses the `model.py` / `parallelize.py` design from `examples/torch/llama3`, with
self-contained local copies to avoid depending on `examples` injecting paths into `sys.path`.

## Files

| File | Description |
|------|------|
| `model.py` | Local copy of `examples/torch/llama3/model.py`, including `Llama3DemoConfig`, `Llama3Model`, and `Llama3BshdSdpaCore`, enabling `ContextParallel` hooks on the BSHD SDPA submodule. |
| `parallelize.py` | Local copy of `examples/torch/llama3/parallelize.py`; `parallelize_llama3` provides a TorchTitan-style TP+SP plan. |
| `_test_llama3_accuracy.py` | Worker launched by torchrun, containing `test_single_card_baseline`, `test_tp_fully_shard_matches_single_card`, and `test_tp_cp_fully_shard_matches_single_card`. |
| `test_llama3_accuracy.py` | Pytest entry point using `arg_mark` and `parallel_run` / `torchrun_case` to launch workers on 1 / 4 / 8 devices. |

## Numerical Alignment Strategy

Each case trains for `_STEPS = 10` and checks loss at **every step** with `rtol=1e-3, atol=1e-3`. Six measures ensure stepwise agreement between distributed and single-device execution:

1. **Identical initialization**: Every rank constructs `Llama3Model` with the same `_INIT_SEED`; rank 0 broadcasts parameters and buffers through `broadcast_state_dict_from_rank0`, ensuring identical initial states.
2. **Identical global batches**: Each worker generates `(tokens, targets)` using `_DATA_SEED`, then rank 0 broadcasts them so every rank sees the same global batch.
3. **Cross-entropy with `reduction="sum"`**: Each rank computes partial summed CE on its `(B/dp, S/cp)` slice. `all_reduce(SUM)` over the `(dp, cp)` plane reconstructs the same global summed loss as the single-device full batch.
4. **`set_reduce_op_type("sum")`**: Use SUM for FSDP gradient reduction, matching the aggregation of rank-local partial gradients from summed-loss backward.
5. **TP backward normalization**: The TP plan produces replicated logits (`use_local_output=True`), so every TP rank computes the same scalar loss. Without normalization, backward sums the gradient `tp_size` times along TP. Passing `1.0 / tp_size` as the gradient seed to `loss.backward(...)` cancels this factor, as in `tests/torch/fully_shard/_test_tp_fully_shard_e2e.py`.
6. **CP+RoPE alignment**: Each CP rank passes `rope_seq_start = cp_rank * (seq_len / cp_size)` into `Llama3Model.forward`, aligning its `freqs_cis` slice with the global position of that CP window.

## Coverage

| Case | Processes | Mesh | Description |
|------|--------|------|------|
| `test_single_card_baseline` | 1 | — | Single-device reference providing a reproducible loss trajectory within the same process for subsequent parallel cases. |
| `test_tp_fully_shard_matches_single_card` | 4 | `(dp=2, tp=2)` | TP + FSDP: DP partitions batches, TP runs `parallelize_llama3` on `mesh["tp"]`, and FSDP shards on `mesh["dp"]`. |
| `test_tp_cp_fully_shard_matches_single_card` | 8 | `(dp=2, cp=2, tp=2)` | TP + CP + FSDP: TP and FSDP follow the preceding case, with Colossal `ContextParallel(ulysses_degree=1)` attached to each layer's `attention.sdpa_core` on `mesh["cp"]`. |

## Run Tests

Run from this directory. The command assembled by `tests/torch/utils.py:torchrun_case` uses a relative worker filename, so pytest's `cwd` must match the launcher directory, as with the repository's other multi-device launchers:

```bash
cd tests/torch/accuracy

# Single-device baseline (1 device).
pytest -s test_llama3_accuracy.py::test_llama3_single_card_baseline

# TP + FSDP (4 devices).
pytest -s test_llama3_accuracy.py::test_llama3_tp_fully_shard_accuracy

# TP + CP + FSDP (8 devices).
pytest -s test_llama3_accuracy.py::test_llama3_tp_cp_fully_shard_accuracy
```

Numerical comparisons use `rtol=1e-3, atol=1e-3`; an assertion fails if the reconstructed global loss differs from the single-device baseline beyond this tolerance at any step.
