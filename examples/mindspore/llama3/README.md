# Llama3-style Tensor Parallel Example (HyperParallel, MindSpore)

This directory mirrors ``examples/torch/llama3`` for **MindSpore on Ascend NPU**: same Llama3-style module layout and ``parallelize_llama3`` plan using ``parallelize_module`` with ``ColwiseParallel``, ``RowwiseParallel``, ``SequenceParallel``, and ``PrepareModuleInput``.

## Contents

| File | Description |
|------|-------------|
| `model.py` | Llama3-style stack as MindSpore ``Cell``s: embeddings, layers, attention (wq/wk/wv/wo), SwiGLU FFN, RMSNorm, output. RoPE uses cos/sin tables (real-valued); attention uses a causal SDPA via ``mint``/``ops``. |
| `parallelize.py` | ``parallelize_llama3()`` with TP+SP; ``build_tp_mesh``; ``broadcast_state_dict_from_rank0``. |
| `tensor_parallel_example.py` | Entry: ``dist.init``, broadcast, TP mesh, ``parallelize_llama3``, short training loop (Adam + ``SkipDTensorDispatch`` around backward/optimizer). |
| `__init__.py` | Re-exports main symbols. |

## Requirements

See ``examples/README.md`` for MindSpore / CANN / HyperParallel versions. Set ``HYPER_PARALLEL_PLATFORM=mindspore`` (the example script sets this before importing HyperParallel).

Source your Ascend or virtualenv activation script before running.

## How to run

Process count must match the TP degree (1-D ``DeviceMesh`` over all ranks). Use your MindSpore multi-process launcher (for example ``msrun``).

### Single node, two NPUs

- Set both ``--worker_num`` and ``--local_worker_num`` to **2** so both worker processes run on this host, matching 2-way tensor parallel; you do not need ``node_rank`` (it is ignored).
- The scheduler listens on ``--master_addr=127.0.0.1`` and ``--master_port=8118`` by default. If **8118 is already in use** on this machine (``Address already in use`` in ``scheduler.log``, or workers cannot reach meta), use a free port, e.g. ``--master_port=29501``.
- Optional: ``ASCEND_RT_VISIBLE_DEVICES=0,1`` to expose only two NPUs to this job (align with your cluster policy).
- On Ascend, pass ``--rank_table_file=...`` to ``msrun`` if your environment requires a rank table (see MindSpore ``msrun`` documentation).

From this directory:

```bash
cd examples/mindspore/llama3

msrun --worker_num=2 --local_worker_num=2 --log_dir=./msrun_log --join=True tensor_parallel_example.py
```

If the default **8118** port is in use, specify a free port explicitly, for example:

```bash
msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 --master_port=29501 --log_dir=./msrun_log --join=True tensor_parallel_example.py
```

From the repo root:

```bash
msrun --worker_num=2 --local_worker_num=2 --log_dir=./msrun_log --join=True examples/mindspore/llama3/tensor_parallel_example.py
```

```bash
msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 --master_port=29501 --log_dir=./msrun_log --join=True examples/mindspore/llama3/tensor_parallel_example.py
```

Adjust ``worker_num`` / ``local_worker_num`` to match available NPUs and the desired TP size.

## Constraints and notes

1. **Sequence parallel**: activations use ``Shard(1)`` on the sequence dimension; **sequence length must divide the TP world size**.
2. **Attention heads**: **``n_heads`` and ``n_kv_heads`` must both divide the TP world size**.
3. **`parallelize_llama3`** follows the **sequence-parallel-enabled** path only (same as the Torch demo).
4. **Training step**: wrap **`loss.backward()`** and the optimizer call with ``SkipDTensorDispatch``; do not wrap the full forward pass (see Torch README).
5. **MindSpore and DTensor**: If forward fails on ``Embedding``/``Gather`` with ``Operator Gather does not contain parallel layout infer func``, HyperParallel’s MindSpore stack does not yet implement parallel layout inference for **Gather-style ops**; that requires a follow-up in the library and is unrelated to single-node two-NPU setup.

## Relation to other examples

| Example | Role |
|---------|------|
| ``examples/mindspore/fully_shard/`` | FSDP-style ``fully_shard`` demo. |
| ``examples/torch/llama3/`` | Same TP+SP story on PyTorch. |
