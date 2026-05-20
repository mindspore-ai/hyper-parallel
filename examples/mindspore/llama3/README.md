# Llama3-style Tensor Parallel Example (HyperParallel, MindSpore)

This directory mirrors ``examples/torch/llama3`` for **MindSpore on Ascend NPU**: same Llama3-style module layout and ``parallelize_llama3`` plan using ``parallelize_module`` with ``ColwiseParallel``, ``RowwiseParallel``, ``SequenceParallel``, and ``PrepareModuleInput``.

## Contents

| File | Description |
|------|-------------|
| `model.py` | Llama3-style stack as MindSpore ``Cell``s: embeddings, layers, attention (wq/wk/wv/wo), SwiGLU FFN, RMSNorm, output. RoPE uses cos/sin tables (real-valued); attention uses a causal SDPA via ``mint``/``ops``. |
| `parallelize.py` | ``parallelize_llama3()`` with TP+SP; ``build_tp_mesh``; ``broadcast_state_dict_from_rank0``. |
| `tensor_parallel_example.py` | Entry: ``dist.init``, broadcast, TP mesh, ``parallelize_llama3``, short training loop (Adam + ``SkipDTensorDispatch`` around backward/optimizer). |
| `fsdp_tp_example.py` | TP + ``fully_shard`` on a 2-D ``(dp, tp)`` mesh (mirrors ``examples/torch/llama3/fsdp_tp_example.py``). |
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

### TP + FSDP（``fully_shard``）

总进程数 = ``worker_num``。设 TP 宽度为 ``LLAMA3_TP_SIZE``（默认 ``2``），则 ``dp_size = worker_num / TP``，且须满足 ``worker_num % TP == 0``。

示例：4 卡、``TP=2``、``DP=2``：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_IF_BASE_PORT=64000
export HCCL_NPU_SOCKET_PORT_RANGE=64000-65000

msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 --master_port=29509 \
  --log_dir=./msrun_log --join=True fsdp_tp_example.py
```

可选环境变量：

| 变量 | 含义 | 默认 |
|------|------|------|
| `LLAMA3_TP_SIZE` | 张量并行宽度 | `2` |

## Constraints and notes

1. **Sequence parallel**: activations use ``Shard(1)`` on the sequence dimension; **sequence length must divide the TP world size**.
2. **Attention heads**: **``n_heads`` and ``n_kv_heads`` must both divide the TP world size**.
3. **`parallelize_llama3`** follows the **sequence-parallel-enabled** path only (same as the Torch demo).
4. **Training step**: wrap **`loss.backward()`** and the optimizer call with ``SkipDTensorDispatch``; do not wrap the full forward pass (see Torch README).
5. **Token embedding**: ``model.py`` uses ``Llama3LocalEmbedding`` (local ``ops.gather`` under ``SkipDTensorDispatch``) because MindSpore ``nn.Embedding`` calls ``Gather``, which is not yet registered for DTensor layout inference.
6. **``fsdp_tp_example.py``**: call ``model.set_reduce_op_type("sum")`` after ``fully_shard``; all ranks use the same random batch for smoke testing (not a strict single-device numerical baseline). On multi-process single-node jobs, set ``HCCL_IF_BASE_PORT`` / ``HCCL_NPU_SOCKET_PORT_RANGE`` if HCCL reports port conflicts.

## Relation to other examples

| Example | Role |
|---------|------|
| ``examples/mindspore/fully_shard/`` | FSDP-style ``fully_shard`` demo. |
| ``examples/torch/llama3/`` | Same TP+SP story on PyTorch. |
