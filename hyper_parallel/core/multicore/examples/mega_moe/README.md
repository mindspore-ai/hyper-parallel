# MegaMoe Qwen optimizer benchmark

This directory contains a deliberately small end-to-end training benchmark:

- `qwen_moe_model.py` defines a random-weight Qwen MoE model with learned TopK
  routing, routed experts, shared experts, attention, and a language-model loss.
- `qwen_moe_benchmark.py` builds identically initialized common-MoE and
  MegaMoe variants, checks their full-model accuracy, and runs `zero_grad`,
  forward, backward, dense-gradient synchronization, and an AdamW step.
- `run_qwen_moe_benchmark.sh` activates CANN and the native payload, verifies
  the editable install, and launches the eight-rank job.

It is a runnable integration example, not a profiler. It does not collect
traces, timelines, HBM samples, or artifact inventories.

## MegaMoe integration

The Qwen Router computes `topk_ids`, `topk_weights`, and
`tokens_per_expert`. The exact Router histogram is passed to
`MegaMoeExperts`, so MegaMoe does not repeat the `bincount` operation.

Each decoder layer keeps independent expert parameters and optimizer state.
All serial MegaMoe layers call `share_execution_resources()` before first
forward and therefore reuse one SHMEM runtime/workspace. The benchmark calls
`QwenMoeModel.close()` before destroying the process group. Execution resources
acquire and release their SHMEM references internally; model code only closes
`MegaMoeExperts` (through `QwenMoeModel.close()`) and never calls SHMEM directly.

The public `MegaMoeExperts` API and this benchmark default
`expert_capacity_factor` to `None`, which reserves the maximum lossless receive
capacity. An explicit factor such as `1.5` keeps the workspace bounded, but a
route that exceeds that capacity fails before native execution with a clear
error. Smaller lossless workspaces via in-kernel multi-wave execution are
future work.

## Workload

| Item | Value |
| --- | --- |
| Topology | TP1, EP8 |
| Dtype | BF16 |
| Batch / sequence per rank | 1 / 1024 |
| Hidden size / layers | 2048 / 8 |
| Routed experts / TopK | 16 / 8 |
| Expert intermediate size | 512 |
| Optimizer | HyperParallel AdamW, FP32 master / BF16 model |
| Comparison order | common A, then MegaMoe B |
| Default timing | 3 warmup + 5 measured optimizer steps |

## Single-node run

Build the Torch multicore payload, install this checkout in editable mode, and
run:

```bash
CANN_ENV_FILE=/path/to/CANN/set_env.sh \
bash hyper_parallel/core/multicore/examples/mega_moe/run_qwen_moe_benchmark.sh \
  --output output/qwen_moe_benchmark.json
```

Useful environment overrides are `NATIVE_ENV_FILE`, `SHMEM_LIB_DIR`,
`ASCEND_RT_VISIBLE_DEVICES`, `MASTER_PORT`, `SHMEM_HOST`, `SHMEM_PORT`, `PYTHON_BIN`,
`TORCHRUN_BIN`, and `TIMEOUT_SECONDS`.

## Multi-node run

Launch the same command on every node with a shared `MASTER_ADDR` and distinct
`NODE_RANK`. For two four-device nodes, set:

```bash
NNODES=2 NPROC_PER_NODE=4 NODE_RANK=<0-or-1> \
MASTER_ADDR=<rank-zero-host> SHMEM_HOST=<rank-zero-ip> \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
bash hyper_parallel/core/multicore/examples/mega_moe/run_qwen_moe_benchmark.sh
```

`NNODES * NPROC_PER_NODE` must equal EP8. This minimal benchmark uses one
SHMEM communicator over the full Torch world. The launcher sets
`HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT` from `SHMEM_HOST` (defaulting to `MASTER_ADDR`) and
`SHMEM_PORT`; subgroup endpoint discovery is intentionally outside this PR.

## Output

Rank zero writes one JSON file containing the fixed topology and shape,
optimizer settings, common/MegaMoe accuracy checks, per-backend first-step
validation, steady-state latency samples and medians, their B/A ratio, measured
losses, and Torch runtime versions. Accuracy starts from exactly equal local
parameters and compares full-model loss/logits, all local gradients, and all
parameters after one AdamW step. The model parameters remain BF16 while
`Float16OptimizerWithFloat16Params` updates FP32 master parameters through
HyperParallel AdamW and copies them back to the model. All comparisons use
`rtol=2e-2` and `atol=2e-3`.
The wrapper's historical `Float16` class name denotes the low-precision model
path; this benchmark explicitly uses BF16 model parameters and FP32 master
parameters.
Gradient snapshots are collected only for the compared first step, so they do
not affect steady-state timing. Timing is plain A/B, not A/B/B/A, and uses the
rank-maximum complete optimizer-step latency after independent warmup. This
random-weight benchmark validates integration; it does not establish checkpoint
convergence.
