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

Select the transport with `--dispatch-mode push|pull` (default: push).
Push starts with `--initial-capacity-factor 1.25` and grows on overflow using
`--capacity-growth-factor 1.25`. Both factors are finite numbers of at least one;
a growth factor of `1.0` allocates only the current demand. Set the initial
factor to the EP size to reserve the full lossless upper bound at startup.
Pull accepts neither factor: its SHMEM contains local source/combine rows,
while received rows use ordinary HBM. Both transports retain all routed tokens.
The old `--expert-capacity-factor` and `--capacity-policy` options were removed.

## Workload

| Item | Value |
| --- | --- |
| Topology | TP1, EP8 |
| Dtype | BF16 |
| Batch / sequence per rank | 1 / 4096 |
| Hidden size / layers | 5120 / 2 |
| Routed experts / TopK | 48 / 8 |
| Routed / shared expert intermediate size | 1792 / 1792 |
| Attention heads / KV heads / head dimension | 40 / 8 / 128 |
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
`SHMEM_PORT`. The public `MegaMoeExperts` API also accepts EP subgroups: each
subgroup bootstraps with a CANN unique ID broadcast only among its members.
`SHMEM_UID_SOCK_IFNAME` can select a host interface reachable by all EP members
for multi-node UID bootstrap. Do not set a fixed `SHMEM_UID_SESSION_ID` for
independent groups. No TP degree is passed to MegaMoE: the caller supplies
rank-local tokens and the actual EP group.

PP stages may retain multiple forward graphs while sharing serial execution
resources; activations needed by delayed backward are owned by each graph.
Close local experts after the pipeline drains and before destroying EP groups.
Only one ordered EP membership may own SHMEM in a process at a time; separate
disjoint PP/DP groups have independent runtimes. This example itself still uses
one full-world EP group rather than a pipeline schedule.

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

### 配置 dispatch 与 push 容量

```bash
bash hyper_parallel/core/multicore/examples/mega_moe/run_qwen_moe_benchmark.sh \
  --dispatch-mode push --initial-capacity-factor 1.25 --capacity-growth-factor 1.25

bash hyper_parallel/core/multicore/examples/mega_moe/run_qwen_moe_benchmark.sh \
  --dispatch-mode pull
```

结果中的 `shape.dispatch_mode`、`shape.initial_capacity_factor` 和 `shape.capacity_growth_factor`
记录生效配置；pull 的两个因子为 `null`。MegaMoE backend 的 `shmem_heap_bytes` 来自实际 runtime，
`heap_growth` 记录扩容前后 heap 大小、各 workspace 容量及各重建阶段耗时，计时来自 rank 0。
容量回落时不缩容；倍数作用于接收行数，而非整个 heap 的字节数。

此 runner 只在首个 optimizer step 比较 common 与 MegaMoE 的数值。
后续参数更新可能导致路由逐渐分叉，稳态耗时不能直接视为相同通信负载下的后端比较。
研究容量策略的开销时，应另用固定参数和固定路由，分别报告稳态与扩容步。
