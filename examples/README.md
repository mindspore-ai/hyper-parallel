# HyperParallel Examples

This directory contains usage examples for the HyperParallel distributed training library, demonstrating how to apply various parallel strategies in PyTorch and MindSpore frameworks.

## Directory Structure

```text
examples/
├── torch/              # PyTorch examples
│   ├── fully_shard/    # FSDP (Fully Sharded Data Parallel) examples
│   ├── llama3/         # Llama3-style tensor parallel + sequence parallel
│   └── moe/            # MoE expert parallel + pipeline parallel examples
└── mindspore/          # MindSpore examples
    ├── fully_shard/    # FSDP (Fully Sharded Data Parallel) examples
    └── llama3/         # Llama3-style tensor parallel + sequence parallel
```

## Environment Requirements

### MindSpore

| Component | Version |
|-----------|---------|
| Python | >=3.9 |
| MindSpore | [nightly (202603)](https://repo.mindspore.cn/mindspore/mindspore/version/202603/20260320/master_20260320160013_54ffea58f5d6f3e96a7e66e2aa981f5440357b22_newest/unified/aarch64/mindspore-2.9.0-cp310-cp310-linux_aarch64.whl) |
| CANN | 8.3.0 |
| HyperParallel | [nightly (202603)](https://repo.mindspore.cn/mindspore/hyper-parallel/version/202603/20260322/master_20260322020005_823f1bea890db254d9cf8ed554fde675137d940b_newest/any/hyper_parallel-0.1.0-py3-none-any.whl) |

### PyTorch

| Component | Version |
|-----------|---------|
| Python | >=3.9 |
| torch & torch_npu | 2.6.0 |
| CANN | 8.3.0 |
| HyperParallel | [nightly (202603)](https://repo.mindspore.cn/mindspore/hyper-parallel/version/202603/20260322/master_20260322020005_823f1bea890db254d9cf8ed554fde675137d940b_newest/any/hyper_parallel-0.1.0-py3-none-any.whl) |

## PyTorch MoE Examples

The `torch/moe/` directory provides Mixture-of-Experts (MoE) distributed training examples, demonstrating expert parallelism and pipeline parallelism with correctness verification.

### Expert Parallelism (EP)

Shards MoE experts across devices using `ExpertParallel`, each device holds a subset of experts.

```bash
torchrun --nproc_per_node=2 expert_parallel_example.py
```

### Pipeline Parallelism + Expert Parallelism (PP + EP)

Combines pipeline parallelism (1F1B schedule) with expert parallelism, verifying that the distributed forward loss matches a standalone reference.

```bash
torchrun --nproc_per_node=4 pp_ep_example.py
```

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MOE_PP_SIZE` | 2 | Pipeline parallel width |
| `MOE_EP_SIZE` | 2 | Expert parallel width |
| `MOE_NUM_STEPS` | 5 | Number of verification steps |
| `MOE_DEVICE_TYPE` | npu | Device type (`npu` or `cuda`) |

Requirements: `world_size == MOE_PP_SIZE * MOE_EP_SIZE`, `num_experts % MOE_EP_SIZE == 0`, `n_layers >= MOE_PP_SIZE`.
