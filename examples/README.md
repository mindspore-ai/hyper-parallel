# HyperParallel Examples

This directory contains usage examples for the HyperParallel distributed training library, demonstrating how to apply various parallel strategies in PyTorch and MindSpore frameworks.

## Directory Structure

```text
examples/
├── torch/              # PyTorch examples
│   └── fully_shard/    # FSDP (Fully Sharded Data Parallel) examples
└── mindspore/          # MindSpore examples
    └── fully_shard/    # FSDP (Fully Sharded Data Parallel) examples
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

