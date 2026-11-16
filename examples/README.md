# HyperParallel Examples

This directory contains usage examples for the HyperParallel distributed training library, demonstrating how to apply various parallel strategies in PyTorch.

## Directory Structure

```text
examples/
├── torch/              # PyTorch examples
│   ├── fully_shard/    # FSDP (Fully Sharded Data Parallel) examples
│   ├── llama3/         # Llama3-style tensor parallel + sequence parallel
│   ├── moe/            # MoE expert parallel + pipeline parallel examples
│   └── pp_overlap/     # Interleaved pipeline + expert parallel overlap
```

## Environment Requirements

| Component | Version |
|-----------|---------|
| Python | >=3.9 |
| torch & torch_npu | 2.6.0 |
| CANN | 8.3.0 |
| HyperParallel | [nightly (202603)](https://repo.mindspore.cn/mindspore/hyper-parallel/version/202603/20260322/master_20260322020005_823f1bea890db254d9cf8ed554fde675137d940b_newest/any/hyper_parallel-0.1.0-py3-none-any.whl) |

## AutoModel Trainer Example

The delivered Qwen3-MoE AutoModel training recipe lives at
`hyper_parallel/models/qwen3_moe/recipes/train.yaml` inside the
package (shipped in the wheel). Model, dataset, optimizer, and parallel
components are selected through typed targets in `train.yaml`.

## Expert Parallel Training

Use the [Qwen3-MoE AutoModel recipe](../hyper_parallel/models/qwen3_moe/recipes/train.yaml) with the [EP guide](../docs/guide/expert_parallel.md).

## PyTorch MoE examples

The [EP](torch/moe/expert_parallel_example.py), [PP+EP](torch/moe/pp_ep_example.py), and [interleaved overlap](torch/pp_overlap/pp_overlap_moe_example.py) examples use the original [ExpertParallel implementation](torch/moe/legacy_expert_parallel.py) and [standalone MoE layer](torch/moe/legacy_moe.py) from the example directory. Migration to the training EP path is tracked in [the second stage of the EP refactor](https://github.com/mindspore-ai/hyper-parallel/issues/1585).

Run from the repository root.

```bash
torchrun --nproc_per_node=2 -m examples.torch.moe.expert_parallel_example
torchrun --nproc_per_node=4 -m examples.torch.moe.pp_ep_example
torchrun --nproc_per_node=4 -m examples.torch.pp_overlap.pp_overlap_moe_example
```

The EP and PP+EP launchers use `MOE_DEVICE_TYPE` (`npu` by default). The PP+EP launcher also accepts `MOE_PP_SIZE` and `MOE_EP_SIZE` (both default to 2) and `MOE_NUM_STEPS` (defaults to 5). The overlap example defaults to four ranks with PP=2 and EP=2.

For PP+EP, set the world size to `MOE_PP_SIZE * MOE_EP_SIZE`. The number of experts must be divisible by `MOE_EP_SIZE`, and the number of layers must be at least `MOE_PP_SIZE`.

Run the grouped-matmul regression cases from the source checkout with a Torch build that provides `torch._grouped_mm`.

```bash
python -m pytest examples/torch/moe/test_legacy_moe.py
```
