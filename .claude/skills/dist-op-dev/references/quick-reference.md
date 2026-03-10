# Quick Reference Card

> **Detailed Information**: This document is a quick reference card. For detailed explanations, please refer to the corresponding workflow documents.

---

## Terminology Definitions

| Term | Definition | Details |
|------|------|----------|
| **Scenario 0/1/2** | Implementation method classification: Scenario 0=fully use base class, Scenario 1=extend base class, Scenario 2=fully custom | See [implementation-decisions.md](./implementation-decisions.md) |
| **WithShape** | A type of `infer_layout_suffix`, used for operators that support broadcasting | See [workflow 03](../workflows/03-yaml-registration.md) |
| **DP** | Data Parallel, data is sharded along batch dimension | - |
| **MP** | Model Parallel, model parameters are sharded along some dimension | - |
| **Hybrid** | Hybrid Parallel, using both DP and MP | - |
| **Partial** | Partial state of a tensor, indicating the tensor only contains partial data along some dimension | See [implementation-decisions.md](./implementation-decisions.md) |
| **TensorMap** | Describes the sharding strategy for each dimension of a tensor, -1 means not sharded | - |

---

## File Location Quick Reference

| Task | File Location | Key Notes |
|------|---------------|-----------|
| YAML Registration | `hyper_parallel/core/shard/ops/yaml/*.yaml` | Configure operator to distributed implementation class mapping |
| Python Implementation | `hyper_parallel/core/shard/ops/parallel_*.py` | Inherit `DistributedOp` or its subclass |
| Unit Test (UT) | `tests/mindspore/ut/parallel_ops_infer/` | Platform-agnostic, verify `infer_layout` logic |
| Integration Test (ST) | `tests/mindspore/st/shard/ops` `tests/torch/shard/ops` | 8-card environment verify distributed execution |
| Base Class Definition | `hyper_parallel/core/shard/ops/parallel_ops.py` | `DistributedOp` base class |
| Dispatch Core | `hyper_parallel/core/shard/_op_dispatch.py` | `OpDispatcher.dispatch` method |
| Layout Definition | `hyper_parallel/core/layout.py` | Distributed layout abstraction |
| DTensor Definition | `hyper_parallel/core/dtensor.py` | Distributed tensor implementation |

---

## Core Method Quick Reference

### DistributedOp Base Class

| Method | Parameters | Return Value | Description |
|--------|------------|--------------|-------------|
| `infer_layout(layouts, extra_args)` | layouts: input layout list<br>extra_args: additional parameters | `Layout` or `tuple[Layout]` | Infer output layout |
| `get_expand_impl(func, output_layout, layouts, extra_args)` | func: original operator<br>output_layout: output layout<br>layouts: input layouts<br>extra_args: additional parameters | `callable` or `None` | Get expand implementation |

### Attributes

| Attribute | Type | Default Value | Description |
|-----------|------|---------------|-------------|
| `_allow_partial_inputs` | `bool` | `False` | Whether to allow partial inputs |

---

## infer_layout_suffix Configuration Quick Reference

> **Detailed Information**: See [workflow 03: YAML Registration](../workflows/03-yaml-registration.md)

| Suffix Value | Usage Scenario |
|--------------|----------------|
| **(no suffix)** | Simple operators (no broadcast) |
| **`WithShape`** | **Operators supporting broadcast** ⭐ |
| `Reshape` | Shape transformation |
| `Slice` | Slice operations |

---

## Base Class Selection Quick Reference

> **Detailed Information**: See [implementation-decisions.md](./implementation-decisions.md)

| Base Class Name | Applicable Scenario | Need Custom |
|-----------------|---------------------|-------------|
| `DistributedOp` | Fully custom semantics | ✅ Must override `infer_layout` |
| `ElementWiseDistributedOp` | Element-wise operations | ⚙️ Optional override |
| `ReduceDistributedOp` | Reduction operations | ⚙️ Optional override |
| `ReshapeDistributedOp` | Shape operations | ⚙️ Optional override |
| `MatMulDistributedOp` | Matrix multiplication | ⚙️ Optional override |
| `GatherDistributedOp` | Index operations | ⚙️ Optional override |

---

## Platform Differences Quick Reference

| Item | MindSpore | PyTorch |
|------|-----------|---------|
| **Operator Name Style** | PascalCase (`Add`, `MatMul`) | snake_case (`add`, `matmul`) |
| **YAML Files** | `element_wise_ops.yaml`, etc. | `torch_*.yaml` |
| **YAML Entry** | `Add:` | `add:` |
| **Test Directories** | `tests/mindspore/` | `tests/torch/` |

> **Important**: MindSpore and PyTorch can reuse the same distributed operator implementation class.

---

## Operator Category Quick Reference

| Category | Operator Examples | Base Class |
|----------|-------------------|------------|
| **ElementWise** | Add, Mul, Greater | `ElementWiseDistributedOp` |
| **Reduce** | Sum, Mean, Max | `ReduceDistributedOp` |
| **Reshape** | Reshape, View | `ReshapeDistributedOp` |
| **MatMul** | MatMul, BatchMatMul | `MatMulDistributedOp` |
| **Gather** | Gather, GatherNd | `GatherDistributedOp` |

---

## Test Scenario Quick Reference

### UT Test Scenarios

| Scenario Type | Description | Required for WithShape |
|---------------|-------------|------------------------|
| **Same Layout** | Two inputs have same tensor_map | ❌ |
| **Different Layout** | Inputs sharded on different dimensions | ❌ |
| **Scalar Broadcast** | One input is scalar | ✅ |
| **Shape Broadcast** | Different rank shapes | ✅ |

### ST Test Scenarios

| Parallel Strategy | Description | Card Count |
|-------------------|-------------|------------|
| **DP** | Data Parallel | 8 |
| **MP** | Model Parallel | 8 |
| **Hybrid** | Hybrid Parallel | 8 |

---

## Lint Check Quick Reference

> **Detailed Information**: See [code-standards.md](./code-standards.md)

| Check Item | Tool | Requirement |
|------------|------|-------------|
| **Pylint** | Pylint | No errors |
| **Lizard** | Lizard | Function length < 100, CCN < 19 |

---

## Git Operations Quick Reference (via autogit)

> **Detailed Information**: See [workflow 06: Git Commit](../workflows/06-git-commit.md)

### Branch Naming Convention

| Branch Type | Naming Format | Example |
|-------------|---------------|---------|
| **New Feature** | `feat/{OpName}-distributed-support` | `feat/MatMul-distributed-support` |
| **Bug Fix** | `fix/{OpName}-{issue}` | `fix/Add-broadcast-error` |

### Commit Message Template

```
feat(shard): add {OpName} operator distributed support

- Add {ClassName} class in {file_name}.py
- Register {OpName} in {yaml_file}.yaml
- Add UT cases for {OpName}
- Add ST cases for {OpName} (8-card)
```

### Common Commands

```bash
# Commit code (with lint check)
/autogit commit -m "feat(shard): add Greater operator support"

# Create PR
/autogit pr

# View PR status
/autogit status #160
```

---

## Common Errors Quick Reference

### YAML Related

| Error | Fix Method |
|-------|------------|
| Operator not registered | Add operator to corresponding YAML file |
| Suffix error | Check if `WithShape` is needed |

### infer_layout Related

| Error | Fix Method |
|-------|------------|
| `KeyError: 'input_shapes'` | Add `infer_layout_suffix: WithShape` in YAML |
| Output tensor_map error | Compare with MindSpore reference code |

### Git/autogit Related

| Error | Fix Method |
|-------|------------|
| Token not found | `export GITCODE_TOKEN=<token>` |
| Push rejected | `git pull --rebase origin <branch>` |
