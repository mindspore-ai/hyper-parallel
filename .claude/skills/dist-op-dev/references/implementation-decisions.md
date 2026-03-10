# Implementation Decisions Reference

This document details key decision points and criteria during HyperParallel distributed operator development.

---

## Decision Points Overview

| Decision Point | Criteria | Details Section |
|----------------|----------|-----------------|
| **Operator Category** | Semantic matching | [Decision 1](#decision-1-operator-category) |
| **Implementation Method** | Need custom logic | [Decision 2](#decision-2-implementation-method) |
| **Broadcast Support** | Support broadcasting | [Decision 3](#decision-3-broadcast-support) |
| **Partial Support** | Handle partial state | [Decision 4](#decision-4-partial-support) |
| **get_expand_impl** | Need expand implementation | [Decision 5](#decision-5-get_expand_impl-requirement) |
| **YAML Suffix** | Operator characteristics | [Decision 6](#decision-6-yaml-suffix-selection) |

---

## Decision 1: Operator Category

### Criteria

Select appropriate operator category based on operator semantics and functional characteristics.

| Category | Characteristics | Operator Examples | Key Decision Points |
|----------|-----------------|-------------------|---------------------|
| **ElementWise** | Element-wise operations, output shape same as input | Add, Mul, Greater, LessEqual | Whether to support broadcast |
| **Reduce** | Aggregate along some dimension | Sum, Mean, Max, Min | dim, keepdim parameter handling |
| **Reshape** | Shape transformation, data unchanged | Reshape, View, Squeeze, Unsqueeze | target_shape vs input_shape relationship |
| **MatMul** | Matrix multiplication (2D/higher-dim) | MatMul, BatchMatMul, Linear | Contracting dimension sharding strategy |
| **Gather** | Index selection | Gather, GatherNd, Index | Index mode, boundary handling |
| **Transpose** | Transpose, permutation | Transpose, Permute | permute parameter handling |

### Decision Impact

| Impact | Description |
|--------|-------------|
| **Base Class Selection** | Determines which base class to inherit |
| **YAML File** | Determines which YAML file to register in |
| **Test Scenarios** | Determines which test scenarios to cover |

---

## Decision 2: Implementation Method

### Criteria

Select one of the following three implementation methods based on operator characteristics and complexity.

| Implementation Method | Criteria | Code Volume | UT Requirement |
|-----------------------|----------|-------------|----------------|
| **Scenario 0: Fully Use Base Class** | ✅ Operator semantics completely match base class<br>✅ No additional validation logic needed<br>✅ No need to customize `infer_layout`<br>✅ No need for `get_expand_impl` | 0 lines | ❌ Not needed |
| **Scenario 1: Extend Base Class** | ✅ Main logic matches base class<br>✅ Need input validation, parameter preprocessing, etc.<br>✅ Don't change core `infer_layout` behavior | 10-30 lines | ✅ Needed |
| **Scenario 2: Fully Custom** | ✅ Operator semantics don't match any existing base class<br>✅ Need fully custom layout inference logic<br>✅ May need complex `get_expand_impl` | 50-200 lines | ✅ Needed |

### Decision Impact

| Impact | Scenario 0 | Scenario 1 | Scenario 2 |
|--------|------------|------------|------------|
| **YAML Config** | Directly specify base class | Specify custom subclass | Specify custom class |
| **Python File** | No need to create new class | Create subclass, override individual methods | Create entirely new class |
| **UT Test** | Not needed (base class covered) | Need to test validation logic | Need to test core logic |
| **Maintenance Cost** | Low | Medium | High |

---

## Decision 3: Broadcast Support

### Criteria

| Operator Characteristic | Support Broadcast | Examples |
|------------------------|-------------------|----------|
| **Single input operator** | ❌ Not needed | ReLU, Sigmoid, Tanh |
| **Multi-input operator, shapes must match** | ❌ Not needed | MatMul, Linear |
| **Multi-input operator, shapes can differ** | ✅ Needed | Add, Mul, Greater, LessEqual |

### Decision Impact

| Item | No Broadcast Support | Broadcast Support (WithShape) |
|------|---------------------|------------------------------|
| **suffix config** | No suffix or empty string | `infer_layout_suffix: WithShape` |
| **extra_args** | `[scalar/non-tensor args]` | `[..., input_shapes]` |
| **infer_layout implementation** | Direct inference | Need to handle input_shapes for broadcast alignment |
| **UT Test** | No broadcast scenarios needed | **Must** cover scalar broadcast, shape broadcast |
| **ST Test** | No broadcast scenarios needed | **Must** cover broadcast scenarios |

> **Detailed Configuration**: See [workflow 03: YAML Registration](../workflows/03-yaml-registration.md)

---

## Decision 4: Partial Support

### Criteria

| Scenario | Need Partial | Description |
|----------|--------------|-------------|
| **Contracting dimension sharded MatMul/Linear** | ✅ Yes | Output may be partial, bias needs scaling |
| **Output partial state inconsistent operators (add)** | ✅ May need | Need to adjust input scaling factors |
| **Simple element-wise operations (ReLU, Sigmoid)** | ❌ No | Output is not partial |
| **Operators needing global communication (TopK)** | ✅ Yes | Need AllReduce first to get global info |

### Decision Impact

| Item | No Partial Needed | Need Partial |
|------|-------------------|--------------|
| `_allow_partial_inputs` | `False` (default) | `True` |
| `infer_layout` | Don't set Partial state | Set `output_layout.set_partial_by_dev_axis()` |
| `get_expand_impl` | Return `None` | Implement expand function to handle partial |

---

## Decision 5: get_expand_impl Requirement

### Criteria

| Scenario | Need get_expand_impl | Description |
|----------|---------------------|-------------|
| **Contracting dimension sharded MatMul/Linear** | ✅ Needed | Output partial state may require bias scaling |
| **Input partial state inconsistent operators (Add)** | ✅ May need | Need to adjust some input's scaling factor |
| **Operators needing global communication (TopK)** | ✅ Needed | Need AllReduce first to get global info |
| **Simple element-wise operations (ReLU, Sigmoid)** | ❌ Not needed | Just return None |

### Decision Impact

| Item | No get_expand_impl Needed | Need get_expand_impl |
|------|---------------------------|---------------------|
| **Implementation Complexity** | No additional implementation | Need to implement expand logic |
| **Communication Overhead** | No additional communication | May have AllReduce/AllGather |
| **Test Coverage** | No need to test expand logic | **Must** test communication and scaling |

---

## Decision 6: YAML Suffix Selection

> **Detailed Configuration**: See [workflow 03: YAML Registration](../workflows/03-yaml-registration.md)

### Criteria

| Suffix Value | Applicable Scenario | extra_args Content |
|--------------|---------------------|-------------------|
| **(no suffix)** | Simple operators (no broadcast) | `[scalar/non-tensor args]` |
| **`WithShape`** | **Operators supporting broadcast** ⭐ | `[..., input_shapes]` |
| `Reshape` | Shape transformation | `[target_shape, input_shape]` |
| `Slice` | Slice operations | `[begin, end, global_shape]` |

### Decision Impact

| Impact | Wrong Config Consequence |
|--------|-------------------------|
| **Broadcast operator not configured WithShape** | ST test broadcast scenarios fail, cannot get input_shapes |
| **Wrong suffix configured** | extra_args format mismatch, infer_layout call fails |

---

## Base Class Selection Reference

Select appropriate base class based on operator category:

| Operator Category | Base Class Name | File Location | Need Custom |
|-------------------|-----------------|---------------|-------------|
| **ElementWise** | `ElementWiseDistributedOp` | `parallel_elementwise.py` | ⚙️ Optional |
| **Reduce** | `ReduceDistributedOp` | `parallel_reduce.py` | ⚙️ Optional |
| **Reshape** | `ReshapeDistributedOp` | `parallel_reshape.py` | ⚙️ Optional |
| **MatMul** | `MatMulDistributedOp` | `parallel_matmul.py` | ⚙️ Optional |
| **Gather** | `GatherDistributedOp` | `parallel_gather.py` | ⚙️ Optional |
| **Transpose** | `TransposeDistributedOp` | `parallel_transpose.py` | ⚙️ Optional |
| **Custom** | `DistributedOp` | `parallel_ops.py` | ✅ Must override `infer_layout` |

---

## References

- **YAML Registration Details**: [workflow 03: YAML Registration](../workflows/03-yaml-registration.md)
- **Python Implementation Details**: [workflow 02: Python Implementation](../workflows/02-python-implementation.md)
- **Quick Reference**: [quick-reference.md](./quick-reference.md)
