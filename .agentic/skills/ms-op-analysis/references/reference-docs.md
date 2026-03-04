# Related File Paths Summary

This document summarizes all file paths that may be needed during MindSpore operator distributed implementation analysis, for quick location.

---

## MindSpore Source Code File Paths

### Operator Primitive Definitions

| File Type | Path | Description |
|-----------|------|-------------|
| **YAML Definitions** | `mindspore/ops/op_def/yaml/` | All operator YAML definition files |
| **Documentation Definitions** | `mindspore/ops/op_def/yaml/doc/` | Operator documentation YAML files |
| **Generation Script** | `mindspore/python/mindspore/ops_generate/gen_ops.py` | YAML → Python/C++ code generator |

**Examples:**
```
mindspore/ops/op_def/yaml/matmul_op.yaml
mindspore/ops/op_def/yaml/add_op.yaml
mindspore/ops/op_def/yaml/doc/matmul_op.yaml
```

### Generated Primitive Code

| File Type | Path | Description |
|-----------|------|-------------|
| **Python Primitives** | `mindspore/python/mindspore/ops/auto_generate/gen_ops_prim.py` | Generated Python operator interfaces |
| **C++ Primitive Definitions** | `mindspore/ops/op_def/auto_generate/` | Generated C++ Primitive definitions |
| **C++ Header Files** | `mindspore/ops/include/primitive/auto_generate/` | Generated C++ header files |
| **PyBoost Code** | `mindspore/pyboost/auto_generate/` | High-performance PyBoost implementation |
| **Device-specific Code** | `mindspore/ops/kernel/{device}/pyboost/auto_generate/` | Device-specific implementation |

### Distributed Implementation (Info Classes)

| File Type | Path | Description |
|-----------|------|-------------|
| **Info Class Source Files** | `mindspore/ccsrc/frontend/parallel/ops_info/` | All operators' distributed implementations |
| **Base Class Definition** | `mindspore/ccsrc/frontend/parallel/operator_info.h` | OperatorInfo base class definition |
| **Intermediate Base Classes** | `mindspore/ccsrc/frontend/parallel/ops_info/` | Base classes for specific operator types |

**Naming Rule:** `{op_name}_info.cc` (Note: Use snake_case version of primitive class)

**Examples:**
```
mindspore/ccsrc/frontend/parallel/ops_info/matmul_info.cc
mindspore/ccsrc/frontend/parallel/ops_info/add_info.cc
mindspore/ccsrc/frontend/parallel/ops_info/flash_attention_score_info.cc
```

### Core Header Files

| Header File | Path | Description |
|-------------|------|-------------|
| `operator_info.h` | `mindspore/ccsrc/frontend/parallel/operator_info.h` | Info class base class definition |
| `tensor_info.h` | `mindspore/ccsrc/frontend/parallel/tensor_info.h` | TensorInfo definition |
| `tensor_layout.h` | `mindspore/ccsrc/frontend/parallel/tensor_layout.h` | TensorLayout definition |
| `strategy.h` | `mindspore/ccsrc/frontend/parallel/strategy.h` | Strategy definition |

---

## Common Commands

### Find Operator YAML File

```bash
# Find specific operator's YAML
cd {mindspore_path}
ls mindspore/ops/op_def/yaml/*matmul*op.yaml
ls mindspore/ops/op_def/yaml/*add*op.yaml

# Find all YAML files
ls mindspore/ops/op_def/yaml/*.yaml | head -20
```

### Find Info Class Implementation

```bash
# Find specific operator's Info class
find . -name "*matmul_info.cc" -o -name "*add_info.cc"

# Find all Info class files
ls mindspore/ccsrc/frontend/parallel/ops_info/*.cc | head -30
```

### Find Base Class Definitions

```bash
# Find OperatorInfo base class
find . -name "operator_info.h"

# Find intermediate base classes
find . -name "arithmetic_info.h"
find . -name "matmul_info_base.h"
```

### Search for Specific Functions

```bash
# Search for InferTensorInfo implementation
grep -r "InferTensorInfo" mindspore/ccsrc/frontend/parallel/ops_info/matmul_info.cc

# Search for replace_graph implementation
grep -r "replace_graph" mindspore/ccsrc/frontend/parallel/ops_info/

# Search for InferOutputTensorInfo implementation
grep -r "InferOutputTensorInfo" mindspore/ccsrc/frontend/parallel/ops_info/
```

---

## Typical Operator File Locations

### Matrix Multiplication Related

| Operator | YAML File | Info Class File |
|----------|-----------|-----------------|
| **MatMul** | `mindspore/ops/op_def/yaml/matmul_op.yaml` | `mindspore/ccsrc/frontend/parallel/ops_info/matmul_info.cc` |
| **BatchMatMul** | `mindspore/ops/op_def/yaml/batch_matmul_op.yaml` | `mindspore/ccsrc/frontend/parallel/ops_info/batch_matmul_info.cc` |

### Element-wise Binary Operators

| Operator | YAML File | Info Class File |
|----------|-----------|-----------------|
| **Add** | `mindspore/ops/op_def/yaml/add_op.yaml` | `mindspore/ccsrc/frontend/parallel/ops_info/add_info.cc` |
| **Sub** | `mindspore/ops/op_def/yaml/sub_op.yaml` | `mindspore/ccsrc/frontend/parallel/ops_info/sub_info.cc` |
| **Mul** | `mindspore/ops/op_def/yaml/mul_op.yaml` | `mindspore/ccsrc/frontend/parallel/ops_info/mul_info.cc` |
| **Div** | `mindspore/ops/op_def/yaml/div_op.yaml` | `mindspore/ccsrc/frontend/parallel/ops_info/div_info.cc` |

### Reduction Operators

| Operator | YAML File | Info Class File |
|----------|-----------|-----------------|
| **ReduceSum** | `mindspore/ops/op_def/yaml/reduce_sum_op.yaml` | `mindspore/ccsrc/frontend/parallel/ops_info/reduce_sum_info.cc` |
| **ReduceMean** | `mindspore/ops/op_def/yaml/reduce_mean_op.yaml` | `mindspore/ccsrc/frontend/parallel/ops_info/reduce_mean_info.cc` |

---

## HyperParallel Local File Paths

### Implementation Files

| File Type | Path | Description |
|-----------|------|-------------|
| **Distributed Operator Implementation** | `hyper_parallel/ops/` | All distributed operators' Python implementations |
| **YAML Registration Files** | `hyper_parallel/ops/` | Operator YAML config |
| **Base Class Definition** | `hyper_parallel/ops/base/` | DistributedOpBase base class |

### Test Files

| File Type | Path | Description |
|-----------|------|-------------|
| **Unit Tests (UT)** | `tests/unit/ops/` | Single-card unit tests |
| **Integration Tests (ST)** | `tests/st/ops/` | Multi-card integration tests |

---

## Directory Structure Overview

```
mindspore/
├── ops/
│   ├── op_def/
│   │   └── yaml/
│   │       ├── matmul_op.yaml          # YAML definition
│   │       ├── add_op.yaml
│   │       └── doc/                    # Documentation definitions
│   ├── include/primitive/
│   │   └── auto_generate/              # C++ header files
│   └── kernel/
│       └── {device}/
│           └── pyboost/auto_generate/  # Device-specific code
├── ccsrc/
│   └── frontend/
│       └── parallel/
│           ├── operator_info.h         # Base class definition
│           ├── tensor_info.h
│           ├── tensor_layout.h
│           └── ops_info/
│               ├── matmul_info.cc      # Info class implementation
│               ├── add_info.cc
│               └── arithmetic_info.h   # Intermediate base class
└── python/
    └── mindspore/
        └── ops/
            ├── auto_generate/
            │   └── gen_ops_prim.py     # Python primitives
            └── generate/
                └── gen_ops.py          # Generator
```
