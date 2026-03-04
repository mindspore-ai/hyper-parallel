# Workflow 2: Operator Primitive Query

## Goal

Read YAML definition, extract operator's input, output, and parameter information.

## Input

- **MindSpore Source Code Path**: From Workflow 1
- **Operator Name**: User-provided operator name

## Output

- **YAML Primitive Definition**: Input parameters, output info, attribute parameters
- **Primitive Class Info**: Generated Python class name, YAML file name, naming mapping

---

## Operator Name Mapping Rules

| Level | Naming Style | Examples |
|-------|--------------|----------|
| **YAML File Name** | snake_case | `matmul_op.yaml` |
| **Operator Definition Name** | snake_case | `matmul:` |
| **Primitive Class Name** | PascalCase | `MatMul` |
| **Distributed Info Class Name** | PascalCase + Info suffix | `MatMulInfo` |

**Conversion Formula:**
```
YAML name: matmul_op.yaml → Operator name: matmul → Class name: MatMul → Info class: MatMulInfo
YAML name: batch_matmul_op.yaml → Operator name: batch_matmul → Class name: BatchMatMul → Info class: BatchMatMulInfo
```

---

## Primitive Definition File Locations

```
mindspore/ops/op_def/yaml/{op_name}_op.yaml        # YAML primitive definition
mindspore/ops/op_def/yaml/doc/{op_name}_op.yaml   # Documentation definition
mindspore/python/mindspore/ops/auto_generate/      # Generated Python classes
```

---

## Primitive Generation Flow

**Generation Command:**
```bash
python mindspore/python/mindspore/ops_generate/gen_ops.py
```

**Generation Process Description**:
1. Input: Read `mindspore/ops/op_def/yaml/*.yaml` and `mindspore/ops/op_def/yaml/doc/*.yaml`
2. Process: Parse YAML definition, extract parameters, attributes, input/output info
3. Output: Generate complete operator primitive class definition

**Generated File Locations:**

| File Type | Generated Path | Content Description |
|-----------|----------------|---------------------|
| Python Primitive Class | `mindspore/python/mindspore/ops/auto_generate/gen_ops_prim.py` | Operator Python interface |
| C++ Primitive Definition | `mindspore/ops/op_def/auto_generate/` | C++ Primitive definition |
| C++ Header Files | `mindspore/ops/include/primitive/auto_generate/` | C++ header files |
| PyBoost Code | `mindspore/pyboost/auto_generate/` | High-performance PyBoost implementation |
| Device-specific Code | `mindspore/ops/kernel/{device}/pyboost/auto_generate/` | Device-specific implementation |

---

## Analysis Points

| Analysis Dimension | Check Items | Description |
|--------------------|-------------|-------------|
| **Input Parameters** | Parameter name, type, required, default value, constraints | Clarify input/output shapes, types, special parameters |
| **Output Info** | Output name, type, shape changes | Shape changes (preserved, transformed, etc.) |
| **Attribute Parameters** | prim_init attributes, default values, constraints | Primitive constructor parameters |
| **Parameter Types** | dtype: tensor/int/float/bool/tuple/list/... | Supported data types |
| **Parameter Handlers** | arg_handler (string conversion, etc.) | Special parameter handling |

---

## Success Criteria

- [ ] Found and read YAML definition file
- [ ] Extracted complete input parameter information
- [ ] Understood parameter constraints
- [ ] Understood output information
- [ ] Recorded naming mapping relationship

---

## Next Step

After primitive query is complete, proceed to **[Workflow 3: Distributed Operator Query](./03-distributed-op-query.md)**

**Input**: MindSpore source code path, operator name, naming mapping
**Goal**: Analyze Info class registration and inheritance structure
