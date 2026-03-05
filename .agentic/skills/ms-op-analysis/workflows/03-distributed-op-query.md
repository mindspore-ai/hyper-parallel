# Workflow 3: Distributed Operator Query

## Goal

Locate and analyze MindSpore operator's distributed implementation class (Info class), understand its registration mechanism and inheritance structure.

## Input

- **MindSpore Source Code Path**: From Workflow 1
- **Operator Name**: User-provided operator name
- **Naming Mapping**: From Workflow 2 (YAML name → Class name → Info class name)

## Output

- **Info Class File Path**: Source file location of distributed implementation
- **Base Class Inheritance**: Which Info base class is inherited
- **Registration Info**: How operator is registered in the system
- **Implementation Summary**: List of key methods for distributed implementation

---

## Info Class Naming Rules

Based on Workflow 2 naming mapping rules, Info class name is: `{OperatorClassName}Info`

| Operator Name | Primitive Class Name | Info Class Name |
|---------------|---------------------|-----------------|
| `matmul` | `MatMul` | `MatMulInfo` |
| `batch_matmul` | `BatchMatMul` | `BatchMatMulInfo` |
| `add` | `Add` | `AddInfo` |
| `flash_attention_score` | `FlashAttentionScore` | `FlashAttentionScoreInfo` |

---

## Info Class File Location

Info class files are located in `mindspore/ccsrc/frontend/parallel/ops_info/` directory:

```
mindspore/ccsrc/frontend/parallel/ops_info/
├── matmul_info.cc              # MatMulInfo
├── batch_matmul_info.cc        # BatchMatMulInfo
├── add_info.cc                 # AddInfo
├── flash_attention_score_info.cc  # FlashAttentionScoreInfo
└── ...
```

**File Naming Rule:** `{op_name}_info.cc` (Note: Use snake_case version of primitive class, not YAML name)

---

## Info Class Analysis Steps

### Step 1: Locate Info Class File

```bash
# Search for Info class definition file
cd {mindspore_path}
find . -name "*matmul_info.cc" -o -name "*add_info.cc"

# Common locations
ls mindspore/ccsrc/frontend/parallel/ops_info/{op}_info.cc
```

### Step 2: Analyze Class Definition

Read Info class file, extract the following information:

```cpp
// Typical Info class structure
class MatMulInfo : public OperatorInfo {  // Inherit base class
 public:
  MatMulInfo(const std::string &name, const Shapes &input_shapes, const Shapes &output_shapes)
      : OperatorInfo(name, input_shapes, output_shapes) {}

  ~MatMulInfo() override = default;

  // Distributed inference core methods
  Status Infer(const OperatorParams &inputs, const OperatorParams &outputs) override;
  Status InferTensorMap() override;           // Strategy flow: Infer TensorMap
  Status InferTensorInfo() override;          // Layout flow: Infer TensorInfo
  std::vector<StrategyPtr> GenerateStrategies(int32_t stage_id) override;

  // Graph replacement methods
  Status ReplaceNodeInputOrAttrs() override;   // Input replacement
  Status replace_graph(const CNodePtr &cnode) override;  // Subgraph replacement

  // Other methods
  Status InferDevMatrixShape() override;
  Status InferMirrorOps() override;
};
```

**Information to Record:**
1. **Base Class**: Inherited from `OperatorInfo` or subclass (e.g., `MatMulInfoBase`)
2. **Key Methods**: Which distributed inference methods are implemented
3. **Custom Logic**: Whether there is special distributed implementation

### Step 3: Analyze Base Class Inheritance Chain

Some operator Info classes may inherit from more specific base classes:

```bash
# Find base class definition
grep -r "class MatMulInfoBase" mindspore/ccsrc/frontend/parallel/ops_info/
```

**Common Inheritance Structures:**

```
OperatorInfo (Base class)
    ↓
ArithmeticInfo (Binary operator base class)
    ↓
AddInfo, SubInfo, MulInfo, ...

OperatorInfo
    ↓
MatMulInfoBase (Matrix multiplication base class)
    ↓
MatMulInfo, BatchMatMulInfo
```

### Step 4: Check Operator Registration

Info classes are associated with operator primitives through registration mechanism:

```cpp
// Registration example
REGISTER(MatMulInfo);
```

**Search for registration code:**
```bash
grep -r "REGISTER(MatMulInfo)" {mindspore_path}/mindspore/ccsrc/
```

---

## Key Method Descriptions

### Strategy Flow Methods

| Method | Description | hyper-parallel Implementation |
|--------|-------------|------------------------------|
| `GenerateStrategies()` | Generate all possible distributed strategies | Strategy inference in `infer_layout` |
| `InferTensorMap()` | Infer output TensorMap distribution | tensor_map return value in `infer_layout` |
| `InferDevMatrixShape()` | Infer device matrix shape | device_mesh usage in `infer_layout` |

### Layout Flow Methods

| Method | Description | hyper-parallel Implementation |
|--------|-------------|------------------------------|
| `InferTensorInfo()` | Infer output TensorInfo | Return value in `infer_layout` |
| `InferOutputTensorInfo()` | Output inference in Layout flow | Core implementation of `infer_layout` |

### Graph Replacement Methods

| Method | Description | hyper-parallel Implementation |
|--------|-------------|------------------------------|
| `ReplaceNodeInputOrAttrs()` | Replace input tensors or modify attributes | Attribute modification in `get_expand_impl` |
| `replace_graph()` | Replace original node with communication operator subgraph | Graph construction in `get_expand_impl` |

---

## Success Criteria

- [ ] Located Info class source file
- [ ] Clarified Info class base class inheritance
- [ ] Identified key implementation methods (InferTensorMap/InferTensorInfo/replace_graph, etc.)
- [ ] Recorded registration info
- [ ] Understood distributed implementation type (Layout only/Strategy only/Layout+Strategy hybrid)

---

## Next Step

After Info class query is complete, proceed to **[Workflow 4: Practical Query Steps](./04-practical-query.md)**

**Input:** MindSpore source code path, operator name, Info class file path, key method list
**Goal:** Execute complete query flow and handle fallback when implementation not found
