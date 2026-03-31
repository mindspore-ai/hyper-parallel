---
name: dist-op-dev
description: Execution-oriented workflow for HyperParallel distributed operator development. Analyzes the operator, implements or updates code and tests.
---

# HyperParallel Distributed Operator Development Workflow

> ✅ 【Unified Entry】When developing HyperParallel distributed operators, **just call this SKILL**, and I will automatically handle the entire process including operator analysis, implementation, testing, etc.
>
> 🚫 **Do not call directly**: `ms-op-analysis` and `pt-op-analysis` are internal tool SKILLs, automatically called by this SKILL, **no need for users to use directly**.

## When to Use This Workflow

Use this workflow when developers need to add distributed operator support for the HyperParallel framework or optimize sharding strategy inference for existing operators.

## How to Use

Call this SKILL directly, providing the operator name and platform type:

```bash
# Develop distributed support for MindSpore operator
/dist-op-dev I want to develop distributed support for MindSpore operator MatMul, I have MindSpore source code locally at /d/workspace/mindspore for reference.

# Develop distributed support for PyTorch operator
/dist-op-dev I want to develop distributed support for PyTorch operator torch.add, I have MindSpore source code locally at /d/workspace/mindspore for reference.
```

---

## Terminology Definitions

| Term | Definition | Details |
|------|------|----------|
| **Scenario 0/1/2** | Implementation method classification: Scenario 0=fully use base class, Scenario 1=extend base class, Scenario 2=fully custom | See [implementation-decisions.md](references/implementation-decisions.md) |
| **WithShape** | A type of `infer_layout_suffix`, used for operators that support broadcasting | See [workflow 03](workflows/03-yaml-registration.md) |
| **DP** | Data Parallel, data is sharded along batch dimension | - |
| **MP** | Model Parallel, model parameters are sharded along some dimension | - |
| **Hybrid** | Hybrid Parallel, using both DP and MP | - |
| **Partial** | Partial state of a tensor, indicating the tensor only contains partial data along some dimension | See [implementation-decisions.md](references/implementation-decisions.md) |
| **TensorMap** | Describes the sharding strategy for each dimension of a tensor, -1 means not sharded | - |

---

## Execution Flow Overview

Distributed operator development follows a **5-step process**, from operator analysis to code push:

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. Operator     │ ──▶ │  2. Python      │ ──▶ │  3. YAML        │
│     Analysis     │     │     Implement   │     │     Registration│
│  Call SKILL      │     │  Inherit/Custom │     │  Configure map  │
│  🔴Output report │     │  infer_layout   │     │  Select suffix  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                           │
            ┌───────────────────────────────────────────────┘
            ▼
┌─────────────────┐     ┌─────────────────┐
│  4. Unit Test    │ ──▶ │  5. Integration │
│     (UT)         │     │     Test (ST)   │
│  Verify inference│     │  8-card verify  │
│  Cover DP/MP     │     │  Compare output │
└─────────────────┘     └─────────────────┘
```

### Workflow Execution Checklist

When using this SKILL to develop distributed operators, create a TODOLIST, then execute the following workflows in order:

- [ ] **[Step 1](workflows/01-operator-analysis.md)**: Operator Analysis

  - The operator analysis process must follow the procedure described in **workflows/01-operator-analysis.md**. Execute each step in order.
  - Goal: Get operator interface definition, distributed implementation plan, implementation reference
  - Input: Operator name, platform type
  - Output: Analysis report file `.claude/skills/dist-op-dev/analysis-results/{OpName}-analysis.md` (🔴required)

- [ ] **[Step 2](workflows/02-python-implementation.md)**: Python Implementation

  - Must: The Python implementation process must follow the procedure described in **workflows/02-python-implementation.md**. Execute each step in order.
  - Goal: Create distributed operator implementation class, implement infer_layout and get_expand_impl
  - Input: Analysis report from Step 1
  - Output: `hyper_parallel/core/shard/ops/parallel_*.py` file

- [ ] **[Step 3](workflows/03-yaml-registration.md)**: YAML Registration

  - Must: The yaml registration process must follow the procedure described in **workflows/03-yaml-registration.md**. Execute each step in order.
  - Goal: Register operator in YAML config file, configure infer_layout_suffix
  - Input: Analysis report from Step 1, Python implementation class info from Step 2
  - Output: `hyper_parallel/core/shard/ops/yaml/*.yaml` entry

- [ ] **[Step 4](workflows/04-unit-testing.md)**: Unit Testing (UT)

  - Must: The test generation process must follow the procedure described in **workflows/04-unit-testing.md**. Execute each step in order.
  - Goal: Verify infer_layout logic correctness, cover various layout combinations
  - Input: YAML config from Step 3, Python implementation class from Step 2
  - Output: `tests/mindspore/ut/parallel_ops_infer/test_parallel_{operator_name}.py` file

- [ ] **[Step 5](workflows/05-integration-testing.md)**: Integration Testing (ST)

  - Must: The test generation process must follow the procedure described in **workflows/05-integration-testing.md**. Execute each step in order.
  - Goal: Verify end-to-end distributed execution correctness in 8-card environment
  - Input: YAML config from Step 3, Python implementation from Step 2, Operator semantics from Step 1
  - Output: `tests/mindspore/st/shard/ops/test_ops_{operator_name}.py` file

---

## Key Decision Points

| Decision Point | Criteria | Options | Impact |
|----------------|----------|---------|--------|
| **Operator Category** | Semantic matching | ElementWise/MatMul/Reduce/Reshape/Gather | Determines base class and YAML file |
| **Implementation Method** | Need custom logic | Scenario 0/Scenario 1/Scenario 2 | Code volume and UT coverage |
| **Broadcast Support** | Support broadcasting | No suffix/WithShape | YAML config and test scenarios |
| **Partial Support** | Handle partial state | _allow_partial_inputs=True/False | get_expand_impl implementation |
**Detailed decision reference:** See [Implementation Decisions](references/implementation-decisions.md)
---

## Quick Reference

### File Location Quick Reference

| Task | File Location | Key Notes |
|------|---------------|-----------|
| Python Implementation | `hyper_parallel/core/shard/ops/parallel_*.py` | Inherit `DistributedOp` or its subclass |
| YAML Registration | `hyper_parallel/core/shard/ops/yaml/*.yaml` | Configure operator to distributed implementation class mapping |
| Unit Test (UT) | `tests/mindspore/ut/parallel_ops_infer/` | Platform-agnostic, verify `infer_layout` `get_expand_impl`(if contains) logic correctness |
| Integration Test (ST) | `tests/mindspore/st/shard/ops` `tests/torch/shard/ops` | 8-card environment verify distributed execution correctness |
**Detailed Test(ST):** See [Quick Reference](references/quick-reference.md)

### Platform Differences

| Item | MindSpore | PyTorch |
|------|-----------|---------|
| **Operator Name Style** | PascalCase (e.g., `Add`, `MatMul`) | snake_case (e.g., `add`, `matmul`) |
| **YAML Files** | `element_wise_ops.yaml`, `matmul_ops.yaml`, etc. | `torch_*.yaml` |
| **Integration Test Directories** | `tests/mindspore/st/shard/` | `tests/torch/shard/ops/` |

Unit Test cases are all placed in the directory `tests/mindspore/ut/parallel_ops_infer/`.
**Important Note:** If MindSpore operator and PyTorch operator have the same semantics, they **can reuse the same distributed operator implementation class**.

---

## Related SKILLs

| SKILL | Purpose | When Called |
|-------|---------|-------------|
| **autogit** | Git workflow automation (commit, pr, status, etc.) | Workflow 6, complete code commit and PR creation |
| **ms-op-analysis** | MindSpore operator analysis | Workflow 1, automatically called when analyzing MindSpore operators |
| **pt-op-analysis** | PyTorch operator analysis | Workflow 1, automatically called when analyzing PyTorch operators |

---

## Reference Document Paths

- **Workflow detailed steps**: `workflows/` directory
- **Knowledge reference documents**: `references/` directory

  - [Quick Reference](references/quick-reference.md)
  - [Implementation Decisions](references/implementation-decisions.md)
  - [Code Standards](references/code-standards.md)

- **Template files**: `templates/operator-analysis-template.md`
