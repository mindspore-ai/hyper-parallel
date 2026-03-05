---
name: dist-op-dev
description: HyperParallel distributed operator development main workflow. Automatically calls analysis tools to complete the entire process from operator analysis to code push. Users only need to call this SKILL, no need to directly call other analysis SKILLs.
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

Distributed operator development follows a **6-step process**, from operator analysis to code push:

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
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  4. Unit Test    │ ──▶ │  5. Integration │ ──▶ │  6. Git Commit   │
│     (UT)         │     │     Test (ST)   │     │  & PR Creation   │
│  Verify inference│     │  8-card verify  │     │  Call autogit    │
│  Cover DP/MP     │     │  Compare output │     │  Create branch   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Workflow Execution Checklist

When using this SKILL to develop distributed operators, create a TODOLIST, then execute the following workflows in order:

- [ ] **[Step 1](workflows/01-operator-analysis.md)**: Operator Analysis

  - Goal: Get operator interface definition, distributed implementation plan, implementation reference
  - Input: Operator name, platform type
  - Output: Analysis report file `.agentic/skills/dist-op-dev/analysis-results/{OpName}-analysis.md` (🔴required)

- [ ] **[Step 2](workflows/02-python-implementation.md)**: Python Implementation

  - Goal: Create distributed operator implementation class, implement infer_layout and get_expand_impl
  - Input: Analysis report from Step 1
  - Output: `hyper_parallel/core/shard/ops/parallel_*.py` file

- [ ] **[Step 3](workflows/03-yaml-registration.md)**: YAML Registration

  - Goal: Register operator in YAML config file, configure infer_layout_suffix
  - Input: Analysis report from Step 1, Python implementation class info from Step 2
  - Output: `hyper_parallel/core/shard/ops/yaml/*.yaml` entry

- [ ] **[Step 4](workflows/04-unit-testing.md)**: Unit Testing (UT)

  - Goal: Verify infer_layout logic correctness, cover various layout combinations
  - Input: Python implementation class from Step 2
  - Output: `tests/mindspore/ut/parallel_ops_infer/test_parallel_*.py`

- [ ] **[Step 5](workflows/05-integration-testing.md)**: Integration Testing (ST)

  - Goal: Verify end-to-end distributed execution correctness in 8-card environment
  - Input: YAML config from Step 3, Python implementation from Step 2
  - Output: `tests/mindspore/st/shard/ops/test_ops_*.py`

- [ ] **[Step 6](workflows/06-git-commit.md)**: Git Commit and PR Creation

  - Goal: Create feature branch, call autogit to complete lint check, commit, push, and create PR if needed
  - Input: All modified code, operator name
  - Output: Feature branch `feat/{OpName}-distributed-support`, commit pushed, PR created (if needed)

---

## Key Decision Points

| Decision Point | Criteria | Options | Impact |
|----------------|----------|---------|--------|
| **Operator Category** | Semantic matching | ElementWise/MatMul/Reduce/Reshape/Gather | Determines base class and YAML file |
| **Implementation Method** | Need custom logic | Scenario 0/Scenario 1/Scenario 2 | Code volume and UT coverage |
| **Broadcast Support** | Support broadcasting | No suffix/WithShape | YAML config and test scenarios |
| **Partial Support** | Handle partial state | _allow_partial_inputs=True/False | get_expand_impl implementation |

> **Detailed decision reference**: See [references/implementation-decisions.md](references/implementation-decisions.md)

---

## Quick Reference

### File Location Quick Reference

| Task | File Location | Key Notes |
|------|---------------|-----------|
| Python Implementation | `hyper_parallel/core/shard/ops/parallel_*.py` | Inherit `DistributedOp` or its subclass |
| YAML Registration | `hyper_parallel/core/shard/ops/yaml/*.yaml` | Configure operator to distributed implementation class mapping |
| Unit Test (UT) | `tests/mindspore/ut/parallel_ops_infer/` | Platform-agnostic, verify `infer_layout` logic correctness |
| Integration Test (ST) | `tests/mindspore/st/shard/ops` `tests/torch/shard/ops` | 8-card environment verify distributed execution correctness |

> **Detailed quick reference**: See [references/quick-reference.md](references/quick-reference.md)

### Platform Differences

| Item | MindSpore | PyTorch |
|------|-----------|---------|
| **Operator Name Style** | PascalCase (e.g., `Add`, `MatMul`) | snake_case (e.g., `add`, `matmul`) |
| **YAML Files** | `element_wise_ops.yaml`, `matmul_ops.yaml`, etc. | `torch_*.yaml` |
| **Test Directories** | `tests/mindspore/ut/parallel_ops_infer/` | `tests/torch/shard/ops/` |

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
