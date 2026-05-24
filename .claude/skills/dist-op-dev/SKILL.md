---
name: dist-op-dev
description: Execution-oriented workflow for HyperParallel distributed operator development. Analyzes the operator, implements or updates code and tests.
---

# HyperParallel Distributed Operator Development Workflow

> ✅ 【Unified Entry】When developing HyperParallel distributed operators, **just call this SKILL**, and I will automatically handle the entire process including operator analysis, implementation, testing, etc.

## When to Use This Workflow

Use this workflow when developers need to add distributed operator support for the HyperParallel framework or optimize sharding strategy inference for existing operators.

## How to Use

Call this SKILL directly, providing the MindSpore mint interface name or PyTorch operator name, along with source code paths:

```bash
# Develop distributed support for MindSpore mint interface
/dist-op-dev I want to develop distributed support for MindSpore mint interface mint.matmul. MindSpore source code is at /root/workspace/mindspore, PyTorch source code is at /root/workspace/pytorch.

# Develop distributed support for PyTorch operator
/dist-op-dev I want to develop distributed support for PyTorch operator torch.nn.functional.linear. MindSpore source code is at /root/workspace/mindspore, PyTorch source code is at /root/workspace/pytorch.
```

**Source code paths are required** — the dist-op-analysis SKILL needs them to locate interface definitions, Primitive mappings, and distributed strategy references.

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
  - Input: MindSpore mint Interface, PyTorch Interface, MindSpore Source Code Path,PyTorch Source Code Path
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
  - Goal: Verify infer_layout and get_expand_impl logic correctness, cover supported/unsupported scenarios
  - Input: Python implementation class from Step 2, analysis report from Step 1
  - Output: `tests/ut/core/shard/ops/test_parallel_*.py`

- [ ] **[Step 5](workflows/05-integration-testing.md)**: Integration Testing (ST)

  - Must: The test generation process must follow the procedure described in **workflows/05-integration-testing.md**. Execute each step in order.
  - Goal: Verify end-to-end distributed execution correctness in 8-card environment
  - Input: YAML config from Step 3, Python implementation from Step 2, analysis report from Step 1
  - Output: `tests/mindspore/st/shard/ops/test_parallel_op_*.py` + `_test_parallel_op_*.py` or `tests/torch/shard/ops/test_parallel_op_*.py` + `_test_parallel_op_*.py`

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

**Detailed decision reference:** See [Implementation Decisions](references/implementation-decisions.md)
---

## Quick Reference

### File Location Quick Reference

| Task | File Location | Key Notes |
|------|---------------|-----------|
| Python Implementation | `hyper_parallel/core/shard/ops/parallel_*.py` | Inherit `DistributedOp` or its subclass |
| YAML Registration | `hyper_parallel/core/shard/ops/yaml/*.yaml` | Configure operator to distributed implementation class mapping |
| Unit Test (UT) | `tests/ut/core/shard/ops/` | Platform-agnostic, verify `infer_layout` and `get_expand_impl` logic |
| Integration Test (ST) | `tests/mindspore/st/shard/ops/` `tests/torch/shard/ops/` | 8-card environment verify distributed execution |

> **Detailed quick reference**: See [references/quick-reference.md](references/quick-reference.md)

### Platform Differences

| Item | MindSpore | PyTorch |
|------|-----------|---------|
| **Interface Name Style** | mint.matmul, mint.nn.functional.relu | torch.matmul, torch.nn.functional.linear |
| **YAML Files** | `element_wise_ops.yaml`, `matmul_ops.yaml`, etc. | `torch_*.yaml` |
| **UT Test Directory** | `tests/ut/core/shard/ops/` (shared) | `tests/ut/core/shard/ops/` (shared) |
| **ST Test Directories** | `tests/mindspore/st/shard/ops/` | `tests/torch/shard/ops/` |

**Important Note:** If MindSpore operator and PyTorch operator have the same semantics, they **can reuse the same distributed operator implementation class**.

---

## Related SKILLs

| SKILL | Purpose | When Called |
|-------|---------|-------------|
| **autogit** | Git workflow automation (commit, pr, status, etc.) | Workflow 6, complete code commit and PR creation |
| **dist-op-analysis** | Internal operator analysis (read-only) | Workflow 1, provides interface specs, distributed strategies, and HyperParallel implementation guidance |

---

## Reference Document Paths

- **Workflow detailed steps**: `workflows/` directory
- **Knowledge reference documents**: `references/` directory

  - [Quick Reference](references/quick-reference.md)
  - [Implementation Decisions](references/implementation-decisions.md)
  - [Code Standards](references/code-standards.md)

- **Template files**: `templates/operator-analysis-template.md`
