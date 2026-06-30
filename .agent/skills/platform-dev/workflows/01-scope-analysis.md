# Workflow 1: Scope Analysis

## Goal

Identify the scope of changes, affected files, existing patterns, and determine the change strategy.

## Steps

### 1.1 Classify the Change Type

| Change Type | Description | Example |
|------------|-------------|---------|
| **New Platform API** | Add new method to Platform base class + both backends | Add `scatter()` collective |
| **Feature Implementation** | Implement feature in one or both backends | Add activation swap for MindSpore |
| **Bug Fix** | Fix existing platform code | Fix stream sync in FSDP scheduler |
| **Refactor** | Restructure without behavior change | Split large state.py into modules |
| **Extension** | Extend DTensorBase or existing feature | Add property to DTensorBase |

### 1.2 Identify Affected Files

Read the following files to understand existing patterns:

```bash
# Base class — understand existing API surface
Read: hyper_parallel/platform/platform.py

# Torch implementation — check existing patterns
Read: hyper_parallel/platform/torch/platform.py

# MindSpore implementation — check parity
Read: hyper_parallel/platform/mindspore/platform.py
```

For feature-specific changes, also read the relevant submodule:

| Feature | Torch Files | MindSpore Files |
|---------|-------------|-----------------|
| FSDP / HSDP | `platform/torch/fully_shard/*.py` + `core/fully_shard/hsdp_*.py` | `platform/mindspore/fully_shard/*.py` + same `core/fully_shard/` |
| Pipeline | `platform/torch/pipeline_parallel/*.py` | `platform/mindspore/pipeline_parallel/*.py` |
| Activation | `platform/torch/activation_checkpoint/*.py` | `platform/mindspore/activation_checkpoint/*.py` |
| DTensor | `platform/torch/dtensor.py` | `platform/mindspore/dtensor.py` |

### 1.3 Check Cross-Platform Impact

For every file you plan to modify, check if the corresponding file on the other backend needs changes:

- `platform/torch/X.py` ↔ `platform/mindspore/X.py`
- `platform/torch/fully_shard/state.py` ↔ `platform/mindspore/fully_shard/state.py`

### 1.4 Identify Callers

Search for usage of the API you're modifying:

```bash
# Find callers in core/
Grep: pattern="platform\\.method_name" path="hyper_parallel/core/"

# Find callers in tests/
Grep: pattern="platform\\.method_name" path="tests/"
```

## Output

- List of files to modify (with change type: new/modify/delete)
- Cross-platform impact assessment
- List of callers that may need updates
- Recommended implementation order

## Next Step

Proceed to **[Workflow 2: Base Class API Design](./02-base-class-api.md)**
