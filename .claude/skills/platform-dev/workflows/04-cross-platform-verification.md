# Workflow 4: Cross-Platform Verification

## Goal

Ensure feature parity between PyTorch and MindSpore backends, consistent semantics, and no abstraction violations.

## Steps

### 4.1 API Parity Check

Compare the two backend implementations side-by-side:

| Check | Criteria |
|-------|----------|
| **Same method signature** | Parameters, types, defaults match |
| **Same return type semantics** | Both return same logical type (even if framework-specific) |
| **Same error behavior** | Same exceptions for same invalid inputs |
| **Same async semantics** | Both support async_op consistently |
| **Same side effects** | Cache updates, global state changes match |

### 4.2 Type Mapping Verification

Verify that cross-platform type differences are handled correctly:

| Concept | Torch | MindSpore | Verification |
|---------|-------|-----------|-------------|
| Device | `torch.device` | `str` | Callers use platform-agnostic API |
| Process group | `ProcessGroup` | `str` | Callers don't assume concrete type |
| Module | `nn.Module` | `nn.Cell` | Callers use `Platform.Module` |
| Parameter | `nn.Parameter` | `ms.Parameter` | Callers use `Platform.Parameter` |

### 4.3 Abstraction Leak Check

Search for direct framework imports in platform-agnostic code:

```bash
# Should find NO matches in core/ (except in platform/ itself)
Grep: pattern="^import torch" path="hyper_parallel/core/"
Grep: pattern="^import mindspore" path="hyper_parallel/core/"
Grep: pattern="^from torch" path="hyper_parallel/core/"
Grep: pattern="^from mindspore" path="hyper_parallel/core/"
```

### 4.4 Caller Compatibility Check

For every caller of the modified API:
- Verify it works with both backend return types
- Verify it handles async/sync modes correctly
- Verify error handling matches both backends

### 4.5 Process Group Cache Consistency

If modifying group management:
- Verify `EXISTING_COMM_GROUPS` key format is consistent
- Verify cache invalidation/cleanup is handled

## Output

- Cross-platform parity report
- List of any parity gaps (with justification if intentional)
- Confirmation of no abstraction violations

## Next Step

Proceed to **[Workflow 5: Testing](./05-testing.md)**
