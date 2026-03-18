# Workflow 6: Git Commit & PR

## Goal

Create feature branch, commit changes with proper message, push, and optionally create PR.

## Steps

### 6.1 Branch Naming

| Change Type | Branch Format | Example |
|------------|---------------|---------|
| New feature | `feat/platform-{feature}` | `feat/platform-scatter-collective` |
| Bug fix | `fix/platform-{issue}` | `fix/platform-fsdp-stream-sync` |
| Refactor | `refactor/platform-{scope}` | `refactor/platform-hsdp-state` |

### 6.2 Commit Message

Follow project conventions:

```
<type>(platform): <description>

- <detail 1>
- <detail 2>
```

**Examples:**
```
feat(platform): add scatter collective operation

- Add scatter() abstract method to Platform base class
- Implement in TorchPlatform using dist.scatter
- Implement in MindSporePlatform using comm_func.scatter
- Add UT for both backends
```

```
fix(platform): fix stream sync in FSDP unshard scheduler

- Add event.wait() before reading all-gather output in TorchFSDPScheduler
- Add matching fix in MindSporeFSDPScheduler
- Add ST to verify correct behavior under async all-gather
```

### 6.3 Commit via autogit

```bash
# Stage, lint check, commit, push
/commit -m "feat(platform): add scatter collective operation"

# Or create PR
/create-pr
```

### 6.4 Pre-Commit Checklist

- [ ] No direct torch/mindspore imports in platform-agnostic code
- [ ] New Platform API defined in base class (`platform.py`)
- [ ] Both backends implemented (or explicit NotImplementedError)
- [ ] Stream safety verified (handle.wait, event sync)
- [ ] Memory safety verified (resize_(0), no freed access)
- [ ] Tests added and passing
- [ ] Pylint clean
- [ ] Apache 2.0 license header on all new files

## Output

- Feature branch with clean commit
- PR created (if requested)
