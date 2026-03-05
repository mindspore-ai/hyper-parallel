# Workflow 3: Call MindSpore Analysis SKILL

## Objective

Reuse the `ms-op-analysis` SKILL to analyze MindSpore operator distributed implementation.

## Input

- **MindSpore Operator Name**: Mapping result from Workflow 2
- **MindSpore Source Code Path**: User-provided path

## Output

- **MindSpore Distributed Implementation Analysis Results**:
  - Info class analysis
  - Layout inference logic
  - Graph replacement logic
  - Reference code annotation

---

## Invocation Methods

### Method 1: Direct SKILL Invocation

Invoke the `ms-op-analysis` SKILL, passing MindSpore operator name and source code path:

```
Invoke /skill ms-op-analysis
Input:
- Operator Name: {MindSpore operator name}
- MindSpore Source Code Path: {path}
```

### Method 2: Reuse Analysis Results

If the MindSpore operator has already been analyzed, directly reuse existing analysis reports:

```
Read existing report:
.agentic/skills/dist-op-dev/analysis-results/{OpName}-analysis.md
```

---

## Analysis Content Reuse

### Directly Reusable Content

| Analysis Content | Reuse Method | Description |
|----------|----------|------|
| **Info Class Structure** | Direct reuse | MindSpore distributed implementation structure |
| **Layout Inference Logic** | Direct reuse | infer_layout implementation reference |
| **Graph Replacement Logic** | Direct reuse | get_expand_impl implementation reference |
| **Communication Operators** | Direct reuse | AllReduce/AllGather, etc. |

### Content Requiring Adaptation

| Analysis Content | Adaptation Method | Description |
|----------|----------|------|
| **Parameter Mapping** | Add mapping code | PyTorch parameters → MindSpore parameters |
| **YAML Registration** | Use PyTorch naming | Register as `torch.{op}` |
| **Test Cases** | Use PyTorch interface | UT/ST use PyTorch syntax |

---

## Analysis Result Integration

### Integration Key Points

1. **Preserve MindSpore References**: Distributed implementation reference sources remain unchanged
2. **Add PyTorch Mapping**: Add PyTorch interface information to analysis report
3. **Annotate Differences**: Clearly annotate differences between the two platforms

### Integration Format

```markdown
## Distributed Implementation Plan

### MindSpore Reference Sources
- **Info Class**: `MatMulInfo`
- **File Location**: `mindspore/ccsrc/frontend/parallel/ops_info/matmul_info.cc`
- **Inference Logic**: ...

### PyTorch Adaptation
- **PyTorch Interface**: `torch.matmul(input, other)`
- **Parameter Mapping**: No difference
- **Behavior Difference**: None
```

---

## Success Criteria

- [ ] Invoked `ms-op-analysis` SKILL
- [ ] Obtained MindSpore distributed implementation analysis results
- [ ] Understood Layout inference logic
- [ ] Understood graph replacement logic (if applicable)
- [ ] Documented reference code sources

---

## Next Step

After MindSpore analysis is complete, proceed to **[Workflow 4: Integrate Analysis Results](./04-integrate-results.md)**

**Input**: PyTorch interface information, MindSpore analysis results
**Objective**: Integrate analysis results from both platforms
