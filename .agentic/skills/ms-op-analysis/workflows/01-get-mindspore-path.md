# Workflow 1: Get MindSpore Source Code Path

## Goal

Obtain the absolute path to MindSpore source code, which is the foundation for all subsequent analysis.

## Output

- **MindSpore Source Code Path**: Absolute path string
- **Path Validation Result**: Whether the path contains required directories

---

## Path Acquisition Methods

### Method 1: Get from Prompt (Priority)

Check if user prompt already provides MindSpore path:
- Example: "I have MindSpore code locally at `/d/workspace/mindspore`"
- Example: "My MindSpore path is `D:\workspace\mindspore`"

### Method 2: Ask User Directly (if not provided in prompt)

If the prompt doesn't provide a path, must proactively ask:

```
I need to access MindSpore source code to analyze operator distributed implementation.

Please provide the absolute path to MindSpore source code, for example:
- Linux/Mac: /home/user/workspace/mindspore
- Windows: D:\workspace\mindspore

Path requirements:
- Contains mindspore/ops/op_def/yaml/ directory (operator definitions)
- Contains mindspore/ccsrc/frontend/parallel/ops_info/ directory (distributed implementations)
```

---

## Path Validation

```bash
# Validate YAML directory exists
ls {mindspore_path}/mindspore/ops/op_def/yaml/ | head -5

# Validate distributed operator directory exists
ls {mindspore_path}/mindspore/ccsrc/frontend/parallel/ops_info/ | head -5
```

**Validation Success Criteria**: Both commands above can output file list normally.

---

## Save Path

After obtaining and validating the path, save it in context for subsequent workflows to use.

---

## Success Criteria

- [ ] Obtained MindSpore source code absolute path
- [ ] Path validation passed (contains required directories)
- [ ] Path saved in context for subsequent use

---

## Next Step

After path acquisition is complete, proceed to **[Workflow 2: Operator Primitive Query](./02-primitive-query.md)**

**Input**: MindSpore source code path, operator name
**Goal**: Analyze YAML definition and generated primitive class
