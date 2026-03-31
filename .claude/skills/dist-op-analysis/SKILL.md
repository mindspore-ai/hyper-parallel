---
name: dist-op-analysis
description: Internal analysis tool for distributed operator development — provides interface specs, Primitive/ATen mappings and HyperParallel layout derivation logic. Used by dist-op-dev workflow. NOT for direct user calls.
---

# HyperParallel Distributed Operator Analysis

> ⚠️ **Internal SKILL** — This SKILL is invoked automatically by `dist-op-dev`. **Do NOT call it directly.**

## Purpose

This SKILL provides read-only analysis for HyperParallel distributed operator development. Given a MindSpore mint or PyTorch op name, it explores framework source code to extract:

- Interface specifications (full parameter signatures)
- Primitive/ATen mappings
- Distributed sharding strategies
- HyperParallel layout derivation logic

## When Used

This SKILL is automatically called by **Workflow 1: Operator Analysis** in the `dist-op-dev` SKILL. It should never be invoked directly by users.

---

## Workflow

### Step 0: Locate Framework Source Paths

If the user provides a path, use it. Otherwise discover installed packages:

```bash
pip show mindspore | grep Location   # → installed location
pip show torch       | grep Location   # → installed location
```

**Source code paths** (preferred for deep analysis): user-provided or ask. Validate:

- MindSpore: must contain `mindspore/python/mindspore/mint/` (conda/pip installs lack `ccsrc/` sources)
- PyTorch: must contain `torch/distributed/tensor/_ops/`

### Step 1: Read Interface Definitions from Source

**MindSpore mint** — Read actual source files to extract full signatures:

- Entry: `<ms_path>/mindspore/python/mindspore/mint/__init__.py` — trace each import to its real definition
- `mint.nn.functional` → `<ms_path>/mindspore/python/mindspore/mint/nn/functional.py`
- `mint.linalg` / `mint.special` → corresponding submodules
- Trace through to underlying function calls to find the **Primitive/kernel name** used
- Important Notice: interface may call function with suffix "_ext" to map to different Primitive/kernel names with suffix "Ext".
- YAML op definitions: `<ms_path>/mindspore/ops/op_def/yaml/{op}_op.yaml` or `func_op/` — extract input/output/attribute parameters

**PyTorch** — Read actual source files:

- `<pt_path>/torch/nn/functional.py` — high-level function signatures
- `<pt_path>/torch/_refs/` — reference implementations showing parameter semantics
- ATen op names: search `<pt_path>/torch/_C/_VariableFunctions.pyi` or use `torch.ops.aten.{op}` patterns

### Step 2: Trace Distributed Implementation & Sharding Strategy

**MindSpore distributed strategy** — conda/pip-installed MindSpore ships compiled binaries with no `.cc` sources. Derive sharding strategy from the HyperParallel side:

- Determine the Primitive name from the YAML op definition found in Step 1
- Look up the `DistributedOp` mapping for that Primitive in `hyper_parallel/core/shard/ops/yaml/*.yaml`
- Read the corresponding `infer_layout()` and `get_expand_impl()` in `core/shard/ops/parallel_*.py` for sharding logic
- For MindSpore distributed design reference, consult user-provided MindSpore source repo `ccsrc/frontend/parallel/ops_info/` (non-conda only)

**PyTorch distributed DTensor strategies** — Read actual Python source:

- `<pt_path>/torch/distributed/tensor/_ops/_matrix_ops.py` — mm, bmm, addmm, linear
- `<pt_path>/torch/distributed/tensor/_ops/_pointwise_ops.py` — element-wise ops with broadcasting
- `<pt_path>/torch/distributed/tensor/_ops/_conv_ops.py` — conv1d/2d/3d
- `<pt_path>/torch/distributed/tensor/_ops/_view_ops.py` — reshape, transpose, permute, slice
- `<pt_path>/torch/distributed/tensor/_ops/_einsum_strategy.py` — einsum strategy generation
- Each uses `@register_op_strategy(aten.xxx)` returning `OpStrategy` with `PlacementStrategy` listing valid input/output placement combinations

**HyperParallel mapping**:

- YAML configs: `hyper_parallel/core/shard/ops/yaml/*.yaml` — maps Primitive/ATen names to `DistributedOp` class
- Implementations: `hyper_parallel/core/shard/ops/parallel_*.py` — `infer_layout()` + optional `get_expand_impl()`

### Step 3: HyperParallel Op Dispatch (`hyper_parallel/core/shard/_op_dispatch.py`)

```
DTensorBase.__torch_function__ / __fallback__
  → platform.get_op_name() → whitelist? → random op?
  → YAML lookup (layout_infer_ops[op_name])
  → infer_layout_suffix routing: "" | "WithShape" | "Reshape" | "WithTupleExpand" | "Slice"
  → distribute_op.infer_layout(input_layouts, extra_args) → output Layout
  → optional get_expand_impl() → execute on local tensors → DTensor.from_local()
```

Cache keyed by `layout.compact_str + rank_id`. 45 YAML files in `core/shard/ops/yaml/`.

### Step 4: DTensor / Layout / DeviceMesh

- **DTensor** (`core/dtensor/dtensor.py`): `from_local()`, `to_local()`, `redistribute()`, `is_partial()` (method, not property)
- **Layout** (`core/dtensor/layout.py`): `alias_tensor_map` maps tensor dims → device axes; `set_partial_by_dev_axis(axis, 'sum')`; `compact_str` for cache
- **DeviceMesh** (`core/dtensor/device_mesh.py`): named axes (`dp`, `tp`, `pp`, `cp`); `get_device_num_along_axis(axis)`

### Step 5: Layout Derivation & Implementation Guidance

When consulted about a specific operator, provide:

- **Interface spec**: full parameter signature, parameter constraints, input/output dtypes/shapes
- **Input analysis**: which dims sharded along which device axes; kwargs affecting sharding (`dim`, `axis`, `transpose_a/b`, `keepdim`)
- **Output layout rules**: element-wise → inherit first input; MatMul → contracting dims must match, sharded contracting → `Partial('sum')`; reduction → sharded dim reduced → `Replicate`; concat → merge on concat axis; reshape → must not cross sharded boundaries
- **get_expand_impl analysis**: identify cases where distributed sharding produces semantically different results from single-device execution. `get_expand_impl` wraps the native op to restore equivalence — common patterns include bias/normalization scaling (Linear), attention score normalization (FlashAttention), index/coordinate remapping (index_select, gather)
- **Suffix recommendation** — based on interface analysis, advise which `infer_layout_suffix` to use:
  - `""` (default): op only needs input layouts + scalar kwargs (e.g., transpose flags). Most element-wise and matrix ops.
  - `"WithShape"`: layout inference depends on input tensor shapes (e.g., broadcasting ops, `expand`, `repeat`).
  - `"Reshape"`: op changes tensor shape in ways requiring specialized shape-to-layout mapping (e.g., `reshape`, `view`, `flatten`, `squeeze`, `unsqueeze`).
  - `"WithTupleExpand"`: op takes tuple/list arguments that need flattening before layout extraction (e.g., `stack`, `cat` with tuple inputs).
  - `"Slice"`: op takes begin/end/stride coordinates that must be transformed per-shard (e.g., `slice`, `slice_scatter`).
- **Base class recommendation** — advise inheritance strategy:
  - **`DistributedOp`** (direct): new op with unique layout logic not shared by existing ops.
  - **Existing `*DistributedOp` subclass**: extend an existing class when the new op shares the same layout inference pattern (e.g., new element-wise variant → inherit `ElementWiseDistributedOp`; new matmul variant → inherit from `MatMulDistributedOp` or `BaseBatchMatMulDistributedOp`).
  - **New intermediate base class**: create when multiple related ops share common layout logic but don't fit any existing class (extract shared logic into a new `BaseXxxDistributedOp`).

---

## Reference Files

- `hyper_parallel/core/shard/_op_dispatch.py` — dispatch mechanism
- `hyper_parallel/core/shard/ops/parallel_ops.py` — `DistributedOp` base class
- `hyper_parallel/core/shard/ops/parallel_ops_register.py` — op registry
- `hyper_parallel/core/shard/ops/yaml/*.yaml` — operator configs (45 files)
- `hyper_parallel/core/shard/ops/parallel_*.py` — distributed implementations
- `hyper_parallel/core/shard/custom_shard.py` — `@custom_shard` decorator
- `hyper_parallel/core/dtensor/{dtensor,layout,device_mesh}.py`
- `hyper_parallel/platform/torch/dtensor.py` — torch interception
- `hyper_parallel/platform/platform.py` — platform abstraction
