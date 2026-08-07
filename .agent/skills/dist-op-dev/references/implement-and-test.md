# Dist-op-dev — Implement & Test Details

Loaded from `SKILL.md` Steps 2–6. Constraints:
`.agent/rules/distributed-op-dev.md`, `.agent/rules/distributed-op-testing.md`.

## Implement — reference templates

| File | Demonstrates |
|------|----------------|
| `parallel_sort.py` | `preprocess` + `infer_layout(cache_values)` + `_MS_PRIMITIVE_OP_NAMES` |
| `parallel_rotary_position_embedding.py` | + `_validate_input_layouts` — most complete new-flow example |
| `parallel_matmul.py` | `preprocess` + `get_expand_impl` |

Most other `parallel_*.py` use **legacy** dispatch — do not copy those signatures.

### Paths

- **A YAML-only:** entry in existing YAML → existing class; no new `.py`
- **B Inherit:** `parallel_{op}.py` + YAML; override only what plan needs
- **C New class:** normalize args → class with `preprocess` /
  `_validate_input_layouts` / `infer_layout(cache_values)` /
  optional `get_expand_impl` + `yaml/{op}_ops.yaml`

## UT

`tests/ut/core/shard/ops/test_parallel_{op_name}.py` — follow
`distributed-op-testing.md`.

## MindSpore ST (if applicable)

`tests/mindspore/st/shard/ops/cases/case_{op_name}.py` — declarative
`OpShardCase` (`ms.mint.*`, never raw Primitives). Tags:
`("npu_level0",)` or `("npu_level1",)`. Placement length == mesh ndim.

## PyTorch ST (if applicable)

`tests/torch/shard/ops/cases/case_{op_name}.py` — same framework with
`torch.*`. Tags often `("cpu_level0", "npu_level0")`.

## Run tests

```bash
npu-smi info >/dev/null 2>&1 && echo "ascend" || echo "no-ascend"

pytest -vs tests/ut/core/shard/ops/test_parallel_{op_name}.py

pytest -vs tests/torch/shard/ops/test_shard_ops_suite.py::test_shard_ops_cpu_level0
pytest -vs tests/torch/shard/ops/test_shard_ops_suite.py::test_shard_ops_cpu_level1
# Ascend only:
pytest -vs tests/torch/shard/ops/test_shard_ops_suite.py::test_shard_ops_ascend_level0
pytest -vs tests/torch/shard/ops/test_shard_ops_suite.py::test_shard_ops_ascend_level1
pytest -vs tests/mindspore/st/shard/ops/test_shard_ops_suite.py::test_shard_ops_ascend_level0
pytest -vs tests/mindspore/st/shard/ops/test_shard_ops_suite.py::test_shard_ops_ascend_level1

HYPER_PARALLEL_SHARD_CASE_FILTER="{op}_ops_*" \
  pytest tests/torch/shard/ops/test_shard_ops_suite.py::test_shard_ops_cpu_level0 -vs
python -m tests.shard_ops.framework --case {op}_ops_dp --num-proc 4
HYPER_PARALLEL_PLATFORM=mindspore \
  python -m tests.shard_ops.framework --framework mindspore --device-type npu \
  --case {op}_ops_dp --num-proc 4
```

`--num-proc` == `math.prod(mesh_shape)`. Non-Ascend: UT + Torch gloo only.

**Failures:** root-cause fix (no skip/suppress workaround); append runs to
`reports/{OpName}_report.md` (Chinese).
