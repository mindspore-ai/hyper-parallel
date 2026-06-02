# {算子名} 分布式算子实现计划

## 1. 接口分析

### 接入平台
- 平台范围：PyTorch / MindSpore / 双平台
- （若单平台）已与用户确认，不接入另一侧。

### MindSpore 接口（如适用）

- mint 接口：`mint.{op_name}`
- functional_overload：是 / 否
- mint → Primitive 映射：`mint.{op_name}` → `{PrimitiveName}`（YAML 注册名）
- Primitive 参数表：

| 参数 | dtype | shape 约束 | kwonlyarg |
|------|-------|-----------|-----------|
| x | float16/bf16/float32 | ... | 否 |
| ... | ... | ... | ... |

- 硬件约束：...

### PyTorch 接口（如适用）

- 接口：`torch.{op_name}`
- 参数表：

| 参数 | 类型 | 默认值 | keyword-only |
|------|------|-------|-------------|
| input | Tensor | — | 否 |
| ... | ... | ... | ... |

- 约束：...

---

## 2. 分布式 Layout 推导规则

### cache_values 构成

`[x_layout, ...]`

| 位置 | 内容 | 说明 |
|------|------|------|
| 0 | x_layout | 主输入 Layout |
| ... | ... | ... |

### 约束规则

| 约束 | 检查方式 | 报错信息含 |
|------|---------|-----------|
| ... | `alias_tensor_map[dim] != "None"` | `"..."`, `"dim {d}"` |

### 允许场景总览

| 场景 | mesh | x.tensor_map | 其他输入 .tensor_map |
|------|------|-------------|---------------------|
| 全 Replicated | (2,) dp | (-1,...) | (-1,...) |
| DP on B | (2,) dp | (0,-1,...) | (-1,...) |
| ... | ... | ... | ... |

---

## 3. 实现文件详情

### 文件一：`hyper_parallel/core/shard/ops/parallel_{op_name}.py`

**基类选择：**

- 选择：`{BaseClass}`（例：`ElementWiseDistributedOp` / `ReshapeDistributedOp` / 新类继承 `DistributedOp`）
- 理由：{选择理由，如"纯逐元素，无需自定义 infer_layout" / "需要 _MS_PRIMITIVE_OP_NAMES 路由" / "输出 layout 非逐元素映射"}
- 若选择纯 YAML 注册（无新 Python 文件），在此注明，并跳过模块结构部分。

**模块结构（仅三阶段 dispatch 新类需要）：**

- `_normalize_{op_name}_args(...)` — 模块级函数，统一 torch/mint/Primitive 接口差异，返回 `(args_tuple, kwargs_dict)`
- `{OpName}DistributedOp({BaseClass})` 类：
  - `_MS_PRIMITIVE_OP_NAMES = frozenset({'{PrimitiveName}'})` （双平台时填入 MindSpore 侧 YAML 注册名）
  - `preprocess(args, kwargs)` — 调用 `_normalize_{op_name}_args`，提取 local tensors，构建 cache_values
  - `@staticmethod _validate_input_layouts(...)` — layout 合法性校验（被 `infer_layout` 调用）
  - `infer_layout(cache_values)` — 调用 `_check_partial_inputs` → `_validate_input_layouts` → 推导输出 layout
  - `get_expand_impl(func, infer_result, cache_values)` — （若需要）说明逻辑；若不需要则注明"不覆盖，继承基类返回 None"

**infer_layout Rules（填入 docstring）：**

1. 输入不得有 Partial 状态
2. ...（算子特有约束）
3. 输出 layout = ...

### 文件二：`hyper_parallel/core/shard/ops/yaml/{op_name}_ops.yaml`

```yaml
{PrimitiveName}:                    # MindSpore Primitive（如适用）
  distributed_op_class: {OpName}DistributedOp
  distributed_op_file: parallel_{op_name}

{torch_op_name}:                    # PyTorch（如适用）
  distributed_op_class: {OpName}DistributedOp
  distributed_op_file: parallel_{op_name}
```

### 文件三：`tests/ut/core/shard/ops/test_parallel_{op_name}.py`

- 类：`Test{OpName}DistributedOp`
- 正向用例（列出每个用例名及场景描述）：
  - `test_{op_name}_all_replicated` — 全复制
  - `test_{op_name}_data_parallel` — DP on B
  - ...
- 错误用例：
  - `test_{op_name}_xxx_failure` — 触发条件及预期报错子串

### 文件四：MindSpore ST（如适用）

- Runner：`tests/mindspore/st/shard/ops/test_parallel_op_{op_name}.py`
- Impl：`tests/mindspore/st/shard/ops/_test_parallel_op_{op_name}.py`
- Impl 中使用 `mint.{op_name}` 接口调用（禁止直接使用 Primitive）

| 用例名 | 卡数 | mesh | 说明 |
|-------|------|------|------|
| `test_{op_name}_data_parallel` | 2 | `(2,)` dp | DP on B，与单机对比 |
| ... | ... | ... | ... |

### 文件五：PyTorch ST（如适用）

- Runner：`tests/torch/shard/ops/test_parallel_op_{op_name}.py`
  - 含 `_gloo` 变体：是 / 否（torch 原生接口需要）
- Impl：`tests/torch/shard/ops/_test_parallel_op_{op_name}.py`

| 用例名 | num_proc | 说明 |
|-------|---------|------|
| `test_{op_name}_data_parallel` | 4 | DP on B，layout + 数值双验证 |
| ... | ... | ... |
