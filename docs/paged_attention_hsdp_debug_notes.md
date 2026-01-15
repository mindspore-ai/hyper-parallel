# PagedAttention + HSDP 适配与问题记录

## 背景
- `ms_custom_ops.paged_attention` 为自定义算子，非 MindSpore 内置算子。
- 在 HSDP + DTensor + Python shard 场景下，需要显式补充分布式规则与调用路径适配。

## 问题现象
- 通过 HSDP 接口调用 `PagedAttentionNet` 时，出现：
  - `Layout is None`（DTensor 封装阶段）
  - `Python fallback failed`（PyBoost backward hook）
  - `Can't access data on Ascend` / `Ascend:0` 设备字符串不被 `Tensor.to` 接受

## 根因分析
1. **自定义算子不触发 DTensor fallback**
   - MindSpore 内置算子通过 `_run_op` C++ 路径执行，会自动触发 `DTensor.__fallback__`。
   - 自定义算子（如 `ms_custom_ops.paged_attention`）不走这条路径，DTensor 传入时不会自动触发 `infer_layout`。
2. **自定义算子名解析不稳定**
   - `ms_custom_ops.paged_attention` 缺少统一 `name`，需从 `__name__` / `__class__.__name__` 推导。
3. **缺少分布式规则注册**
   - 自定义算子必须通过 YAML + `DistributedOp` 显式注册。
4. **CPU 参数语义未保持**
   - `context_lens` / `q_seq_lens` 必须在 CPU 执行，否则算子异常。
5. **CPU Tensor 被 DTensor 包装后强制迁移到 Ascend**
   - `DistributedCustomOp` 在只有 layout、没有 DTensor 输入时，会将普通 Tensor 包装为 DTensor。
   - `DTensor.from_local` 最终走到 `DTensorBase.__new__`，默认 `device="Ascend"`，执行 `local_tensor.to("Ascend")`。
   - 如果该 Tensor 原本应该驻留 CPU（如 `context_lens/q_seq_lens`），会被错误迁移到 Ascend，
     进而导致算子输出异常，最终在 `DTensor.from_local(output, layout)` 处触发 `Fatal Python error: Aborted`。

相关代码片段：
```26:41:hyper_parallel/platform/mindspore/dtensor.py
def __new__(cls, local_tensor, layout=None, device="Ascend"):
    ...
    device_local_tensor = local_tensor if local_tensor.has_init else local_tensor.to(device)
```

## 解决方案：DistributedCustomOp

为自定义算子提供 `DistributedCustomOp` 封装类，使其能像内置算子一样自动触发 `infer_layout`。

### 接口设计

```python
from hyper_parallel import DistributedCustomOp
import ms_custom_ops

# 在 Cell.__init__ 中封装自定义算子
self.paged_attention = DistributedCustomOp(ms_custom_ops.paged_attention)

# 在 construct 中直接调用，自动检测 DTensor 并走 infer_layout
output = self.paged_attention(query, key_cache, value_cache, ...)
```

### 工作原理
1. `DistributedCustomOp.__call__` 检测输入中是否有 DTensor
2. 有 DTensor 时，调用 `_op_dispatch._with_layout_infer` 触发 `infer_layout`
3. 无 DTensor 时，直接调用原始算子
4. 输出自动封装为 DTensor（通过 `_with_layout_infer` 内部逻辑）

### 与内置算子的对比

| 特性 | 内置算子 | 自定义算子 + DistributedCustomOp |
|------|----------|----------------------------------|
| DTensor 触发方式 | 自动（C++ fallback） | 自动（Python 检测） |
| infer_layout 调用 | 通过 dispatch | 通过 dispatch |
| 使用方式 | 直接调用 | 需先封装 |

## 适配点（与非内置算子相关）
1. **算子名称解析**
   - `platform.get_op_name` 增加 `__name__` / `__class__` 回退。
2. **分布式规则与 YAML**
   - 新增 `parallel_paged_attention.py` 和 `paged_attention_ops.yaml`。
3. **CPU 参数迁移**
   - 在 `PagedAttentionNet.construct` 中强制把 `context_lens/q_seq_lens` 迁移到 CPU。
4. **DistributedCustomOp 封装**
   - 使用 `DistributedCustomOp` 包装自定义算子，使其支持自动 DTensor dispatch。

## 风险与后续
- `get_op_name` 对非内置算子改为模块限定名，避免自定义算子名称冲突。
- 已补充 `q_seq_lens` 的 infer_layout 单测与可选输入 smoke 用例，后续可按算子签名细化。

## MindSpore 内置算子调用路径（PyNative）
内置算子不会走 Python fallback，因此不会被 `CellBackwardHook` 干扰。

核心分支逻辑（简化）：
1) `Cell.__call__` 判断是否有 hook / shard / recompute 等逻辑
2) 无 hook 直接 `construct`；有 hook 走 `_run_construct`
3) `Primitive.__call__` 直接进入 `_run_op`（C++ 执行）

更详细的执行分支：
```text
Cell.__call__:
  if no hooks/recompute/shard and not requires_grad:
      return construct(...)
  else:
      return _run_construct(...)  # 会执行 forward/backward hooks

Primitive.__call__:
  return _run_op(...)  # 直接进入 C++ 执行器
```

代码参考（关键判断点）：
```1353:1383:mindspore/mindspore/python/mindspore/nn/cell.py
    def __call__(self, *args, **kwargs):
        ...
        if not (self.requires_grad or self._dynamic_shape_inputs or self.mixed_precision_type):
            if not (self._forward_pre_hook or self._forward_hook or self._backward_pre_hook or self._backward_hook or
                    self._shard_fn or self._recompute_cell or (self.has_bprop and _pynative_executor.requires_grad())):
                return self.construct(*args, **kwargs)

            return self._run_construct(*args, **kwargs)
```

```390:397:mindspore/mindspore/python/mindspore/ops/primitive.py
    def __call__(self, *args):
        ...
        return _run_op(self, self.name, args)
```

```999:1005:mindspore/mindspore/python/mindspore/ops/primitive.py
def _run_op(obj, op_name, args):
    res = _pynative_executor.run_op_async(obj, op_name, args)
    ...
    return res
```

结论：
- 内置算子在 PyNative 下直接进入 C++ 执行器，不依赖 Python 的 `__fallback__`
- 因此 DTensor 的 fallback 机制只对 Python 层调用有效，而不会影响内置算子

## ms_custom_ops 算子调用路径（当前用例）
自定义算子通过 `DistributedCustomOp` 封装后，自动触发 `infer_layout`。

1) `PagedAttentionNet.__init__`: 创建 `DistributedCustomOp(ms_custom_ops.paged_attention)`
2) `PagedAttentionNet.construct`: 调用 `self.paged_attention(...)`
3) `DistributedCustomOp.__call__`: 检测 DTensor 输入
4) `_op_dispatch._with_layout_infer`: 触发 layout 推导
5) `PagedAttentionDistributedOp.infer_layout`: 推导输出 layout
6) 真实算子执行 `ms_custom_ops.paged_attention(...)`
7) `DTensor.from_local`: 封装输出

代码参考：
```32:82:tests/mindspore/st/shard/ops/paged_attention/parallel_paged_attention_forward_only.py
class PagedAttentionNet(nn.Cell):
    def __init__(self, q_head_num, kv_head_num, head_size, qk_scale):
        ...
        # Wrap the custom op with DistributedCustomOp for automatic DTensor handling
        self.paged_attention = DistributedCustomOp(ms_custom_ops.paged_attention)

    def construct(self, query, key_cache, value_cache, block_tables, context_lens, q_seq_lens):
        ...
        # DistributedCustomOp automatically detects DTensor and routes through infer_layout
        return self.paged_attention(query, key_cache, value_cache, ...)
```

```27:97:hyper_parallel/core/shard/api.py
class DistributedCustomOp:
    def __init__(self, op: Callable, infer_layout_suffix: Optional[str] = None):
        self._op = op
        ...

    def __call__(self, *args, **kwargs):
        has_dtensor = any(isinstance(arg, DTensor) for arg in args)
        ...
        if not has_dtensor:
            return self._op(*args, **kwargs)
        # Has DTensor inputs, route through dispatch
        return dispatcher._with_layout_infer(self._op, *args, **kwargs)
```

```185:252:hyper_parallel/core/shard/_op_dispatch.py
    def _with_layout_infer(self, func: callable, *args, **kwargs) -> Tensor:
        ...
        distribute_op = cache_manager.distributed_op(func_name)
        output_layout = distribute_op.infer_layout(*all_args)
        py_output = func(*input_args, **kwargs)
        ...
        return DTensor.from_local(py_output, output_layout)
```

```37:129:hyper_parallel/core/shard/ops/parallel_paged_attention.py
    def infer_layout(self, layouts, extra_args):
        ...
        return output_layout(*query_map)
```

## 相关文件
- `hyper_parallel/core/shard/api.py` - `DistributedCustomOp` 定义
- `hyper_parallel/core/shard/ops/parallel_paged_attention.py` - `infer_layout` 实现
- `hyper_parallel/core/shard/ops/yaml/paged_attention_ops.yaml` - 分布式规则注册
- `hyper_parallel/platform/mindspore/platform.py` - `get_op_name` 实现
- `hyper_parallel/core/shard/_op_dispatch.py` - dispatch 逻辑
- `tests/mindspore/st/shard/ops/paged_attention/parallel_paged_attention_forward_only.py` - 测试用例
