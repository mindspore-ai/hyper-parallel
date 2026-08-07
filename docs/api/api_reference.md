# HyperParallel API 参考文档

本文档按特性模块组织，每个模块包含概述 + 接口详细说明。

---

## HSDP / FSDP 数据并行

HyperParallel 提供 FSDP/HSDP 数据并行能力，支持参数/梯度/优化器状态的分布式切分，显著降低单卡内存占用。

### `fully_shard`

对模型应用 FSDP/HSDP 参数切分。

```python
fully_shard(
    module: Module,
    mesh: DeviceMesh,
    *,
    reshard_after_forward: bool = True,
    mixed_precision: Optional[MixedPrecisionPolicy] = None,
    offload_policy: Optional[OffloadPolicy] = None,
    comm_fusion: bool = True,
    comm_fusion_zero_copy: Optional[bool] = None,
) -> Module
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `module` | `Module` | — | 要切分的模块 |
| `mesh` | `DeviceMesh` | — | DP 维度的 DeviceMesh |
| `reshard_after_forward` | `bool` | `True` | 正向后是否 reshard（节省内存） |
| `mixed_precision` | `MixedPrecisionPolicy` | `None` | 混合精度策略 |
| `offload_policy` | `OffloadPolicy` | `None` | Offload 策略 |
| `comm_fusion` | `bool` | `True` | 通信融合 |
| `comm_fusion_zero_copy` | `bool` | `None` | 通信融合零拷贝（PyTorch 默认 True，MindSpore 默认 False） |

**返回值：** 切分后的模块（原地修改，返回同一对象）。

---

### `HSDPModule`

HSDP 模块封装类。

```python
class HSDPModule
```

HSDPModule 封装了 HSDP 参数切分的核心逻辑，包括参数 unshard/reshard、梯度同步等。

---

### `hsdp_sync_stream`

HSDP 梯度同步流管理。

```python
hsdp_sync_stream()
```

返回 HSDP 使用的梯度同步流，确保梯度 all-reduce 与计算流正确同步。

---

## DTensor 分布式张量

DTensor 是 HyperParallel 的核心分布式张量抽象，封装 local shard + DeviceMesh + Placements。

### `DTensor`

分布式张量类。

```python
class DTensor
```

**核心方法：**

| 方法 | 说明 |
|------|------|
| `from_local(local_tensor, layout)` | 从本地张量创建 DTensor |
| `to_local()` | 获取本地 shard 张量 |
| `redistribute(device_mesh, placements)` | 重分布到目标 layout |
| `full_tensor()` | 获取完整（聚合后的）张量 |
| `is_partial()` | 检查是否为 partial 状态 |
| `reduce_partial()` | 对 partial DTensor 执行 reduce |

---

### `Layout`

张量到 mesh 的排布映射。

```python
Layout(mesh_shape, mesh_dim_names)
```

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `mesh_shape` | `tuple` | mesh 形状，如 `(4, 8)` |
| `mesh_dim_names` | `tuple` | mesh 维度名，如 `("dp", "tp")` |

**使用方式：**

```python
layout = Layout((dp, tp), ("dp", "tp"))
x_layout = layout("dp", "tp")      # 指定各维度的排布
w_layout = layout("tp", "None")    # "None" 表示不切分
out_layout = layout()               # 空 layout 表示 Replicate
```

---

### `DeviceMesh`

多维设备拓扑管理。

```python
class DeviceMesh
```

**核心方法：**

| 方法 | 说明 |
|------|------|
| `__getitem__(dim_name)` | 切片子 mesh，如 `mesh["dp"]` |
| `get_device_num_along_axis(dim_name)` | 获取指定维度设备数 |

---

### `init_device_mesh`

创建 DeviceMesh。

```python
init_device_mesh(
    device_type: str,
    mesh_shape: tuple,
    mesh_dim_names: tuple,
) -> DeviceMesh
```

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `device_type` | `str` | 设备类型，如 `"npu"` 或 `"cuda"` |
| `mesh_shape` | `tuple` | mesh 形状 |
| `mesh_dim_names` | `tuple` | mesh 维度名 |

---

### `get_current_mesh`

获取当前活跃的 DeviceMesh。

```python
get_current_mesh() -> DeviceMesh
```

---

### `distribute_module`

将模块的参数按指定 layout 分布。

```python
distribute_module(module, device_mesh, ...)
```

---

### `init_parameters`

分片参数初始化。

```python
init_parameters(model) -> Module
```

对模型参数按 DTensor layout 进行分片初始化，避免先初始化完整参数再切分导致的内存峰值。

---

### `init_empty_weights`

空权重初始化上下文。

```python
init_empty_weights()
```

延迟初始化上下文，创建参数时不分配内存，后续通过 `init_parameters` 分片初始化。

---

### `init_on_device`

在设备上初始化参数。

```python
init_on_device()
```

---

### `manual_seed`

分布式随机数种子控制（PyTorch parity）。

```python
manual_seed(seed: int)
```

设置分布式随机数种子，确保各 rank 的随机数生成一致。

---

## Shard / TP 张量并行

### `shard_module`

声明式并行策略接口。

```python
shard_module(
    module: Module,
    sharding_plan: dict,
) -> Module
```

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `module` | `Module` | 要切分的模块 |
| `sharding_plan` | `dict` | 切分配置，包含 `forward`（input/output）和 `parameter` |

**sharding_plan 格式：**

```python
sharding_plan = {
    "forward": {
        "input": (x_layout,),
        "output": (out_layout,),
    },
    "parameter": {
        "weight": w_layout,
    },
}
```

---

### `custom_shard`

自定义并行接入 DTensor 并行流程。

```python
custom_shard(...)
```

---

### `DFunction`

自定义分布式 autograd 函数基类。

```python
class DFunction(platform.Function):
    _op_name: str = None

    @staticmethod
    def forward(ctx, *args, **kwargs) -> Tensor: ...

    @staticmethod
    def backward(ctx, *grad_outputs) -> ...: ...

    @classmethod
    def apply(cls, *args, **kwargs) -> Tensor | DTensor: ...
```

**详细文档：** 参见 [DFunction 文档](./dfunction.md)。

---

### `parallelize_value_and_grad`

并行化的值与梯度计算。

```python
parallelize_value_and_grad(...)
```

---

### `SkipDTensorDispatch`

梯度 hook 中绕过 DTensor dispatch 的标记。

```python
class SkipDTensorDispatch
```

在 FSDP/HSDP 的梯度 hook 中使用，直接操作 local tensor 而不经过 DTensor dispatch 系统。

---

## TP Styles 声明式张量并行

### `ColwiseParallel`

列切分并行策略。

```python
class ColwiseParallel(ParallelStyle):
    input_layouts: Placement = Replicate()
    output_layouts: Placement = Shard(-1)
    use_local_output: bool = True
```

---

### `RowwiseParallel`

行切分并行策略。

```python
class RowwiseParallel(ParallelStyle):
    input_layouts: Placement = Shard(-1)
    output_layouts: Placement = Replicate()
    use_local_output: bool = True
```

---

### `SequenceParallel`

序列并行策略。

```python
class SequenceParallel(ParallelStyle)
```

---

### `PrepareModuleInput`

模块输入准备钩子。

```python
class PrepareModuleInput(ParallelStyle)
```

---

### `PrepareModuleOutput`

模块输出准备钩子。

```python
class PrepareModuleOutput(ParallelStyle)
```

---

### `PrepareModuleInputOutput`

模块输入/输出准备钩子。

```python
class PrepareModuleInputOutput(ParallelStyle)
```

---

### `ParallelStyle`

并行策略基类。

```python
class ParallelStyle
```

所有并行策略的抽象基类，定义 `apply(module, device_mesh)` 接口。

---

### `parallelize_module`

声明式 TP 应用接口。

```python
parallelize_module(
    module: Module,
    device_mesh: DeviceMesh,
    parallelize_plan: dict | ParallelStyle,
    *,
    src_data_rank: int = 0,
) -> Module
```

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `module` | `Module` | 要并行化的模块 |
| `device_mesh` | `DeviceMesh` | 1-D TP DeviceMesh |
| `parallelize_plan` | `dict` 或 `ParallelStyle` | 并行策略配置，支持 fnmatch glob pattern |
| `src_data_rank` | `int` | 数据源 rank（默认 0） |

---

## PP 流水线并行

### `PipelineStage`

流水线 stage 封装。

```python
class PipelineStage:
    def __init__(
        self,
        module: Module,
        stage_index: int,
        stage_num: int,
    )
```

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `module` | `Module` | stage 对应的子模块 |
| `stage_index` | `int` | 当前 stage 的索引 |
| `stage_num` | `int` | 总 stage 数 |

---

### `Schedule1F1B`

1F1B 调度策略。

```python
class Schedule1F1B:
    def __init__(
        self,
        stage: PipelineStage,
        micro_batch_num: int,
    )
```

---

### `ScheduleGPipe`

GPipe 调度策略。

```python
class ScheduleGPipe:
    def __init__(
        self,
        stage: PipelineStage,
        micro_batch_num: int,
    )
```

---

### `ScheduleInterleaved1F1B`

VPP（交错 1F1B）调度策略。

```python
class ScheduleInterleaved1F1B:
    def __init__(
        self,
        stages: list,
        micro_batch_num: int,
        *,
        overlap_p2p: bool = False,
        overlap_b_f: bool = False,
    )
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `stages` | `list` | — | PipelineStage 列表 |
| `micro_batch_num` | `int` | — | micro-batch 数 |
| `overlap_p2p` | `bool` | `False` | 是否启用 P2P prefetch |
| `overlap_b_f` | `bool` | `False` | 是否启用 B/F 通算掩盖 |

---

### `MetaStep`

调度单元抽象。

```python
class MetaStep
```

描述调度单元，包含 `step_type` 和可选的 `sub_steps`。

---

### `MetaStepType`

调度单元类型枚举。

```python
class MetaStepType(enum.Enum):
    FORWARD = "F"
    BACKWARD = "B"
    OVERLAP_F_B = "OFB"
    OVERLAP_B_F = "OBF"
    FWD_RECV = "FR"
    BWD_RECV = "BR"
    FWD_SEND = "FS"
    BWD_SEND = "BS"
```

---

### `BatchDimSpec`

批量维度规格。

```python
class BatchDimSpec
```

---

### `CommComputeOverlap`

双线程通算掩盖协调器。

```python
class CommComputeOverlap:
    def wrap_dispatch(self, dispatch_fn, ...) -> Callable
    def wrap_combine(self, combine_fn, is_last_layer: bool = False) -> Callable
    def run(self, fwd_fn: Callable, bwd_fn: Callable) -> None
```

---

### `HookCoordinator` / `HookRole`

COMM-first rendezvous 原语。

```python
class HookCoordinator:
    def enable(self) -> None
    def disable(self) -> None
    def rendezvous(self, hook_name: str, role: HookRole) -> None

class HookRole(enum.Enum):
    COMM = "COMM"
    COMPUTE = "COMPUTE"
```

---

## CP 上下文并行

### `ContextParallel`

基础上下文并行。

```python
class ContextParallel(ParallelStyle)
```

---

### `AsyncContextParallel`

异步上下文并行。

```python
class AsyncContextParallel(ParallelStyle)
```

---

### DSA 系列

| 类 | 说明 |
|----|------|
| `DSAIndexerContextParallel` | DSA 索引器 CP |
| `AsyncDSAIndexerContextParallel` | 异步 DSA 索引器 CP |
| `DSAIndexerLossContextParallel` | DSA 索引器 + Loss CP |
| `AsyncDSAIndexerLossContextParallel` | 异步 DSA 索引器 + Loss CP |
| `DSASparseAttentionContextParallel` | DSA Sparse Attention CP |
| `AsyncDSASparseAttentionContextParallel` | 异步 DSA Sparse Attention CP |

所有 DSA 类均继承 `ParallelStyle`，支持 `apply(module, device_mesh)` 接口。

---

## EP 专家并行

### `ExpertParallel`

标准 all-to-all EP。

```python
class ExpertParallel(BaseExpertParallel)
```

每个 rank 持有 `num_experts / ep_degree` 个本地专家，通过 differentiable all-to-all 实现 token dispatch/combine。

**Mesh 需求：** 1-D mesh，如 `mesh_dim_names=("ep",)`。

---

### `TensorParallel`（EP 内）

TP-only 权重切分（EP degree = 1 时使用）。

```python
class TensorParallel(BaseExpertParallel)
```

---

### `ExpertTensorParallel`

EP+TP 二维并行。

```python
class ExpertTensorParallel(BaseExpertParallel)
```

**Mesh 需求：** 2-D mesh，如 `mesh_dim_names=("ep", "tp")`。

---

## Process Group 进程组管理

### `init_process_group`

初始化进程组。

```python
init_process_group(backend: str, ...)
```

---

### `destroy_process_group`

销毁进程组。

```python
destroy_process_group()
```

---

### `get_process_group_ranks`

获取进程组内所有 rank。

```python
get_process_group_ranks(group) -> list
```

---

### `get_backend`

获取当前进程组的 backend 类型。

```python
get_backend() -> str
```

---

### `split_group`

从父进程组中分裂出子进程组。

```python
split_group(group, split_size, ...)
```

---

### `get_group_local_rank`

获取进程组内的 local rank。

```python
get_group_local_rank(group) -> int
```

---

### `mark_created_groups`

标记已创建的进程组。

```python
mark_created_groups(groups)
```

---

## Optimizer 优化器

### `AdamW`

标准 AdamW 优化器。

```python
class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: list,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        ...
    )
```

---

### `Muon`

Muon（momentum-based）优化器。

```python
class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: list,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        nesterov: bool = True,
        weight_decay: float = 0.1,
        ...
    )
```

---

### `ChainedOptimizer`

链式优化器（Muon+AdamW 组合）。

```python
class ChainedOptimizer:
    def __init__(
        self,
        model: nn.Module,
        optimizers: Dict[str, Optimizer],
        flatten: bool = False,
    )

    def step(self) -> None
    def zero_grad(self, set_to_none: bool = True) -> None
    def __iter__(self)
```

---

### `get_hyper_optimizer`

优化器工厂函数。

```python
get_hyper_optimizer(
    model: nn.Module,
    muon_params: List[Dict[str, Any]],
    adamw_params: List[Dict[str, Any]],
    muon_kwargs: Optional[Dict[str, Any]] = None,
    adamw_kwargs: Optional[Dict[str, Any]] = None,
) -> ChainedOptimizer
```

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `nn.Module` | 模型 |
| `muon_params` | `List[Dict]` | Muon 参数组，空列表禁用 Muon |
| `adamw_params` | `List[Dict]` | AdamW 参数组，空列表禁用 AdamW |
| `muon_kwargs` | `Dict` | Muon 配置，支持 `muon_` 前缀自动去前缀 |
| `adamw_kwargs` | `Dict` | AdamW 配置，支持 `adamw_` 前缀自动去前缀 |

---

### `get_hyper_lr_scheduler`

学习率调度器工厂函数。

```python
get_hyper_lr_scheduler(
    optimizer: ChainedOptimizer,
    total_steps: int,
    warmup_steps: int = 0,
    warmup_ratio: float = 0.0,
    decay_style: str = "cosine",
    lr: float = 1e-4,
    lr_min: float = 1e-7,
    lr_start: float = 0.0,
    lr_decay_ratio: float = 1.0,
    wsd_decay_steps: Optional[int] = None,
    lr_wsd_decay_style: str = "exponential",
    override_opt_param_scheduler: bool = False,
) -> LRSchedulersContainer
```

**支持的 decay_style：** `"constant"` / `"linear"` / `"cosine"` / `"WSD"`

**支持的 WSD decay_style：** `"linear"` / `"cosine"` / `"exponential"` / `"minus_sqrt"`

---

## Activation Checkpoint / Swap

> PyTorch 与 MindSpore 后端均已实现。公共 API 位于 `hyper_parallel.core.activation_checkpoint`。

### `checkpoint`

函数式激活重计算（`use_reentrant=False`）。

```python
checkpoint(
    function,
    *args,
    swap_inputs: bool = False,
    policy_fn: Optional[Callable] = None,
    context_fn: Optional[Callable[[], Tuple[object, object]]] = None,
    group_swap: bool = False,
    early_stop: bool = True,
    **kwargs,
)
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `function` | `callable` | — | 要 checkpoint 的函数或模块 |
| `swap_inputs` | `bool` | `False` | 是否将 checkpoint 保存的输入 offload 到 CPU |
| `policy_fn` | `callable` | `None` | SAC 逐算子策略：`(ctx, op, *args, **kwargs) -> CheckpointPolicy` |
| `context_fn` | `callable` | `None` | 返回 `(forward_ctx, recompute_ctx)` 的无参工厂；可与 `policy_fn` 组合 |
| `group_swap` | `bool` | `False` | 是否对 `MUST_SWAP` tensor 启用分组 copy 融合 |
| `early_stop` | `bool` | `True` | 是否在产生全部 backward 所需 tensor 后提前停止重计算 |

Torch 2.6、2.7、2.9 eager 模式统一使用 HyperParallel non-reentrant 实现。`early_stop` 只接受单次调用
关键字或 `checkpoint_kwargs` 配置，不继承外层 `torch.utils.checkpoint.set_checkpoint_early_stop()`。
Torch backend 还会消费 `preserve_rng_state`、`determinism_check` 和 `debug` 等 checkpoint 保留关键字。

---

### `swap`

函数式激活 swap（不重计算）。须在使用时调用 `SwapManager.set_forward_prefetch_layer` 建立层间预取关系。

```python
swap(
    function,
    *args,
    policy_fn: Optional[Callable] = None,
    **kwargs,
)
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `function` | `callable` | — | 要 swap 的函数或模块 |
| `policy_fn` | `callable` | `None` | 逐 tensor 策略：`(tensor) -> CheckpointPolicy`；仅 `MUST_SAVE` / `MUST_SWAP` 有效 |

---

### `checkpoint_wrapper`

模块级 checkpoint 装饰器（`CheckpointWrapper`，继承 `ActivationWrapper`）。

```python
checkpoint_wrapper(module, **checkpoint_kwargs) -> CheckpointWrapper
```

`checkpoint_kwargs` 与 `checkpoint` 一致，例如 `policy_fn`、`swap_inputs`、`early_stop`。

---

### `checkpoint_exclude_wrapper`

将 checkpoint 内部的指定函数或 Cell 标记为不重算区域。前向阶段保留该区域的输出及反向所需 tensor，
重算阶段直接复用前向输出，以额外显存占用换取计算开销降低。

```python
checkpoint_exclude_wrapper(module) -> CheckpointExcludeWrapper
```

当前仅支持 MindSpore PyNative 模式，并且需要在 HyperParallel 的 `checkpoint` 或
`checkpoint_wrapper`（`use_reentrant=False`）内部使用。支持包装 MindSpore Cell 和普通 callable。

---

### `swap_wrapper`

模块级纯 swap 装饰器（不重计算）。

```python
swap_wrapper(
    module,
    policy_fn: Optional[Callable] = None,
) -> SwapWrapper
```

---

### `swap_tensor_wrapper`

在 `forward`/`construct` 内将指定 tensor 注册到当前 swap group。

```python
swap_tensor_wrapper(target, tag: Optional[str] = None)
```

---

### `CheckpointPolicy`

重计算 / swap 策略枚举。

```python
class CheckpointPolicy(enum.Enum):
    MUST_SAVE = 0        # 必须保存（留在设备）
    PREFER_SAVE = 1      # 优先保存
    MUST_RECOMPUTE = 2   # 必须重计算
    PREFER_RECOMPUTE = 3 # 优先重计算
    MUST_SWAP = 4        # offload 到 HOST，反向 reload（需 SwapManager）
```

---

### `SwapManager`

Swap 分组管理器（单例）。

```python
class SwapManager:
    def set_forward_prefetch_layer(self, first_layer, second_layer) -> None
    def launch_offload(self, group_name: str, copy_stream=None) -> None
    def wait_offload(self, group_name: str) -> None
    def launch_load(self, group_name: str, copy_stream=None) -> None
    def wait_load(self, group_name: str) -> None
```

---

## 平台抽象

### `get_platform`

获取当前平台对象。

```python
get_platform() -> Platform
```

返回 PyTorch 或 MindSpore 平台实现，用于访问平台特定功能。

---

## 其他导出接口

### 主接口导出（`hyper_parallel/__init__.py`）

```python
__all__ = [
    "get_platform", "DFunction", "fully_shard", "hsdp_sync_stream", "HSDPModule",
    "DTensor", "Layout", "DeviceMesh", "init_device_mesh", "get_current_mesh",
    "distribute_module", "init_parameters", "init_empty_weights", "init_on_device",
    "shard_module", "custom_shard", "parallelize_value_and_grad", "SkipDTensorDispatch",
    "MetaStep", "MetaStepType", "BatchDimSpec", "PipelineStage", "ScheduleInterleaved1F1B",
    "init_process_group", "destroy_process_group", "get_process_group_ranks", "get_backend",
    "split_group", "get_group_local_rank", "mark_created_groups",
    "ContextParallel", "AsyncContextParallel",
    "AsyncDSAIndexerContextParallel", "AsyncDSAIndexerLossContextParallel",
    "AsyncDSASparseAttentionContextParallel",
    "DSAIndexerContextParallel", "DSAIndexerLossContextParallel", "DSASparseAttentionContextParallel",
    "ColwiseParallel", "RowwiseParallel", "SequenceParallel",
    "PrepareModuleInput", "PrepareModuleInputOutput", "PrepareModuleOutput",
    "ParallelStyle", "parallelize_module", "manual_seed",
]
```
