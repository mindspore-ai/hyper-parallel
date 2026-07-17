# 模型脚本生成器（Model Script Generator）

> 将经过 DTensor 分片 + PrecompiledBoundary 通信包装后的模型，生成为一份**独立可运行的 Python 脚本**，便于调试、离线分析和代码审查。

---

## 1. 模块职责

定义 hyper_parallel 的模型脚本生成能力。核心目标：在 `apply_sharding_plan()` 完成运行时包装后，将模型"还原"为一份**自包含的 Python 源码**，其中通信逻辑不再是隐藏的 monkey-patch wrapper，而是显式的函数调用。

### 核心文件

| 文件 | 职责 |
|------|------|
| `hyper_models/components/distributed/script_generator.py` | `ModelScriptGenerator`：从运行时模型 + ShardingPlan → 生成独立脚本 |
| `hyper_models/components/distributed/scriptgen/__init__.py` | 子包入口，暴露 `generate_model_script` 便捷函数 |
| `hyper_models/components/distributed/scriptgen/codegen.py` | `ScriptCodeGenerator`：AST 解析 + 重写 + 源码生成 |
| `hyper_models/components/distributed/scriptgen/boundary_extractor.py` | `BoundaryExtractor`：从运行时模型中提取 PrecompiledBoundary 信息 |
| `hyper_models/components/distributed/scriptgen/templates.py` | Jinja2 模板（wrapper 类模板、import 模板、main 模板） |

### 设计哲学：与 VeOmni patchgen 的关系

| 维度 | VeOmni patchgen | Hyper-Parallel scriptgen |
|------|----------------|--------------------------|
| **输入** | `PatchConfig`（手写补丁声明） | `ShardingPlan`（自动推导）+ 运行时模型 |
| **补丁内容** | 类替换（Liger RMSNorm）、方法覆写（optimized forward） | forward 通信注入（pre/post boundary）、参数分片（DTensor） |
| **补丁来源** | 人工定义（优化 kernel 库） | 自动生成（从 `ModuleShardingSpec` 和 `PrecompiledBoundary` 推导） |
| **输出** | `patched_modeling_gpu.py`（性能优化） | `standalone_qwen3_tp4.py`（调试 + 离线分析） |
| **运行时依赖** | 无 patchgen 依赖 | 无 hyper_parallel 核心依赖（仅需 torch + DTensor） |
| **生成时机** | 开发期（手动运行 CLI） | 训练初始化后（`_build_model` 末尾自动/手动触发） |

**共同点**：
- 都基于 AST 解析原始 HF 建模代码（`inspect.getsource` + `ast.parse`）
- 都通过 AST 重写产生自包含的输出文件（清晰标注哪些被修改过）
- 都保持原始代码结构不变（未修改的部分原样保留）
- 都支持 `ruff` 格式化 + CI drift 检查

**关键差异 —— patchgen 的 `PatchConfig` 是手写的，scriptgen 的"补丁"是自动推导的**：

```python
# VeOmni patchgen: 人工声明要替换什么
@config.replace_class("Qwen3RMSNorm")
class LigerRMSNorm(nn.Module): ...

# Hyper-Parallel scriptgen: 从 ShardingPlan 自动生成所有"补丁"
for spec in plan.module_specs:
    if spec.is_boundary:
        boundary_patches.append(
            ForwardWrapPatch(module=spec.module_name, boundary=spec.boundary)
        )
```

---

## 2. 调用时序

### 2.1 在 `main()` 调用链中的位置

`scriptgen` 在 `_build_model()` 末尾触发——此时模型已完成分片和通信包装，处于**可运行状态**。生成脚本是可选的，由配置控制。

```
main() → recipe.setup(cfg)                                              # 01_hf_compatibility_layer.md §4
└─④.3 model = cfg.model.instantiate(distributed_setup=...)
    └─ HyperAutoModelForCausalLM.from_pretrained(...)                    # 01 §6
        └─ _build_model(...)                                             # 01 §6.3
            │
            ├─④.3.2 instantiate_infrastructure(distributed_setup, device)
            │   ├─ sharding_planner = ShardingPlanner()                  # 05 §5
            │   └─ script_generator = ModelScriptGenerator()             # ★ 本文档：可选
            │
            ├─④.3.5.2 _init_model() → meta device 空壳模型              # 01 §7
            ├─④.3.5.5 load_weights() → 权重加载                        # 01 §10
            │
            ├─④.3.5.7 plan = sharding_planner.plan(model, mesh, ...)     # 05 §5: 编译期规划
            └─④.3.5.8 apply_sharding_plan(model, plan, mesh, ...)        # 05 §6/§7/§8: 运行时应用
                │
                └─④.3.5.9 [if cfg.scriptgen.enabled]                     # ★ 本文档：可选脚本生成
                    └─ script_generator.generate(
                           model, plan, mesh,
                           output_dir="generated/",
                           approach="wrapper",  # 或 "inline"
                       )
```

### 2.2 独立使用：不需要训练流程

```python
from transformers import AutoModelForCausalLM
from hyper_models.components.distributed import (
    MeshContext, ShardingPlanner, apply_sharding_plan,
    generate_model_script,  # ★ 新增
)

# 1. 加载模型 + 分片（标准流程）
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B")
mesh = MeshContext.build(strategy_config, sizes)
plan = ShardingPlanner().plan(model, mesh.device_mesh, tp_size=4)
apply_sharding_plan(model, plan, mesh.device_mesh)

# 2. 生成独立脚本（新增）
generate_model_script(
    model, plan, mesh.device_mesh,
    output_path="generated/standalone_qwen3_tp4.py",
    approach="wrapper",
)
```

### 2.3 触发方式

scriptgen 支持三种触发方式：

| 方式 | 触发位置 | 适用场景 |
|------|---------|---------|
| **配置自动触发** | `cfg.scriptgen.enabled = true` | 常规训练任务，自动生成调试脚本 |
| **API 手动调用** | 调用 `generate_model_script()` | 独立使用场景、临时调试 |
| **CLI 命令** | `hyper-parallel generate-script --config train.yaml` | 基于已有配置文件生成 |

---

## 3. 核心设计

### 3.1 设计目标

1. **自包含**：生成的脚本仅依赖 `torch`、`torch.distributed` 和 `transformers`，**不依赖** `hyper_parallel`
2. **可调试**：所有通信操作是显式的函数调用，可以单步调试、print tensor shape、加断点
3. **可读性**：清晰标注哪些是原始代码、哪些是插入的通信逻辑
4. **可复现**：生成的脚本独立运行结果与原始 wrapped 模型一致
5. **版本标记**：记录生成时的 transformers 版本、分片配置等元信息

### 3.2 生成什么

一份独立的 Python 脚本，包含：

```
standalone_qwen3_tp4.py
├── [HEADER]    生成元信息（时间、配置、分片策略摘要）
├── [IMPORTS]   必要的 import（torch, DTensor, distribute_tensor, DeviceMesh）
├── [MESH]      DeviceMesh 构建代码（tp_size, dp_size, mesh_shape）
├── [CLASSES]   所有模型类（原始 HF 类 + 通信注入）
│   ├── Qwen3RMSNorm          → 无 boundary（norm 模块：Replicate）
│   ├── Qwen3Attention        → 有 boundary（in/out 重分布 + _local_ctx）
│   ├── Qwen3MLP              → 有 boundary（in/out 重分布 + _local_ctx）
│   ├── Qwen3DecoderLayer     → 无 boundary（容器模块）
│   ├── Qwen3Model            → 无 boundary
│   └── Qwen3ForCausalLM      → 无 boundary（仅参数 shard）
├── [PARAM_SHARDING]  参数分片代码（distribute_tensor 调用）
├── [MAIN]            __main__ 入口：加载权重 → 分片 → 配置 boundary → 测试 forward
└── [TEST]            可选的 smoke test（随机输入 forward + 输出校验）
```

### 3.3 两种生成方式对比

| 维度 | Wrapper 类继承（推荐） | AST 内联改写（高级） |
|------|----------------------|---------------------|
| **原理** | 生成子类，override `forward` 显式调用 boundary | 直接修改原始 `forward` 方法的 AST，插入 boundary 调用 |
| **代码形态** | `class Qwen3Attention_TP(Qwen3Attention)` | 改写后的 `Qwen3Attention.forward` 源码 |
| **可调试性** | ⭐⭐⭐⭐ 可进入 `super().forward()` 对比 | ⭐⭐⭐⭐⭐ 通信与计算在同一函数体内 |
| **鲁棒性** | ⭐⭐⭐⭐⭐ 不修改原始代码，wrapper 失败只影响外层 | ⭐⭐⭐ 需处理各种 forward 签名、return 形态 |
| **生成难度** | 低（模板驱动） | 中（AST 重写 + 变量名推断） |
| **与原始代码对比** | 需对照两个类 | 同一方法体内对比 |
| **适用场景** | 所有模型、快速调试 | 单步深度调试、代码审查 |

**推荐策略**：默认使用 wrapper 类继承方案；对特定模块（如 `Qwen3Attention.forward`）可按需使用 AST 内联方案。

### 3.4 Wrapper 类继承方案（推荐）

#### 3.4.1 基本原理

对每个有 `PrecompiledBoundary` 的模块，生成一个**子类**，其 `forward` 方法显式调用 `boundary.pre_forward` / `boundary.post_forward`。

```
原始 HF 代码（不修改）:
  class Qwen3Attention(nn.Module):
      def forward(self, hidden_states, ...):
          # 原始计算逻辑
          ...
          return output

生成脚本（wrapper 类）:
  class Qwen3Attention_TP(Qwen3Attention):          # ← 新类，继承原类
      def __init__(self, config, layer_idx, *, boundary_config):
          super().__init__(config, layer_idx)
          self._boundary = self._build_boundary(boundary_config)

      def forward(self, hidden_states, ...):
          # === [HP BOUNDARY] pre_forward: Shard(0) → Shard(1) ===
          hidden_states = self._boundary.redistribute_input(
              "hidden_states", hidden_states
          )
          # === 原始计算 ===
          output = super().forward(hidden_states, ...)
          # === [HP BOUNDARY] post_forward: Partial → Shard(0) ===
          output = self._boundary.redistribute_output(output)
          return output
```

#### 3.4.2 优势

- **零侵入**：原始 HF 类定义完整保留在脚本中，方便对比
- **健壮**：不解析 forward 内部逻辑，只在外层包装
- **可单步调试**：在 `super().forward()` 前后设断点，精确观察通信前后的 tensor
- **模板化生成**：所有 boundary wrapper 结构相同，模板化程度高

#### 3.4.3 生成逻辑

```
for each ModuleShardingSpec in plan.module_specs:
    if spec.is_boundary:
        1. 获取原始类名（如 Qwen3Attention）
        2. 从 HF 源码中提取原始类定义（inspect.getsource）
        3. 生成 wrapper 类：
           a. 类名 = 原始类名 + _TP (或 _TP{size}_CP{size})
           b. __init__: 调用 super().__init__ + 构建 boundary
           c. forward: 模板化包装（见上）
        4. 注册类名映射：Qwen3Attention → Qwen3Attention_TP
    else:
        保留原始类定义不变
```

### 3.5 AST 内联方案（高级）

#### 3.5.1 基本原理

直接解析原始 `forward` 方法的 AST，在 AST 层面插入 boundary 调用节点，然后 `ast.unparse` 回源码。

```
原始 forward 的 AST:
  FunctionDef(name='forward', args=...)
    body=[
      Assign(...),          # qkv = self.qkv_proj(hidden_states)
      Assign(...),          # attn_output = ...
      Return(attn_output),  # return attn_output
    ]

改写后 forward 的 AST:
  FunctionDef(name='forward', args=...)
    body=[
      # === 插入: boundary pre_forward ===
      Expr(Comment('# [HP BOUNDARY] pre_forward: Replicate → Shard(1)')),
      Assign('hidden_states', Call(self._boundary.pre_forward, [Name('hidden_states')])),
      # === 原始代码 ===
      Assign(...),          # qkv = self.qkv_proj(hidden_states)
      Assign(...),          # attn_output = ...
      # === 插入: boundary post_forward ===
      Expr(Comment('# [HP BOUNDARY] post_forward: Partial → Shard(0)')),
      Assign('attn_output', Call(self._boundary.post_forward, [Name('attn_output')])),
      Return(attn_output),
    ]
```

#### 3.5.2 关键挑战与处理

| 挑战 | 处理方式 |
|------|---------|
| **识别 hidden_states 参数** | 默认取 `forward` 的第一个非 self 参数（HF 惯例）；如果 `ModuleShardingSpec.in_dst` 指定了多输入，按参数名匹配 |
| **识别输出变量** | 找到所有 `Return` 节点，取 `return` 的变量名；对 `return tuple(...)` 形态只对第一个元素做重分布 |
| **处理 early return** | 每个 `Return` 前都插入 `post_forward`（如果有 early return 路径） |
| **处理多输出** | `return a, b, c` → 只对 `a` 做重分布（`hidden_states` 的输出），`b`, `c` 透传 |
| **注释保留** | 插入的 boundary 调用使用 `# [HP BOUNDARY]` 前缀注释，与原始注释区分 |
| **缩进保持** | AST 重写后通过 `ruff format` 统一格式化，不手动维护缩进 |

#### 3.5.3 变量名推断

```python
def _infer_hidden_states_name(forward_node: ast.FunctionDef) -> str:
    """推断 forward 方法的 hidden_states 参数名。"""
    # 策略 1: 参数名直接匹配 "hidden_states"
    for arg in forward_node.args.args:
        if arg.arg == "hidden_states":
            return "hidden_states"

    # 策略 2: 第一个非 self 参数即为 hidden_states（HF 惯例）
    if len(forward_node.args.args) > 1:
        return forward_node.args.args[1].arg

    # 策略 3: 只有 self → 无输入 tensor（极少数情况）
    return None
```

### 3.6 参数分片代码生成

除了 forward 通信注入，生成的脚本还需要包含**显式的参数分片代码**。运行时 `apply_sharding_plan` 通过 `distribute_tensor` + `setattr` 完成了参数 → DTensor 的转换；生成脚本中需要等价代码。

```python
# 生成脚本中的参数分片部分

def _shard_parameters(model, mesh, sharding_map):
    """显式的参数分片，等价于 apply_sharding_plan 的 Phase A。"""
    for param_path, placements in sharding_map.items():
        module_path, param_name = param_path.rsplit(".", 1)
        module = _resolve_module(model, module_path)
        param = getattr(module, param_name)
        dt = distribute_tensor(param, mesh, placements)
        setattr(module, param_name, nn.Parameter(dt, requires_grad=param.requires_grad))

# sharding_map 从 ShardingPlan 自动生成
SHARDING_MAP = {
    "model.layers.0.self_attn.q_proj.weight": [Shard(0)],
    "model.layers.0.self_attn.k_proj.weight": [Shard(0)],
    "model.layers.0.self_attn.v_proj.weight": [Shard(0)],
    "model.layers.0.self_attn.o_proj.weight": [Shard(1)],
    "model.layers.0.mlp.gate_proj.weight": [Shard(0)],
    "model.layers.0.mlp.up_proj.weight": [Shard(0)],
    "model.layers.0.mlp.down_proj.weight": [Shard(1)],
    # ...
}
```

---

## 4. 数据流与组件

### 4.1 组件关系图

```
┌─────────────────────────────────────────────────────────────┐
│                   ModelScriptGenerator                       │
│  generate(model, plan, mesh, approach, output_dir) → Path   │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
┌─────────────────┐ ┌─────────────┐ ┌──────────────────┐
│ BoundaryExtractor│ │ScriptCodeGen│ │  TemplateRenderer │
│                  │ │             │ │                  │
│ - extract_specs  │ │ - load_hf   │ │ - render_header   │
│ - extract_params │ │   _source   │ │ - render_imports  │
│ - extract_mesh   │ │ - build_ast │ │ - render_wrapper  │
│ - extract_class  │ │ - insert_bc │ │ - render_main     │
│   _hierarchy     │ │ - unparse   │ │ - render_test     │
└─────────────────┘ └─────────────┘ └──────────────────┘
```

### 4.2 ModelScriptGenerator

```python
@dataclass
class ScriptGenConfig:
    """脚本生成配置。

    通常在 RecipeConfig 中以 `scriptgen:` 子配置提供。
    """
    enabled: bool = False                    # 是否启用脚本生成
    output_dir: str = "generated"            # 输出目录（相对于项目根目录）
    approach: str = "wrapper"                # "wrapper" | "inline" | "both"
    include_smoke_test: bool = True          # 是否包含 smoke test
    include_diff: bool = True                # 是否生成 unified diff
    ruff_format: bool = True                 # 是否 ruff 格式化生成文件
    ruff_extra_ignore: tuple[str, ...] = ("E501", "E402", "B007")
    single_file: bool = True                 # 单文件输出 vs 多文件（模型+config）
    keep_original_classes: bool = True       # 是否在脚本中保留原始类定义
```

```python
class ModelScriptGenerator:
    """从运行时 wrapped 模型 + ShardingPlan 生成独立脚本。

    统一入口，内部根据 approach 参数调度不同的代码生成策略。
    """

    def __init__(self, config: ScriptGenConfig = ScriptGenConfig()):
        self.config = config
        self._extractor = BoundaryExtractor()
        self._codegen = ScriptCodeGenerator()

    def generate(
        self,
        model: nn.Module,
        plan: ShardingPlan,
        mesh: DeviceMesh,
        *,
        output_dir: str | Path | None = None,
        approach: str | None = None,
    ) -> Path:
        """生成独立模型脚本。

        Args:
            model: 已完成 apply_sharding_plan 的 wrapped 模型
            plan: ShardingPlanner 产出的分片计划
            mesh: DeviceMesh 实例
            output_dir: 输出目录，覆盖 config
            approach: "wrapper" | "inline" | "both"，覆盖 config

        Returns:
            生成的脚本文件路径
        """
        approach = approach or self.config.approach
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: 从运行时模型中提取 boundary 信息
        boundary_specs = self._extractor.extract_specs(model, plan)

        # Step 2: 提取类层次结构（模块树 → 类依赖图）
        class_hierarchy = self._extractor.extract_class_hierarchy(model)

        # Step 3: 提取参数分片映射
        sharding_map = self._extractor.extract_sharding_map(model, plan)

        # Step 4: 提取 mesh 构建代码
        mesh_code = self._extractor.extract_mesh_code(mesh)

        # Step 5: 加载 HF 原始源码（用于保留原始类定义）
        hf_sources = self._codegen.load_hf_sources(
            class_hierarchy.source_modules
        )

        # Step 6: 使用指定 approach 生成代码
        if approach in ("wrapper", "both"):
            wrapper_path = self._codegen.generate_wrapper_script(
                hf_sources=hf_sources,
                boundary_specs=boundary_specs,
                sharding_map=sharding_map,
                mesh_code=mesh_code,
                class_hierarchy=class_hierarchy,
                config=self.config,
                output_dir=output_dir,
            )

        if approach in ("inline", "both"):
            inline_path = self._codegen.generate_inline_script(
                hf_sources=hf_sources,
                boundary_specs=boundary_specs,
                sharding_map=sharding_map,
                mesh_code=mesh_code,
                class_hierarchy=class_hierarchy,
                config=self.config,
                output_dir=output_dir,
            )

        return wrapper_path if approach == "wrapper" else inline_path
```

### 4.3 BoundaryExtractor

```python
class BoundaryExtractor:
    """从运行时 wrapped 模型中提取 boundary 相关信息。

    核心假设：
    - 每个被 _wrap_forward 修改过的模块，其 forward.__wrapped__ 指向原始 forward
    - 每个被包装的模块上有 `_hp_boundary` 属性（PrecompiledBoundary 实例）
    - ShardingPlan 中有完整的 ModuleShardingSpec 描述
    """

    def extract_specs(
        self, model: nn.Module, plan: ShardingPlan
    ) -> list[BoundarySpec]:
        """提取每个模块的 boundary 规格。

        Returns:
            [BoundarySpec(module_path, in_plan, out_plan, ...), ...]
        """
        specs = []
        for module_path, spec in plan.module_specs.items():
            if not spec.is_boundary:
                continue

            module = _resolve_module(model, module_path)
            boundary = getattr(module, "_hp_boundary", None)
            if boundary is None:
                continue

            specs.append(BoundarySpec(
                module_path=module_path,
                module_class=type(module).__name__,
                in_plan=boundary.in_plan,      # list[RedistOp]
                out_plan=boundary.out_plan,    # list[RedistOp]
                in_src=spec.in_src,            # NamedPlacement
                in_dst=spec.in_dst,
                out_src=spec.out_src,
                out_dst=spec.out_dst,
                is_terminal=spec._is_terminal,
            ))
        return specs

    def extract_sharding_map(
        self, model: nn.Module, plan: ShardingPlan
    ) -> dict[str, list[Placement]]:
        """提取参数 → placement 映射。

        Returns:
            {"model.layers.0.self_attn.q_proj.weight": [Shard(0)], ...}
        """
        sharding_map = {}
        for module_path, spec in plan.module_specs.items():
            for param_name, placements in spec.params.items():
                full_path = f"{module_path}.{param_name}"
                sharding_map[full_path] = placements
        return sharding_map

    def extract_class_hierarchy(
        self, model: nn.Module
    ) -> ClassHierarchy:
        """提取模型的类层次结构。

        Returns 模块名 → (类名, 源文件路径, 基类名) 的映射，
        用于确定哪些 HF 源码文件需要被加载和重写。
        """
        hierarchy = ClassHierarchy()
        for name, module in model.named_modules():
            cls = type(module)
            source_file = inspect.getfile(cls)
            source_module = _file_to_module(source_file)
            hierarchy.add(name, cls.__name__, source_module,
                         [b.__name__ for b in cls.__bases__])
        return hierarchy

    def extract_mesh_code(self, mesh: DeviceMesh) -> str:
        """生成 DeviceMesh 构建代码。"""
        mesh_shape = mesh.shape
        mesh_dim_names = mesh.mesh_dim_names
        return (
            f"mesh = DeviceMesh(\n"
            f"    device_type='cuda',\n"
            f"    mesh=torch.arange({math.prod(mesh_shape)}).reshape({mesh_shape}),\n"
            f"    mesh_dim_names={mesh_dim_names},\n"
            f")"
        )
```

### 4.4 ScriptCodeGenerator

```python
class ScriptCodeGenerator:
    """AST 驱动的代码生成器。

    负责：
    1. 加载 HF 原始源码
    2. 解析 AST
    3. 注入 wrapper / inline boundary 调用
    4. 生成自包含脚本
    """

    def load_hf_sources(
        self, source_modules: set[str]
    ) -> dict[str, str]:
        """加载所有需要的 HF 源码。

        Returns: {module_name: source_code}
        """
        sources = {}
        for module_name in source_modules:
            # 优先文件查找（不 import HF 模块，避免触发 CUDA kernel 加载）
            file_path = _find_module_file(module_name)
            if file_path:
                sources[module_name] = file_path.read_text(encoding="utf-8")
            else:
                # 回退：import + inspect.getsource
                mod = importlib.import_module(module_name)
                sources[module_name] = inspect.getsource(mod)
        return sources

    def generate_wrapper_script(self, ...) -> Path:
        """生成 wrapper 类继承方案的独立脚本。"""
        # 1. 生成 HEADER
        # 2. 生成 IMPORTS
        # 3. 生成 MESH 构建代码
        # 4. 遍历 class_hierarchy，对每个类：
        #    a. 如果有 boundary → 生成 wrapper 子类
        #    b. 如果无 boundary → 保留原始类定义
        # 5. 生成类名替换映射（Qwen3Attention → Qwen3Attention_TP）
        # 6. 生成 _build_model 函数（用替换映射构建模型）
        # 7. 生成 _shard_parameters 函数
        # 8. 生成 __main__ 入口（加载权重 → 分片 → 配置 boundary → forward 测试）
        ...

    def generate_inline_script(self, ...) -> Path:
        """生成 AST 内联方案的独立脚本。"""
        # 1-3: 同上
        # 4. 遍历 class_hierarchy，对每个有 boundary 的类：
        #    a. AST parse forward 方法
        #    b. 在 body 首尾插入 boundary 调用节点
        #    c. ast.unparse → 生成新源码
        # 5. 生成参数分片代码
        # 6-8: 同上
        ...
```

---

## 5. 生成流程详解（Wrapper 方案）

### 5.1 Phase 1: 收集信息

```
generate() 入口
│
├─ Step 1-2: BoundaryExtractor 从运行时模型提取
│   ├─ boundary_specs:  每个有 boundary 的模块的 in/out plan
│   ├─ sharding_map:    参数路径 → placement 映射
│   ├─ class_hierarchy: 模块树 → 类名/源文件映射
│   └─ mesh_code:        DeviceMesh 构建代码
│
└─ Step 3: ScriptCodeGenerator 加载 HF 原始源码
    └─ hf_sources: {module_name: source_code}
```

### 5.2 Phase 2: 确定需要生成的类

```
对于模型中的每个模块:
  if 模块类型在 class_hierarchy 中已处理过:
      skip
  if 模块有 boundary:
      标记为 WRAPPER_CLASS（需要生成 wrapper 子类）
      标记原始类为 KEEP_ORIGINAL（保留原始类供继承）
  else:
      标记为 KEEP_ORIGINAL（保留原始类定义）
```

### 5.3 Phase 3: 生成脚本

```
输出文件 = HEADER + IMPORTS + MESH + CLASSES + SHARDING + MAIN [+ TEST]

各部分的生成逻辑:

HEADER:
  # ================================================================================
  #  AUTO-GENERATED FILE - DO NOT EDIT DIRECTLY
  # ================================================================================
  #
  #  Source: transformers.models.qwen3.modeling_qwen3
  #  Based on: transformers==4.52.0
  #  Generated by: hyper_parallel v1.0.0, ModelScriptGenerator
  #  Generation time: 2026-07-16 15:30:00 UTC
  #  Approach: wrapper
  #
  #  DTensor Configuration:
  #    TP size: 4
  #    CP size: 1
  #    DP size: 2
  #    Mesh shape: (4, 1, 2)
  #
  #  Modules with communication boundaries:
  #    - Qwen3Attention (24 layers): in: Replicate→Shard(1), out: Partial→Shard(0)
  #    - Qwen3MLP (24 layers):       in: Shard(0)→Shard(1), out: Partial→Shard(0)
  #
  #  This file is fully self-contained. It does NOT depend on hyper_parallel.
  #  Run with: torchrun --nproc_per_node=4 standalone_qwen3_tp4.py
  #
  # ================================================================================

IMPORTS:
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
  from torch.distributed.tensor import DTensor, Shard, Replicate, Partial
  from torch.distributed.tensor.parallel import (
      distribute_tensor, distribute_module,
  )
  from transformers import AutoConfig
  # ... 以及 HF 原始代码中的 import（去重后）

CLASSES:
  对于每个类:

  # === 原始类（保留不变）===
  class Qwen3Attention(nn.Module):     # ← 原始 HF 类，一字不改
      def __init__(self, config, layer_idx):
          ...
      def forward(self, hidden_states, ...):
          ...

  # === Wrapper 子类 ===
  class Qwen3Attention_TP(Qwen3Attention):
      """[HP GENERATED] TP-sharded Qwen3Attention with explicit boundary calls.

      Boundary spec:
        in:  Replicate → Shard(1)   (all-gather not needed, input already Replicate)
        out: Partial → Shard(0)     (reduce-scatter)
      """
      def __init__(self, config, layer_idx, *, mesh, boundary_config=None):
          super().__init__(config, layer_idx)
          self._mesh = mesh
          self._boundary = PrecompiledBoundary(
              in_plan=[
                  RedistOp(
                      arg_name="hidden_states",
                      arg_index=0,
                      mesh=mesh,
                      src_placements=(Replicate(),),
                      dst_placements=(Shard(1),),
                      collective_type="all_gather",
                  ),
              ],
              out_plan=[
                  RedistOp(
                      arg_name="output",
                      arg_index=None,
                      mesh=mesh,
                      src_placements=(Partial(),),
                      dst_placements=(Shard(0),),
                      collective_type="reduce_scatter",
                  ),
              ],
          )
          self._dtensor_params = {}  # 在 _shard_parameters 中填充

      @contextlib.contextmanager
      def _local_ctx(self):
          """零拷贝替换 DTensor → local tensor，等价于 _local_params_context。"""
          # 保存 DTensor 引用
          saved = {}
          for name in self._dtensor_params:
              saved[name] = getattr(self, name)
              setattr(self, name, saved[name]._local_tensor)
          try:
              yield
          finally:
              for name, val in saved.items():
                  setattr(self, name, val)

      def forward(
          self,
          hidden_states: torch.Tensor,
          position_embeddings: tuple[torch.Tensor, torch.Tensor],
          attention_mask: torch.Tensor | None = None,
          **kwargs,
      ):
          # === [HP BOUNDARY] pre_forward ===
          hidden_states = self._boundary.redistribute_inputs(
              (hidden_states,),
              {"position_embeddings": position_embeddings,
               "attention_mask": attention_mask},
          )[0][0]

          # === 原始计算（通过 super().forward）===
          with self._local_ctx():
              output = super().forward(
                  hidden_states,
                  position_embeddings=position_embeddings,
                  attention_mask=attention_mask,
                  **kwargs,
              )

          # === [HP BOUNDARY] post_forward ===
          output = self._boundary.redistribute_outputs(output)

          return output

  # === 无 boundary 的类：保留原始定义 ===
  class Qwen3RMSNorm(nn.Module):       # ← norm 模块：Replicate，无需通信
      ...

SHARDING:
  def _shard_parameters(model, mesh, sharding_map):
      """显式参数分片。"""
      for param_path, placements in sharding_map.items():
          module_path, param_name = param_path.rsplit(".", 1)
          module = _resolve_module(model, module_path)
          param = getattr(module, param_name)
          dt = distribute_tensor(param, mesh, placements)
          setattr(module, param_name, nn.Parameter(dt))

  SHARDING_MAP = {
      "model.layers.0.self_attn.q_proj.weight": [Shard(0)],
      "model.layers.0.self_attn.o_proj.weight": [Shard(1)],
      # ... (从 ShardingPlan 自动生成)
  }

MAIN:
  if __name__ == "__main__":
      # 1. 初始化分布式环境
      dist.init_process_group(backend="nccl")
      local_rank = int(os.environ["LOCAL_RANK"])
      torch.cuda.set_device(local_rank)

      # 2. 构建 DeviceMesh
      mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("tp",))

      # 3. 加载模型配置
      config = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B")

      # 4. 构建模型（使用 wrapper 类）
      model = _build_model_with_wrappers(config, mesh)

      # 5. 加载权重
      state_dict = load_hf_state_dict("Qwen/Qwen3.5-4B")
      model.load_state_dict(state_dict, strict=False)

      # 6. 参数分片
      _shard_parameters(model, mesh, SHARDING_MAP)

      # 7. Smoke test
      _smoke_test(model, mesh, local_rank)

SMOKE_TEST:
  def _smoke_test(model, mesh, local_rank):
      """简单的 forward 测试，验证分片正确性。"""
      batch_size, seq_len = 2, 128
      input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
      input_ids = input_ids.cuda(local_rank)

      model.eval()
      with torch.no_grad():
          output = model(input_ids=input_ids)

      if local_rank == 0:
          print(f"[Smoke Test] PASS - output shape: {output.logits.shape}")
          print(f"[Smoke Test] output mean: {output.logits.mean().item():.6f}")
```

### 5.4 Phase 4: 格式化和输出

```
生成 raw 脚本
│
├─ ruff check --fix --ignore E402,B007,E501 <script_path>
├─ ruff format <script_path>
│
├─ [if config.include_diff] 生成 unified diff
│   └─ standalone_qwen3_tp4.diff（与原始 HF 代码对比）
│
└─ 输出:
    generated/
    ├── standalone_qwen3_tp4.py      # 完整独立脚本
    └── standalone_qwen3_tp4.diff    # unified diff（可选）
```

---

## 6. PrecompiledBoundary 的独立副本

生成的脚本需要 `PrecompiledBoundary` 和 `RedistOp` 类，但不依赖 hyper_parallel。解决方案：

### 6.1 内联核心类

生成脚本中嵌入一个精简版的 `PrecompiledBoundary` 和 `RedistOp`（~80 行），与 05 文档 §7 的实现逻辑完全一致，但去除了 hyper_parallel 内部依赖。

```python
# ================================================================================
# [HP INLINED] PrecompiledBoundary — standalone copy, no hyper_parallel dependency
# ================================================================================

@dataclass
class RedistOp:
    """一个预编译的 redistribute 操作。"""
    arg_name: str
    arg_index: int | None
    mesh: DeviceMesh
    src_placements: tuple
    dst_placements: tuple
    collective_type: str  # "identity" | "all_gather" | "reduce_scatter" | "all_reduce"

    def execute(self, tensor: torch.Tensor, *, as_dtensor: bool = False):
        if self.collective_type == "identity":
            if as_dtensor and not isinstance(tensor, DTensor):
                return DTensor.from_local(tensor, self.mesh, list(self.src_placements), run_check=False)
            return tensor

        dt = DTensor.from_local(tensor, self.mesh, list(self.src_placements), run_check=False)
        dt = dt.redistribute(placements=list(self.dst_placements), async_op=True)
        return dt if as_dtensor else dt.to_local()


class PrecompiledBoundary:
    """编译期通信计划。与 hyper_parallel 中的实现等价，独立副本。"""

    def __init__(self, in_plan: list[RedistOp], out_plan: list[RedistOp]):
        self.in_plan = in_plan
        self.out_plan = out_plan

    def redistribute_inputs(self, args, kwargs, *, as_dtensor=False):
        for op in self.in_plan:
            if op.arg_index is not None and op.arg_index < len(args):
                args = list(args)
                args[op.arg_index] = op.execute(args[op.arg_index], as_dtensor=as_dtensor)
                args = tuple(args)
            elif op.arg_name in kwargs:
                kwargs[op.arg_name] = op.execute(kwargs[op.arg_name], as_dtensor=as_dtensor)
        return args, kwargs

    def redistribute_outputs(self, outputs, *, as_dtensor_input=False):
        for op in self.out_plan:
            if as_dtensor_input:
                outputs = op.execute(outputs, as_dtensor=True)
            else:
                outputs = op.execute(outputs, as_dtensor=False)
        return outputs
```

### 6.2 为什么不 import hyper_parallel？

生成的脚本的目标是**独立可运行**。引入 `import hyper_parallel` 会：
1. 增加环境依赖（脚本只能在没有安装 hyper_parallel 的环境运行? 不，用户已在用 hyper_parallel）
2. 脚本行为取决于 hyper_parallel 版本（未来升级可能导致脚本行为变化）

**实际上，因为 hyper_parallel 本来就已安装**，更好的做法是让生成的脚本**可选地**依赖 hyper_parallel：
- 默认模式：内联精简版 PrecompiledBoundary（~80 行），保证独立性
- 可选模式：`from hyper_models.components.distributed.precompiled_boundary import PrecompiledBoundary`（更简洁，版本锁定）——通过 `ScriptGenConfig.inline_boundary_class = False` 控制

---

## 7. 类名替换映射（class_name_map）

生成脚本中，所有引用被 wrapper 类替换的地方都需要更新类名。例如 `Qwen3DecoderLayer` 中使用 `Qwen3Attention` 的地方要替换为 `Qwen3Attention_TP`。

### 7.1 映射表生成

```python
# 从 class_hierarchy + boundary_specs 自动推导
CLASS_NAME_MAP = {
    # 有 boundary 的模块 → wrapper 类名
    "Qwen3Attention": "Qwen3Attention_TP",
    "Qwen3MLP": "Qwen3MLP_TP",
    # 无 boundary 的模块 → 保持原名（不在 map 中）
}

# 在容器类（如 Qwen3DecoderLayer）的 __init__ 中，
# 所有 self.self_attn = Qwen3Attention(...) 的引用会自动替换。
```

### 7.2 自动替换策略

```
方式 A（简单）：在生成的脚本中，在所有类定义的源码级做字符串替换
  - 优点：简单粗暴，不需要 AST 改写
  - 缺点：可能误替换 docstring 中的类名

方式 B（精确）：AST 解析容器类的 __init__，找到赋值语句中的类名引用，
            仅替换这些位置
  - 优点：精确，不会误伤
  - 缺点：复杂，需要处理各种写法

推荐：方式 B（AST 精确替换），对无法解析的类 fallback 到方式 A。
```

---

## 8. 使用示例

### 8.1 配置自动触发（推荐）

```yaml
# train_qwen3_tp4.yaml
scriptgen:
  enabled: true
  output_dir: "generated/"
  approach: "wrapper"         # wrapper | inline | both
  include_smoke_test: true
  include_diff: true
  single_file: true
  inline_boundary_class: true # 内联 PrecompiledBoundary（vs import hyper_parallel）

model:
  _target_: hyper_models._transformers.HyperAutoModelForCausalLM
  model_name_or_path: "Qwen/Qwen3.5-4B"
  # ...

distributed:
  tp_size: 4
  dp_size: 2
  # ...
```

### 8.2 API 手动调用

```python
from hyper_models.components.distributed import (
    MeshContext, ShardingPlanner, apply_sharding_plan,
    generate_model_script,
)

# 标准流程
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B")
mesh = MeshContext.build(strategy_config, sizes)
plan = ShardingPlanner().plan(model, mesh.device_mesh, tp_size=4)
apply_sharding_plan(model, plan, mesh.device_mesh)

# 生成脚本
script_path = generate_model_script(
    model, plan, mesh.device_mesh,
    output_path="generated/standalone_qwen3_tp4.py",
    approach="wrapper",
)
print(f"Script generated: {script_path}")
```

### 8.3 生成脚本的使用

```bash
# 1. 生成脚本（在训练任务初始化阶段自动完成，或手动执行）
python train_qwen3.py --config train_qwen3_tp4.yaml

# 2. 查看生成脚本
cat generated/standalone_qwen3_tp4.py

# 3. 独立运行生成脚本（不需要 hyper_parallel）
torchrun --nproc_per_node=4 generated/standalone_qwen3_tp4.py

# 4. 单 GPU 调试（可设 tp_size=1 或设置 CUDA_VISIBLE_DEVICES=0）
CUDA_VISIBLE_DEVICES=0 python generated/standalone_qwen3_tp4.py

# 5. 查看与原始 HF 代码的 diff
cat generated/standalone_qwen3_tp4.diff

# 6. pdb 调试每个通信步骤
python -m pdb generated/standalone_qwen3_tp4.py
```

### 8.4 调试场景

```python
# 在生成的脚本中，你可以随意添加调试代码：

class Qwen3Attention_TP(Qwen3Attention):
    def forward(self, hidden_states, ...):
        # --- 添加调试代码 ---
        print(f"[DEBUG] {self.layer_idx} input shape: {hidden_states.shape}")
        print(f"[DEBUG] {self.layer_idx} input device: {hidden_states.device}")

        # 在通信前设断点
        import pdb; pdb.set_trace()  # ← 检查通信前的 tensor

        hidden_states = self._boundary.redistribute_inputs(...)

        # 在通信后设断点
        print(f"[DEBUG] {self.layer_idx} after pre_forward shape: {hidden_states.shape}")

        with self._local_ctx():
            output = super().forward(hidden_states, ...)

        # 检查原始输出
        print(f"[DEBUG] {self.layer_idx} output shape: {output.shape}")

        output = self._boundary.redistribute_outputs(output)

        print(f"[DEBUG] {self.layer_idx} after post_forward shape: {output.shape}")

        return output
```

---

## 9. 与 VeOmni patchgen 的集成关系

scriptgen 不是要**替代** patchgen，而是解决不同的需求。两者可以组合使用：

```
HF 源码
  │
  ├─ [VeOmni patchgen] 静态补丁
  │   └─ 替换类/方法为优化版本（Liger kernel 等）
  │   └─ 输出: patched_model.py
  │
  ├─ [hyper_parallel] 运行时包装
  │   ├─ apply_sharding_plan → DTensor + PrecompiledBoundary
  │   └─ 输出: 内存中的 wrapped model
  │
  └─ [hyper_parallel scriptgen] 脚本生成 ★
      ├─ 从 wrapped model 回溯生成显式通信脚本
      └─ 输出: standalone_model.py（用于调试）
```

如果项目同时使用 patchgen 和 hyper_parallel，理想流程是：

```bash
# Step 1: patchgen → 生成优化后的建模代码（用 Liger kernel 等）
patchgen --all --diff

# Step 2: 训练（使用 hyper_parallel + patchgen 优化的模型）
python train.py --config train_config.yaml

# Step 3: scriptgen → 在训练初始化阶段自动生成调试脚本
# （train_config.yaml 中 scriptgen.enabled=true）
# → 生成 generated/standalone_qwen3_tp4.py

# Step 4: 用生成的脚本调试
torchrun --nproc_per_node=4 generated/standalone_qwen3_tp4.py
```

---

## 10. 配置完整示例

```yaml
# train_config.yaml

scriptgen:
  # === 基础配置 ===
  enabled: true
  output_dir: "generated/"
  approach: "wrapper"               # wrapper | inline | both

  # === 输出控制 ===
  single_file: true                 # 单文件输出
  include_smoke_test: true          # 包含 forward smoke test
  include_diff: true                # 生成 unified diff
  keep_original_classes: true       # 保留原始 HF 类定义

  # === 格式化 ===
  ruff_format: true                 # ruff 格式化
  ruff_extra_ignore: ["E501", "E402", "B007"]

  # === 依赖模式 ===
  inline_boundary_class: true       # 内联 PrecompiledBoundary（vs import hyper_parallel）

  # === 高级选项 ===
  target_modules: []                # 空=全部，指定则只生成这些模块
  # target_modules: ["Qwen3Attention", "Qwen3MLP"]
  exclude_modules: []               # 排除特定模块
  # exclude_modules: ["Qwen3RMSNorm"]

  # === 调试增强 ===
  add_debug_logging: false          # 生成脚本中包含 DEBUG 日志
  add_pdb_hooks: false              # 在每个 boundary 调用前后插入 pdb.set_trace()

model:
  _target_: hyper_models._transformers.HyperAutoModelForCausalLM
  model_name_or_path: "Qwen/Qwen3.5-4B"
  trust_remote_code: true
  torch_dtype: "bfloat16"

distributed:
  tp_size: 4
  cp_size: 1
  dp_size: 2
  ep_size: 1

# ... 其余训练配置
```

---

## 11. 需求分解

### 11.1 模块需求

| 子需求编号 | 子需求 | 说明 | 工时(AI人·日) |
|-----------|--------|------|--------------|
| **N_A: ScriptGenConfig** | | | **0.5** |
| N_A-1 | `ScriptGenConfig` 数据类 | `ScriptGenConfig` dataclass，包含所有配置字段：`enabled`, `output_dir`, `approach`, `include_smoke_test`, `include_diff`, `ruff_format`, `inline_boundary_class`, `target_modules`, `exclude_modules`, `add_debug_logging`, `add_pdb_hooks` | 0.3 |
| N_A-2 | RecipeConfig 集成 | 在 `RecipeConfig` 中添加 `scriptgen:` 子配置，支持 `cfg.scriptgen.instantiate()` 和 `cfg.scriptgen.build()` | 0.2 |
| **N_B: BoundaryExtractor** | | | **0.8** |
| N_B-1 | `extract_specs` | 从运行时模型 + ShardingPlan 中提取每个模块的 `BoundarySpec`（包含 `in_plan`, `out_plan`, `in_src`, `in_dst`, `out_src`, `out_dst`） | 0.3 |
| N_B-2 | `extract_sharding_map` | 从 `ShardingPlan.module_specs` 中提取参数分片映射 `{param_path: placements}` | 0.15 |
| N_B-3 | `extract_class_hierarchy` | 遍历模型 `named_modules()`，提取 `{module_path: (class_name, source_file, bases)}` | 0.2 |
| N_B-4 | `extract_mesh_code` | 从 `DeviceMesh` 实例生成 mesh 构建代码字符串 | 0.15 |
| **N_C: ScriptCodeGenerator (wrapper)** | | | **2.0** |
| N_C-1 | `load_hf_sources` | 加载 HF 原始建模代码（文件查找优先，import 回退），与 patchgen 的 `get_module_source` 等价 | 0.3 |
| N_C-2 | `generate_wrapper_script` 主流程 | 编排 HEADER → IMPORTS → MESH → CLASSES → SHARDING → MAIN → TEST 各阶段的生成 | 0.5 |
| N_C-3 | Wrapper 类生成 | 为每个有 boundary 的模块生成 wrapper 子类：`__init__` 构建 `PrecompiledBoundary`，`forward` 模板化调用 `pre_forward`/`post_forward` | 0.5 |
| N_C-4 | 类名替换映射 | AST 解析容器类的 `__init__`，将 `ClassName(...)` 替换为 `ClassName_TP(...)` | 0.4 |
| N_C-5 | `_local_ctx` 生成 | 在 wrapper 类中生成 `_local_params_context` 等价代码 | 0.3 |
| **N_D: ScriptCodeGenerator (inline)** | | | **2.5** |
| N_D-1 | forward AST 解析 | 解析原始 `forward` 方法的 AST，识别参数和 `return` 语句 | 0.5 |
| N_D-2 | `insert_boundary_calls` | 在 forward body 首尾插入 `pre_forward`/`post_forward` 调用 AST 节点 | 0.8 |
| N_D-3 | 变量名推断 | 识别 `hidden_states` 参数名 + 输出变量名 | 0.3 |
| N_D-4 | 多 return 处理 | 对 early return、多输出等模式正确处理 `post_forward` 插入位置 | 0.5 |
| N_D-5 | `ast.unparse` + 注释保留 | 生成保持可读性和原始注释的内联代码 | 0.4 |
| **N_E: PrecompiledBoundary 内联副本** | | | **0.5** |
| N_E-1 | 独立 `RedistOp` 类 | 在生成脚本中嵌入 `RedistOp` dataclass（与 05 文档 §7 一致） | 0.15 |
| N_E-2 | 独立 `PrecompiledBoundary` 类 | 在生成脚本中嵌入 `PrecompiledBoundary` 类（~60 行） | 0.2 |
| N_E-3 | 可选 import 模式 | `inline_boundary_class=false` 时从 hyper_parallel import（版本锁定） | 0.15 |
| **N_F: 输出与格式化** | | | **0.8** |
| N_F-1 | `ruff` 格式化集成 | 调用 `ruff check --fix` + `ruff format`，与 patchgen 的 `ruff_fix_and_format` 等价 | 0.3 |
| N_F-2 | Unified diff 生成 | 生成脚本 vs 原始 HF 代码的 unified diff（与 patchgen 的 `build_unified_diff` 一致） | 0.2 |
| N_F-3 | Smoke test 生成 | 生成 `_smoke_test()` 函数：随机输入 forward + 输出校验 | 0.3 |
| **N_G: 集成与 CLI** | | | **1.0** |
| N_G-1 | `ModelScriptGenerator` 主类 | 统一入口，调度 Extractor + Codegen | 0.3 |
| N_G-2 | `generate_model_script` 便捷函数 | 顶层 API，from `hyper_models/components/distributed/__init__.py` | 0.1 |
| N_G-3 | 配置自动触发 | `_build_model()` 末尾调用 scriptgen（如果 `cfg.scriptgen.enabled`） | 0.2 |
| N_G-4 | CLI 命令 | `hyper-parallel generate-script --config train.yaml` 或 `--model-path Qwen/Qwen3.5-4B --tp-size 4` | 0.4 |
| **N_H: 测试** | | | **1.0** |
| N_H-1 | Wrapper 方案正确性测试 | 生成脚本执行结果 vs 原始 wrapped 模型 forward 结果（atol 容差校验） | 0.4 |
| N_H-2 | Inline 方案正确性测试 | 同上，对 inline 方案 | 0.3 |
| N_H-3 | 独立运行测试 | 验证生成脚本可在 `torchrun` 下独立运行（无 hyper_parallel import） | 0.3 |
| **总计** | | | **9.1 AI人·日** |

### 11.2 与各设计文档的关系

| 依赖文档 | 依赖内容 | 方式 |
|---------|---------|------|
| [01_hf_compatibility_layer.md] §6.3 | `_build_model` 末尾调用 scriptgen | 扩展现有函数 |
| [05_dual_mode_dtensor_parallel_strategy.md] §7 | `PrecompiledBoundary` / `RedistOp` 的实现 | 内联精简副本 |
| [05_dual_mode_dtensor_parallel_strategy.md] §6 | `_wrap_forward` / `_local_params_context` | 生成等价代码 |
| 05 §3 | `ShardingPlan` / `ModuleShardingSpec` 数据结构 | 读取 boundary_specs |
| 01 §6.4 | PEFT 注入（LoRA 等） | scriptgen 在 PEFT 注入后运行（LoRA 参数也需要生成） |
| VeOmni patchgen | AST 解析 / 源码加载 / ruff 格式化 / drift check | 复用设计模式 |

---

## 12. 与其他文档的关系

```
main()
│
├─ ① ConfigNode / RecipeConfig (01 §2-3)
├─ ② 分布式基础设施 (06 §2)     ← DistributedSetup / MeshContext
├─ ③ HF 兼容层 (01 §5-6)        ← MODEL_ARCH_MAPPING / from_pretrained
├─ ④ _build_model (01 §6.3)
│   ├─ ④.3.1 分布式环境初始化 (06 §3)
│   ├─ ④.3.2 instantiate_infrastructure (01 §8)
│   ├─ ④.3.5.2 _init_model (01 §7)
│   ├─ ④.3.5.4 权重加载 (01 §10)
│   ├─ ④.3.5.7 ShardingPlanner.plan() (05 §5)     ← 编译期规划
│   ├─ ④.3.5.8 apply_sharding_plan() (05 §6/§7/§8) ← 运行时应用
│   └─ ④.3.5.9 scriptgen.generate() ★ 本文档       ← 脚本生成（可选）
│
├─ ⑤ run_train_validation_loop() (03 §6)
│   └─ ⑤.1.2 _forward_backward_step()
│       └─ PrecompiledBoundary 执行 (05 §7)
│
├─ ⑥ Checkpointer (04 §5)
├─ ⑦ Optimizer/Loss (03 §9-10)
├─ ⑧ Data Pipeline (02 §3)
└─ ⑨ 模型实现 (01 §7 + 05 §4)
```

---

## 13. 总结

`ModelScriptGenerator` 是 hyper_parallel 并行能力的"出口"——将运行时动态包装的模型还原为一份**可读、可调试、可独立运行**的 Python 源码。它与 VeOmni 的 patchgen 共享 AST 操作基础设施，但方向相反：patchgen 从声明到源码（正向生成），scriptgen 从运行时状态回溯源码（反向生成）。

核心价值：
1. **调试效率**：在生成的脚本中单步调试每个通信操作，而不是在 monkey-patch wrapper 的黑盒中猜
2. **代码审查**：生成 diff 清晰展示通信注入的位置，便于审查分片策略的正确性
3. **环境独立**：产出脚本可在最小依赖下运行，适合 CI 和离线分析
4. **与 patchgen 互补**：patchgen 做静态算子优化，scriptgen 做动态通信可视化
